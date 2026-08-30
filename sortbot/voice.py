"""Voice I/O: ElevenLabs TTS/STT with graceful fallbacks, plus a persistent RulesStore.

Deviations from spec: no sounddevice/pyaudio is installed, so mic capture uses `ffmpeg -f avfoundation`
(macOS) in fixed 4 s chunks; push-to-talk is not implemented (no keyboard lib). Without ffmpeg or an
ELEVENLABS_API_KEY the STT side falls back to a stdin line reader. TTS playback uses afplay/ffplay,
else the text is just logged.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger("sortbot.voice")
REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_FILE = REPO_ROOT / "sortbot" / "calib" / "rules.json"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
CHUNK_SECONDS = 4.0


# ---------------------------------------------------------------- classifier
@dataclass
class Intent:
    kind: str  # "rule" | "action" | "unknown"
    text: str  # normalised rule text or the raw command


_RULE_PATTERNS = [
    r"\b(put|place|sort|move)\b.+\b(with|into|in|to|next to|on)\b.+",
    r".+\b(go|goes|belong|belongs)\b.+\b(with|into|in|to|on)\b.+",
    r".+\b(on|to) the (left|right|top|bottom|front|back)\b.*",
    r"\b(always|never|from now on|don't|do not|only)\b.+",
    r".+\b(are|is|count as|counts as)\b.+",
]
_ACTION_PATTERNS = [
    r"^(stop|halt|wait|pause|freeze|home|retract|go home)\b",
    r"^(open|close|release|drop|let go)\b",
    r"^(undo|redo|skip|next|continue|resume|done|finish)\b",
    r"^(pick|grab|take|get)\b",
    r"^(turn|rotate|lift|lower|raise)\b",
    r"^(that'?s|this is)\s+(wrong|not)\b",
    r"^(no|nope|wrong)\b",
]


def classify(cmd: str) -> Intent:
    t = re.sub(r"\s+", " ", cmd.strip().lower().rstrip(".!"))
    if not t:
        return Intent("unknown", "")
    for p in _ACTION_PATTERNS:
        if re.search(p, t):
            return Intent("action", t)
    for p in _RULE_PATTERNS:
        if re.fullmatch(p, t):
            return Intent("rule", t)
    return Intent("unknown", t)


# ---------------------------------------------------------------- rules
class RulesStore:
    def __init__(self, path: Path = RULES_FILE):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rules: list[str] = []
        if self.path.exists():
            try:
                self._rules = [str(r) for r in json.loads(self.path.read_text())]
            except (json.JSONDecodeError, TypeError):
                log.warning("bad rules file %s, starting empty", self.path)

    def append(self, rule: str) -> None:
        rule = rule.strip()
        with self._lock:
            if rule and rule not in self._rules:
                self._rules.append(rule)
                self._save()

    def clear(self) -> None:
        with self._lock:
            self._rules = []
            self._save()

    def list(self) -> list[str]:
        with self._lock:
            return list(self._rules)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._rules, indent=2))


# ---------------------------------------------------------------- voice io
class VoiceIO:
    """start() launches a listener thread; drain() returns commands heard since last call."""

    def __init__(self, voice_id: str = DEFAULT_VOICE_ID, stdin=None, force_text: bool = False):
        load_dotenv(REPO_ROOT / ".env")
        self.voice_id = voice_id
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stdin = stdin or sys.stdin
        self._client = None
        key = os.environ.get("ELEVENLABS_API_KEY")
        if key:
            try:
                from elevenlabs.client import ElevenLabs

                self._client = ElevenLabs(api_key=key)
            except Exception as e:  # noqa: BLE001
                log.warning("elevenlabs client unavailable: %s", e)
        self.mode = "text"
        if not force_text and self._client and shutil.which("ffmpeg") and sys.platform == "darwin":
            self.mode = "mic"
        self._player = next((p for p in ("afplay", "ffplay") if shutil.which(p)), None)
        log.info("VoiceIO mode=%s tts=%s player=%s", self.mode, bool(self._client), self._player)

    # -- lifecycle
    def start(self) -> None:
        target = self._mic_loop if self.mode == "mic" else self._text_loop
        self._thread = threading.Thread(target=target, daemon=True, name="voice-listener")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self.mode == "mic":
            self._thread.join(timeout=CHUNK_SECONDS + 2)

    def drain(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def push(self, text: str) -> None:
        """Inject a command (HUD / tests)."""
        if text.strip():
            self._q.put(text.strip())

    # -- listeners
    def _text_loop(self) -> None:
        for line in self._stdin:
            if self._stop.is_set():
                break
            self.push(line)

    def _mic_loop(self) -> None:
        while not self._stop.is_set():
            wav = self._record(CHUNK_SECONDS)
            if wav is None:
                log.warning("mic capture failed, switching to text input")
                self.mode = "text"
                return self._text_loop()
            try:
                text = self.transcribe(wav)
            except Exception as e:  # noqa: BLE001
                log.warning("STT failed: %s", e)
                continue
            finally:
                Path(wav).unlink(missing_ok=True)
            if text:
                log.info("heard: %s", text)
                self.push(text)

    @staticmethod
    def _record(seconds: float) -> str | None:
        path = tempfile.mktemp(suffix=".wav")
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "avfoundation", "-i", ":0",
               "-t", str(seconds), "-ar", "16000", "-ac", "1", path]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=seconds + 10)
        except (subprocess.TimeoutExpired, OSError):
            return None
        return path if r.returncode == 0 and Path(path).exists() else None

    def transcribe(self, wav_path: str) -> str:
        if not self._client:
            return ""
        with open(wav_path, "rb") as f:
            res = self._client.speech_to_text.convert(model_id="scribe_v1", file=f, tag_audio_events=False)
        text = (getattr(res, "text", "") or "").strip()
        return "" if re.fullmatch(r"[\s\W]*", text) else text

    # -- speaking
    def speak(self, text: str) -> None:
        print(f"[robot says] {text}", flush=True)
        if not self._client or not self._player:
            return
        try:
            audio = b"".join(self._client.text_to_speech.convert(
                self.voice_id, text=text, model_id="eleven_turbo_v2_5", output_format="mp3_22050_32"))
        except Exception as e:  # noqa: BLE001
            log.warning("TTS failed: %s", e)
            return
        threading.Thread(target=self._play, args=(audio,), daemon=True).start()

    def _play(self, audio: bytes) -> None:
        try:
            if self._player == "ffplay":
                subprocess.run(["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", "-"],
                               input=audio, capture_output=True, timeout=60)
            else:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(audio)
                subprocess.run(["afplay", f.name], capture_output=True, timeout=60)
                Path(f.name).unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            log.warning("playback failed: %s", e)


# ---------------------------------------------------------------- selftest
def _selftest() -> None:
    import io
    import time

    tmp = Path(tempfile.mkdtemp()) / "rules.json"
    rules = RulesStore(tmp)
    assert rules.list() == []

    fake_stdin = io.StringIO("put the red wires with the black ones\nstop\nscrews go in the left bin\n\n")
    v = VoiceIO(stdin=fake_stdin, force_text=True)
    assert v.mode == "text"
    v.start()
    time.sleep(0.3)
    cmds = v.drain()
    assert cmds == ["put the red wires with the black ones", "stop", "screws go in the left bin"], cmds
    assert v.drain() == []
    for c in cmds:
        i = classify(c)
        if i.kind == "rule":
            rules.append(i.text)
    kinds = [classify(c).kind for c in cmds]
    assert kinds == ["rule", "action", "rule"], kinds
    assert classify("that's wrong, open the gripper").kind == "action"
    assert classify("resistors on the left").kind == "rule"
    assert classify("hello there").kind == "unknown"
    assert RulesStore(tmp).list() == rules.list() == cmds[0::2]
    rules.append(cmds[0])  # dedupe
    assert len(RulesStore(tmp).list()) == 2
    v.speak("selftest speaking")  # logs only unless key + player available
    v.stop()
    print("selftest OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--selftest" in sys.argv:
        _selftest()
    else:  # interactive: type or speak, prints classified commands
        v, rules = VoiceIO(), RulesStore()
        v.start()
        print(f"mode={v.mode}; rules={rules.list()}; Ctrl-C to quit")
        try:
            while True:
                for c in v.drain():
                    i = classify(c)
                    print(f"{i.kind}: {i.text}")
                    if i.kind == "rule":
                        rules.append(i.text)
                threading.Event().wait(0.2)
        except KeyboardInterrupt:
            v.stop()
