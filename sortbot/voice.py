"""Voice I/O: ElevenLabs TTS/STT with graceful fallbacks, plus a persistent RulesStore.

Two queues live here (see the MULTI-QUEUE block at the top of sortbot/main.py):
  q_heard -- everything the human says (mic PTT / Listening toggle / the HUD text box) -> push()/drain().
             The "luna-chat" worker is its consumer whenever a Session is running.
  q_say   -- lines to speak; ONE TTS worker serializes synth + playback so the bot never talks over
             itself. speak(text, priority=True) drops the whole stale backlog first, so a fresh
             conversational reply always wins over queued planner chatter.
The regex pre-filter (urgent_kind / bare_command) runs BEFORE any LLM: a stop/pause must never wait on
a model, and a bare "open"/"close"/"home" needs no model at all.

Deviations from spec: no sounddevice/pyaudio is installed, so mic capture uses `ffmpeg -f avfoundation`
(macOS) in fixed 4 s chunks; push-to-talk is not implemented (no keyboard lib). Without ffmpeg or an
ELEVENLABS_API_KEY the STT side falls back to a stdin line reader. TTS playback uses afplay/ffplay,
else the text is just logged.
"""
from __future__ import annotations

import asyncio
import base64
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
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger("sortbot.voice")
REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_FILE = REPO_ROOT / "sortbot" / "calib" / "rules.json"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_TTS_MODEL = "eleven_turbo_v2_5"
DEFAULT_STT_MODEL = "scribe_v2"
CHUNK_SECONDS = 4.0            # legacy fallback: fixed-length ffmpeg captures, transcribed one at a time
STREAM_MODEL = "scribe_v2_realtime"   # WebSocket streaming STT (partial + endpointed committed transcripts)
STREAM_SAMPLE_RATE = 16000
STREAM_CHUNK_BYTES = 3200      # 100 ms of pcm_s16le @ 16 kHz
#: How long VAD must hear silence before it endpoints the utterance (API range 0.3 .. 3.0 s). Measured:
#: 0.5 s -> the transcript lands ~520 ms after you stop talking, 0.8 s -> ~650 ms. 0.7 is the compromise:
#: the extra ~130 ms buys tolerance for a mid-sentence pause, and a sentence split in half costs a whole
#: extra reply (and Luna answering only the second half), which is far more visible than 130 ms.
STREAM_VAD_SILENCE_S = 0.7
#: an identical utterance arriving twice inside this window is a double delivery (push-to-talk firing
#: while the Listening stream is also running), not the human saying the same thing twice
DEDUPE_S = 2.5


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
# --- urgent pre-filter: E-STOP-shaped speech NEVER waits for an LLM (see main.ChatWorker) ---
_URGENT_PATTERNS = [
    (r"^(stop|halt|freeze|abort|cancel|quit|e-?stop|emergency)\b", "stop"),
    (r"^(no|nope)[ ,]+(stop|halt|freeze)\b", "stop"),
    (r"\b(stop|halt|freeze)\s+(now|right now|immediately|everything|please)\b", "stop"),
    (r"^(pause|wait|hold on|hold up|hang on|hold it|stand ?by|one (second|sec|moment))\b", "pause"),
    (r"\bpause\s+(now|please|for a (sec|second|moment))\b", "pause"),
]
#: bare one/two-word commands the loop can execute itself -- no model call, no conversation
_BARE_COMMANDS = {"open": "open", "release": "open", "drop": "open", "let": "open",
                  "close": "close", "home": "home", "retract": "home"}

_ACTION_PATTERNS = [
    r"^(stop|halt|wait|pause|freeze|home|retract|go home)\b",
    r"^(open|close|release|drop|let go)\b",
    r"^(undo|redo|skip|next|continue|resume|done|finish)\b",
    r"^(pick|grab|take|get)\b",
    r"^(turn|rotate|lift|lower|raise)\b",
    r"^(that'?s|this is)\s+(wrong|not)\b",
    r"^(no|nope|wrong)\b",
]


#: leading filler / the bot's name: "Luna, stop!" and "hey Luna, open" must hit the same fast path as
#: "stop" and "open" -- people address the robot, and an E-STOP cannot depend on them not doing so.
_ADDRESS_RE = re.compile(r"^(?:hey|hi|yo|ok|okay|please|um|uh|luna|robot|bot|sortbot)\b[\s,;:]*")


def _norm(cmd: str) -> str:
    return re.sub(r"\s+", " ", cmd.strip().lower().rstrip(".!"))


def _addressed(cmd: str) -> str:
    """_norm() with any leading address stripped ("hey luna, stop" -> "stop")."""
    t = _norm(cmd)
    while True:
        t2 = _ADDRESS_RE.sub("", t, count=1).strip()
        if t2 == t or not t2:
            return t
        t = t2


def urgent_kind(cmd: str) -> str:
    """Regex-only pre-filter: "none" | "stop" | "pause". Must stay cheap and synchronous -- the chat
    worker calls it before deciding whether an utterance is worth an LLM round trip at all."""
    t = _addressed(cmd)
    for pat, kind in _URGENT_PATTERNS:
        if re.search(pat, t):
            return kind
    return "none"


def bare_command(cmd: str) -> str | None:
    """"open" / "close it" / "go home" -> the gripper/home command the ACTION loop should run, else None.
    Only short bare phrases: "drop the red one in the left bin" is a sentence for the chat model."""
    t = _addressed(cmd)
    words = t.split()
    if not (1 <= len(words) <= 2):
        return None
    w = words[0]
    if w == "go" and len(words) == 2:
        w = words[1]
    return _BARE_COMMANDS.get(w)


def classify(cmd: str) -> Intent:
    t = _norm(cmd)
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

    def delete(self, i: int) -> str | None:
        """Remove rule at index i; returns the removed rule or None if out of range."""
        with self._lock:
            if not 0 <= i < len(self._rules):
                return None
            r = self._rules.pop(i)
            self._save()
            return r

    def move(self, i: int, delta: int) -> bool:
        """Move rule at index i by delta (-1 = up / earlier, +1 = down / later)."""
        with self._lock:
            j = i + delta
            if not (0 <= i < len(self._rules) and 0 <= j < len(self._rules) and i != j):
                return False
            self._rules.insert(j, self._rules.pop(i))
            self._save()
            return True

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

    def __init__(self, voice_id: str = DEFAULT_VOICE_ID, stdin=None, force_text: bool = False,
                 tts_model: str = DEFAULT_TTS_MODEL, stt_model: str = DEFAULT_STT_MODEL):
        load_dotenv(REPO_ROOT / ".env")
        self.voice_id = voice_id
        self.tts_model, self.stt_model = tts_model, stt_model
        self.last_transcript = ""
        self.last_said = ""
        self._q: queue.Queue[str] = queue.Queue()  # q_heard
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
        self.mode = "text"  # mic NEVER auto-starts; toggle with mic_on()/mic_off()
        self._mic_ok = bool(self._client and shutil.which("ffmpeg") and sys.platform == "darwin")
        self._mic_thread = None
        # --- streaming STT (see _stream_loop): which path the mic is actually using right now ---
        self.stt_mode = ""              # "" (mic off) | "stream" | "chunk"
        self.last_partial = ""          # newest interim transcript, for the HUD
        #: fired from an INTERIM transcript the moment stop/pause-shaped speech is recognised, so an
        #: E-STOP never waits for endpointing, let alone for a model. Set by main.Session.
        self.urgent_hook = None         # callable(kind: "stop"|"pause", text: str)
        self._partial_urgent = None     # the urgent kind already fired for the utterance in progress
        self._last_push: tuple[str, float] = ("", 0.0)
        self._stream_proc = None
        self.stream_model = STREAM_MODEL
        self._player = next((p for p in ("ffplay", "afplay") if shutil.which(p)), None)
        # ONE speech worker: synth + playback are serialized so the bot never talks over itself,
        # and the sorting loop never blocks on the TTS network call. Backlog is capped: stale
        # lines are dropped rather than droning on long after the moment has passed.
        self._speak_q: queue.Queue[str] = queue.Queue()
        self._speak_thread: threading.Thread | None = None
        log.info("VoiceIO mode=%s tts=%s player=%s", self.mode, bool(self._client), self._player)

    # -- lifecycle
    def start(self) -> None:
        target = self._mic_loop if self.mode == "mic" else self._text_loop
        self._thread = threading.Thread(target=target, daemon=True, name="voice-listener")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.mic_off()

    @property
    def listening(self) -> bool:
        return bool(self._mic_thread and self._mic_thread.is_alive())

    def mic_on(self) -> str:
        """Start continuous mic capture. Only ever called from an explicit user toggle."""
        if not self._mic_ok:
            return "mic unavailable (need ELEVENLABS_API_KEY + ffmpeg on macOS)"
        if self.listening:
            return "already listening"
        self._mic_stop = threading.Event()
        self.mode = "mic"
        self._mic_thread = threading.Thread(target=self._mic_loop, daemon=True, name="voice-mic")
        self._mic_thread.start()
        return "listening"

    def mic_off(self) -> str:
        if self._mic_thread:
            getattr(self, "_mic_stop", self._stop).set()
            # the streaming path blocks on ffmpeg's stdout: killing the capture is what unblocks it
            self._kill_stream_proc()
            self._mic_thread.join(timeout=CHUNK_SECONDS + 2)
            self._mic_thread = None
        self.stt_mode = ""
        self.mode = "text"
        return "mic off"

    def drain(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def push(self, text: str) -> None:
        """q_heard: inject an utterance (streaming STT, push-to-talk, the HUD text box, tests).

        Push-to-talk and the Listening stream can both hear the same sentence (holding the mic button
        while the toggle is on), so an identical utterance repeated inside DEDUPE_S is dropped as a
        double delivery. A human really saying the same thing twice that fast is not a case worth
        serving, and a duplicate would cost a second model call and a second spoken reply."""
        text = text.strip()
        if not text:
            return
        now, key = time.time(), " ".join(text.lower().split())
        if key == self._last_push[0] and now - self._last_push[1] < DEDUPE_S:
            log.info("dropping duplicate utterance %r (heard twice within %.1fs)", text, DEDUPE_S)
            return
        self._last_push = (key, now)
        self._q.put(text)

    def peek(self) -> list[str]:
        """Snapshot of pending (undrained) commands, oldest first."""
        return list(self._q.queue)

    # -- listeners
    def _text_loop(self) -> None:
        for line in self._stdin:
            if self._stop.is_set():
                break
            self.push(line)

    # ---------------------------------------------------------------- listening
    def _mic_loop(self) -> None:
        """The Listening toggle's thread. Prefers STREAMING STT and falls back to the fixed-chunk path
        if the socket cannot be opened or dies -- the toggle keeps working either way, and the mode in
        use is logged and served in /state (voice.stt_mode)."""
        if self._stream_ok():
            try:
                asyncio.run(self._stream_loop())
                self.stt_mode = ""
                return
            except Exception as e:  # noqa: BLE001 - any streaming failure falls back, never kills the mic
                log.warning("streaming STT unavailable (%s); falling back to %.0f s chunks",
                            e, CHUNK_SECONDS)
            finally:
                self._kill_stream_proc()
        stop = getattr(self, "_mic_stop", self._stop)
        if stop.is_set() or self._stop.is_set():
            self.stt_mode = ""
            return
        self.stt_mode = "chunk"
        log.info("mic: fixed %.0f s chunks (transcript arrives on the chunk boundary)", CHUNK_SECONDS)
        self._chunk_loop()
        self.stt_mode = ""

    def _stream_ok(self) -> bool:
        """Streaming needs a key, ffmpeg, macOS avfoundation and an SDK new enough to have realtime STT."""
        if not (self._client and shutil.which("ffmpeg") and sys.platform == "darwin"):
            return False
        try:
            from elevenlabs.realtime.connection import RealtimeEvents  # noqa: F401
            from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy  # noqa: F401
        except Exception as e:  # noqa: BLE001
            log.info("realtime STT not available in this elevenlabs SDK (%s)", e)
            return False
        return True

    # -- streaming path
    async def _stream_loop(self) -> None:
        """Continuous ffmpeg capture -> ElevenLabs realtime STT over a WebSocket.

        Two kinds of transcript come back and they are used for DIFFERENT things:
          partial_transcript   -- interim, arrives WHILE the human is still talking. Only the urgent
                                  pre-filter reads these: "stop" must fire the moment the word is
                                  recognised, not after endpointing and certainly not after a model call.
          committed_transcript -- endpointed by VAD once they stop talking (~0.3 s), the real utterance.
                                  This is what goes onto q_heard for the chat worker.
        """
        from elevenlabs.client import AsyncElevenLabs
        from elevenlabs.realtime.connection import RealtimeEvents
        from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy

        stop = getattr(self, "_mic_stop", self._stop)
        client = AsyncElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
        conn = await client.speech_to_text.realtime.connect({
            "model_id": self.stream_model,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": STREAM_SAMPLE_RATE,
            "commit_strategy": CommitStrategy.VAD,
            "vad_silence_threshold_secs": STREAM_VAD_SILENCE_S,
        })
        conn.on(RealtimeEvents.PARTIAL_TRANSCRIPT, lambda d: self._on_partial(str(d.get("text", ""))))
        conn.on(RealtimeEvents.COMMITTED_TRANSCRIPT, lambda d: self._on_final(str(d.get("text", ""))))
        for ev in (RealtimeEvents.ERROR, RealtimeEvents.AUTH_ERROR, RealtimeEvents.QUOTA_EXCEEDED,
                   RealtimeEvents.TRANSCRIBER_ERROR, RealtimeEvents.INVALID_REQUEST):
            conn.on(ev, lambda d, ev=ev: log.warning("streaming STT %s: %s", ev.value, str(d)[:200]))
        self.stt_mode = "stream"
        log.info("mic: STREAMING STT (%s), endpointed after %.1f s of silence",
                 self.stream_model, STREAM_VAD_SILENCE_S)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-loglevel", "error", "-f", "avfoundation", "-i", ":0",
            "-ar", str(STREAM_SAMPLE_RATE), "-ac", "1", "-f", "s16le", "pipe:1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        self._stream_proc = proc
        try:
            while not (self._stop.is_set() or stop.is_set()):
                chunk = await proc.stdout.read(STREAM_CHUNK_BYTES)
                if not chunk:
                    raise RuntimeError("mic capture ended (ffmpeg closed the stream)")
                await self.feed_audio(conn, chunk)
        finally:
            self._kill_stream_proc()
            try:
                await conn.close()
            except Exception as e:  # noqa: BLE001
                log.debug("streaming STT close: %s", e)

    @staticmethod
    async def feed_audio(conn, pcm: bytes) -> None:
        """One pcm_s16le @ 16 kHz chunk to the socket. Split out so a test (and the latency probe) can
        drive the real connection with recorded audio instead of a microphone."""
        await conn.send({"audio_base_64": base64.b64encode(pcm).decode()})

    def _kill_stream_proc(self) -> None:
        proc, self._stream_proc = self._stream_proc, None
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except Exception as e:  # noqa: BLE001
                log.debug("ffmpeg kill: %s", e)

    def _on_partial(self, text: str) -> None:
        """INTERIM transcript. The ONLY thing allowed to act on it is the urgent pre-filter -- a partial
        is not a finished sentence, so it never reaches q_heard, but "stop" cannot wait for the sentence
        to finish. Fires at most once per utterance per kind."""
        text = text.strip()
        if not text:
            return
        self.last_partial = text
        kind = urgent_kind(text)
        if kind == "none" or kind == self._partial_urgent:
            return
        self._partial_urgent = kind
        log.warning("URGENT %r heard mid-sentence: %r", kind, text)
        hook = self.urgent_hook
        if hook is not None:
            try:
                hook(kind, text)
            except Exception as e:  # noqa: BLE001 - the mic must survive a bad hook
                log.warning("urgent hook: %s", e)

    def _on_final(self, text: str) -> None:
        """Endpointed utterance -> q_heard, exactly like a push-to-talk clip."""
        self._partial_urgent = None
        self.last_partial = ""
        text = text.strip()
        if not text:
            return
        self.last_transcript = text
        log.info("heard: %s", text)
        self.push(text)

    # -- fallback path (pre-streaming behaviour, unchanged)
    def _chunk_loop(self) -> None:
        stop = getattr(self, "_mic_stop", self._stop)
        while not (self._stop.is_set() or stop.is_set()):
            wav = self._record(CHUNK_SECONDS)
            if wav is None:
                log.warning("mic capture failed; mic off")
                self.mode = "text"
                self._mic_thread = None
                return
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
        with open(wav_path, "rb") as f:
            return self.transcribe_bytes(f.read(), "audio/wav")

    def transcribe_bytes(self, data: bytes, mime: str = "audio/webm") -> str:
        """STT on an in-memory clip (HUD push-to-talk posts webm/opus; the API takes the bytes as-is)."""
        if not self._client:
            raise RuntimeError("ElevenLabs unavailable (no ELEVENLABS_API_KEY)")
        ext = (mime.split("/")[-1].split(";")[0] or "webm") if "/" in mime else "webm"
        res = self._client.speech_to_text.convert(
            model_id=self.stt_model, file=(f"clip.{ext}", data, mime), tag_audio_events=False)
        text = (getattr(res, "text", "") or "").strip()
        text = "" if re.fullmatch(r"[\s\W]*", text) else text
        if text:
            self.last_transcript = text
        return text

    # -- speaking
    def speak(self, text: str, priority: bool = False) -> None:
        """q_say: non-blocking, queued for the single speech worker (serialized synth + playback, so the
        bot never talks over itself). priority=True is the CONVERSATION channel: a fresh reply to the human
        drops the ENTIRE stale backlog so it is spoken next, instead of queueing behind planner chatter."""
        print(f"[robot says] {text}", flush=True)
        self.last_said = text
        if not self._client or not self._player:
            return
        # keep speech current: drop the oldest unspoken lines (all of them for a fresh reply)
        while self._speak_q.qsize() >= (1 if priority else 2):
            try:
                dropped = self._speak_q.get_nowait()
                log.info("TTS backlog: dropping %r", dropped)
            except queue.Empty:
                break
        self._speak_q.put(text)
        if self._speak_thread is None or not self._speak_thread.is_alive():
            self._speak_thread = threading.Thread(target=self._speak_loop, daemon=True, name="voice-tts")
            self._speak_thread.start()

    def pending_say(self) -> list[str]:
        """q_say snapshot (undrained lines), oldest first -- for the HUD."""
        return list(self._speak_q.queue)

    def _speak_loop(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._speak_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # stream: first audio chunk plays while the rest is still being synthesized
                chunks = self._client.text_to_speech.stream(
                    self.voice_id, text=text, model_id=self.tts_model, output_format="mp3_22050_32")
            except Exception as e:  # noqa: BLE001
                log.warning("TTS failed: %s", e)
                continue
            self._play(chunks)  # blocking on purpose: one utterance at a time, never talking over itself

    def _play(self, audio) -> None:
        """audio: bytes or an iterator of byte chunks. ffplay streams (starts on chunk 1); afplay
        needs a complete file, so an iterator is buffered first."""
        try:
            if self._player == "ffplay":
                proc = subprocess.Popen(["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", "-"],
                                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                try:
                    for chunk in ([audio] if isinstance(audio, bytes) else audio):
                        if self._stop.is_set():
                            break
                        proc.stdin.write(chunk)
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                proc.wait(timeout=60)
                return
            data = audio if isinstance(audio, bytes) else b"".join(audio)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(data)
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

    fake_stdin = io.StringIO("put the red pieces with the black ones\nstop\nscrews go in the left bin\n\n")
    v = VoiceIO(stdin=fake_stdin, force_text=True)
    assert v.mode == "text"
    v.start()
    time.sleep(0.3)
    cmds = v.drain()
    assert cmds == ["put the red pieces with the black ones", "stop", "screws go in the left bin"], cmds
    assert v.drain() == []
    for c in cmds:
        i = classify(c)
        if i.kind == "rule":
            rules.append(i.text)
    kinds = [classify(c).kind for c in cmds]
    assert kinds == ["rule", "action", "rule"], kinds
    assert classify("that's wrong, open the gripper").kind == "action"
    assert classify("round things on the left").kind == "rule"
    assert classify("hello there").kind == "unknown"
    assert RulesStore(tmp).list() == rules.list() == cmds[0::2]
    rules.append(cmds[0])  # dedupe
    assert len(RulesStore(tmp).list()) == 2
    rules.append("third rule here")
    assert rules.move(2, -1) and rules.list()[1] == "third rule here"
    assert not rules.move(0, -1) and not rules.move(9, 1)
    assert rules.delete(1) == "third rule here" and rules.delete(99) is None
    assert RulesStore(tmp).list() == rules.list() and len(rules.list()) == 2
    assert v.peek() == [] and v.last_transcript == ""
    # regex pre-filter: urgent speech never waits on a model, bare commands never need one
    for s, k in [("stop", "stop"), ("STOP!", "stop"), ("stop right now", "stop"), ("e-stop", "stop"),
                 ("no, stop", "stop"), ("pause", "pause"), ("hold on a moment", "pause"),
                 ("wait", "pause"), ("what are you doing", "none"), ("put red things left", "none"),
                 ("Luna, stop!", "stop"), ("hey luna, pause", "pause"), ("please stop", "stop"),
                 ("luna", "none"), ("don't stop until the table is clear", "none")]:
        assert urgent_kind(s) == k, (s, urgent_kind(s), k)
    for s, c in [("open", "open"), ("close it", "close"), ("go home", "home"), ("release", "open"),
                 ("drop the red one in the left bin", None), ("hello", None), ("home", "home"),
                 ("Luna, open", "open"), ("hey luna go home", "home")]:
        assert bare_command(s) == c, (s, bare_command(s), c)
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
