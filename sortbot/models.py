"""Model registry for the HUD MODELS group.

get(current) -> {openai: [...ids], elevenlabs: {tts, stt, voices: [{id,name}], current}, current: {...}, notes: [...]}.
OpenAI ids come from client.models.list() filtered to the vision-capable families (gpt-5 / gpt-4.1 / gpt-4o /
o3 / o4), cached 5 min; ElevenLabs voices from client.voices.search() (first page, ~30). A missing key or API
error lands in `notes`, never raises. yaml_set() persists a choice into config.yaml with a minimal text edit
so comments survive.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_S = 300.0  # 5 min
OPENAI_PREFIXES = ("gpt-5", "gpt-4.1", "gpt-4o", "o3", "o4")
OPENAI_EXCLUDE = ("audio", "realtime", "transcribe", "tts", "search", "moderation",
                  "embedding", "image", "deep-research", "codex")
ELEVENLABS_TTS = ["eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2", "eleven_v3"]
ELEVENLABS_STT = ["scribe_v2", "scribe_v1"]
PROVIDERS = ("openai", "elevenlabs_tts", "elevenlabs_stt", "elevenlabs_voice")


def _errmsg(e: Exception) -> str:
    """Short human-readable note for an API failure (ElevenLabs ApiError dumps full headers in str())."""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        det = body.get("detail")
        if isinstance(det, dict) and det.get("message"):
            return str(det["message"])
    return str(e)[:200]


def yaml_set(path: Path, section: str, key: str, value: str) -> None:
    """Set `section:\\n  key: value` in a 2-level yaml file by editing the text minimally (comments preserved).
    Missing key/section lines are inserted; everything else is left byte-for-byte as it was."""
    path = Path(path)
    lines = path.read_text().splitlines()
    key_re = re.compile(rf"^(\s+{re.escape(key)}\s*:\s*)([^#]*?)(\s*#.*)?$")
    out, in_sec, done = [], False, False
    for line in lines:
        top = re.match(r"^([A-Za-z_]\w*)\s*:", line)
        if top:
            if in_sec and not done:  # leaving the section without having found the key
                out.append(f"  {key}: {value}")
                done = True
            in_sec = top.group(1) == section
            out.append(line)
            continue
        if in_sec and not done:
            m = key_re.match(line)
            if m:
                out.append(f"{m.group(1)}{value}{m.group(3) or ''}")
                done = True
                continue
        out.append(line)
    if not done:
        if not in_sec:
            out.append(f"{section}:")
        out.append(f"  {key}: {value}")
    path.write_text("\n".join(out) + "\n")


class ModelRegistry:
    """Lists selectable models; clients are created lazily from .env keys (or injected for tests)."""

    def __init__(self, openai_client=None, el_client=None):
        load_dotenv(REPO_ROOT / ".env")
        self._openai_client, self._el_client = openai_client, el_client
        self._lock = threading.Lock()
        self._openai_cache: tuple[float, list[str]] | None = None
        self._voices_cache: tuple[float, list[dict]] | None = None

    # -- clients
    def _openai(self):
        if self._openai_client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("no OPENAI_API_KEY")
            from openai import OpenAI
            self._openai_client = OpenAI()
        return self._openai_client

    def _el(self):
        if self._el_client is None:
            key = os.environ.get("ELEVENLABS_API_KEY")
            if not key:
                raise RuntimeError("no ELEVENLABS_API_KEY")
            from elevenlabs.client import ElevenLabs
            self._el_client = ElevenLabs(api_key=key)
        return self._el_client

    # -- listings
    def openai_ids(self, current: str, notes: list[str]) -> list[str]:
        with self._lock:
            c = self._openai_cache
            ids = list(c[1]) if c and time.time() - c[0] < CACHE_S else None
        if ids is None:
            try:
                ids = sorted({m.id for m in self._openai().models.list()
                              if m.id.startswith(OPENAI_PREFIXES) and not any(x in m.id for x in OPENAI_EXCLUDE)})
                with self._lock:
                    self._openai_cache = (time.time(), list(ids))
            except Exception as e:  # noqa: BLE001
                notes.append(f"openai: {_errmsg(e)}")
                ids = []
        if current:  # current first (kept even if the listing failed / lacks it)
            ids = [current] + [i for i in ids if i != current]
        return ids

    def voices(self, notes: list[str]) -> list[dict]:
        with self._lock:
            c = self._voices_cache
            if c and time.time() - c[0] < CACHE_S:
                return list(c[1])
        try:
            res = self._el().voices.search(page_size=30)
            vs = [{"id": v.voice_id, "name": v.name} for v in (res.voices or [])]
            with self._lock:
                self._voices_cache = (time.time(), list(vs))
            return vs
        except Exception as e:  # noqa: BLE001
            notes.append(f"elevenlabs: {_errmsg(e)}")
            return []

    def get(self, current: dict) -> dict:
        """current: {provider: value} for the PROVIDERS above."""
        notes: list[str] = []
        return {"openai": self.openai_ids(str(current.get("openai", "")), notes),
                "elevenlabs": {"tts": list(ELEVENLABS_TTS), "stt": list(ELEVENLABS_STT),
                               "voices": self.voices(notes), "current": current.get("elevenlabs_voice", "")},
                "current": dict(current), "notes": notes}


def _selftest() -> None:
    import tempfile

    # yaml_set: value replaced, comments + unrelated lines untouched, missing keys/sections inserted
    p = Path(tempfile.mkdtemp()) / "c.yaml"
    p.write_text("# top comment\nvlm:\n  model: gpt-5   # the model\nvoice:\n  elevenlabs_voice_id: abc\nhud:\n  port: 8765\n")
    yaml_set(p, "vlm", "model", "gpt-4o")
    yaml_set(p, "voice", "tts_model", "eleven_v3")
    yaml_set(p, "extra", "k", "v")
    t = p.read_text()
    assert "  model: gpt-4o   # the model" in t and "# top comment" in t, t
    assert "elevenlabs_voice_id: abc" in t and "  tts_model: eleven_v3" in t, t
    assert t.index("tts_model") < t.index("hud:") and "extra:\n  k: v" in t, t
    import yaml as _y
    d = _y.safe_load(t)
    assert d["vlm"]["model"] == "gpt-4o" and d["voice"]["tts_model"] == "eleven_v3" and d["extra"]["k"] == "v"
    yaml_set(p, "hud", "port", "9000")  # last section, key present
    assert _y.safe_load(p.read_text())["hud"]["port"] == 9000

    # registry with fake clients
    class _O:  # noqa: N801
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class FakeOpenAI:
        class models:  # noqa: N801
            calls = 0

            @classmethod
            def list(cls):
                cls.calls += 1
                return [_O(id=i) for i in ["gpt-4o", "gpt-5", "o3", "gpt-4o-audio-preview", "whisper-1", "gpt-3.5-turbo"]]

    class FakeEL:
        class voices:  # noqa: N801
            @staticmethod
            def search(page_size=30):
                return _O(voices=[_O(voice_id="v1", name="Rachel")])

    reg = ModelRegistry(openai_client=FakeOpenAI(), el_client=FakeEL())
    d = reg.get({"openai": "gpt-5", "elevenlabs_tts": "eleven_turbo_v2_5", "elevenlabs_stt": "scribe_v2",
                 "elevenlabs_voice": "v1"})
    assert d["openai"] == ["gpt-5", "gpt-4o", "o3"], d["openai"]
    assert d["elevenlabs"]["voices"] == [{"id": "v1", "name": "Rachel"}] and d["elevenlabs"]["current"] == "v1"
    assert d["elevenlabs"]["tts"][0] == "eleven_flash_v2_5" and d["notes"] == []
    reg.get({"openai": "gpt-5"})
    assert FakeOpenAI.models.calls == 1, "5-min cache not used"

    # graceful without keys (pop env AFTER the constructor: __init__'s load_dotenv would restore them)
    reg2 = ModelRegistry()
    old = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "ELEVENLABS_API_KEY")}
    try:
        d = reg2.get({"openai": "gpt-5"})
        assert d["openai"] == ["gpt-5"] and d["elevenlabs"]["voices"] == [] and len(d["notes"]) == 2, d
    finally:
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v
    print("selftest OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        reg = ModelRegistry()
        from sortbot import config
        cfg = config.load()
        d = reg.get({"openai": cfg.openai_model, "elevenlabs_tts": cfg.tts_model,
                     "elevenlabs_stt": cfg.stt_model, "elevenlabs_voice": cfg.elevenlabs_voice_id})
        import json
        print(json.dumps(d, indent=1))
