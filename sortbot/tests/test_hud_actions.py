"""HTTP tests for the HUD action endpoints (server-first main, RUN/ROBOT/VOICE/RULES/MODELS groups).

`python -m sortbot.tests.test_hud_actions` or pytest. Every feature is exercised over POST /action/<name>
against a live HUD server, exactly as the page drives it. No hardware: the test doubles from
sortbot/testing.py are injected through the Session(factories=...) seam.
"""
from __future__ import annotations

import base64
import json
import logging
import shutil
import socket
import tempfile
import threading
import time
import urllib.request
from argparse import Namespace
from pathlib import Path

from sortbot import main as m
from sortbot import testing


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _mk(base):
    def get(path):
        with urllib.request.urlopen(base + path, timeout=5) as r:
            return json.load(r)

    def post(name, body=None):
        req = urllib.request.Request(base + "/action/" + name, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)

    return get, post


def _wait(fn, timeout=20.0, what=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for {what}")


def test_hud_actions() -> None:
    logging.disable(logging.WARNING)
    port = _free_port()
    tmp = Path(tempfile.mkdtemp())
    cfg_copy = tmp / "config.yaml"  # set_model persists here, never into the real config.yaml
    shutil.copy(Path(m.__file__).parent / "config.yaml", cfg_copy)
    args = Namespace(max_steps=40, no_voice=True, hud_port=port,
                     rules_file=str(tmp / "rules.json"), config=str(cfg_copy))
    session = m.serve(args, factories=testing.session_factories())
    session.step_delay_s = 0.3  # slow the loop enough that pause/stop hit it mid-run
    get, post = _mk(f"http://127.0.0.1:{port}")
    try:
        # --- server-first: HUD is up with NOTHING connected ---
        st = get("/state")
        assert st["run"]["phase"] == "idle" and st["run"]["step"] == 0, st["run"]
        assert st["run"]["connected"] == {"robot": False, "cams": False, "vlm": False}
        assert st["robot"] is None
        acts = {a["name"]: a for a in get("/actions")}
        for name, group in (("connect_robot", "run"), ("connect_cameras", "run"), ("connect_vlm", "run"),
                            ("start", "run"), ("pause", "run"), ("resume", "run"),
                            ("stop", "run"), ("step_once", "run"), ("set_max_steps", "run"), ("set_task", "run"),
                            ("home", "robot"), ("open_gripper", "robot"), ("close_gripper", "robot"),
                            ("jog", "robot"), ("goto", "robot"), ("torque_off", "robot"), ("torque_on", "robot"),
                            ("say_to_robot", "voice"), ("calib_start", "calibration")):
            assert name in acts and acts[name]["group"] == group, (name, acts.get(name))
        assert [p["name"] for p in acts["jog"]["params"]] == ["axis", "delta"]
        assert [p["name"] for p in acts["goto"]["params"]] == ["x", "y", "z"]
        r = post("start")
        assert not r["ok"] and "not connected" in r["message"], r
        assert not post("home")["ok"] and not post("torque_off")["ok"]

        # --- connect errors are reported in the response, never raised ---
        vlm_factory = session.factories["vlm"]

        def _boom(s):
            raise RuntimeError("no key configured")

        session.factories["vlm"] = _boom
        r = post("connect_vlm")
        assert not r["ok"] and "no key configured" in r["message"], r
        assert get("/state")["run"]["connected"]["vlm"] is False
        session.factories["vlm"] = vlm_factory

        # --- connect the (injected double) devices: robot, cameras, vlm ---
        assert post("connect_robot")["ok"]
        assert post("connect_cameras")["ok"]
        assert post("connect_vlm")["ok"]
        st = get("/state")
        assert st["run"]["connected"] == {"robot": True, "cams": True, "vlm": True}
        assert st["robot"]["torque"] is True and st["robot"]["gripper_open"] is True
        assert st["calibration"]["state"] == "idle" and "target" in st["calibration"]

        # --- ROBOT group ---
        assert post("home")["ok"]
        assert post("goto", {"x": 275, "y": 0, "z": 60})["ok"]
        p = get("/state")["robot"]["ee_pose"]
        assert abs(p["x"] - 275) < 6 and abs(p["y"]) < 6 and abs(p["z"] - 60) < 6, p
        assert post("jog", {"axis": "x", "delta": 20})["ok"]
        assert abs(get("/state")["robot"]["ee_pose"]["x"] - 295) < 6
        assert post("jog", {"axis": "roll", "delta": 15})["ok"]
        assert abs(get("/state")["robot"]["ee_pose"]["roll_deg"] - 15) < 1
        assert post("jog", {"axis": "roll", "delta": -15})["ok"]
        assert not post("jog", {"axis": "warp", "delta": 5})["ok"]
        assert post("close_gripper")["ok"] and get("/state")["robot"]["gripper_open"] is False
        assert post("open_gripper")["ok"] and get("/state")["robot"]["gripper_open"] is True
        r = post("goto", {"x": 900, "y": 0, "z": 60})  # outside the safety envelope
        assert not r["ok"] and "workspace" in r["message"], r

        # --- E-STOP ---
        assert post("torque_off")["ok"]
        assert get("/state")["robot"]["torque"] is False
        r = post("home")
        assert not r["ok"] and "torque" in r["message"].lower(), r
        assert post("torque_on")["ok"] and get("/state")["robot"]["torque"] is True
        assert post("home")["ok"]

        # --- PERCEPTION group is overlay-only now (the VLM does the seeing): zone drops + px->mm.
        # No detector actions may exist at all.
        acts = {a["name"]: a for a in get("/actions")}
        for name, group in (("set_zone_drop", "perception"), ("px_to_mm", "perception"), ("log_clear", "log")):
            assert name in acts and acts[name]["group"] == group, (name, acts.get(name))
            assert acts[name]["help"], f"{name} has no help text in GET /actions"
        for gone in ("set_detector_params", "redetect", "toggle_mask", "pick"):
            assert gone not in acts, f"detector-era action {gone!r} still registered"
        p0 = get("/state")["perception"]
        assert "params" not in p0 and "mask" not in p0, p0  # detector state keys are gone
        assert {z["name"] for z in p0["zones"]} == {"LEFT", "MIDDLE", "RIGHT"}, p0
        # px -> mm conversion (drives the click-to-set-drop mode); needs the first preview frame
        _wait(lambda: post("px_to_mm", {"u": 320, "v": 240})["ok"], what="first preview frame for px_to_mm")
        r = post("px_to_mm", {"u": 320, "v": 240})
        assert r["ok"] and 120 <= r["data"]["x"] <= 420 and -220 <= r["data"]["y"] <= 220, r
        # zone drop points: moved live + persisted (rect untouched)
        import yaml as _y
        r = post("set_zone_drop", {"name": "MIDDLE", "x": 280, "y": 5})
        assert r["ok"], r
        z = next(z for z in get("/state")["perception"]["zones"] if z["name"] == "MIDDLE")
        assert z["drop"] == [280, 5], z
        y = _y.safe_load(cfg_copy.read_text())
        assert y["zones"]["MIDDLE"]["drop"] == [280, 5], y["zones"]
        assert y["zones"]["MIDDLE"]["rect"] == [[150.0, 60.0], [400.0, -60.0]], y["zones"]
        assert not post("set_zone_drop", {"name": "NOPE", "x": 280, "y": 5})["ok"]
        assert not post("set_zone_drop", {"name": "LEFT", "x": 9999, "y": 0})["ok"]

        # --- run config ---
        assert post("set_task", {"text": "sort it however makes sense"})["ok"]
        assert post("set_max_steps", {"n": 30})["ok"]
        st = get("/state")["run"]
        assert st["task"] == "sort it however makes sense" and st["max_steps"] == 30

        # --- step_once from idle: starts paused, runs exactly one step ---
        assert post("step_once")["ok"]
        _wait(lambda: get("/state")["run"]["phase"] == "paused" and get("/state")["run"]["step"] == 1,
              what="paused after 1 step")
        assert post("step_once")["ok"]
        _wait(lambda: get("/state")["run"]["step"] == 2 and get("/state")["run"]["phase"] == "paused",
              what="paused after 2 steps")
        r = post("connect_vlm")  # device change refused while a run is up
        assert not r["ok"] and "stop" in r["message"], r

        # --- resume to completion ---
        assert post("resume")["ok"]
        _wait(lambda: get("/state")["run"]["phase"] == "done", what="run done")
        st = get("/state")["run"]
        assert st["result"].startswith("done"), st

        # --- restart without restarting the process (fresh SimScene) ---
        assert post("start")["ok"]
        _wait(lambda: get("/state")["run"]["phase"] == "running", what="running again")
        assert post("pause")["ok"]
        _wait(lambda: get("/state")["run"]["phase"] == "paused", what="paused")
        # E-STOP while paused, then resume is refused until torque_on
        assert post("torque_off")["ok"]
        r = post("resume")
        assert not r["ok"] and "torque" in r["message"].lower(), r
        assert post("torque_on")["ok"]
        assert post("stop")["ok"]
        _wait(lambda: get("/state")["run"]["phase"] == "stopped", what="stopped")
        # ... and a third run goes all the way through again
        assert post("start")["ok"]
        _wait(lambda: get("/state")["run"]["phase"] == "done", what="second full run done")
        assert get("/state")["run"]["result"].startswith("done")

        # --- E-STOP mid-run pauses the loop ---
        assert post("start")["ok"]
        _wait(lambda: get("/state")["run"]["phase"] == "running", what="running for estop")
        assert post("torque_off")["ok"]
        _wait(lambda: get("/state")["run"]["phase"] == "paused", what="paused by E-STOP")
        assert get("/state")["robot"]["torque"] is False
        assert post("torque_on")["ok"] and post("stop")["ok"]
        _wait(lambda: get("/state")["run"]["phase"] == "stopped", what="stopped after estop")

        # --- voice from the page ---
        assert post("say_to_robot", {"text": "put white things on the left"})["ok"]

        # --- calibration from the page while idle (fixture teleop session, temp calib file) ---
        assert post("calib_start")["ok"]
        _wait(lambda: get("/state")["calibration"]["state"] == "running", what="calibration running")
        # hammer /state while the teleop session runs: the single bus lock + cached pose must keep every
        # response consistent (regression: concurrent bus reads -> feetech "Port is in use!")
        hammer_errs: list = []
        hammer_stop = threading.Event()

        def _hammer():
            while not hammer_stop.is_set():
                try:
                    s = get("/state")
                    assert s["robot"] is not None and "torque" in s["robot"], s["robot"]
                    assert s["calibration"]["state"] in ("running", "fitted"), s["calibration"]
                except Exception as e:  # noqa: BLE001
                    hammer_errs.append(e)
                    return

        hammer = threading.Thread(target=_hammer, daemon=True)
        hammer.start()
        r = post("connect_robot", {"connect": False})  # refused while calibrating: the teleop thread owns the arm
        assert not r["ok"] and "calibration" in r["message"], r
        assert get("/state")["calibration"]["state"] in ("running", "fitted"), "calibration must survive device changes"
        _wait(lambda: get("/state")["calibration"]["state"] == "fitted", timeout=40, what="calibration fitted")
        hammer_stop.set()
        hammer.join(timeout=5)
        assert not hammer_errs, hammer_errs[:1]
        c = get("/state")["calibration"]
        assert c["n"] >= 4 and c["residual_mean_mm"] is not None and c["residual_mean_mm"] < 3.0, c
        assert session._calib_out is not None and session._calib_out.exists()
        # a click on the bare mat (gray pixel) must be rejected, keeping the previous target
        r = post("calib_sample", {"u": 5, "v": 5})
        assert not r["ok"] and "target" in r["message"].lower() and r["data"]["det"] is None, r
        r = post("connect_vlm")  # the guard releases once the session has finished (reconnect is fine)
        assert r["ok"], r

        # --- VOICE group: say_to_bot / transcribe / speak (fake ElevenLabs client; the HTTP path is real) ---
        class _O:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class _FakeEL:
            class speech_to_text:
                last: dict = {}

                @classmethod
                def convert(cls, **kw):
                    cls.last = kw
                    return _O(text="always put round things on the left")

            class text_to_speech:
                @staticmethod
                def convert(voice_id, **kw):
                    return iter([b"mp3bytes"])

            class voices:
                @staticmethod
                def search(page_size=30):
                    return _O(voices=[_O(voice_id="v1", name="Rachel"), _O(voice_id="v2", name="Sam")])

        session.voice._client = _FakeEL()
        session.voice._player = None  # no audio playback during tests
        acts = {a["name"]: a for a in get("/actions")}
        for name, group in (("say_to_bot", "voice"), ("transcribe", "voice"), ("speak", "voice"),
                            ("add_rule", "rules"), ("delete_rule", "rules"), ("move_rule", "rules"),
                            ("clear_hints", "rules"), ("get_models", "models"), ("set_model", "models")):
            assert name in acts and acts[name]["group"] == group, (name, acts.get(name))
            assert acts[name]["help"], f"{name} has no help text in GET /actions"
        r = post("say_to_bot", {"text": "screws go in the right bin"})  # idle + rule -> persisted now
        assert r["ok"] and r["data"]["kind"] == "rule", r
        assert "screws go in the right bin" in get("/state")["rules"]["list"]
        r = post("say_to_bot", {"text": "open"})  # idle + immediate command -> queued for the next run
        assert r["ok"] and r["data"]["kind"] == "action" and "queued" in r["message"], r
        assert "open" in get("/state")["voice"]["queue"]
        r = post("transcribe", {"audio_b64": base64.b64encode(b"\x1aE\xdf\xa3 fake webm").decode(),
                                "mime": "audio/webm"})
        assert r["ok"] and r["data"]["text"] == "always put round things on the left", r
        assert _FakeEL.speech_to_text.last["model_id"] == "scribe_v2"
        assert _FakeEL.speech_to_text.last["file"][2] == "audio/webm"
        assert get("/state")["voice"]["last_transcript"] == "always put round things on the left"
        assert "always put round things on the left" in get("/state")["rules"]["list"]  # classified as a rule
        assert not post("transcribe", {"audio_b64": ""})["ok"]
        assert post("speak", {"text": "test sentence"})["ok"]
        assert not post("speak", {"text": ""})["ok"]

        # --- RULES group ---
        rules0 = get("/state")["rules"]["list"]
        assert post("add_rule", {"text": "red things go in LEFT"})["ok"]
        lst = get("/state")["rules"]["list"]
        assert lst == rules0 + ["red things go in LEFT"], lst
        i = lst.index("red things go in LEFT")
        while i > 0:  # walk it to the top
            assert post("move_rule", {"i": i, "dir": "up"})["ok"]
            i -= 1
        assert get("/state")["rules"]["list"][0] == "red things go in LEFT"
        assert not post("move_rule", {"i": 0, "dir": "up"})["ok"]
        assert not post("move_rule", {"i": 0, "dir": "sideways"})["ok"]
        assert post("delete_rule", {"i": 0})["ok"]
        assert "red things go in LEFT" not in get("/state")["rules"]["list"]
        assert not post("delete_rule", {"i": 99})["ok"]
        assert post("clear_hints")["ok"]

        # --- MODELS group (fake clients injected; caching, filtering and persistence are real) ---
        class _FakeOpenAI:
            class models:
                calls = 0

                @classmethod
                def list(cls):
                    cls.calls += 1
                    return [_O(id=i) for i in ("gpt-4o", "gpt-5", "gpt-5-mini", "o3",
                                               "gpt-4o-audio-preview", "whisper-1", "gpt-3.5-turbo")]

        session.models._openai_client = _FakeOpenAI()
        session.models._el_client = _FakeEL()
        r = post("get_models")
        assert r["ok"], r
        d = r["data"]
        assert d["openai"][0] == d["current"]["openai"] and set(d["openai"]) >= {"gpt-4o", "gpt-5", "gpt-5-mini", "o3"}, d
        assert "gpt-4o-audio-preview" not in d["openai"] and "whisper-1" not in d["openai"] and "gpt-3.5-turbo" not in d["openai"]
        assert d["elevenlabs"]["tts"] == ["eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2", "eleven_v3"]
        assert d["elevenlabs"]["stt"] == ["scribe_v2", "scribe_v1"]
        assert {v["id"] for v in d["elevenlabs"]["voices"]} == {"v1", "v2"}
        post("get_models")
        assert _FakeOpenAI.models.calls == 1, "openai listing not cached for 5 min"
        r = post("set_model", {"provider": "elevenlabs_stt", "value": "scribe_v1"})
        assert r["ok"] and "persisted" in r["message"] and session.voice.stt_model == "scribe_v1", r
        assert post("set_model", {"provider": "openai", "value": "gpt-4o"})["ok"] and session.cfg.openai_model == "gpt-4o"
        assert post("set_model", {"provider": "elevenlabs_voice", "value": "v2"})["ok"] and session.voice.voice_id == "v2"
        assert post("set_model", {"provider": "elevenlabs_tts", "value": "eleven_v3"})["ok"] and session.voice.tts_model == "eleven_v3"
        assert not post("set_model", {"provider": "bananas", "value": "x"})["ok"]
        import yaml as _yaml
        ytext = cfg_copy.read_text()
        y = _yaml.safe_load(ytext)
        assert y["vlm"]["model"] == "gpt-4o" and y["voice"]["stt_model"] == "scribe_v1", y
        assert y["voice"]["tts_model"] == "eleven_v3" and y["voice"]["elevenlabs_voice_id"] == "v2", y
        assert "# indices verified" in ytext, "config.yaml comments were clobbered"
        d = post("get_models")["data"]
        assert d["openai"][0] == "gpt-4o" and d["current"]["elevenlabs_voice"] == "v2", d
        assert get("/state")["vlm"]["model"] == "mock"  # injected test VLM; latency card present

        # --- LOG group: ring buffer of decisions + events at GET /log, newest first ---
        lg = get("/log")
        assert isinstance(lg, list) and len(lg) > 5, len(lg)
        assert all(set(e) >= {"i", "step", "t", "tool", "args", "result", "ok", "say", "latency_ms", "thumb_b64"}
                   for e in lg), lg[0]
        assert lg[0]["i"] > lg[-1]["i"], "log not newest-first"
        tools = [e["tool"] for e in lg]
        assert "connect_robot" in tools and "voice" in tools and "set_zone_drop" in tools, tools
        picks = [e for e in lg if e["tool"] == "pick_at"]
        assert picks and any(e["thumb_b64"] for e in picks), "no pick decisions with overlay thumbnails"
        assert all("x_cm" in e["args"] for e in picks), picks  # the VLM surface is coordinates in cm
        assert any(e["thumb_b64"] and e["thumb_b64"].startswith("/9j/") for e in lg), "thumb is not a jpeg"
        assert any(e["tool"] == "torque_off" and e["ok"] is False for e in lg), "E-STOP not logged as red"
        assert any(e["tool"] == "place_in_zone" and e["ok"] for e in lg), tools
        assert post("log_clear")["ok"]
        assert get("/log") == []

        # --- UX redesign: help on EVERY action, mic toggle endpoints, page/action cross-check ---
        acts = {a["name"]: a for a in get("/actions")}
        missing_help = [n for n, a in acts.items() if not a.get("help")]
        assert not missing_help, f"actions without help text: {missing_help}"
        st = get("/state")
        assert st["voice"]["listening"] is False, "mic must be OFF by default"
        assert st["perception"]["calibrated"] is True, st["perception"]  # injected fixed H: always calibrated
        session.voice._mic_ok = False  # never start a real ffmpeg mic capture from tests
        r = post("mic_on")
        assert r["ok"] and "unavailable" in r["message"], r
        assert get("/state")["voice"]["listening"] is False
        assert post("mic_off")["ok"]
        assert get("/state")["voice"]["listening"] is False
        assert any(e["tool"] == "mic_on" for e in get("/log")), "mic toggle not in the decision log"
        # the page: every act('...') target in its JS must be a registered action; safety/UX chrome present
        import re as _re
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as rr:
            page = rr.read().decode()
        refs = set(_re.findall(r"act\('([a-z_0-9]+)'", page))
        assert len(refs) > 15, f"page references too few actions: {sorted(refs)}"
        unknown = refs - set(acts)
        assert not unknown, f"page JS calls actions not in /actions: {sorted(unknown)}"
        for frag in ("MIC LIVE", "mic_on", "mic_off", 'id="checklist"', 'id="demo"', "E-STOP",
                     'data-tab="setup"', 'data-tab="operate"', 'data-tab="tune"', 'data-tab="debug"',
                     "Not a straight line"):
            assert frag in page, f"page missing {frag!r}"
    finally:
        session.shutdown()
        session.voice.stop()
        if session.hud is not None:
            session.hud.stop()
    print("hud actions OK: RUN/ROBOT/PERCEPTION/VOICE/RULES/MODELS/LOG groups, E-STOP, restart, calibration, "
          "push-to-talk transcribe, model hot-swap, zone drops + px->mm + /log, "
          "help on every action, mic toggle, page/action cross-check all over HTTP")


if __name__ == "__main__":
    test_hud_actions()
