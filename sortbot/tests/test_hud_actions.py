"""HTTP tests for the HUD action endpoints (server-first main, RUN + ROBOT groups), all in mock mode.

`python -m sortbot.tests.test_hud_actions` or pytest. Every feature is exercised over POST /action/<name>
against a live HUD server, exactly as the page drives it.
"""
from __future__ import annotations

import json
import logging
import socket
import tempfile
import time
import urllib.request
from argparse import Namespace
from pathlib import Path

from sortbot import main as m


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
    args = Namespace(mock=False, real=False, max_steps=40, no_hud=False, no_voice=True, hud_port=port,
                     live_vlm=False, rules_file=str(Path(tempfile.mkdtemp()) / "rules.json"), config=None)
    session = m.serve(args)
    session.step_delay_s = 0.3  # slow the mock loop enough that pause/stop hit it mid-run
    get, post = _mk(f"http://127.0.0.1:{port}")
    try:
        # --- server-first: HUD is up with NOTHING connected ---
        st = get("/state")
        assert st["run"]["phase"] == "idle" and st["run"]["step"] == 0, st["run"]
        assert st["run"]["connected"] == {"robot": False, "cams": False, "vlm": False}
        assert st["robot"] is None
        acts = {a["name"]: a for a in get("/actions")}
        for name, group in (("set_mode", "run"), ("start", "run"), ("pause", "run"), ("resume", "run"),
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

        # --- set_mode: bad values are reported, not raised ---
        r = post("set_mode", {"robot": "banana"})
        assert not r["ok"] and "banana" in r["message"], r

        # --- connect mock robot + sim cams + mock vlm ---
        r = post("set_mode", {"robot": "mock", "cams": "sim", "vlm": "mock"})
        assert r["ok"], r
        st = get("/state")
        assert st["run"]["connected"] == {"robot": True, "cams": True, "vlm": True}
        assert st["run"]["mode"] == {"robot": "mock", "cams": "sim", "vlm": "mock"}
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
        r = post("set_mode", {"vlm": "mock"})  # mode change refused while a run is up
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
        assert post("say_to_robot", {"text": "put white things in wires"})["ok"]

        # --- calibration from the page while idle (mock teleop session, temp calib file) ---
        assert post("calib_start")["ok"]
        _wait(lambda: get("/state")["calibration"]["state"] == "fitted", timeout=40, what="calibration fitted")
        c = get("/state")["calibration"]
        assert c["n"] >= 4 and c["residual_mean_mm"] is not None and c["residual_mean_mm"] < 3.0, c
        assert session._calib_out is not None and session._calib_out.exists()
    finally:
        session.shutdown()
        session.voice.stop()
        if session.hud is not None:
            session.hud.stop()
    print("hud actions OK: server-first, RUN/ROBOT groups, E-STOP, restart, calibration all over HTTP")


if __name__ == "__main__":
    test_hud_actions()
