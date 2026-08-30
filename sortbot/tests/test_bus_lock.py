"""Regression test for the bus-lock invariant (sortbot.main module docstring): EXACTLY ONE THREAD MAY
TOUCH THE FEETECH SERIAL BUS AT A TIME. Seen live as lerobot's "Failed to sync read 'Present_Position'
... [TxRxResult] Port is in use!" / pyserial "device reports readiness to read but returned no data"
whenever a second thread (the HUD /state poller, or the Loop drawing the HUD) read the bus while the
Loop thread was mid-motion.

The robot double's bus methods detect concurrent entry (non-reentrant flag) and raise exactly like the
real port does; the Loop, the /state pollers and the idle preview thread all run concurrently, and the
run must finish with zero overlaps. SORTBOT_BUS_ASSERT=1 is on, so any bus call made without holding
Session.robot_lock raises immediately as well.

`python -m sortbot.tests.test_bus_lock` or pytest.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import threading
import time
import urllib.request
from argparse import Namespace
from pathlib import Path

os.environ["SORTBOT_BUS_ASSERT"] = "1"  # the Session reads this when the robot connects

import numpy as np

from sortbot import main as m
from sortbot import testing
from sortbot.testing import MockRobot


class BusMockRobot(MockRobot):
    """MockRobot whose bus methods fail on concurrent entry, like the real Feetech serial port."""

    def __init__(self, cfg):
        super().__init__(cfg, realtime=False)
        self._bus_flag = threading.Lock()  # non-reentrant: models exclusive ownership of the port
        self.overlaps: list[str] = []
        self.touches = 0

    def _touch(self):
        if not self._bus_flag.acquire(blocking=False):
            who = threading.current_thread().name
            self.overlaps.append(who)
            raise RuntimeError("Port is in use! (concurrent bus access from thread %s)" % who)
        try:
            self.touches += 1
            time.sleep(0.0002)  # widen the race window a little
        finally:
            self._bus_flag.release()

    def _read_joints(self) -> np.ndarray:
        self._touch()
        return super()._read_joints()

    def _write_joints(self, q) -> None:
        self._touch()
        super()._write_joints(q)


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


def _wait(fn, timeout=60.0, what=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for {what}")


def test_bus_lock() -> None:
    logging.disable(logging.WARNING)
    robots: list[BusMockRobot] = []

    def make_robot(cfg):
        robots.append(BusMockRobot(cfg))
        return robots[-1]

    port = _free_port()
    tmp = Path(tempfile.mkdtemp())
    args = Namespace(max_steps=12, no_voice=True, hud_port=port, rules_file=str(tmp / "rules.json"), config=None)
    session = m.serve(args, factories=testing.session_factories(make_robot=make_robot))
    session.step_delay_s = 0.05  # step boundaries where the pollers can win the lock
    get, post = _mk(f"http://127.0.0.1:{port}")
    try:
        assert post("connect_robot")["ok"] and post("connect_cameras")["ok"] and post("connect_vlm")["ok"]
        robot = robots[0]

        # --- the SORTBOT_BUS_ASSERT proxy is armed: an unlocked bus call fails loudly ---
        assert type(session.robot).__name__ == "BusAssertRobot", type(session.robot)
        caught = ""
        try:
            session.robot.get_ee_pose()  # no lock held -> must raise
        except AssertionError as e:
            caught = str(e)
        assert "BUS LOCK VIOLATION" in caught, f"unlocked bus call was not caught: {caught!r}"
        with session.robot_lock:  # locked: goes straight through to the robot
            session.robot.get_ee_pose()
        assert not robot.overlaps

        # --- Loop + /state pollers + preview thread all at once: zero bus overlap allowed ---
        errs: list = []
        stop = threading.Event()

        def hammer():
            while not stop.is_set():
                try:
                    s = get("/state")
                    assert s["robot"] is not None and "torque" in s["robot"], s["robot"]
                except Exception as e:  # noqa: BLE001
                    errs.append(e)
                    return

        threads = [threading.Thread(target=hammer, daemon=True, name=f"state-hammer-{i}") for i in range(3)]
        assert post("start")["ok"]
        for t in threads:
            t.start()
        _wait(lambda: get("/state")["run"]["phase"] == "running", what="run started")
        post("say_to_bot", {"text": "open"})  # drain_voice's immediate-command path must also hold the lock
        r = post("jog", {"axis": "x", "delta": 5})  # refused while running (never a second bus toucher)
        assert not r["ok"] and "pause" in r["message"], r
        _wait(lambda: get("/state")["run"]["phase"] in ("done", "error", "stopped"), what="run finished")
        stop.set()
        for t in threads:
            t.join(timeout=5)

        st = get("/state")["run"]
        assert st["phase"] == "done" and st["result"].startswith("done"), st
        assert robot.touches > 100, robot.touches  # the run really went over the bus
        assert not robot.overlaps, f"concurrent bus access detected: {robot.overlaps[:5]}"
        assert not errs, errs[:1]
        bad = [e for e in get("/log") if "Port is in use" in str(e.get("result", ""))
               or "BUS LOCK VIOLATION" in str(e.get("result", ""))]
        assert not bad, bad[:2]
        assert not (session.loop and any("Port is in use" in h["result"] or "BUS LOCK VIOLATION" in h["result"]
                                         for h in session.loop.history)), session.loop.history
    finally:
        session.shutdown()
        session.voice.stop()
        if session.hud is not None:
            session.hud.stop()
    print(f"bus lock OK: {robot.touches} bus touches, 0 overlaps, run done with 3 /state hammers + preview; "
          "SORTBOT_BUS_ASSERT proxy catches unlocked calls")


if __name__ == "__main__":
    test_bus_lock()
