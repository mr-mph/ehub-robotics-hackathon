"""End-to-end run with the test doubles injected straight into the Loop (no mock mode exists in
the app; the doubles live in sortbot/testing.py): `python -m sortbot.tests.test_e2e` or pytest.

Also covers the HARD REQUIREMENT that the claw NEVER closes on an object without first checking the
overhead AND wrist views (main.Loop.verify_alignment), in the three scripted cases:
aligned first try / corrected then aligned / never aligned so the pick is ABORTED.
"""
from __future__ import annotations

import dataclasses
import io
import json
import logging
import socket
import tempfile
from pathlib import Path

from sortbot import config as cfgmod
from sortbot import main as m
from sortbot.testing import MockRobot, MockVLM, SimScene
from sortbot.voice import RulesStore, VoiceIO


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cfg():
    """The suite pins workspace.z_trim_mm to 0: the live value is the operator's calibration of THEIR
    table (see robot.grasp_z_mm), not something these fixtures should depend on."""
    return dataclasses.replace(cfgmod.load(), z_trim_mm=0.0)


def test_e2e(with_hud: bool = True) -> None:
    logging.disable(logging.WARNING)
    cfg = _cfg()
    scene = SimScene(MockRobot(cfg), cfg)
    n_objects = len(scene.blobs)
    voice = VoiceIO(stdin=io.StringIO("put white things on the left\n"), force_text=True)
    voice.start()
    hud = None
    if with_hud:
        from sortbot.hud import HUD
        hud = HUD(port=_free_port())
        hud.start()
    rules_path = Path(tempfile.mkdtemp()) / "rules.json"
    loop = m.Loop(cfg, scene, MockVLM(), voice, hud, m.Homography(cfg, scene.H), RulesStore(rules_path), max_steps=12)
    try:
        result = loop.run()
    finally:
        voice.stop()
        if hud:
            hud.stop()
    assert result.startswith("done"), result
    assert loop.placed == n_objects, (loop.placed, n_objects)
    assert loop.holding is None and scene.held is None
    drops_mm = [(x * 10.0, y * 10.0) for x, y in MockVLM.DROPS_CM]
    for xy, _ in scene.blobs:  # every blob now sits at one of the drop coordinates it was placed at
        assert min(abs(xy[0] - dx) + abs(xy[1] - dy) for dx, dy in drops_mm) < 5.0, xy
    assert json.loads(rules_path.read_text()) == ["put white things on the left"]
    assert [h["tool"] for h in loop.history].count("pick_at") == n_objects
    # a rejected command is fed back as a history entry, not raised; coordinate tools are validated
    # against the workspace AABB (cm on the VLM surface), not an object-id list
    world = m.WorldState()
    assert "workspace" in loop.validate(m.Command("pick_at", {"x_cm": 90.0, "y_cm": 0.0}), world)
    assert "cm" in loop.validate(m.Command("pick_at", {"x_cm": 5.0, "y_cm": 0.0}), world)
    assert loop.validate(m.Command("place_at", {"x_cm": 27.5, "y_cm": 0.0}), world)  # not holding anything
    assert loop.validate(m.Command("bogus", {}), world)
    # wrist angle: absolute (turn_to) and relative (turn_by) are both known low-level tools
    assert loop.validate(m.Command("turn_to", {"deg": 30}), world) is None
    assert loop.validate(m.Command("turn_by", {"deg": -10}), world) is None
    assert loop.execute(m.Command("turn_to", {"deg": 30}), world).ok
    assert abs(scene.get_ee_pose().roll_deg - 30) < 1e-6
    assert loop.execute(m.Command("turn_by", {"deg": -12}), world).ok
    assert abs(scene.get_ee_pose().roll_deg - 18) < 1e-6, scene.get_ee_pose()
    assert not loop.execute(m.Command("turn_by", {"deg": 400}), world).ok  # stays inside -90..90
    print(f"e2e OK: {result}, placed {loop.placed}/{n_objects}, hud={'on' if hud else 'off'}")


class SpyRobot(MockRobot):
    """MockRobot that records every gripper CLOSE, so the test can prove no close happens unchecked."""

    def __init__(self, cfg, events):
        super().__init__(cfg, realtime=False)
        self.events = events

    def close_gripper(self):
        self.events.append("close")
        return super().close_gripper()


class SpyVLM(MockVLM):
    """MockVLM that records every alignment check into the same ordered event list as the closes."""

    def __init__(self, events, **kw):
        super().__init__(**kw)
        self.events = events

    def verify_grasp(self, overhead_jpeg, wrist_jpeg, x_cm, y_cm, attempt=1):
        self.events.append("verify")
        return super().verify_grasp(overhead_jpeg, wrist_jpeg, x_cm, y_cm, attempt)


def _grasp_case(verify_script, target_cm=(20.0, 12.0), max_steps=4):
    """One scripted pick against the SimScene, with the alignment check driven by `verify_script`."""
    cfg = _cfg()
    events: list = []
    scene = SimScene(SpyRobot(cfg, events), cfg)
    vlm = SpyVLM(events, targets_cm=[target_cm], verify_script=verify_script)
    voice = VoiceIO(stdin=io.StringIO(""), force_text=True)
    rules_path = Path(tempfile.mkdtemp()) / "rules.json"
    loop = m.Loop(cfg, scene, vlm, voice, None, m.Homography(cfg, scene.H), RulesStore(rules_path),
                  max_steps=max_steps)
    loop.dlog = m.DecisionLog()
    result = loop.run()
    return loop, vlm, scene, events, result


def test_grasp_verification() -> None:
    """The claw must never close without BOTH camera views agreeing the jaws are on the object."""
    logging.disable(logging.WARNING)
    cfg = _cfg()
    assert cfg.grasp_verify is True, "grasp.verify must default to true"

    # --- 1. aligned on the first try: exactly one check, and it happens BEFORE the close -----------
    loop, vlm, scene, events, result = _grasp_case(["aligned"])
    assert vlm.verify_calls, "no alignment check was made at all"
    assert vlm.verify_calls[0] == {"x_cm": 20.0, "y_cm": 12.0, "attempt": 1}, vlm.verify_calls
    assert "close" in events and events.index("verify") < events.index("close")
    for i, e in enumerate(events):  # EVERY close is immediately preceded by a check
        assert e != "close" or events[i - 1] == "verify", (i, events)
    assert loop.placed == 1 and loop.holding is None, (loop.placed, loop.holding)
    picks = [h for h in loop.history if h["tool"] == "pick_at"]
    assert picks and picks[0]["result"].startswith("ok:") and "alignment confirmed" in picks[0]["result"], picks
    v = loop.last_verdict
    assert v["accepted"] and v["try"] == 1 and v["confidence"] >= cfg.grasp_min_confidence, v
    aligned_events = list(events)

    # --- 2. off by 1 cm, corrected, then aligned: two checks, the second at the corrected point ----
    loop, vlm, scene, events, result = _grasp_case(["off:1.0:0.5", "aligned"])
    assert [c["attempt"] for c in vlm.verify_calls] == [1, 2], vlm.verify_calls
    assert vlm.verify_calls[0]["x_cm"] == 20.0 and vlm.verify_calls[1]["x_cm"] == 21.0, vlm.verify_calls
    assert vlm.verify_calls[1]["y_cm"] == 12.5, vlm.verify_calls  # dy applied, in table cm
    assert events.count("verify") == 2 and events.index("close") > 1, events
    for i, e in enumerate(events):
        assert e != "close" or events[i - 1] == "verify", (i, events)
    assert loop.placed == 1, loop.placed
    assert "alignment confirmed on check 2/3" in [h["result"] for h in loop.history if h["tool"] == "pick_at"][0]
    assert loop.last_verdict["accepted"] and loop.last_verdict["try"] == 2, loop.last_verdict

    # a correction is BOUNDED: a wild 12 cm verdict is clamped to grasp.max_correction_cm
    loop, vlm, _, _, _ = _grasp_case(["off:12.0:-9.0", "aligned"])
    dx = vlm.verify_calls[1]["x_cm"] - vlm.verify_calls[0]["x_cm"]
    dy = vlm.verify_calls[1]["y_cm"] - vlm.verify_calls[0]["y_cm"]
    assert abs(dx) <= cfg.grasp_max_correction_cm + 1e-6 and abs(dy) <= cfg.grasp_max_correction_cm + 1e-6, (dx, dy)

    # --- 3. never aligned: max_retries+1 checks, NO close at all, a FAILED result for the planner --
    loop, vlm, scene, events, result = _grasp_case(["blind"])
    tries = cfg.grasp_max_retries + 1
    assert len(vlm.verify_calls) == tries, vlm.verify_calls
    assert "close" not in events, f"the claw closed without a confirmed alignment: {events}"
    assert loop.placed == 0 and loop.holding is None and scene.held is None
    picks = [h for h in loop.history if h["tool"] == "pick_at"]
    assert picks and picks[0]["result"].startswith("FAILED:"), picks
    assert f"alignment not confirmed after {tries} tries" in picks[0]["result"], picks[0]["result"]
    assert "too dark" in picks[0]["result"], picks[0]["result"]  # the reason reaches the planner verbatim
    assert not loop.last_verdict["accepted"] and loop.last_verdict["try"] == tries, loop.last_verdict
    # the verdicts (with a both-cameras thumbnail) are in the decision log, so the human can see why
    checks = [e for e in loop.dlog.entries() if e["tool"] == "verify_grasp"]
    assert len(checks) == tries and all(e["ok"] is False for e in checks), checks
    assert any(e["thumb_b64"] for e in checks), "no verification image in the decision log"

    # a low-confidence "aligned" is NOT aligned
    loop, vlm, _, events, _ = _grasp_case([{"aligned": True, "confidence": 0.2}])
    assert "close" not in events and loop.placed == 0, events

    print(f"grasp verification OK: aligned-first-try {aligned_events}, corrected-then-aligned, "
          f"never-aligned aborts after {tries} checks with no close")


if __name__ == "__main__":
    test_e2e()
    test_grasp_verification()
