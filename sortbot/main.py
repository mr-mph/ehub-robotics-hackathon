"""Main loop: home -> capture -> homography -> detect -> overlay -> HUD -> voice -> VLM -> validate -> execute.

--mock: MockRobot wrapped in SimScene (synthetic overhead frames with movable blobs), MockVLM, text voice.
--real: SO101Robot + OpenAI VLM + TableHomography (fitted H from calib.json and/or the ArUco mat, config calibration.mode).
SafetyError / rejected commands are fed back to the VLM as the tool result, never raised.
HUD calibration: "Start calibration" requests a teleoperated CalibSession (sortbot.calibrate); the loop pauses at
its next step boundary, the leader arm drives the follower, Finish/Cancel resumes with the reloaded homography.
In --mock a VirtualLeader scripts the whole session against SimScene's ground-truth H.
"""
from __future__ import annotations

import argparse
import io
import logging
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from sortbot import config as cfgmod
from sortbot import perception
from sortbot.types import Command, DetectedObject, ExecResult, RobotAPI, WorldState
from sortbot.vlm import VLM, MockVLM
from sortbot.voice import RulesStore, VoiceIO, classify

log = logging.getLogger("sortbot.main")

HIGH_LEVEL = {"pick", "place_in_zone", "place_at", "done", "say"}
LOW_LEVEL = {"move_to", "open", "close", "turn_to"}
SIM_OBJECTS = [((200.0, 120.0), (220, 40, 40)), ((240.0, -70.0), (40, 200, 60)),
               ((180.0, 70.0), (240, 220, 50)), ((270.0, 70.0), (255, 255, 255))]


class SimScene:
    """Wraps MockRobot: keeps blob positions on a synthetic mat and moves them with pick/place."""

    def __init__(self, robot: RobotAPI, cfg: cfgmod.Config, objects=SIM_OBJECTS):
        self.robot, self.cfg = robot, cfg
        self.w, self.h = cfg.overhead_cam.width, cfg.overhead_cam.height
        self.H = perception.synth_homography(cfg, self.w, self.h)
        self.blobs = [[list(xy), col] for xy, col in objects]
        self.held: list | None = None

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def capture(self, name: str) -> np.ndarray:
        if name != "overhead":
            return self.robot.capture(name)
        img = np.random.default_rng(0).integers(20, 40, (self.h, self.w, 3), dtype=np.uint8)
        for xy, col in self.blobs:
            cx, cy = perception.mm_to_px(self.H, [xy])[0].astype(int)
            cv2.rectangle(img, (cx - 22, cy - 15), (cx + 22, cy + 15), col, -1)
        return img

    def pick(self, obj: DetectedObject) -> ExecResult:
        r = self.robot.pick(obj)
        if r.ok and self.blobs:
            i = min(range(len(self.blobs)), key=lambda i: np.hypot(*(np.array(self.blobs[i][0]) - obj.centroid_mm)))
            if np.hypot(*(np.array(self.blobs[i][0]) - obj.centroid_mm)) < 30:
                self.held = self.blobs.pop(i)
        return r

    def place_at(self, x: float, y: float) -> ExecResult:
        r = self.robot.place_at(x, y)
        if r.ok and self.held:
            self.held[0] = [x, y]
            self.blobs.append(self.held)
            self.held = None
        return r

    def open_gripper(self) -> ExecResult:
        r = self.robot.open_gripper()
        if self.held:  # dropped wherever the EE is
            p = self.robot.get_ee_pose()
            self.held[0] = [p.x, p.y]
            self.blobs.append(self.held)
            self.held = None
        return r


class Homography:
    """Uniform px->mm H source: fixed (mock) or TableHomography (real: fitted calib.json H and/or ArUco tags)."""

    def __init__(self, cfg: cfgmod.Config, fixed: np.ndarray | None = None):
        self.cfg, self.fixed = cfg, fixed
        self.tracker = None if fixed is not None else __import__("sortbot.calibration", fromlist=["x"]).TableHomography(cfg)

    @property
    def method(self) -> str:
        return "ball" if self.tracker is None else self.tracker.method

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        if self.fixed is not None:
            return self.fixed
        return self.tracker.H if self.tracker.update(frame) else None

    def reload(self, path: Path | None = None) -> None:
        """After a calibration: pick up the new H (real) or the fitted one from `path` (mock)."""
        if self.tracker is not None:
            self.tracker.reload(path)
        elif path is not None:
            H = __import__("sortbot.calibration", fromlist=["x"]).load_calib_dict(path)["H_px_to_mm"]
            if H is not None:
                self.fixed = H


def png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


class Loop:
    def __init__(self, cfg, robot, vlm, voice, hud, homography, rules: RulesStore, max_steps: int):
        self.cfg, self.robot, self.vlm, self.voice, self.hud = cfg, robot, vlm, voice, hud
        self.homog, self.rules, self.max_steps = homography, rules, max_steps
        self.history: list[dict] = []
        self.hints: list[str] = []
        self.holding: int | None = None
        self.holding_obj: DetectedObject | None = None  # ids are renumbered every frame; keep what was picked
        self.filled: set[str] = set()
        self.placed = 0
        self.last_say = ""
        self.calib = None  # CalibController (see attach_calibration)
        self._calib_requested = threading.Event()
        self.last_overhead: np.ndarray | None = None

    # ---- voice ----
    def drain_voice(self) -> bool:
        """Returns False if a stop was requested."""
        for text in self.voice.drain():
            it = classify(text)
            log.info("voice %s: %s", it.kind, it.text)
            if it.kind == "rule":
                self.rules.append(it.text)
            elif it.kind == "action":
                w = it.text.split()[0]
                if w in ("stop", "halt", "freeze"):
                    return False
                # only short bare commands act immediately; "drop the red one in WIRES" is a hint for the VLM
                bare = len(it.text.split()) <= 2
                if bare and w in ("open", "release", "drop", "let"):
                    self.record("open", {}, self.robot.open_gripper())
                    self.holding = self.holding_obj = None
                elif bare and w == "close":
                    self.record("close", {}, self.robot.close_gripper())
                elif bare and w in ("home", "retract", "go"):
                    self.record("home", {}, self.robot.home())
                else:
                    self.hints.append(f"(human) {it.text}")
            else:
                self.hints.append(f"(human) {it.text}")
        return True

    def record(self, tool: str, args: dict, r: ExecResult) -> None:
        self.history.append({"tool": tool, "args": args, "result": ("ok: " if r.ok else "FAILED: ") + r.message})
        log.info("%s(%s) -> %s", tool, args, self.history[-1]["result"])

    # ---- validate + execute ----
    def validate(self, cmd: Command, world: WorldState) -> str | None:
        a, t = cmd.args, cmd.tool
        if t not in HIGH_LEVEL | LOW_LEVEL:
            return f"unknown tool {t!r}"
        lo, hi = self.cfg.aabb_min_mm, self.cfg.aabb_max_mm
        if t == "pick":
            if self.holding is not None:
                return f"already holding #{self.holding}; place it first"
            if not any(o.id == a.get("id") for o in world.objects):
                return f"object id {a.get('id')} not in candidates {[o.id for o in world.objects]}"
        elif t == "place_in_zone":
            if self.holding is None:
                return "not holding anything"
            if self.cfg.zone(str(a.get("zone", ""))) is None:
                return f"unknown zone {a.get('zone')!r}; zones: {[z.name for z in self.cfg.zones]}"
        elif t in ("place_at", "move_to"):
            x, y = a.get("x_mm", a.get("x")), a.get("y_mm", a.get("y"))
            if t == "place_at" and self.holding is None:
                return "not holding anything"
            if x is None or y is None:
                return "missing x/y"
            if not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]):
                return f"({x},{y}) outside workspace x[{lo[0]},{hi[0]}] y[{lo[1]},{hi[1]}]"
            if t == "move_to" and not (lo[2] <= a.get("z", -1) <= hi[2]):
                return f"z outside [{lo[2]},{hi[2]}]"
        return None

    def execute(self, cmd: Command, world: WorldState) -> ExecResult:
        a, t, r = cmd.args, cmd.tool, self.robot
        if t == "pick":
            obj = next(o for o in world.objects if o.id == a["id"])
            res = r.pick(obj)
            if res.ok:
                self.holding, self.holding_obj = obj.id, obj
            return res
        if t == "place_in_zone":
            z = self.cfg.zone(a["zone"])
            res = r.place_at(*z.drop_point_mm)
            if res.ok:
                self.holding, self.holding_obj, self.placed = None, None, self.placed + 1
                self.filled.add(z.name)
            return res
        if t == "place_at":
            res = r.place_at(float(a.get("x_mm", a.get("x"))), float(a.get("y_mm", a.get("y"))))
            if res.ok:
                self.holding, self.holding_obj, self.placed = None, None, self.placed + 1
            return res
        if t == "move_to":
            return r.move_to(float(a["x"]), float(a["y"]), float(a["z"]))
        if t == "open":
            res = r.open_gripper()
            self.holding = self.holding_obj = None
            return res
        if t == "close":
            return r.close_gripper()
        if t == "turn_to":
            return r.turn_to(float(a["deg"]))
        if t == "say":
            self.last_say = str(a.get("text", ""))
            self.voice.speak(self.last_say)
            return ExecResult(True, "said")
        return ExecResult(True, str(a.get("summary", "")))

    # ---- calibration from the HUD ----
    def attach_calibration(self, ctrl, mock_out: Path | None = None) -> None:
        """Register the calibration actions; Start is deferred to the next step boundary so the teleop session never
        fights an in-flight motion. mock_out: where the mock session writes (never the real calib.json)."""
        self.calib, self._calib_out = ctrl, mock_out
        if self.hud is None:
            return
        ctrl.register(self.hud)
        self.hud.register("calib_start", self._request_calib, "Start calibration", "calibration")

    def _request_calib(self) -> dict:
        if self.calib.active or self._calib_requested.is_set():
            return {"ok": False, "message": "calibration already running/requested", "data": None}
        self._calib_requested.set()
        return {"ok": True, "message": "calibration will start at the next step boundary (loop pauses)", "data": None}

    def _calib_done(self, session) -> None:
        if session.state == "fitted":
            self.homog.reload(self._calib_out)
            if self._calib_out is None:  # real: the z offset may have changed too
                r = getattr(self.robot, "robot", self.robot)
                r.table_T_base = __import__("sortbot.robot", fromlist=["x"]).load_calib(self.cfg.calib_file)
                r.base_T_table = np.linalg.inv(r.table_T_base)
        log.info("calibration %s: %s", session.state, session.message)

    def _maybe_calibrate(self) -> None:
        if self.calib is None or not self._calib_requested.is_set():
            return
        self._calib_requested.clear()
        r = self.calib.start()
        log.info("calibration: %s", r["message"])
        while self.calib.active:
            time.sleep(0.2)

    # ---- main ----
    def run(self) -> str:
        from sortbot.robot import SafetyError

        for step in range(1, self.max_steps + 1):
            t0 = time.time()
            self._maybe_calibrate()
            hr = self.robot.home()
            if not hr.ok:  # routine homing is not worth a history slot; failures are
                self.record("home", {}, hr)
            if not self.drain_voice():  # before capture so voice actions cannot stale the frame
                return "stopped by voice"
            overhead, wrist = self.robot.capture("overhead"), self.robot.capture("wrist")
            self.last_overhead = overhead
            H = self.homog.update(overhead)
            if H is None:
                log.warning("no homography (no calib.json H and ArUco tags not visible); run calibration")
                self.hud_update(overhead, wrist, step, t0, "no homography - calibrate")
                time.sleep(0.5)
                continue
            objects = perception.detect_objects(overhead, H, self.cfg, filled_zones=sorted(self.filled), method=self.homog.method)
            rules = self.rules.list() + self.hints
            if self.holding_obj is not None:
                o = self.holding_obj
                rules = rules + [f"holding: the {o.color_hint} object picked from ({o.centroid_mm[0]:.0f},{o.centroid_mm[1]:.0f}); object ids below are renumbered"]
            world = WorldState(objects, self.cfg.zones, self.robot.get_ee_pose(), self.robot.gripper_open, self.holding, rules)
            overlay = perception.render_overlay(overhead, H, objects, self.cfg.zones, world.ee_pose, rules)
            self.hud_update(overlay, wrist, step, t0, "planning")
            try:
                cmd = self.vlm.plan_step(png(overlay), png(wrist), world, self.history)
            except Exception as e:  # noqa: BLE001
                log.error("VLM failed: %s", e)
                self.history.append({"tool": "vlm", "args": {}, "result": f"error {e}"})
                continue
            err = self.validate(cmd, world)
            if err:
                self.record(cmd.tool, cmd.args, ExecResult(False, f"rejected: {err}"))
                continue
            try:
                res = self.execute(cmd, world)
            except SafetyError as e:
                res = ExecResult(False, f"safety: {e}")
            args = dict(cmd.args)
            if cmd.tool == "pick" and res.ok:
                o = self.holding_obj
                args.update(color=o.color_hint, at=[round(o.centroid_mm[0]), round(o.centroid_mm[1])])
            self.record(cmd.tool, args, res)
            self.hud_update(overlay, wrist, step, t0, f"{cmd.tool} -> {res.message}")
            if cmd.tool == "done":
                return f"done: {res.message}"
        return f"max_steps {self.max_steps} reached"

    def hud_update(self, overlay, wrist, step, t0, status) -> None:
        if self.hud is None:
            return
        self.hud.update(overlay, wrist, dict(
            step=step, status=status, ee_pose=self.robot.get_ee_pose(), holding=self.holding,
            gripper_open=self.robot.gripper_open, last_call=self.history[-1] if self.history else None,
            say=self.last_say, rules=self.rules.list() + self.hints, latency_ms=int((time.time() - t0) * 1000)))


def build(args) -> Loop:
    cfg = cfgmod.load(args.config) if args.config else cfgmod.load()
    if args.mock:
        from sortbot.robot import MockRobot
        scene = SimScene(MockRobot(cfg, realtime=False), cfg)
        robot, vlm, homog = scene, (VLM() if args.live_vlm else MockVLM()), Homography(cfg, scene.H)
    else:
        from sortbot.robot import SO101Robot
        robot, vlm, homog = SO101Robot(cfg), VLM(), Homography(cfg)
    voice = VoiceIO(cfg.elevenlabs_voice_id, stdin=io.StringIO("") if args.no_voice else None, force_text=args.mock or args.no_voice)
    voice.start()
    hud = None
    if not args.no_hud:
        from sortbot.hud import HUD
        hud = HUD(port=args.hud_port or cfg.hud_port)
        hud.start()
    rules = RulesStore(args.rules_file) if args.rules_file else RulesStore()
    loop = Loop(cfg, robot, vlm, voice, hud, homog, rules, args.max_steps)
    loop.attach_calibration(*make_calibration(cfg, robot, loop, mock=args.mock))
    return loop


def make_calibration(cfg, robot, loop: Loop, mock: bool):
    """(CalibController, mock_out). mock: FakeRig renders the target at SimScene's ground-truth projection and a
    VirtualLeader scripts the captures, writing to a temp file so the real calib.json is never touched."""
    from sortbot import calibrate as cal

    target = cal.ColorTarget.parse(cfg.calib_target)
    if mock:
        out = Path(tempfile.mkdtemp()) / "calib_mock.json"
        rig, leader = cal.mock_rig(cfg, robot.robot, np.linalg.inv(robot.H), lambda: robot.capture("overhead"), target)
        ctrl = cal.CalibController(cfg, lambda: rig, lambda: leader, target, out, loop._calib_done,
                                   lambda: loop.last_overhead, driver=lambda ld, c: ld.drive(c, dwell_s=0.4))
        return ctrl, out
    rig = cal.RobotRig(robot)
    return cal.CalibController(cfg, lambda: rig, lambda: cal.open_leader(cfg), target, cfg.calib_file,
                               loop._calib_done, lambda: loop.last_overhead), None


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true", help="mock robot/VLM/frames, text voice")
    g.add_argument("--real", action="store_true", help="SO101 + cameras + OpenAI")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--no-voice", action="store_true", help="do not read stdin for corrections")
    ap.add_argument("--hud-port", type=int, default=0)
    ap.add_argument("--live-vlm", action="store_true", help="with --mock: use the real OpenAI VLM")
    ap.add_argument("--rules-file", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    loop = build(args)
    try:
        result = loop.run()
    finally:
        loop.voice.stop()
        if loop.hud:
            loop.hud.stop()
        if hasattr(loop.robot, "disconnect"):
            loop.robot.disconnect()
    log.info("finished: %s (placed %d)", result, loop.placed)
    return result


if __name__ == "__main__":
    main()
