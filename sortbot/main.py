"""Server-first entry point: `python -m sortbot.main` starts the HUD immediately with NOTHING connected;
robot / cameras / VLM are chosen from the page (RUN group: set_mode) and connected lazily. --mock / --real
are shortcuts that pre-select a combo and auto-start the sorting loop.

Modes (any combination; "real cams + mock robot" tunes perception with no arm):
  robot: mock (MockRobot kinematic sim) | real (SO101 follower) | off
  cams:  sim (SimScene synthetic blobs) | real (overhead+wrist OpenCV) | off
  vlm:   mock (deterministic) | live (OpenAI) | off

The Loop is startable/pausable/stoppable from the page repeatedly without restarting the process (a fresh
SimScene per start in sim cams). SafetyError / rejected commands are fed back to the VLM as the tool result,
never raised. torque_off is the E-STOP: cuts motor torque (real: bus.disable_torque()) and pauses the loop.
HUD calibration: "Start calibration" runs immediately when idle, else at the next step boundary (loop pauses).
"""
from __future__ import annotations

import argparse
import base64
import collections
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
from sortbot.models import PROVIDERS, ModelRegistry, yaml_set
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


class CamRig:
    """Any robot + real cameras: capture() reads OpenCV cams, everything else is the robot."""

    def __init__(self, robot: RobotAPI, cams):
        self.robot, self.cams = robot, cams

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def capture(self, name: str) -> np.ndarray:
        return self.cams.read(name)


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


class Control:
    """Pause/resume/stop/single-step gate shared between the Loop thread and the HUD actions."""

    def __init__(self):
        self.stop_ev, self.pause_ev = threading.Event(), threading.Event()
        self._steps, self._slock = 0, threading.Lock()
        self.phase = "idle"  # idle|running|paused|done|stopped|error

    def grant_steps(self, n: int = 1) -> None:
        with self._slock:
            self._steps += n

    def gate(self) -> bool:
        """Blocks while paused (unless a step token is granted); False = stop requested."""
        while not self.stop_ev.is_set():
            if not self.pause_ev.is_set():
                self.phase = "running"
                return True
            with self._slock:
                if self._steps > 0:
                    self._steps -= 1
                    self.phase = "running"
                    return True
            self.phase = "paused"
            time.sleep(0.05)
        return False


def png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


class DecisionLog:
    """Ring buffer (last 200) of decisions/events for the HUD LOG tab, served newest-first at GET /log.
    Entries: {i, step, t, tool, args, result, ok, say, latency_ms, thumb_b64 (160px overlay jpeg)}."""

    def __init__(self, maxlen: int = 200):
        self._d: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._n = 0

    def add(self, tool: str, args: dict, result: str, ok: bool = True, step: int = 0, say: str = "",
            latency_ms: int | None = None, frame: np.ndarray | None = None) -> None:
        thumb = None
        if frame is not None:
            try:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (160, max(1, round(h * 160 / w))), interpolation=cv2.INTER_AREA)
                okj, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                if okj:
                    thumb = base64.b64encode(buf).decode()
            except Exception as e:  # noqa: BLE001
                log.debug("log thumb: %s", e)
        with self._lock:
            self._n += 1
            self._d.append({"i": self._n, "step": int(step), "t": round(time.time(), 2), "tool": tool,
                            "args": args, "result": result, "ok": bool(ok), "say": say,
                            "latency_ms": latency_ms, "thumb_b64": thumb})

    def entries(self) -> list[dict]:
        """Newest first."""
        with self._lock:
            return list(self._d)[::-1]

    def clear(self) -> int:
        with self._lock:
            n = len(self._d)
            self._d.clear()
            return n


class Loop:
    def __init__(self, cfg, robot, vlm, voice, hud, homography, rules: RulesStore, max_steps: int,
                 task: str = "", ctl: Control | None = None, lock: threading.RLock | None = None,
                 step_delay_s: float = 0.0):
        self.cfg, self.robot, self.vlm, self.voice, self.hud = cfg, robot, vlm, voice, hud
        self.homog, self.rules, self.max_steps = homography, rules, max_steps
        self.task, self.ctl, self.lock = task, ctl, lock or threading.RLock()
        self.step_delay_s = step_delay_s
        self.step = 0
        self.history: list[dict] = []
        self.hints: list[str] = []
        self.holding: int | None = None
        self.holding_obj: DetectedObject | None = None  # ids are renumbered every frame; keep what was picked
        self.filled: set[str] = set()
        self.placed = 0
        self.last_say = ""
        self.calib = None  # CalibController (see attach_calibration / Session)
        self._calib_requested = threading.Event()
        self.last_overhead: np.ndarray | None = None
        self.frame_sink = None  # optional callable(overhead) so the Session sees the latest raw frame
        self.detector = None    # shared perception.ClassicalDetector (Session-tuned); default per detect_objects
        self.dlog: DecisionLog | None = None
        self.view = None        # optional callable(overlay, raw) -> frame shown on /overhead.mjpg (mask view)
        self._overlay: np.ndarray | None = None  # overlay at decision time (log thumbnails)
        self._t0: float | None = None

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
        if self.dlog is not None:
            self.dlog.add(tool, args, self.history[-1]["result"], ok=r.ok, step=self.step,
                          say=self.last_say if tool == "say" else "",
                          latency_ms=int((time.time() - self._t0) * 1000) if self._t0 else None,
                          frame=self._overlay)

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
        self.hud.register("calib_start", self._request_calib, "Start calibration", "calibration",
                          help="Begin the teleoperated camera calibration at the next step boundary (the loop pauses).")

    def _request_calib(self) -> dict:
        if self.calib.active or self._calib_requested.is_set():
            return {"ok": False, "message": "calibration already running/requested", "data": None}
        self._calib_requested.set()
        return {"ok": True, "message": "calibration will start at the next step boundary (loop pauses)", "data": None}

    def _calib_done(self, session) -> None:
        if session.state == "fitted":
            self.homog.reload(getattr(self, "_calib_out", None))
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

        while self.step < self.max_steps:
            if self.ctl is not None and not self.ctl.gate():
                return "stopped"
            self.step += 1
            step = self.step
            t0 = time.time()
            self._t0 = t0
            self._maybe_calibrate()
            with self.lock:
                hr = self.robot.home()
            if not hr.ok:  # routine homing is not worth a history slot; failures are
                self.record("home", {}, hr)
            if not self.drain_voice():  # before capture so voice actions cannot stale the frame
                return "stopped by voice"
            overhead, wrist = self.robot.capture("overhead"), self.robot.capture("wrist")
            self.last_overhead = overhead
            if self.frame_sink:
                self.frame_sink(overhead)
            H = self.homog.update(overhead)
            if H is None:
                log.warning("no homography (no calib.json H and ArUco tags not visible); run calibration")
                self.hud_update(overhead, wrist, step, t0, "no homography - calibrate")
                time.sleep(0.5)
                continue
            objects = perception.detect_objects(overhead, H, self.cfg, detector=self.detector,
                                                filled_zones=sorted(self.filled), method=self.homog.method)
            rules = ([f"GOAL: {self.task}"] if self.task else []) + self.rules.list() + self.hints
            if self.holding_obj is not None:
                o = self.holding_obj
                rules = rules + [f"holding: the {o.color_hint} object picked from ({o.centroid_mm[0]:.0f},{o.centroid_mm[1]:.0f}); object ids below are renumbered"]
            world = WorldState(objects, self.cfg.zones, self.robot.get_ee_pose(), self.robot.gripper_open, self.holding, rules)
            overlay = perception.render_overlay(overhead, H, objects, self.cfg.zones, world.ee_pose, rules)
            self._overlay = overlay
            self.hud_update(overlay, wrist, step, t0, "planning")
            try:
                cmd = self.vlm.plan_step(png(overlay), png(wrist), world, self.history)
            except Exception as e:  # noqa: BLE001
                log.error("VLM failed: %s", e)
                self.history.append({"tool": "vlm", "args": {}, "result": f"error {e}"})
                if self.dlog is not None:
                    self.dlog.add("vlm", {}, f"error {e}", ok=False, step=step, frame=overlay,
                                  latency_ms=int((time.time() - t0) * 1000))
                continue
            err = self.validate(cmd, world)
            if err:
                self.record(cmd.tool, cmd.args, ExecResult(False, f"rejected: {err}"))
                continue
            try:
                with self.lock:
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
            if self.step_delay_s:
                time.sleep(self.step_delay_s)
        return f"max_steps {self.max_steps} reached"

    def hud_update(self, overlay, wrist, step, t0, status) -> None:
        if self.hud is None:
            return
        if self.view is not None:
            overlay = self.view(overlay, self.last_overhead)
        self.hud.update(overlay, wrist, dict(
            step=step, status=status, ee_pose=self.robot.get_ee_pose(), holding=self.holding,
            gripper_open=self.robot.gripper_open, last_call=self.history[-1] if self.history else None,
            say=self.last_say, rules=self.rules.list() + self.hints, latency_ms=int((time.time() - t0) * 1000)))


class Session:
    """Owns the devices (robot / cams / vlm), one Loop thread at a time, and the RUN + ROBOT HUD actions.
    Everything is reachable over POST /action/*; nothing requires the terminal."""

    ROBOT_MODES, CAM_MODES, VLM_MODES = ("mock", "real", "off"), ("sim", "real", "off"), ("mock", "live", "off")
    JOG_AXES = ("x", "y", "z", "roll")

    def __init__(self, cfg: cfgmod.Config, hud, voice: VoiceIO, rules: RulesStore,
                 max_steps: int = 40, step_delay_s: float = 0.25):
        self.cfg, self.hud, self.voice, self.rules = cfg, hud, voice, rules
        self.mode: dict[str, str | None] = {"robot": None, "cams": None, "vlm": None}
        self.robot = None          # MockRobot | SO101Robot
        self.cams = None           # robot_mod.Cameras (real cams only)
        self.vlm = None            # MockVLM | VLM
        self.rig = None            # SimScene | CamRig (what the Loop drives)
        self.homog_real: Homography | None = None  # persistent TableHomography for real cams
        self.loop: Loop | None = None
        self.ctl: Control | None = None
        self.thread: threading.Thread | None = None
        self.max_steps, self.task = max_steps, ""
        self.step_delay_s = step_delay_s
        self.last_error, self.result = "", ""
        self.robot_lock = threading.RLock()
        self._robot_cache: dict = {}
        self.calib, self._calib_out = None, None
        self.models = ModelRegistry()
        self.det_params = perception.DetectorParams(cfg.det_v_min, cfg.det_s_min, cfg.det_area_min, cfg.det_area_max)
        self.detector = perception.ClassicalDetector(self.det_params)  # shared: loop + redetect + mask view
        self.mask_view = False
        self.dlog = DecisionLog()
        self._overlay_hold: tuple[float, np.ndarray] | None = None  # (until_ts, overlay) shown while idle after redetect
        self._calib_flag: tuple[float, bool] | None = None  # (calib.json mtime, has fitted H) cache
        self.last_overhead: np.ndarray | None = None
        self._shutdown = threading.Event()
        self._register()
        threading.Thread(target=self._preview_loop, daemon=True, name="preview").start()

    # ------------------------------------------------ registration
    def _register(self) -> None:
        if self.hud is None:
            return
        h = self.hud
        h.register("set_mode", self.set_mode, None, "run",  # driven by the Setup mode cards / pills
                   help="Connect or disconnect devices: robot=mock|real|off, cams=sim|real|off, vlm=mock|live|off. Any combination; connected lazily.")
        h.register("start", self.start, "Start", "run",
                   help="Start a sorting run with the connected devices (needs robot + cams + VLM).")
        h.register("pause", self.pause, "Pause", "run",
                   help="Pause the run; the current step finishes first.")
        h.register("resume", self.resume, "Resume", "run",
                   help="Continue a paused run (refused while torque is off after an E-STOP).")
        h.register("stop", self.stop, "Stop", "run",
                   help="End the run at the next step boundary; devices stay connected for the next Start.")
        h.register("step_once", self.step_once, "Step once", "run",
                   help="Run exactly one step; from idle the run is started paused first.")
        h.register("set_max_steps", self.set_max_steps, "Set max steps", "run",
                   help="Cap the number of steps for this and future runs (n >= 1).")
        h.register("set_task", self.set_task, "Set task", "run",
                   help="Free-text goal sent to the VLM as GOAL: ... (empty = sort sensibly into the zones).")
        h.register("home", lambda: self._robot_act(lambda r: r.home()), "Home", "robot", params=[],
                   help="Move the arm to its HOME pose above the table.")
        h.register("open_gripper", lambda: self._robot_act(lambda r: r.open_gripper()), "Open gripper", "robot", params=[],
                   help="Open the gripper (drops whatever it is holding).")
        h.register("close_gripper", lambda: self._robot_act(lambda r: r.close_gripper()), "Close gripper", "robot", params=[],
                   help="Close the gripper.")
        h.register("jog", self.jog, "Jog", "robot",
                   help="Nudge the arm along one axis (axis: x|y|z|roll; delta in mm, degrees for roll), through the safety envelope.")
        h.register("goto", self.goto, "Go to", "robot",
                   help="Move the gripper to table-frame (x, y, z) mm, through the safety envelope.")
        h.register("torque_off", self.torque_off, "E-STOP (torque off)", "robot", params=[],
                   help="E-STOP: cut motor torque immediately and pause the run; all motion is refused until torque_on.")
        h.register("torque_on", self.torque_on, "Torque on", "robot", params=[],
                   help="Re-enable motor torque after an E-STOP.")
        h.register("say_to_bot", self.say_to_bot, "Send to bot", "voice",
                   help="Send a typed correction through the voice classifier (rule / hint / immediate command), exactly like speech.")
        h.register("say_to_robot", self.say_to_bot, None, "voice",
                   help="Alias of say_to_bot, kept for older pages/scripts.")
        h.register("transcribe", self.transcribe, None, "voice",
                   help="Push-to-talk: POST base64 audio (webm/opus and friends); ElevenLabs STT transcribes it and the text goes through say_to_bot.")
        h.register("speak", self.speak, "Speak (TTS test)", "voice",
                   help="Say the text aloud via ElevenLabs TTS with the current voice/model (plays on the server machine).")
        h.register("mic_on", lambda: self._voice_event("mic_on", self.voice.mic_on()), "Listening ON", "voice",
                   help="Start hands-free listening: the mic transcribes continuously until toggled off. A MIC LIVE chip shows while on.")
        h.register("mic_off", lambda: self._voice_event("mic_off", self.voice.mic_off()), "Listening OFF", "voice",
                   help="Stop hands-free listening. Push-to-talk and the text box keep working.")
        h.register("add_rule", self.add_rule, "Add rule", "rules",
                   help="Append a persistent rule; rules ride along with every VLM prompt and survive restarts.")
        h.register("delete_rule", self.delete_rule, None, "rules",
                   help="Delete the rule at 0-based index i (the x button in the list).")
        h.register("move_rule", self.move_rule, None, "rules",
                   help="Move rule i one slot up or down (dir: up|down); earlier rules read as higher priority.")
        h.register("clear_hints", self.clear_hints, "Clear hints", "rules",
                   help="Drop the one-shot (human) hints accumulated during the current run; persisted rules are kept.")
        h.register("get_models", self.get_models, "Refresh models", "models",
                   help="List selectable OpenAI / ElevenLabs models and voices (OpenAI listing cached 5 min).")
        h.register("set_model", self.set_model, None, "models",
                   help="Hot-swap a model (provider: openai|elevenlabs_tts|elevenlabs_stt|elevenlabs_voice) on the live clients and persist it to config.yaml.")
        h.register("set_detector_params", self.set_detector_params, None, "perception",
                   help="Tune the blob detector live (v_min/s_min HSV foreground thresholds, area_min/area_max px^2); persists to config.yaml perception:. Driven by the sliders.")
        h.register("redetect", self.redetect, "Redetect", "perception", params=[],
                   help="Re-run detection on the latest overhead frame and refresh the overlay, without a robot step.")
        h.register("toggle_mask", self.toggle_mask, "Toggle mask view", "perception", params=[],
                   help="Stream the detector's binary foreground mask on /overhead.mjpg instead of the overlay (toggle back for the overlay).")
        h.register("set_zone_drop", self.set_zone_drop, "Set zone drop", "perception",
                   help="Move a zone's drop point to table-frame (x, y) mm and persist it to config.yaml zones; or use 'set drop' + click on the overhead image.")
        h.register("px_to_mm", self.px_to_mm, None, "perception",
                   help="Convert an overhead pixel (u, v) to table-frame mm via the current homography (used by the click-to-set-drop mode).")
        h.register("log_clear", self.log_clear, "Clear log", "log", params=[],
                   help="Empty the decision log (ring buffer of the last 200 decisions/events, served at GET /log).")
        h.add_state_source("run", self._run_state)
        h.add_state_source("robot", self._robot_state)
        h.add_state_source("voice", self._voice_state)
        h.add_state_source("rules", self._rules_state)
        h.add_state_source("vlm", self._vlm_state)
        h.add_state_source("perception", self._perception_state)
        h.add_route("/log", self.dlog.entries)
        self._register_calib_placeholders()

    def _register_calib_placeholders(self) -> None:
        if self.hud is None:
            return
        no = lambda **kw: {"ok": False, "message": "no robot connected; pick one in the header first", "data": None}  # noqa: E731
        for name, label in (("calib_start", "Start calibration"), ("calib_touch", "Touch table"),
                            ("calib_capture", "Capture"), ("calib_undo", "Undo"),
                            ("calib_finish", "Finish"), ("calib_cancel", "Cancel"), ("calib_sample", None)):
            self.hud.register(name, no, label, "calibration", params=[],
                              help="Calibration needs a robot: pick a mode in Setup first.")
        self.hud.add_state_source("calibration", lambda: {"state": "idle", "n": 0, "message": "connect a robot first"})

    # ------------------------------------------------ mode / devices
    def _running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def set_mode(self, robot: str | None = None, cams: str | None = None, vlm: str | None = None) -> dict:
        """Lazily (dis)connect devices; errors are reported in the response, never raised."""
        if self._running():
            return {"ok": False, "message": "stop the run before changing mode", "data": None}
        errs, done = [], []
        for key, val, allowed, fn in (("robot", robot, self.ROBOT_MODES, self._set_robot),
                                      ("cams", cams, self.CAM_MODES, self._set_cams),
                                      ("vlm", vlm, self.VLM_MODES, self._set_vlm)):
            if val is None:
                continue
            val = str(val).lower()
            if val not in allowed:
                errs.append(f"{key}={val!r} (allowed: {'|'.join(allowed)})")
                continue
            try:
                fn(val)
                done.append(f"{key}={val}")
            except Exception as e:  # noqa: BLE001
                self.mode[key] = None
                errs.append(f"{key}={val}: {e}")
        self._refresh_rig()
        if errs:
            self.last_error = "; ".join(errs)
        msg = "; ".join(done + [f"ERROR {e}" for e in errs]) or "nothing changed"
        if done or errs:
            self.dlog.add("set_mode", {k: v for k, v in (("robot", robot), ("cams", cams), ("vlm", vlm)) if v is not None},
                          msg, ok=not errs, step=self.loop.step if self.loop is not None else 0)
        return {"ok": not errs, "message": msg, "data": {"mode": self.mode, "connected": self._connected()}}

    def _set_robot(self, v: str) -> None:
        if v == self.mode["robot"]:
            return
        if self.robot is not None and hasattr(self.robot, "disconnect"):
            try:
                self.robot.disconnect()
            except Exception as e:  # noqa: BLE001
                log.warning("robot disconnect: %s", e)
        self.robot, self.calib, self.mode["robot"] = None, None, None
        self._register_calib_placeholders()
        if v == "off":
            return
        from sortbot.robot import MockRobot, SO101Robot
        self.robot = MockRobot(self.cfg, realtime=False) if v == "mock" else SO101Robot(self.cfg, with_cameras=False)
        self.mode["robot"] = v
        self._attach_calib()

    def _set_cams(self, v: str) -> None:
        if v == self.mode["cams"]:
            return
        if self.cams is not None:
            self.cams.disconnect()
            self.cams = None
        self.mode["cams"] = None
        if v == "off":
            return
        if v == "real":
            from sortbot.robot import Cameras
            self.cams = Cameras(self.cfg)
        self.mode["cams"] = v

    def _set_vlm(self, v: str) -> None:
        self.vlm, self.mode["vlm"] = None, None
        if v == "off":
            return
        self.vlm = MockVLM() if v == "mock" else VLM(self.cfg.openai_model)
        self.mode["vlm"] = v

    def _refresh_rig(self) -> None:
        """Rig used for idle previews and as the template for the next start."""
        if self.mode["cams"] == "real" and self.cams is not None and self.robot is not None:
            self.rig = CamRig(self.robot, self.cams)
        elif self.mode["cams"] == "sim" and self.robot is not None:
            if not isinstance(self.rig, SimScene) or self.rig.robot is not self.robot:
                self.rig = SimScene(self.robot, self.cfg)
        else:
            self.rig = None

    def _connected(self) -> dict:
        return {"robot": self.robot is not None,
                "cams": self.mode["cams"] == "sim" or self.cams is not None,
                "vlm": self.vlm is not None}

    # ------------------------------------------------ calibration
    def _attach_calib(self) -> None:
        from sortbot import calibrate as cal

        target = cal.ColorTarget.parse(self.cfg.calib_target)
        if self.mode["robot"] == "mock":
            self._calib_out = Path(tempfile.mkdtemp()) / "calib_mock.json"
            w, h = self.cfg.overhead_cam.width, self.cfg.overhead_cam.height
            H_sim = perception.synth_homography(self.cfg, w, h)
            pair = []

            def make():
                if not pair:  # dark-mat background: the scene's own green blob must not distract the detector
                    pair.extend(cal.mock_rig(self.cfg, self.robot, np.linalg.inv(H_sim), None, target))
                return pair

            self.calib = cal.CalibController(self.cfg, lambda: make()[0], lambda: make()[1], target, self._calib_out,
                                             self._calib_done, lambda: self.last_overhead,
                                             driver=lambda ld, c: ld.drive(c, dwell_s=0.2))
        else:
            self._calib_out = None
            rig = cal.RobotRig(self.robot, lambda: self._grab_frame("overhead"))
            self.calib = cal.CalibController(self.cfg, lambda: rig, lambda: cal.open_leader(self.cfg), target,
                                             self.cfg.calib_file, self._calib_done, lambda: self.last_overhead)
        if self.hud is not None:
            self.calib.register(self.hud)
            self.hud.register("calib_start", self._calib_start, "Start calibration", "calibration", params=[],
                              help="Begin the teleoperated camera calibration (put the target ball in the gripper first); "
                                   "mid-run it starts at the next step boundary.")

    def _calib_start(self) -> dict:
        if self.calib is None:
            return {"ok": False, "message": "no robot connected", "data": None}
        if self._running() and self.ctl is not None and not self.ctl.pause_ev.is_set():
            if self.calib.active or (self.loop and self.loop._calib_requested.is_set()):
                return {"ok": False, "message": "calibration already running/requested", "data": None}
            self.loop._calib_requested.set()
            return {"ok": True, "message": "calibration will start at the next step boundary (loop pauses)", "data": None}
        return self.calib.start()

    def _calib_done(self, session) -> None:
        if session.state == "fitted":
            if self.mode["robot"] == "mock":
                if self.loop is not None and self.loop.homog.fixed is not None:
                    self.loop.homog.reload(self._calib_out)
            else:  # real: the z offset may have changed too
                import sortbot.robot as robot_mod
                self.robot.table_T_base = robot_mod.load_calib(self.cfg.calib_file)
                self.robot.base_T_table = np.linalg.inv(self.robot.table_T_base)
                if self.homog_real is not None:
                    self.homog_real.reload()
        log.info("calibration %s: %s", session.state, session.message)

    def _grab_frame(self, name: str) -> np.ndarray:
        if self.cams is not None:
            return self.cams.read(name)
        if self.rig is not None:
            return self.rig.capture(name)
        raise RuntimeError("no cameras connected (set_mode cams)")

    # ------------------------------------------------ RUN actions
    def start(self, paused: bool = False) -> dict:
        if self._running():
            return {"ok": False, "message": "already running; stop first", "data": None}
        missing = [k for k, ok in self._connected().items() if not ok]
        if missing:
            return {"ok": False, "message": f"not connected: {', '.join(missing)} -- pick modes in the header (set_mode)", "data": None}
        if self.mode["cams"] == "sim":
            self.rig = SimScene(self.robot, self.cfg)  # fresh scene per start
            homog = Homography(self.cfg, self.rig.H)
        else:
            self.rig = CamRig(self.robot, self.cams)
            if self.homog_real is None:
                self.homog_real = Homography(self.cfg)
            homog = self.homog_real
        self.ctl = Control()
        if paused:
            self.ctl.pause_ev.set()
        delay = self.step_delay_s if self.mode["robot"] == "mock" else 0.0
        self.loop = Loop(self.cfg, self.rig, self.vlm, self.voice, self.hud, homog, self.rules,
                         self.max_steps, task=self.task, ctl=self.ctl, lock=self.robot_lock, step_delay_s=delay)
        self.loop.calib, self.loop._calib_out = self.calib, self._calib_out
        self.loop.frame_sink = lambda f: setattr(self, "last_overhead", f)
        self.loop.detector, self.loop.dlog, self.loop.view = self.detector, self.dlog, self._overhead_view
        self.result = ""
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="sortbot-loop")
        self.thread.start()
        return {"ok": True, "message": "run started" + (" (paused)" if paused else ""), "data": None}

    def _run_loop(self) -> None:
        try:
            self.result = self.loop.run()
            self.ctl.phase = "stopped" if self.ctl.stop_ev.is_set() else "done"
        except Exception as e:  # noqa: BLE001
            log.exception("loop crashed")
            self.last_error = str(e)
            self.result = f"error: {e}"
            self.ctl.phase = "error"
        log.info("finished: %s (placed %d)", self.result, self.loop.placed)

    def pause(self) -> dict:
        if not self._running():
            return {"ok": False, "message": "not running", "data": None}
        self.ctl.pause_ev.set()
        return {"ok": True, "message": "pausing (current step finishes first)", "data": None}

    def resume(self) -> dict:
        if not self._running():
            return {"ok": False, "message": "not running (use start)", "data": None}
        if self.robot is not None and not getattr(self.robot, "torque", True):
            return {"ok": False, "message": "torque is off (E-STOP); press torque_on first", "data": None}
        self.ctl.pause_ev.clear()
        return {"ok": True, "message": "resumed", "data": None}

    def stop(self) -> dict:
        if not self._running():
            return {"ok": False, "message": "not running", "data": None}
        self.ctl.stop_ev.set()
        return {"ok": True, "message": "stop requested (takes effect at the step boundary)", "data": None}

    def step_once(self) -> dict:
        if self._running():
            if not self.ctl.pause_ev.is_set():
                return {"ok": False, "message": "already running; pause first to single-step", "data": None}
            self.ctl.grant_steps(1)
            return {"ok": True, "message": "stepping once", "data": None}
        r = self.start(paused=True)
        if not r["ok"]:
            return r
        self.ctl.grant_steps(1)
        return {"ok": True, "message": "started paused; stepping once", "data": None}

    def set_max_steps(self, n: int) -> dict:
        n = int(n)
        if n < 1:
            return {"ok": False, "message": "n must be >= 1", "data": None}
        self.max_steps = n
        if self.loop is not None:
            self.loop.max_steps = n
        return {"ok": True, "message": f"max_steps = {n}", "data": None}

    def set_task(self, text: str = "") -> dict:
        self.task = str(text).strip()
        if self.loop is not None:
            self.loop.task = self.task
        return {"ok": True, "message": f"task: {self.task or '(default: sort sensibly)'}", "data": None}

    def say_to_bot(self, text: str) -> dict:
        """Same path as spoken corrections: classifier -> rule / immediate command / VLM hint."""
        text = str(text).strip()
        if not text:
            return {"ok": False, "message": "empty text", "data": None}
        it = classify(text)
        if it.kind == "rule" and not self._running():
            self.rules.append(it.text)  # nothing drains the queue while idle; persist now so the page shows it
            self.dlog.add("voice", {"text": text}, f"rule added: {it.text}",
                          step=self.loop.step if self.loop is not None else 0)
            return {"ok": True, "message": f"rule added: {it.text}", "data": {"kind": "rule"}}
        self.voice.push(text)
        kind = {"rule": "rule", "action": "command", "unknown": "hint"}[it.kind]
        when = "" if self._running() else " (queued until the next run)"
        self.dlog.add("voice", {"text": text}, f"heard as {kind}: {it.text}{when}",
                      step=self.loop.step if self.loop is not None else 0)
        return {"ok": True, "message": f"heard as {kind}: {it.text}{when}", "data": {"kind": it.kind}}

    def transcribe(self, audio_b64: str, mime: str = "audio/webm") -> dict:
        """HUD push-to-talk: base64 audio -> ElevenLabs STT -> say_to_bot."""
        try:
            data = base64.b64decode(str(audio_b64))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": f"bad base64 audio: {e}", "data": None}
        if not data:
            return {"ok": False, "message": "empty audio", "data": None}
        text = self.voice.transcribe_bytes(data, str(mime))  # raises without a key -> reported by the HUD wrapper
        if not text:
            return {"ok": False, "message": "no speech recognized", "data": None}
        r = self.say_to_bot(text)
        return {"ok": True, "message": f'"{text}" -> {r["message"]}',
                "data": {"text": text, "kind": (r.get("data") or {}).get("kind")}}

    def speak(self, text: str) -> dict:
        text = str(text).strip()
        if not text:
            return {"ok": False, "message": "empty text", "data": None}
        self.voice.speak(text)
        if self.voice._client is None:
            return {"ok": True, "message": "no ELEVENLABS_API_KEY: logged to the server console only", "data": None}
        return {"ok": True, "message": f"speaking ({self.voice.tts_model}, voice {self.voice.voice_id})", "data": None}

    # ------------------------------------------------ RULES actions
    def add_rule(self, text: str) -> dict:
        text = str(text).strip()
        if not text:
            return {"ok": False, "message": "empty rule", "data": None}
        self.rules.append(text)
        return {"ok": True, "message": f"rule added: {text}", "data": {"rules": self.rules.list()}}

    def delete_rule(self, i: int) -> dict:
        r = self.rules.delete(int(i))
        if r is None:
            return {"ok": False, "message": f"no rule at index {i}", "data": None}
        return {"ok": True, "message": f"deleted rule {i}: {r}", "data": {"rules": self.rules.list()}}

    def move_rule(self, i: int, dir: str = "up") -> dict:
        d = str(dir).lower()
        if d not in ("up", "down"):
            return {"ok": False, "message": "dir must be up|down", "data": None}
        if not self.rules.move(int(i), -1 if d == "up" else 1):
            return {"ok": False, "message": f"cannot move rule {i} {d}", "data": None}
        return {"ok": True, "message": f"moved rule {i} {d}", "data": {"rules": self.rules.list()}}

    def clear_hints(self) -> dict:
        n = len(self.loop.hints) if self.loop is not None else 0
        if self.loop is not None:
            self.loop.hints = []
        return {"ok": True, "message": f"cleared {n} hint(s)", "data": None}

    # ------------------------------------------------ MODELS actions
    def _model_current(self) -> dict:
        return {"openai": self.cfg.openai_model, "elevenlabs_tts": self.voice.tts_model,
                "elevenlabs_stt": self.voice.stt_model, "elevenlabs_voice": self.voice.voice_id}

    def get_models(self) -> dict:
        d = self.models.get(self._model_current())
        msg = "; ".join(d["notes"]) or f"{len(d['openai'])} openai models, {len(d['elevenlabs']['voices'])} voices"
        return {"ok": True, "message": msg, "data": d}

    def set_model(self, provider: str, value: str) -> dict:
        """Hot-swap on the live objects (VLM / VoiceIO) + persist to config.yaml (comments preserved)."""
        provider, value = str(provider).lower(), str(value).strip()
        if provider not in PROVIDERS:
            return {"ok": False, "message": f"provider must be one of {'|'.join(PROVIDERS)}", "data": None}
        if not value:
            return {"ok": False, "message": "empty value", "data": None}
        if provider == "openai":
            self.cfg.openai_model = value
            if isinstance(self.vlm, VLM):
                self.vlm.model = value
            sec, key = "vlm", "model"
        elif provider == "elevenlabs_tts":
            self.voice.tts_model = self.cfg.tts_model = value
            sec, key = "voice", "tts_model"
        elif provider == "elevenlabs_stt":
            self.voice.stt_model = self.cfg.stt_model = value
            sec, key = "voice", "stt_model"
        else:
            self.voice.voice_id = self.cfg.elevenlabs_voice_id = value
            sec, key = "voice", "elevenlabs_voice_id"
        try:
            yaml_set(self.cfg.source_path, sec, key, value)
            persisted = f"persisted to {self.cfg.source_path.name}"
        except Exception as e:  # noqa: BLE001
            persisted = f"NOT persisted: {e}"
        return {"ok": True, "message": f"{provider} = {value} (live; {persisted})",
                "data": {"current": self._model_current()}}

    # ------------------------------------------------ PERCEPTION + LOG actions
    def _current_H(self) -> np.ndarray | None:
        """Homography for the latest overhead frame: the sim scene's fixed H, or the real TableHomography."""
        if isinstance(self.rig, SimScene):
            return self.rig.H
        if self.last_overhead is None:
            return None
        if self.homog_real is None:
            self.homog_real = Homography(self.cfg)
        return self.homog_real.update(self.last_overhead)

    def _homog_method(self) -> str:
        if isinstance(self.rig, SimScene) or self.homog_real is None:
            return "ball"
        return self.homog_real.method

    def set_detector_params(self, v_min=None, s_min=None, area_min=None, area_max=None) -> dict:
        """Any subset; applied to the shared DetectorParams (live) and persisted under config.yaml perception:."""
        p, sets = self.det_params, {}
        for k, v, lo, hi in (("v_min", v_min, 0, 255), ("s_min", s_min, 0, 255),
                             ("area_min", area_min, 0, 10**7), ("area_max", area_max, 1, 10**7)):
            if v is None:
                continue
            v = int(float(v))
            if not (lo <= v <= hi):
                return {"ok": False, "message": f"{k}={v} out of range [{lo}, {hi}]", "data": {"params": p.to_dict()}}
            sets[k] = v
        if not sets:
            return {"ok": False, "message": "nothing to set (v_min, s_min, area_min, area_max)", "data": {"params": p.to_dict()}}
        trial = {**p.to_dict(), **sets}
        if trial["area_min"] >= trial["area_max"]:
            return {"ok": False, "message": f"area_min {trial['area_min']} must be < area_max {trial['area_max']}",
                    "data": {"params": p.to_dict()}}
        for k, v in sets.items():
            setattr(p, k, v)
            setattr(self.cfg, f"det_{k}", v)
        try:
            for k, v in sets.items():
                yaml_set(self.cfg.source_path, "perception", k, str(v))
            persisted = f"persisted to {self.cfg.source_path.name}"
        except Exception as e:  # noqa: BLE001
            persisted = f"NOT persisted: {e}"
        return {"ok": True, "message": ", ".join(f"{k}={v}" for k, v in sets.items()) + f" (live; {persisted})",
                "data": {"params": p.to_dict()}}

    def redetect(self) -> dict:
        """Detection + overlay on the latest overhead frame, no robot step (idle: shown for a few seconds)."""
        frame = self.last_overhead
        if frame is None:
            return {"ok": False, "message": "no overhead frame yet (connect cameras with set_mode)", "data": None}
        H = self._current_H()
        if H is None:
            return {"ok": False, "message": "no homography (run calibration / show the ArUco tags)", "data": None}
        filled = sorted(self.loop.filled) if self.loop is not None else []
        objs = perception.detect_objects(frame, H, self.cfg, detector=self.detector,
                                         filled_zones=filled, method=self._homog_method())
        overlay = perception.render_overlay(frame, H, objs, self.cfg.zones, None, self.rules.list())
        self._overlay_hold = (time.time() + 5.0, overlay)
        if self.hud is not None and not self._running():
            self.hud.update(self._overhead_view(overlay, frame), None, {})
        data = {"n": len(objs),
                "objects": [{"id": o.id, "x_mm": round(o.centroid_mm[0], 1), "y_mm": round(o.centroid_mm[1], 1),
                             "px": [int(round(c)) for c in perception.mm_to_px(H, [o.centroid_mm])[0]],
                             "area_px": o.area_px, "color": o.color_hint} for o in objs]}
        return {"ok": True, "message": f"{len(objs)} object(s) detected", "data": data}

    def toggle_mask(self) -> dict:
        self.mask_view = not self.mask_view
        return {"ok": True, "message": "mask view ON (binary detector mask on /overhead.mjpg)" if self.mask_view
                else "mask view off (overlay)", "data": {"mask": self.mask_view}}

    def _overhead_view(self, overlay: np.ndarray, raw: np.ndarray | None) -> np.ndarray:
        """What /overhead.mjpg shows: the overlay, or the detector's binary mask while toggle_mask is on."""
        if not self.mask_view or raw is None:
            return overlay
        m = cv2.cvtColor(self.detector.fg_mask(raw), cv2.COLOR_GRAY2RGB)
        p = self.det_params
        cv2.putText(m, f"MASK  v>{p.v_min} | s>{p.s_min}   area {p.area_min}-{p.area_max}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return m

    def set_zone_drop(self, name: str, x: float, y: float) -> dict:
        """Move a zone's drop point (table mm); persists into config.yaml zones (rect kept as is)."""
        z = self.cfg.zone(str(name))
        if z is None:
            return {"ok": False, "message": f"unknown zone {name!r}; zones: {[q.name for q in self.cfg.zones]}", "data": None}
        x, y = round(float(x), 1), round(float(y), 1)
        lo, hi = self.cfg.aabb_min_mm, self.cfg.aabb_max_mm
        if not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]):
            return {"ok": False, "message": f"({x}, {y}) outside workspace x[{lo[0]},{hi[0]}] y[{lo[1]},{hi[1]}]", "data": None}
        z.drop_point_mm = (x, y)
        warn = "" if z.contains(x, y) else " (note: outside the zone rectangle)"
        (x0, y0), (x1, y1) = z.polygon_mm[0], z.polygon_mm[2]
        try:
            yaml_set(self.cfg.source_path, "zones", z.name,
                     f"{{rect: [[{x0}, {y0}], [{x1}, {y1}]], drop: [{x}, {y}]}}")
            persisted = f"persisted to {self.cfg.source_path.name}"
        except Exception as e:  # noqa: BLE001
            persisted = f"NOT persisted: {e}"
        self.dlog.add("set_zone_drop", {"name": z.name, "x": x, "y": y}, f"drop moved{warn}; {persisted}",
                      step=self.loop.step if self.loop is not None else 0)
        if not self._running():
            self.redetect()  # re-render the overlay with the new drop marker
        return {"ok": True, "message": f"{z.name} drop -> ({x}, {y}){warn}; {persisted}",
                "data": {"zones": self._perception_state()["zones"]}}

    def px_to_mm(self, u: float, v: float) -> dict:
        H = self._current_H()
        if H is None:
            return {"ok": False, "message": "no homography (connect cameras / calibrate first)", "data": None}
        x, y = perception.px_to_mm(H, [(float(u), float(v))])[0]
        return {"ok": True, "message": f"({u}, {v}) px -> ({x:.1f}, {y:.1f}) mm",
                "data": {"x": round(float(x), 1), "y": round(float(y), 1)}}

    def log_clear(self) -> dict:
        n = self.dlog.clear()
        return {"ok": True, "message": f"cleared {n} log entries", "data": None}

    def _calibrated(self) -> bool:
        """Is a px->mm homography available for the current cams? (sim: always; real: fitted H in calib.json).
        Drives the page's 'Not calibrated' banner + first-run checklist; cached on the calib file's mtime."""
        if self.mode["cams"] == "sim":
            return True
        p = self.cfg.calib_file
        try:
            mt = p.stat().st_mtime
        except OSError:
            return False
        if self._calib_flag is None or self._calib_flag[0] != mt:
            try:
                from sortbot.calibration import load_calib_dict
                self._calib_flag = (mt, load_calib_dict(p)["H_px_to_mm"] is not None)
            except Exception:  # noqa: BLE001
                self._calib_flag = (mt, False)
        return self._calib_flag[1]

    def _perception_state(self) -> dict:
        return {"params": self.det_params.to_dict(), "mask": self.mask_view, "calibrated": self._calibrated(),
                "zones": [{"name": z.name, "drop": [z.drop_point_mm[0], z.drop_point_mm[1]],
                           "rect": [list(z.polygon_mm[0]), list(z.polygon_mm[2])]} for z in self.cfg.zones]}

    # ------------------------------------------------ ROBOT actions
    def _robot_act(self, fn, allow_while_running: bool = False) -> dict:
        if self.robot is None:
            return {"ok": False, "message": "no robot connected (set_mode robot=mock|real)", "data": None}
        if not allow_while_running and self._running() and not self.ctl.pause_ev.is_set():
            return {"ok": False, "message": "loop is running; pause or stop first", "data": None}
        if not self.robot_lock.acquire(timeout=2.0):
            return {"ok": False, "message": "robot busy (finishing a motion); retry", "data": None}
        try:
            r = fn(self.robot)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": str(e), "data": None}
        finally:
            self.robot_lock.release()
        return {"ok": r.ok, "message": r.message, "data": None}

    def jog(self, axis: str, delta: float = 10.0) -> dict:
        """delta: mm for x|y|z, degrees for roll. Goes through the normal safety envelope."""
        axis = str(axis).lower()
        if axis not in self.JOG_AXES:
            return {"ok": False, "message": f"axis must be one of {'|'.join(self.JOG_AXES)}", "data": None}
        d = float(delta)

        def do(r):
            p = r.get_ee_pose()
            if axis == "roll":
                return r.turn_to(p.roll_deg + d)
            x, y, z = p.x + d * (axis == "x"), p.y + d * (axis == "y"), p.z + d * (axis == "z")
            return r.move_to(x, y, z)

        return self._robot_act(do)

    def goto(self, x: float, y: float, z: float) -> dict:
        return self._robot_act(lambda r: r.move_to(float(x), float(y), float(z)))

    def torque_off(self) -> dict:
        """E-STOP: flag first (any in-flight motion aborts within one tick), pause the loop, then cut bus torque."""
        if self.robot is None:
            return {"ok": False, "message": "no robot connected", "data": None}
        self.robot.torque = False
        if self.ctl is not None:
            self.ctl.pause_ev.set()
        got = self.robot_lock.acquire(timeout=1.0)
        try:
            self.robot._set_torque(False)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": f"torque off (flag set) but bus call failed: {e}", "data": None}
        finally:
            if got:
                self.robot_lock.release()
        self.dlog.add("torque_off", {}, "TORQUE OFF (E-STOP); loop paused", ok=False,  # ok=False -> red in the log
                      step=self.loop.step if self.loop is not None else 0)
        return {"ok": True, "message": "TORQUE OFF (E-STOP); loop paused", "data": None}

    def torque_on(self) -> dict:
        r = self._robot_act(lambda r: r.torque_on())
        if r["ok"]:
            self.dlog.add("torque_on", {}, "torque on", step=self.loop.step if self.loop is not None else 0)
        return r

    def _voice_event(self, tool: str, msg: str) -> dict:
        self.dlog.add(tool, {}, msg, step=self.loop.step if self.loop is not None else 0)
        return {"ok": True, "message": msg}

    # ------------------------------------------------ state
    def _run_state(self) -> dict:
        return {"mode": dict(self.mode),
                "phase": self.ctl.phase if self.ctl is not None else "idle",
                "step": self.loop.step if self.loop is not None else 0,
                "max_steps": self.loop.max_steps if self.loop is not None else self.max_steps,
                "task": self.task, "last_error": self.last_error, "result": self.result,
                "connected": self._connected()}

    def _voice_state(self) -> dict:
        v = self.voice
        return {"mode": v.mode, "listening": v.listening, "queue": v.peek(), "last_transcript": v.last_transcript,
                "tts_model": v.tts_model, "stt_model": v.stt_model, "voice_id": v.voice_id}

    def _rules_state(self) -> dict:
        return {"list": self.rules.list(), "hints": list(self.loop.hints) if self.loop is not None else []}

    def _vlm_state(self) -> dict | None:
        v = self.vlm
        if v is None:
            return None
        return {"model": getattr(v, "model", "?"), "last_latency_ms": getattr(v, "last_latency_ms", None),
                "last_cost_usd": getattr(v, "last_cost_usd", None), "last_usage": getattr(v, "last_usage", None)}

    def _robot_state(self) -> dict | None:
        r = self.robot
        if r is None:
            return None
        if self.robot_lock.acquire(timeout=0.2):
            try:
                p = r.get_ee_pose()
                self._robot_cache = {"ee_pose": {"x": round(p.x, 1), "y": round(p.y, 1), "z": round(p.z, 1),
                                                "roll_deg": round(p.roll_deg, 1)},
                                     "joints_deg": [round(float(v), 1) for v in r.get_joints_deg()]}
            except Exception as e:  # noqa: BLE001
                self._robot_cache.setdefault("ee_pose", None)
                self._robot_cache["error"] = str(e)
            finally:
                self.robot_lock.release()
        return {**self._robot_cache, "gripper_open": r.gripper_open, "torque": bool(getattr(r, "torque", True)),
                "holding": self.loop.holding if self.loop is not None else None}

    # ------------------------------------------------ idle preview + teardown
    def _preview_loop(self) -> None:
        """Streams camera frames to the HUD while the loop is not running (so mode/cam checks need no terminal)."""
        while not self._shutdown.is_set():
            try:
                busy = (self.ctl is not None and self.ctl.phase == "running") or (self.calib is not None and self.calib.active)
                if self.hud is not None and not busy and self.rig is not None:
                    ov, wr = self.rig.capture("overhead"), self.rig.capture("wrist")
                    self.last_overhead = ov
                    hold = self._overlay_hold
                    show = hold[1] if hold is not None and time.time() < hold[0] else ov
                    self.hud.update(self._overhead_view(show, ov), wr, {})
            except Exception as e:  # noqa: BLE001
                log.debug("preview: %s", e)
            time.sleep(0.4)

    def shutdown(self) -> None:
        self._shutdown.set()
        if self.ctl is not None:
            self.ctl.stop_ev.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        for dev in (self.cams, self.robot):
            if dev is not None and hasattr(dev, "disconnect"):
                try:
                    dev.disconnect()
                except Exception:  # noqa: BLE001
                    pass


def serve(args) -> Session:
    """Build the HUD + Session (used by main() and the HTTP tests)."""
    cfg = cfgmod.load(args.config) if args.config else cfgmod.load()
    hud = None
    if not args.no_hud:
        from sortbot.hud import HUD
        hud = HUD(port=args.hud_port or cfg.hud_port)
        hud.start()
    voice = VoiceIO(cfg.elevenlabs_voice_id, stdin=io.StringIO("") if (args.no_voice or args.mock) else None,
                    force_text=True, tts_model=cfg.tts_model, stt_model=cfg.stt_model)  # mic only via explicit mic_on toggle
    voice.start()
    rules = RulesStore(args.rules_file) if args.rules_file else RulesStore()
    return Session(cfg, hud, voice, rules, max_steps=args.max_steps)


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_true", help="shortcut: robot=mock cams=sim vlm=mock (or live with --live-vlm), auto-start")
    g.add_argument("--real", action="store_true", help="shortcut: robot=real cams=real vlm=live, auto-start")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--no-hud", action="store_true", help="headless (requires --mock/--real); exits when the run finishes")
    ap.add_argument("--no-voice", action="store_true", help="do not read stdin for corrections")
    ap.add_argument("--hud-port", type=int, default=0)
    ap.add_argument("--live-vlm", action="store_true", help="with --mock: use the real OpenAI VLM")
    ap.add_argument("--rules-file", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    if args.no_hud and not (args.mock or args.real):
        ap.error("--no-hud requires --mock or --real (nothing else could start the run)")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    session = serve(args)
    result = ""
    try:
        if args.mock or args.real:
            combo = (dict(robot="mock", cams="sim", vlm="live" if args.live_vlm else "mock") if args.mock
                     else dict(robot="real", cams="real", vlm="live"))
            r = session.set_mode(**combo)
            if not r["ok"]:
                log.error("set_mode: %s", r["message"])
            else:
                log.info("start: %s", session.start()["message"])
        if args.no_hud:
            if session.thread is not None:
                session.thread.join()
            result = session.result
        else:
            while True:  # server-first: keep serving; everything else happens from the page
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        session.shutdown()
        session.voice.stop()
        if session.hud is not None:
            session.hud.stop()
    log.info("finished: %s", result or session.result)
    return result or session.result


if __name__ == "__main__":
    main()
