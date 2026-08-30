"""test fixtures — not reachable from the app.

Doubles for running sortbot with no hardware and no API keys: MockRobot (kinematic sim), MockVLM
(deterministic planner), SimScene (synthetic overhead scene that doubles as the robot), SceneCams,
FakeRig / VirtualLeader / mock_rig (scripted teleop calibration), ArucoFakeRig (legacy aruco flow)
and session_factories() — the injection seam for main.Session(factories=...).

Imported ONLY by sortbot/tests/* and module --selftest blocks; nothing in the app imports this.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from sortbot import config as cfgmod
from sortbot import perception
from sortbot.calibrate import MOTORS, RobotRig
from sortbot.calibration import _ip, render_mat
from sortbot.robot import HOME_JOINTS, _KinematicBase
from sortbot.types import Command, ExecResult, RobotAPI, WorldState


# ---------------------------------------------------------------- robot / vlm doubles


class MockRobot(_KinematicBase):
    """Pure kinematic sim: joint state in memory, real FK, blank frames. `log` keeps every commanded q."""

    def __init__(self, cfg: cfgmod.Config | None = None, realtime: bool = False):
        super().__init__(cfg or cfgmod.load())
        self.realtime = realtime
        self.q = HOME_JOINTS.copy()
        self.log: list[np.ndarray] = []

    def _read_joints(self) -> np.ndarray:
        return self.q.copy()

    def _write_joints(self, q: np.ndarray) -> None:
        self.q = np.asarray(q, float).copy()
        self.log.append(self.q)

    def _sleep(self, s: float) -> None:
        if self.realtime:
            time.sleep(s)

    def capture(self, name: str) -> np.ndarray:
        cam = self.cfg.overhead_cam if name == "overhead" else self.cfg.wrist_cam
        return np.zeros((cam.height, cam.width, 3), np.uint8)


class MockVLM:
    """Deterministic coordinate planner (the app's VLM does its own seeing, so this double works from a
    script): pick_at each target in turn, place_at the matching drop coordinate, done when the script is
    exhausted. All coordinates are cm, like the real tool surface. Default targets: the SIM_OBJECTS
    positions, so it 'sees' the default SimScene.

    It also doubles for the two FAST calls the real VLM makes on the small model, both scriptable:
      verify_grasp() -> the pre-grasp alignment verdict. `verify_script` is a list of verdicts (dicts, or
        the shorthands "aligned" / "off:DX:DY" / "blind") consumed in order; the LAST entry repeats
        forever, so a one-element script fixes the behaviour for the whole run. Default: always aligned.
      chat() -> {reply, rules, hints, urgent}. `chat_script` is consumed the same way; the default echoes
        the utterance back as a hint so a Session with no script still behaves sanely.
    `verify_calls` / `chat_calls` record every call for assertions."""

    #: where picked items are put down (cm) -- three tidy clusters, cycled
    DROPS_CM = [(27.5, 14.0), (27.5, 0.0), (27.5, -14.0)]
    ALIGNED = {"aligned": True, "dx_cm": 0.0, "dy_cm": 0.0, "reason": "jaws centred on the object",
               "confidence": 0.92}

    def __init__(self, model: str | None = None, targets_cm: list[tuple[float, float]] | None = None,
                 drops_cm: list[tuple[float, float]] | None = None,
                 verify_script: list | None = None, chat_script: list | None = None):
        self.model = model or "mock"
        self.chat_model = self.verify_model = self.model
        # mm -> cm: the doubles speak the same cm tool surface as the real VLM (see sortbot.vlm)
        self.targets = list(targets_cm) if targets_cm is not None \
            else [(x / 10.0, y / 10.0) for (x, y), _ in SIM_OBJECTS]
        self.drops = list(drops_cm) if drops_cm is not None else list(self.DROPS_CM)
        self._i = 0
        self._drop_i = 0
        self.last_latency_ms: int | None = None
        self.last_usage = self.last_cost_usd = None
        self.last_chat_latency_ms = self.last_verify_latency_ms = None
        self.verify_script = list(verify_script) if verify_script else None
        self.chat_script = list(chat_script) if chat_script else None
        self.verify_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self._v_i = self._c_i = 0

    def reset(self) -> None:
        """Fresh script for a fresh run (Session.start resets every device that supports it)."""
        self._i = self._drop_i = 0

    # -- scripted fast calls ------------------------------------------------
    @staticmethod
    def _next(script: list, i: int):
        """Consume `script` in order; the last entry repeats forever."""
        return script[min(i, len(script) - 1)]

    @staticmethod
    def _verdict(spec) -> dict:
        """dict | "aligned" | "off:DX:DY" (not aligned, correct by DX/DY cm) | "blind" (cannot tell)."""
        if isinstance(spec, dict):
            return {**MockVLM.ALIGNED, **spec}
        if spec == "aligned":
            return dict(MockVLM.ALIGNED)
        if spec == "blind":
            return {"aligned": False, "dx_cm": 0.0, "dy_cm": 0.0, "confidence": 0.1,
                    "reason": "wrist view too dark to tell what is under the jaws"}
        if isinstance(spec, str) and spec.startswith("off:"):
            _, dx, dy = spec.split(":")
            return {"aligned": False, "dx_cm": float(dx), "dy_cm": float(dy), "confidence": 0.8,
                    "reason": f"object is {abs(float(dx)):g} cm forward / {abs(float(dy)):g} cm left of the jaws"}
        raise ValueError(f"bad verify spec {spec!r}")

    def verify_grasp(self, overhead_jpeg, wrist_jpeg, x_cm: float, y_cm: float, attempt: int = 1) -> dict:
        assert overhead_jpeg and wrist_jpeg, "the grasp check must get BOTH camera views"
        self.verify_calls.append({"x_cm": round(float(x_cm), 2), "y_cm": round(float(y_cm), 2),
                                  "attempt": int(attempt)})
        self.last_verify_latency_ms = 7
        if self.verify_script is None:
            return dict(self.ALIGNED)
        v = self._verdict(self._next(self.verify_script, self._v_i))
        self._v_i += 1
        return v

    def chat(self, heard: str, context: str, overhead_jpeg=None, wrist_jpeg=None) -> dict:
        self.chat_calls.append({"heard": heard, "context": context,
                                "images": int(bool(overhead_jpeg)) + int(bool(wrist_jpeg))})
        self.last_chat_latency_ms = 9
        if self.chat_script is None:
            return {"reply": f"Got it: {heard}", "rules": [], "hints": [heard], "urgent": "none"}
        d = self._next(self.chat_script, self._c_i)
        self._c_i += 1
        return {"reply": "", "rules": [], "hints": [], "urgent": "none", **d}

    def plan_step(self, overhead_overlay_png: bytes, wrist_png: bytes, world: WorldState, history: list,
                  workspace_mm=None) -> Command:
        if world.holding is not None:
            x, y = self.drops[self._drop_i % len(self.drops)]
            self._drop_i += 1
            return Command("place_at", {"x_cm": float(x), "y_cm": float(y)})
        if self._i < len(self.targets):
            x, y = self.targets[self._i]
            self._i += 1
            return Command("pick_at", {"x_cm": float(x), "y_cm": float(y)})
        return Command("done", {"summary": f"sorted {self._drop_i} objects"})


# ---------------------------------------------------------------- synthetic scene (robot + camera in one)

SIM_OBJECTS = [((200.0, 120.0), (220, 40, 40)), ((240.0, -70.0), (40, 200, 60)),
               ((180.0, 70.0), (240, 220, 50)), ((270.0, 70.0), (255, 255, 255))]


class SimScene:
    """Wraps MockRobot: keeps blob positions on a synthetic mat and moves them with pick/place.
    Satisfies RobotAPI (everything not overridden is forwarded to the wrapped robot), so it can be
    injected as the Session's robot; SceneCams exposes its frames as the cameras."""

    def __init__(self, robot: RobotAPI, cfg: cfgmod.Config, objects=SIM_OBJECTS):
        self.robot, self.cfg = robot, cfg
        self.w, self.h = cfg.overhead_cam.width, cfg.overhead_cam.height
        self.H = perception.synth_homography(cfg, self.w, self.h)
        self._objects = objects
        self.blobs: list = []
        self.held: list | None = None
        self.reset()

    def reset(self) -> None:
        """Fresh blobs (Session.start calls this so every run starts with an unsorted table)."""
        self.blobs = [[list(xy), col] for xy, col in self._objects]
        self.held = None

    def __getattr__(self, name):
        return getattr(self.robot, name)

    @property
    def torque(self) -> bool:  # the E-STOP flag must land on the wrapped robot, not on this wrapper
        return self.robot.torque

    @torque.setter
    def torque(self, v: bool) -> None:
        self.robot.torque = v

    def capture(self, name: str) -> np.ndarray:
        if name != "overhead":
            return self.robot.capture(name)
        img = np.random.default_rng(0).integers(20, 40, (self.h, self.w, 3), dtype=np.uint8)
        for xy, col in self.blobs:
            cx, cy = perception.mm_to_px(self.H, [xy])[0].astype(int)
            cv2.rectangle(img, (cx - 22, cy - 15), (cx + 22, cy + 15), col, -1)
        return img

    def pick(self, x_mm: float, y_mm: float) -> ExecResult:
        r = self.robot.pick(x_mm, y_mm)
        if r.ok and self.blobs:
            at = np.array([x_mm, y_mm], float)
            i = min(range(len(self.blobs)), key=lambda i: np.hypot(*(np.array(self.blobs[i][0]) - at)))
            if np.hypot(*(np.array(self.blobs[i][0]) - at)) < 30:
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


class SceneCams:
    """Cameras-shaped view of a SimScene: read(name) -> the scene's frames."""

    def __init__(self, scene_fn):
        self._scene_fn = scene_fn

    def read(self, name: str) -> np.ndarray:
        return self._scene_fn().capture(name)

    def disconnect(self) -> None:
        pass


# ---------------------------------------------------------------- teleop calibration doubles


class FakeRig(RobotRig):
    """MockRobot follower whose overhead frames are rendered with the target at the *true* projection of the
    FK xy (hidden ground-truth H_true: base mm -> px) on top of `background` (or a dark mat)."""

    def __init__(self, robot, H_true_mm_to_px: np.ndarray, background=None, z_offset_true_mm: float = 25.0,
                 target=None):
        super().__init__(robot)
        self.H_true, self.z_off_true, self.bg = np.asarray(H_true_mm_to_px, float), z_offset_true_mm, background
        self.color = (40, 220, 60) if target is None or target.name == "green" else (255, 150, 20)
        self.w, self.h = robot.cfg.overhead_cam.width, robot.cfg.overhead_cam.height

    def frame(self) -> np.ndarray:
        img = self.bg() if callable(self.bg) else (self.bg.copy() if self.bg is not None
                                                   else np.full((self.h, self.w, 3), 30, np.uint8))
        p = self.fk_base_mm(self.read_joints())
        u, v = cv2.perspectiveTransform(np.array([[[p[0], p[1]]]]), self.H_true).ravel()
        cv2.circle(img, _ip((u, v)), 16, self.color, -1)
        return img


class VirtualLeader:
    """Scripted 'human' for hardware-free calibration: get_action() returns the joints of the current scripted
    pose (base-frame IK on the follower); drive(session) walks the poses, capturing each, and finishes."""

    def __init__(self, robot, poses_base_mm, touch_z_base_mm: float):
        self.robot = robot
        self.poses = [self._ik(*p) for p in poses_base_mm]
        self.touch = self._ik(180.0, 0.0, touch_z_base_mm)
        self.q = self.robot._read_joints()

    def _ik(self, x, y, z) -> np.ndarray:
        q5, err = self.robot.ik.solve(x, y, z, 0.0)
        assert err < 5.0, (x, y, z, err)
        return np.array([*q5, 20.0])

    def get_action(self) -> dict[str, float]:
        return {f"{m}.pos": float(v) for m, v in zip(MOTORS, self.q)}

    def goto(self, q: np.ndarray, settle_s: float = 0.15) -> None:
        self.q = q
        time.sleep(settle_s)

    def drive(self, ctrl, dwell_s: float = 0.15) -> None:
        """Script the human against the controller (so HUD status / on_done fire exactly as with real triggers)."""
        self.goto(self.touch, dwell_s)
        print("  " + ctrl.trigger("touch_table")["message"])
        for q in self.poses:
            self.goto(q, dwell_s)
            print("  " + ctrl.trigger("capture")["message"])
        print("  " + ctrl.trigger("finish")["message"])

    def disconnect(self) -> None:
        pass


MOCK_POSES_MM = [(180, -120, 40), (180, 120, 40), (290, 80, 40), (290, -80, 40),  # a quad first: 4th sample fits
                 (180, 0, 40), (240, -120, 40), (240, 0, 45), (240, 120, 40), (210, 60, 60)]
MOCK_Z_OFF_TRUE = 25.0


def mock_rig(cfg: cfgmod.Config, robot=None, H_mm_to_px: np.ndarray | None = None, background=None,
             target=None) -> tuple[FakeRig, VirtualLeader]:
    """MockRobot follower + FakeRig frames + VirtualLeader. Default H_true: 2 px/mm, x fwd = up, y left = left,
    plus a little perspective so a true homography (not an affinity) is being recovered."""
    robot = robot or MockRobot(cfg)
    if H_mm_to_px is None:
        w, h = cfg.overhead_cam.width, cfg.overhead_cam.height
        H_mm_to_px = np.array([[0.0, -2.0, w / 2], [-2.0, 0.0, h / 2 + 2 * 250], [1e-4, 5e-5, 1.0]]) @ np.eye(3)
    rig = FakeRig(robot, H_mm_to_px, background, MOCK_Z_OFF_TRUE, target)
    return rig, VirtualLeader(robot, MOCK_POSES_MM, -MOCK_Z_OFF_TRUE)


class ArucoFakeRig:
    """Legacy aruco selftest: move_to(x, y, z) executes in the *base* frame (identity prior); the hidden
    transform says where that really is on the table, and the overhead frame is rendered with the ball there."""

    def __init__(self, cfg, cam_h, cam_xy, r_mm):
        ang = np.deg2rad(4.0)
        self.T_true = np.eye(4)
        self.T_true[:3, :3] = [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]
        self.T_true[:3, 3] = [8.0, -3.0, 25.0]
        self.cfg, self.cam_h, self.cam_xy, self.r_mm = cfg, cam_h, cam_xy, r_mm
        self.mat, self.truth = render_mat(cfg.aruco_tags_mm, cfg.aruco_dict)
        self.q_mm = self.p_mm = np.zeros(3)

    def move_to(self, x, y, z):
        self.q_mm = np.array([x, y, z], float)
        self.p_mm = self.T_true[:3, :3] @ self.q_mm + self.T_true[:3, 3]

    def fk_base_m(self):
        return self.q_mm / 1000.0

    def frame(self):
        s = (self.cam_h - self.p_mm[2]) / self.cam_h
        plane = self.cam_xy + (self.p_mm[:2] - self.cam_xy) / s
        img = self.mat.copy()
        cv2.circle(img, tuple(int(round(v)) for v in self.truth(*plane)), 30, (255, 150, 20), -1)
        return img

    def touch_z_base_m(self):
        return -self.T_true[2, 3] / 1000.0


# ---------------------------------------------------------------- Session injection seam


def session_factories(make_robot=None, make_vlm=None) -> dict:
    """Factories dict for main.Session(factories=...) / main.serve(args, factories=...): a MockRobot-backed
    SimScene serves as both the robot and (via SceneCams) the cameras, MockVLM plans, the homography is the
    scene's fixed synthetic H, and the calibration controller is a FakeRig/VirtualLeader teleop session that
    writes to a temp file (never the real calib.json). Connect the robot before the cameras.
    make_robot: optional cfg -> robot override for the wrapped base robot (default MockRobot; the bus-lock
    regression test injects a robot whose bus methods detect concurrent access).
    make_vlm: optional () -> planner override (a MockVLM with a verify_script / chat_script)."""
    box: dict = {}

    def robot(s):
        base = make_robot(s.cfg) if make_robot is not None else MockRobot(s.cfg, realtime=False)
        box["scene"] = SimScene(base, s.cfg)
        return box["scene"]

    def cams(s):
        if "scene" not in box:
            raise RuntimeError("session_factories: connect the robot before the cameras")
        return SceneCams(lambda: box["scene"])

    def vlm(s):
        return make_vlm() if make_vlm is not None else MockVLM()

    def homography(s):
        from sortbot.main import Homography
        return Homography(s.cfg, box["scene"].H)

    def calib(s):
        from sortbot import calibrate as cal
        target = cal.ColorTarget.parse(s.cfg.calib_target)
        out = Path(tempfile.mkdtemp()) / "calib_mock.json"
        w, h = s.cfg.overhead_cam.width, s.cfg.overhead_cam.height
        H_sim = perception.synth_homography(s.cfg, w, h)
        pair: list = []

        def make():
            if not pair:  # dark-mat background: the scene's own green blob must not distract the detector
                pair.extend(mock_rig(s.cfg, s.robot, np.linalg.inv(H_sim), None, target))
            return pair

        ctrl = cal.CalibController(s.cfg, lambda: make()[0], lambda: make()[1], target, out,
                                   s._calib_done, lambda: s.last_overhead,
                                   driver=lambda ld, c: ld.drive(c, dwell_s=0.2), bus_lock=s.robot_lock)
        return ctrl, out

    return {"robot": robot, "cams": cams, "vlm": vlm, "homography": homography, "calib": calib}
