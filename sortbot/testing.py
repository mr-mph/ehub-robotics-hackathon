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
from sortbot.types import Command, DetectedObject, ExecResult, RobotAPI, WorldState


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
    """Deterministic: pick lowest id -> place in zones round-robin -> done when no objects remain."""

    def __init__(self, model: str | None = None):
        self.model = model or "mock"
        self._zone_i = 0
        self.last_latency_ms: int | None = None
        self.last_usage = self.last_cost_usd = None

    def plan_step(self, overhead_overlay_png: bytes, wrist_png: bytes, world: WorldState, history: list) -> Command:
        if world.holding is not None:
            if not world.zones:
                return Command("place_at", {"x_mm": 275.0, "y_mm": 0.0})
            z = world.zones[self._zone_i % len(world.zones)]
            self._zone_i += 1
            return Command("place_in_zone", {"zone": z.name})
        if world.objects:
            return Command("pick", {"id": min(o.id for o in world.objects)})
        return Command("done", {"summary": f"sorted {self._zone_i} objects"})


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


def session_factories() -> dict:
    """Factories dict for main.Session(factories=...) / main.serve(args, factories=...): a MockRobot-backed
    SimScene serves as both the robot and (via SceneCams) the cameras, MockVLM plans, the homography is the
    scene's fixed synthetic H, and the calibration controller is a FakeRig/VirtualLeader teleop session that
    writes to a temp file (never the real calib.json). Connect the robot before the cameras."""
    box: dict = {}

    def robot(s):
        box["scene"] = SimScene(MockRobot(s.cfg, realtime=False), s.cfg)
        return box["scene"]

    def cams(s):
        if "scene" not in box:
            raise RuntimeError("session_factories: connect the robot before the cameras")
        return SceneCams(lambda: box["scene"])

    def vlm(s):
        return MockVLM()

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
