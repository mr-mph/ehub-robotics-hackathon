"""Robot layer: SO101Robot (real SO101Follower) and MockRobot (kinematic sim), both satisfy RobotAPI.

Table frame = mm, origin at base on tabletop, x fwd, y left, z up. Base frame = meters (URDF base_link,
whose origin is ~2 mm above the base's underside, so identity is a sane uncalibrated default).
calib.json: {"table_T_base": 4x4} mapping base-frame points (in mm) into the table frame.

Deviations from DESIGN, with reasons:
- IK: placo's solver (RobotKinematics.inverse_kinematics) is a single velocity-limited QP step and got stuck
  at joint limits / local minima for most of the workspace (100+ mm errors). We keep RobotKinematics for FK
  and use our own damped-least-squares IK over (pan, lift, elbow, wrist_flex) with a coarse FK-grid seed
  and URDF joint limits. Wrist roll is set directly (roll_deg -> wrist_roll).
- "Straight down" is a *minimised* tilt, not a hard constraint: with wrist_flex limited to +/-97 deg and a
  ~160 mm wrist-to-fingertip lever, an exactly vertical gripper is only reachable for z <~ 100 mm and
  radius <~ 300 mm. Poses at grasp height are vertical; at travel_z / home the gripper tilts back.
  Position accuracy (FK(IK) within 5 mm) is always enforced; tilt is reported in ExecResult messages.
- Gripper units are the motor's 0-100 normalized range (SO101 gripper uses MotorNormMode.RANGE_0_100).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from sortbot import config as cfgmod
from sortbot.types import DetectedObject, ExecResult, Pose, RobotAPI

log = logging.getLogger("sortbot.robot")

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IK_TOL_MM = 5.0
GRIPPER_OPEN, GRIPPER_CLOSED = 60.0, 5.0  # motor 0-100 units
MOVE_RATE_HZ = 50.0
MOVE_STEP_DEG = 2.0  # max joint delta per interpolation tick
XY_SUBSTEP_MM = 40.0  # cartesian waypoint spacing for the translate segment
# pan, lift, elbow, wrist_flex: from the URDF (+/-110, +/-100, +/-96.8, +/-95); the motor calibration ranges agree
JOINT_LIMITS_DEG = np.array([[-110, 110], [-100, 100], [-96.8, 96.8], [-95, 95]], float)
SETTLE_TOL_DEG, SETTLE_TIMEOUT_S = 1.5, 1.5
TILT_W = 3.0  # mm of cost per radian of gripper tilt (position dominates)
HOME_JOINTS = np.array([0.0, -30.0, 60.0, 60.0, 0.0, GRIPPER_OPEN])


class SafetyError(RuntimeError):
    pass


def load_calib(path: Path) -> np.ndarray:
    if path.exists():
        return np.asarray(json.loads(path.read_text())["table_T_base"], float).reshape(4, 4)
    log.warning("*** %s not found -- using identity table_T_base (UNCALIBRATED) ***", path)
    return np.eye(4)


class DownIK:
    """DLS IK for a down-pointing gripper at (x, y, z) mm in the base frame with fixed wrist roll."""

    def __init__(self, kin: RobotKinematics):
        self.kin = kin
        q, p = [], []
        (l0, l1), (e0, e1), (w0, w1) = (JOINT_LIMITS_DEG[1:].round().astype(int)).tolist()
        for lift in range(l0, l1 + 1, 5):
            for elbow in range(e0, e1 + 1, 5):
                for wrist in range(w0, w1 + 1, 5):
                    T = kin.forward_kinematics(np.array([0, lift, elbow, wrist, 0, 0], float))
                    q.append((lift, elbow, wrist))
                    p.append((T[0, 3] * 1000, T[2, 3] * 1000, math.acos(max(-1.0, min(1.0, -T[2, 2])))))
        self.seed_q, self.seed_p = np.array(q, float), np.array(p)

    def _f(self, q4: np.ndarray, roll: float) -> np.ndarray:
        T = self.kin.forward_kinematics(np.array([*q4, roll, 0.0]))
        tilt = math.atan2(math.hypot(T[0, 2], T[1, 2]), -T[2, 2])
        return np.array([*(T[:3, 3] * 1000), tilt * TILT_W])

    def _dls(self, q: np.ndarray, tgt: np.ndarray, roll: float) -> tuple[float, np.ndarray]:
        lo, hi = JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1]
        for it in range(50):
            f = self._f(q, roll)
            e = tgt - f
            if it > 2 and np.linalg.norm(e[:3]) < 0.05:
                break
            J = np.zeros((4, 4))
            for j in range(4):
                qq = q.copy()
                qq[j] += 0.5
                J[:, j] = (self._f(qq, roll) - f) / math.radians(0.5)
            free = np.ones(4, bool)
            dq = np.zeros(4)
            for _ in range(4):  # active-set: freeze joints that would leave their limits
                Jf = J[:, free]
                dq[:] = 0
                dq[free] = np.degrees(np.linalg.solve(Jf.T @ Jf + np.eye(int(free.sum())), Jf.T @ e))
                hit = ((q + dq < lo) | (q + dq > hi)) & free
                if not hit.any():
                    break
                q[hit] = np.clip(q[hit], lo[hit], hi[hit])
                free &= ~hit
            q = np.clip(q + dq, lo, hi)
        return float(np.linalg.norm(self._f(q, roll)[:3] - tgt[:3])), q

    def solve(self, x: float, y: float, z: float, roll: float) -> tuple[np.ndarray, float]:
        """Returns (joints_deg[5], position_error_mm). Base frame, mm."""
        rho, pan = math.hypot(x, y), -math.degrees(math.atan2(y, x))
        tgt = np.array([x, y, z, 0.0])
        cost = np.hypot(self.seed_p[:, 0] - rho, self.seed_p[:, 1] - z) + TILT_W * self.seed_p[:, 2]
        best = None
        for i in np.argsort(cost)[:60:20]:
            err, q = self._dls(np.array([pan, *self.seed_q[i]]), tgt, roll)
            if best is None or err < best[1]:
                best = (np.array([*q, roll]), err)
            if err < 0.5:
                break
        return best


class _KinematicBase:
    """Frame math, safety envelope and motion planning. Subclasses: _read_joints/_write_joints/capture."""

    def __init__(self, cfg: cfgmod.Config):
        self.cfg = cfg
        self.kin = RobotKinematics(str(cfg.urdf), "gripper_frame_link")
        self.ik = DownIK(self.kin)
        self.table_T_base = load_calib(cfg.calib_file)
        self.base_T_table = np.linalg.inv(self.table_T_base)
        self.roll_deg = 0.0
        self.gripper_open = True

    # ---- frames ----
    def fk_table(self, q: np.ndarray) -> np.ndarray:
        T = self.kin.forward_kinematics(q).copy()
        T[:3, 3] *= 1000.0
        return self.table_T_base @ T

    def tilt_deg(self, q: np.ndarray) -> float:
        T = self.fk_table(q)
        return math.degrees(math.atan2(math.hypot(T[0, 2], T[1, 2]), -T[2, 2]))

    def ik_table(self, q_seed: np.ndarray, x: float, y: float, z: float, roll_deg: float) -> np.ndarray:
        """Joints for table-frame (x, y, z) mm; raises SafetyError if FK(IK) is off by > IK_TOL_MM."""
        pb = (self.base_T_table @ np.array([x, y, z, 1.0]))[:3]
        q5, _ = self.ik.solve(*pb, roll_deg)
        q = np.array([*q5, q_seed[5]])
        err = float(np.linalg.norm(self.fk_table(q)[:3, 3] - [x, y, z]))
        if err > IK_TOL_MM:
            raise SafetyError(f"IK sanity failed for ({x:.0f},{y:.0f},{z:.0f}) mm: FK(IK) off by {err:.1f} mm (unreachable)")
        return q

    # ---- safety ----
    def check_target(self, x: float, y: float, z: float) -> None:
        lo, hi = self.cfg.aabb_min_mm, self.cfg.aabb_max_mm
        zmin = self.cfg.table_z_mm + self.cfg.gripper_clearance_mm
        if z < zmin - 1e-6:
            raise SafetyError(f"target z={z:.1f} mm below clearance {zmin:.1f} mm")
        for i, (v, name) in enumerate(zip((x, y, z), "xyz")):
            if not lo[i] <= v <= hi[i]:
                raise SafetyError(f"target {name}={v:.1f} mm outside workspace [{lo[i]}, {hi[i]}]")
        p = self.get_ee_pose()
        d = math.dist((p.x, p.y, p.z), (x, y, z))
        if d > self.cfg.max_step_mm:
            raise SafetyError(f"step of {d:.0f} mm exceeds max_step {self.cfg.max_step_mm:.0f} mm")

    # ---- state ----
    def get_joints_deg(self) -> np.ndarray:
        return self._read_joints()

    def get_ee_pose(self) -> Pose:
        p = self.fk_table(self._read_joints())[:3, 3]
        return Pose(float(p[0]), float(p[1]), float(p[2]), self.roll_deg)

    # ---- motion ----
    def _goto_joints(self, q_target: np.ndarray) -> None:
        q = self._read_joints()
        n = max(1, int(math.ceil(np.abs(q_target - q).max() / MOVE_STEP_DEG)))
        for i in range(1, n + 1):
            self._write_joints(q + (q_target - q) * (i / n))
            self._sleep(1.0 / MOVE_RATE_HZ)
        self._settle(q_target)

    def _settle(self, q_target: np.ndarray) -> None:
        """Re-send the goal until the motors reach it (the stream can outrun max_relative_target clipping).
        The gripper is excluded from the tolerance: closing on an object legitimately stalls short."""
        t_end = time.monotonic() + SETTLE_TIMEOUT_S
        while True:
            err = np.abs(self._read_joints() - q_target)
            if err[:5].max() < SETTLE_TOL_DEG:
                return
            if time.monotonic() > t_end:
                log.warning("settle timeout: joints off by %s deg", np.round(err, 1))
                return
            self._write_joints(q_target)
            self._sleep(1.0 / MOVE_RATE_HZ)

    def _plan(self, waypoints: list[tuple[float, float, float]]) -> list[np.ndarray]:
        q, plan = self._read_joints(), []
        for wp in waypoints:
            q = self.ik_table(q, *wp, self.roll_deg)
            plan.append(q)
        return plan

    def move_to(self, x: float, y: float, z: float) -> ExecResult:
        """Lift to travel_z -> translate XY (cartesian sub-waypoints) -> descend. Fully planned before moving."""
        try:
            self.check_target(x, y, z)
            p = self.get_ee_pose()
            tz = max(self.cfg.travel_z_mm, z, p.z)
            wps = [(p.x, p.y, tz)]
            n = max(1, int(math.ceil(math.hypot(x - p.x, y - p.y) / XY_SUBSTEP_MM)))
            wps += [(p.x + (x - p.x) * i / n, p.y + (y - p.y) * i / n, tz) for i in range(1, n + 1)]
            wps.append((x, y, z))
            plan = self._plan(wps)
        except SafetyError as e:
            log.error("move_to rejected: %s", e)
            return ExecResult(False, str(e))
        for q in plan:
            self._goto_joints(q)
        return ExecResult(True, f"at ({x:.0f},{y:.0f},{z:.0f}) tilt {self.tilt_deg(plan[-1]):.0f}deg")

    def turn_to(self, deg: float) -> ExecResult:
        if not -90.0 <= deg <= 90.0:
            return ExecResult(False, f"roll {deg} outside [-90, 90]")
        p = self.get_ee_pose()
        try:
            q = self.ik_table(self._read_joints(), p.x, p.y, p.z, float(deg))
        except SafetyError as e:
            return ExecResult(False, str(e))
        self.roll_deg = float(deg)
        self._goto_joints(q)
        return ExecResult(True, f"roll {deg:.0f}")

    def _set_gripper(self, value: float) -> None:
        q = self._read_joints()
        q[5] = value
        self._goto_joints(q)

    def open_gripper(self) -> ExecResult:
        self._set_gripper(GRIPPER_OPEN)
        self.gripper_open = True
        return ExecResult(True, "gripper open")

    def close_gripper(self) -> ExecResult:
        self._set_gripper(GRIPPER_CLOSED)
        self.gripper_open = False
        return ExecResult(True, "gripper closed")

    def home(self) -> ExecResult:
        h = self.cfg.home
        self.roll_deg = h.roll_deg
        r = self.move_to(h.x, h.y, h.z)
        if r.ok:
            return ExecResult(True, "home")
        try:  # arm far from home (e.g. right after connect): go directly in joint space
            self._goto_joints(self.ik_table(self._read_joints(), h.x, h.y, h.z, h.roll_deg))
        except SafetyError as e:
            return ExecResult(False, str(e))
        return ExecResult(True, "home (direct)")

    def _run(self, label: str, *steps) -> ExecResult:
        for step in steps:
            r = step()
            if not r.ok:
                return ExecResult(False, f"{label} failed: {r.message}")
        return ExecResult(True, label)

    def pick(self, obj: DetectedObject) -> ExecResult:
        x, y = obj.centroid_mm
        zg = max(self.cfg.grasp_z_mm, self.cfg.table_z_mm + self.cfg.gripper_clearance_mm)
        return self._run(f"picked #{obj.id}",
                         self.open_gripper,
                         lambda: self.move_to(x, y, zg),
                         self.close_gripper,
                         lambda: self.move_to(x, y, self.cfg.travel_z_mm))

    def place_at(self, x: float, y: float) -> ExecResult:
        zp = max(self.cfg.grasp_z_mm, self.cfg.table_z_mm + self.cfg.gripper_clearance_mm) + 10.0
        return self._run(f"placed at ({x:.0f},{y:.0f})",
                         lambda: self.move_to(x, y, zp),
                         self.open_gripper,
                         lambda: self.move_to(x, y, self.cfg.travel_z_mm))

    def _sleep(self, s: float) -> None:
        time.sleep(s)


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


class SO101Robot(_KinematicBase):
    """Real arm via lerobot SO101Follower; cameras 'overhead' and 'wrist' are read through the follower."""

    def __init__(self, cfg: cfgmod.Config | None = None, with_cameras: bool = True):
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        super().__init__(cfg or cfgmod.load())
        cams = {}
        if with_cameras:
            for name, c in (("overhead", self.cfg.overhead_cam), ("wrist", self.cfg.wrist_cam)):
                cams[name] = OpenCVCameraConfig(index_or_path=c.index, fps=c.fps, width=c.width, height=c.height)
        cal_dir = self.cfg.robot_calibration_dir
        if cal_dir is None or not (cal_dir / f"{self.cfg.robot_id}.json").exists():
            raise FileNotFoundError(f"motor calibration {cal_dir}/{self.cfg.robot_id}.json missing (robot.calibration_dir)")
        self.robot = SO101Follower(SO101FollowerConfig(
            port=self.cfg.robot_port, id=self.cfg.robot_id, cameras=cams, calibration_dir=cal_dir,
            max_relative_target=MOVE_STEP_DEG * 3))
        self.robot.connect()

    def _read_joints(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([float(obs[f"{m}.pos"]) for m in MOTORS])

    def _write_joints(self, q: np.ndarray) -> None:
        self.robot.send_action({f"{m}.pos": float(v) for m, v in zip(MOTORS, q)})

    def capture(self, name: str) -> np.ndarray:
        return self.robot.get_observation()[name]

    def disconnect(self) -> None:
        self.robot.disconnect()


def _selftest() -> None:
    cfg = cfgmod.load()
    r = MockRobot(cfg)
    assert isinstance(r, RobotAPI)
    assert r.home().ok
    p = r.get_ee_pose()
    assert math.dist((p.x, p.y, p.z), (cfg.home.x, cfg.home.y, cfg.home.z)) < IK_TOL_MM, p

    # FK(IK) round trip on a grid of the physically reachable core workspace (see module docstring)
    zg = cfg.grasp_z_mm + cfg.gripper_clearance_mm
    worst, n, tilt_lo = 0.0, 0, 0.0
    for x in np.linspace(150, 300, 6):
        for y in np.linspace(-200, 200, 7):
            for z in (zg, 60.0, cfg.travel_z_mm):
                if math.hypot(x, y) > 310:
                    continue
                q = r.ik_table(HOME_JOINTS, x, y, z, 0.0)
                err = np.linalg.norm(r.fk_table(q)[:3, 3] - [x, y, z])
                worst, n = max(worst, err), n + 1
                if z == zg:
                    tilt_lo = max(tilt_lo, r.tilt_deg(q))
    print(f"IK grid: {n} points, worst FK(IK) error {worst:.2f} mm, max tilt at grasp height {tilt_lo:.1f} deg")
    assert worst < IK_TOL_MM and tilt_lo < 10.0

    assert r.move_to(275, 0, 60).ok
    assert r.turn_to(45).ok and abs(r.get_ee_pose().roll_deg - 45) < 1e-9 and abs(r.q[4] - 45) < 1e-9
    assert r.turn_to(0).ok
    obj = DetectedObject(1, (300.0, 120.0), (0, 0, 10, 10), 100.0, "red")
    n0 = len(r.log)
    assert r.pick(obj).ok and not r.gripper_open
    zs = [r.fk_table(q)[2, 3] for q in r.log[n0:]]
    assert min(zs) >= cfg.gripper_clearance_mm - 2.0, min(zs)
    assert r.place_at(*cfg.zone("SENSORS").drop_point_mm).ok and r.gripper_open
    p = r.get_ee_pose()
    assert abs(p.x - 275) < IK_TOL_MM and abs(p.y) < IK_TOL_MM and abs(p.z - cfg.travel_z_mm) < IK_TOL_MM, p

    for bad in ((275, 0, 5), (50, 0, 60), (275, 300, 60), (500, 0, 60), (275, 0, 300)):
        assert not r.move_to(*bad).ok, bad
        try:
            r.check_target(*bad)
        except SafetyError:
            pass
        else:
            raise AssertionError(bad)
    assert r.home().ok
    assert not r.move_to(400, -200, 60).ok  # > max_step_mm from home
    assert r.move_to(275, 0, 60).ok
    res = r.move_to(420, 0, 60)  # inside AABB but kinematically unreachable -> IK sanity rejects
    assert not res.ok and "IK sanity" in res.message, res
    assert isinstance(r.capture("overhead"), np.ndarray) and r.capture("wrist").shape == (480, 640, 3)
    print("robot selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.selftest:
        _selftest()
    else:
        ap.print_help()
