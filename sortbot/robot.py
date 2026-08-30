"""Robot layer: SO101Robot (real SO101Follower via lerobot), satisfies RobotAPI; _KinematicBase carries the
frame math, safety envelope and motion planning (sortbot/testing.py's MockRobot subclasses it for tests).

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
from sortbot.types import ExecResult, Pose, RobotAPI

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
SETTLE_STEP_DEG = MOVE_STEP_DEG * 2   # settle re-sends are step-limited BELOW lerobot's clamp (see _settle)
SETTLE_STALL_TICKS = 10               # give up early when the error stops improving (gravity-loaded joints)
TILT_W = 3.0  # mm of cost per radian of gripper tilt (position dominates)
HOME_JOINTS = np.array([0.0, -30.0, 60.0, 60.0, 0.0, GRIPPER_OPEN])


# ---------------------------------------------------------------- table plane / grasp depth
#: How far workspace.z_trim_mm may shift the plane. The trim is the USER's knob (their table can genuinely
#: sit far below the assumed plane -- a high-mounted base, or a wrong table_z_mm), so this is deliberately
#: generous; the backstop is workspace.z_floor_mm, which exists only to catch a typo or a runaway.
Z_TRIM_LIMIT_MM = 150.0
#: |trim| above this is legal but unusual -- warned about at startup and in the HUD, never blocked.
LARGE_TRIM_MM = 40.0
DEFAULT_Z_FLOOR_MM = -150.0


def z_trim_mm(cfg) -> float:
    """workspace.z_trim_mm, clamped. NEGATIVE lowers the plane the gripper descends to -- the one knob for
    "the jaws stop short of the table" (raise it if the gripper presses into the table instead)."""
    v = float(getattr(cfg, "z_trim_mm", 0.0) or 0.0)
    return max(-Z_TRIM_LIMIT_MM, min(Z_TRIM_LIMIT_MM, v))


def table_plane_mm(cfg) -> float:
    """The table plane the ARM works to: the configured table_z_mm shifted by the trim."""
    return float(cfg.table_z_mm) + z_trim_mm(cfg)


def grasp_z_mm(cfg) -> float:
    """The commanded z of every grasp/pre-place descent, and the floor of the safety envelope.
    EVERY z-floor in the app comes from here, so one trim moves them all together."""
    return max(float(cfg.grasp_z_mm), float(cfg.table_z_mm) + float(cfg.gripper_clearance_mm)) + z_trim_mm(cfg)


def hard_floor_mm(cfg) -> float:
    """workspace.z_floor_mm: the absolute minimum commanded z, in table-frame mm, independent of the trim.
    A backstop against a typo or a runaway -- not a policy limit; lower it if your table really is deeper."""
    return float(getattr(cfg, "z_floor_mm", DEFAULT_Z_FLOOR_MM))


def large_trim(cfg) -> bool:
    """Legal but worth a warning: at this depth the gripper can genuinely drive into the table."""
    return abs(z_trim_mm(cfg)) > LARGE_TRIM_MM


class SafetyError(RuntimeError):
    pass


def load_calib(path: Path) -> np.ndarray:
    if path.exists():
        return np.asarray(json.loads(path.read_text())["table_T_base"], float).reshape(4, 4)
    log.warning("*** %s not found -- using identity table_T_base (UNCALIBRATED) ***", path)
    return np.eye(4)


class DownIK:
    """DLS IK for a down-pointing gripper at (x, y, z) mm in the base frame with fixed wrist roll.

    All positions are the TOOL point (the grasp point between the jaws) = the URDF gripper_frame_link
    origin plus `tool_offset` expressed in that frame. FK and IK apply the same offset, so they stay
    mutually consistent."""

    def __init__(self, kin: RobotKinematics, tool_offset: np.ndarray | None = None):
        self.kin = kin
        self.tool = np.zeros(3) if tool_offset is None else np.asarray(tool_offset, float)
        q, p = [], []
        (l0, l1), (e0, e1), (w0, w1) = (JOINT_LIMITS_DEG[1:].round().astype(int)).tolist()
        for lift in range(l0, l1 + 1, 5):
            for elbow in range(e0, e1 + 1, 5):
                for wrist in range(w0, w1 + 1, 5):
                    T = kin.forward_kinematics(np.array([0, lift, elbow, wrist, 0, 0], float))
                    pt = T[:3, 3] * 1000.0 + T[:3, :3] @ self.tool
                    q.append((lift, elbow, wrist))
                    p.append((pt[0], pt[2], math.acos(max(-1.0, min(1.0, -T[2, 2])))))
        self.seed_q, self.seed_p = np.array(q, float), np.array(p)

    def _f(self, q4: np.ndarray, roll: float) -> np.ndarray:
        T = self.kin.forward_kinematics(np.array([*q4, roll, 0.0]))
        tilt = math.atan2(math.hypot(T[0, 2], T[1, 2]), -T[2, 2])
        return np.array([*(T[:3, 3] * 1000.0 + T[:3, :3] @ self.tool), tilt * TILT_W])

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
        # What every xyz in this app MEANS: the grasp point between the jaws, not the URDF's
        # gripper_frame_link dummy (which sits ~7.8 mm off the jaw centreline). See config workspace.
        self.tool_offset_mm = np.asarray(getattr(cfg, "tool_offset_mm", (0.0, 0.0, 0.0)), float)
        self.ik = DownIK(self.kin, self.tool_offset_mm)
        self.table_T_base = load_calib(cfg.calib_file)
        self.base_T_table = np.linalg.inv(self.table_T_base)
        self.roll_deg = 0.0
        self.gripper_open = True
        self.torque = True  # False after torque_off() (E-STOP): every motion is refused until torque_on()

    # ---- frames ----
    def fk_table(self, q: np.ndarray) -> np.ndarray:
        """Table-frame pose of the TOOL POINT (between the jaws), in mm -- not the raw URDF frame."""
        T = self.kin.forward_kinematics(q).copy()
        T[:3, 3] = T[:3, 3] * 1000.0 + T[:3, :3] @ self.tool_offset_mm
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
            trim = z_trim_mm(self.cfg)
            # tell the two causes apart: "the arm cannot go there" vs "the arm cannot go that DEEP"
            deep = trim < 0 and z < max(float(self.cfg.grasp_z_mm),
                                        float(self.cfg.table_z_mm) + float(self.cfg.gripper_clearance_mm))
            why = (f"unreachable at this grasp depth (z={z:.0f} mm comes from the grasp depth trim "
                   f"{trim:+.0f} mm) -- reduce workspace.z_trim_mm" if deep else "unreachable")
            raise SafetyError(f"IK sanity failed for ({x:.0f},{y:.0f},{z:.0f}) mm: FK(IK) off by {err:.1f} mm ({why})")
        return q

    # ---- safety ----
    def check_target(self, x: float, y: float, z: float) -> None:
        lo, hi = self.cfg.aabb_min_mm, self.cfg.aabb_max_mm
        zmin, hard = grasp_z_mm(self.cfg), hard_floor_mm(self.cfg)
        if z < hard - 1e-6:  # trim-independent backstop: workspace.z_floor_mm
            raise SafetyError(f"target z={z:.1f} mm is below the absolute floor workspace.z_floor_mm = "
                              f"{hard:.1f} mm; raise the grasp depth trim (workspace.z_trim_mm, currently "
                              f"{z_trim_mm(self.cfg):+.1f} mm) or lower z_floor_mm if the table really is deeper")
        if z < zmin - 1e-6:
            raise SafetyError(f"target z={z:.1f} mm below clearance {zmin:.1f} mm "
                              f"(grasp depth trim {z_trim_mm(self.cfg):+.1f} mm)")
        # a negative trim lowers the z floor with it, so the AABB's own z minimum must not veto it
        lo = (lo[0], lo[1], min(lo[2], zmin))
        for i, (v, name) in enumerate(zip((x, y, z), "xyz")):
            if not lo[i] <= v <= hi[i]:
                raise SafetyError(f"target {name}={v:.1f} mm outside workspace [{lo[i]}, {hi[i]}]")
        p = self.get_ee_pose()
        # max_step_mm bounds the TRANSLATION across the table, so it is measured in XY only.
        # move_to never travels the straight line between two poses: it lifts to travel height, crosses in
        # bounded cartesian sub-steps and then descends straight down. Folding the vertical leg into this
        # number measured a path the arm does not take, and made the limit depend on the grasp depth --
        # a deeper workspace.z_trim_mm silently shrank the reachable table until far drop points started
        # failing as "unreachable". Vertical travel is already bounded, by the workspace AABB and the z
        # floor; this is the horizontal bound.
        d = math.hypot(x - p.x, y - p.y)
        if d > self.cfg.max_step_mm:
            raise SafetyError(f"XY step of {d:.0f} mm across the table exceeds max_step "
                              f"{self.cfg.max_step_mm:.0f} mm")

    # ---- state ----
    def get_joints_deg(self) -> np.ndarray:
        return self._read_joints()

    def get_ee_pose(self) -> Pose:
        p = self.fk_table(self._read_joints())[:3, 3]
        return Pose(float(p[0]), float(p[1]), float(p[2]), self.roll_deg)

    # ---- torque / E-STOP ----
    def torque_off(self) -> ExecResult:
        """E-STOP: cut motor torque (real: bus.disable_torque()); motions raise SafetyError until torque_on()."""
        self.torque = False
        self._set_torque(False)
        return ExecResult(True, "torque OFF (E-STOP)")

    def torque_on(self) -> ExecResult:
        self._set_torque(True)
        self.torque = True
        return ExecResult(True, "torque on")

    def _set_torque(self, on: bool) -> None:
        pass

    # ---- motion ----
    def _goto_joints(self, q_target: np.ndarray) -> None:
        if not self.torque:
            raise SafetyError("torque is off (E-STOP); press torque_on first")
        q = self._read_joints()
        n = max(1, int(math.ceil(np.abs(q_target - q).max() / MOVE_STEP_DEG)))
        for i in range(1, n + 1):
            if not self.torque:
                raise SafetyError("torque cut mid-motion (E-STOP)")
            self._write_joints(q + (q_target - q) * (i / n))
            self._sleep(1.0 / MOVE_RATE_HZ)
        self._settle(q_target)

    def _settle(self, q_target: np.ndarray) -> None:
        """Re-send the goal until the motors reach it (the stream can outrun max_relative_target clipping).
        Each re-send is itself step-limited toward the target (SETTLE_STEP_DEG < lerobot's
        max_relative_target), so lerobot never has to clamp -- re-sending the full q_target made it log
        'Relative goal position magnitude had to be clamped to be safe' on every tick while it walked the
        gap in tiny clipped steps. Also exits early once the error stops improving: a gravity-loaded joint
        can stall a couple of degrees short and would otherwise spin until the timeout. The gripper is
        excluded from the tolerance: closing on an object legitimately stalls short."""
        t_end = time.monotonic() + SETTLE_TIMEOUT_S
        best, stall = float("inf"), 0
        while True:
            q = self._read_joints()
            err = np.abs(q - q_target)
            e = float(err[:5].max())
            if e < SETTLE_TOL_DEG:
                return
            if e < best - 0.05:
                best, stall = e, 0
            else:
                stall += 1
            timed_out = time.monotonic() > t_end
            if timed_out or stall >= SETTLE_STALL_TICKS:
                # close-enough residuals (load/friction) are routine -> INFO; big gaps stay a WARNING
                log.log(logging.INFO if e < 3.0 else logging.WARNING,
                        "settle stopped (%s): joints off by %s deg",
                        "timeout" if timed_out else "no further progress", np.round(err, 1))
                return
            if not self.torque:
                log.warning("settle abandoned: torque cut (E-STOP)")
                return
            self._write_joints(q + np.clip(q_target - q, -SETTLE_STEP_DEG, SETTLE_STEP_DEG))
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
        try:
            for q in plan:
                self._goto_joints(q)
        except SafetyError as e:  # torque off (E-STOP) mid-motion
            return ExecResult(False, str(e))
        return ExecResult(True, f"at ({x:.0f},{y:.0f},{z:.0f}) tilt {self.tilt_deg(plan[-1]):.0f}deg")

    def turn_to(self, deg: float) -> ExecResult:
        if not -90.0 <= deg <= 90.0:
            return ExecResult(False, f"roll {deg} outside [-90, 90]")
        p = self.get_ee_pose()
        try:
            q = self.ik_table(self._read_joints(), p.x, p.y, p.z, float(deg))
        except SafetyError as e:
            return ExecResult(False, str(e))
        try:
            self._goto_joints(q)
        except SafetyError as e:
            return ExecResult(False, str(e))
        self.roll_deg = float(deg)
        return ExecResult(True, f"roll {deg:.0f}")

    def _set_gripper(self, value: float) -> None:
        q = self._read_joints()
        q[5] = value
        self._goto_joints(q)

    def open_gripper(self) -> ExecResult:
        try:
            self._set_gripper(GRIPPER_OPEN)
        except SafetyError as e:
            return ExecResult(False, str(e))
        self.gripper_open = True
        return ExecResult(True, "gripper open")

    def close_gripper(self) -> ExecResult:
        try:
            self._set_gripper(GRIPPER_CLOSED)
        except SafetyError as e:
            return ExecResult(False, str(e))
        self.gripper_open = False
        return ExecResult(True, "gripper closed")

    def home(self) -> ExecResult:
        h = self.cfg.home
        self.roll_deg = h.roll_deg
        r = self.move_to(h.x, h.y, h.z)
        if r.ok:
            return ExecResult(True, "home")
        if not self.torque:
            return r
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

    def pick(self, x_mm: float, y_mm: float) -> ExecResult:
        x, y = float(x_mm), float(y_mm)
        zg = grasp_z_mm(self.cfg)  # trimmed by workspace.z_trim_mm (see the helpers at the top)
        # label carries no bare coordinates: failure strings reach the VLM, whose surface is cm
        # (the inner SafetyError text stays mm but always says "mm" explicitly)
        return self._run("pick",
                         self.open_gripper,
                         lambda: self.move_to(x, y, zg),
                         self.close_gripper,
                         lambda: self.move_to(x, y, self.cfg.travel_z_mm))

    def place_at(self, x: float, y: float) -> ExecResult:
        zp = grasp_z_mm(self.cfg) + 10.0
        return self._run("place",  # no bare mm coordinates in the label (see pick)
                         lambda: self.move_to(x, y, zp),
                         self.open_gripper,
                         lambda: self.move_to(x, y, self.cfg.travel_z_mm))

    def _sleep(self, s: float) -> None:
        time.sleep(s)


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
        self._with_cameras = with_cameras
        self.robot = SO101Follower(SO101FollowerConfig(
            port=self.cfg.robot_port, id=self.cfg.robot_id, cameras=cams, calibration_dir=cal_dir,
            # generous clamp: a last-resort safety net for big jumps only. Our own interpolation
            # (MOVE_STEP_DEG) and settle re-sends (SETTLE_STEP_DEG) stay well below it, so lerobot
            # never clamps (and never spams its 'had to be clamped' warning) in normal operation.
            max_relative_target=MOVE_STEP_DEG * 5))
        self.robot.connect()

    def _read_joints(self) -> np.ndarray:
        # Motor bus ONLY -- get_observation() would also grab a frame from every attached camera,
        # which is wasted work at 50 Hz and made joint reads hold the bus far longer than needed.
        obs = self.robot.bus.sync_read("Present_Position", num_retry=self.robot.config.num_read_retries)
        return np.array([float(obs[m]) for m in MOTORS])

    def _write_joints(self, q: np.ndarray) -> None:
        self.robot.send_action({f"{m}.pos": float(v) for m, v in zip(MOTORS, q)})

    def capture(self, name: str) -> np.ndarray:
        if not self._with_cameras:
            # The Session owns the cameras separately (robot.Cameras); a capture() here would sync-read
            # the motor bus too, becoming a hidden second bus toucher. Fail loudly instead of silently
            # doing a full get_observation().
            raise RuntimeError("SO101Robot was connected with_cameras=False: frames come from robot.Cameras "
                               "(the Session's cams), not through the follower")
        return self.robot.get_observation()[name]

    def _set_torque(self, on: bool) -> None:
        (self.robot.bus.enable_torque if on else self.robot.bus.disable_torque)()

    def disconnect(self) -> None:
        self.robot.disconnect()


class Cameras:
    """Overhead + wrist OpenCV cameras opened directly (not through the follower), so 'real cams + mock robot'
    works for tuning perception without an arm. read(name) -> RGB HxWx3; frames are cached per name."""

    def __init__(self, cfg: cfgmod.Config | None = None, names=("overhead", "wrist")):
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        self.cfg = cfg or cfgmod.load()
        self.cams = {}
        try:
            for name in names:
                c = self.cfg.overhead_cam if name == "overhead" else self.cfg.wrist_cam
                cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=c.index, fps=c.fps, width=c.width, height=c.height))
                cam.connect(warmup=True)
                self.cams[name] = cam
        except Exception:
            self.disconnect()
            raise

    def read(self, name: str) -> np.ndarray:
        return np.ascontiguousarray(self.cams[name].async_read(timeout_ms=1000))

    def disconnect(self) -> None:
        for cam in self.cams.values():
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self.cams = {}


def _selftest() -> None:
    import sortbot.robot as _robot  # run as __main__, the fixture raises sortbot.robot.SafetyError, not ours
    from sortbot.testing import MockRobot  # test fixture; only reachable from this selftest

    cfg = cfgmod.load()
    r = MockRobot(cfg)
    assert isinstance(r, RobotAPI)

    # every xyz in the app is the TOOL point (between the jaws), not the raw URDF gripper_frame_link:
    # FK must be displaced from the URDF frame by exactly |tool_offset|, and IK must agree with FK
    off = np.asarray(cfg.tool_offset_mm, float)
    if np.linalg.norm(off) > 0:
        q0 = HOME_JOINTS.copy()
        raw = r.kin.forward_kinematics(q0)[:3, 3] * 1000.0
        tool = r.fk_table(q0)[:3, 3] - r.table_T_base[:3, 3]
        assert abs(np.linalg.norm(tool - raw) - np.linalg.norm(off)) < 1e-6, (tool, raw, off)
        q_ik = r.ik_table(HOME_JOINTS, 250.0, 40.0, 60.0, 0.0)  # IK targets the same tool point
        assert np.linalg.norm(r.fk_table(q_ik)[:3, 3] - [250.0, 40.0, 60.0]) < IK_TOL_MM
    assert r.home().ok
    p = r.get_ee_pose()
    assert math.dist((p.x, p.y, p.z), (cfg.home.x, cfg.home.y, cfg.home.z)) < IK_TOL_MM, p

    # FK(IK) round trip on a grid of the physically reachable core workspace (see module docstring)
    zg = grasp_z_mm(cfg)
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
    n0 = len(r.log)
    assert r.pick(300.0, 120.0).ok and not r.gripper_open
    zs = [r.fk_table(q)[2, 3] for q in r.log[n0:]]
    assert min(zs) >= grasp_z_mm(cfg) - 2.0, min(zs)
    assert r.place_at(275.0, 0.0).ok and r.gripper_open
    p = r.get_ee_pose()
    assert abs(p.x - 275) < IK_TOL_MM and abs(p.y) < IK_TOL_MM and abs(p.z - cfg.travel_z_mm) < IK_TOL_MM, p

    # --- workspace.z_trim_mm: a negative trim really does lower BOTH the commanded grasp z and the floor,
    # and the absolute floor (table - 40 mm) still refuses anything below it whatever the trim says ---
    import dataclasses
    # the live workspace.z_trim_mm is the OPERATOR's calibration of their table, so pin a baseline here
    base = dataclasses.replace(cfg, z_trim_mm=0.0)
    zg0 = grasp_z_mm(base)
    low = dataclasses.replace(base, z_trim_mm=-6.0)
    assert abs(grasp_z_mm(low) - (zg0 - 6.0)) < 1e-9, (grasp_z_mm(low), zg0)
    assert abs(table_plane_mm(low) - (base.table_z_mm - 6.0)) < 1e-9
    assert z_trim_mm(dataclasses.replace(base, z_trim_mm=-999.0)) == -Z_TRIM_LIMIT_MM  # clamped at 150 mm
    assert Z_TRIM_LIMIT_MM >= 150.0 and not large_trim(base) and large_trim(dataclasses.replace(base, z_trim_mm=-55.0))
    # the floor is a configurable backstop, not a policy limit: a deep table is reachable by lowering it
    deep = dataclasses.replace(base, z_trim_mm=-90.0, z_floor_mm=-150.0)
    assert grasp_z_mm(deep) < -70.0 and hard_floor_mm(deep) == -150.0
    try:
        MockRobot(deep).check_target(250.0, 0.0, -160.0)
        raise AssertionError("z_floor_mm not enforced")
    except (SafetyError, _robot.SafetyError) as e:
        assert "z_floor_mm" in str(e), e
    rl = MockRobot(low)
    rl.home()
    assert rl.move_to(250.0, 0.0, grasp_z_mm(low)).ok, "the trimmed grasp height must be reachable"
    zs_low = [rl.fk_table(q)[2, 3] for q in rl.log[-1:]]
    assert min(zs_low) < zg0 - 3.0, (min(zs_low), zg0)  # genuinely lower than the untrimmed plane
    assert not rl.move_to(250.0, 0.0, grasp_z_mm(low) - 3.0).ok, "below the trimmed floor must be refused"
    try:  # the absolute floor is trim-independent
        rl.check_target(250.0, 0.0, hard_floor_mm(low) - 1.0)
        raise AssertionError("absolute floor not enforced")
    except (SafetyError, _robot.SafetyError) as e:
        assert "absolute floor" in str(e), e
    # "too low" is relative to the configured grasp height, not a hardcoded number: the operator's
    # workspace.z_trim_mm moves that floor (see grasp_z_mm)
    for bad in ((275, 0, grasp_z_mm(cfg) - 5.0), (50, 0, 60), (275, 300, 60), (500, 0, 60), (275, 0, 300)):
        assert not r.move_to(*bad).ok, bad
        try:
            r.check_target(*bad)
        except (SafetyError, _robot.SafetyError):
            pass
        else:
            raise AssertionError(bad)
    assert r.home().ok
    assert r.torque_off().ok and not r.torque
    res = r.move_to(275, 0, 60)
    assert not res.ok and "torque" in res.message, res  # E-STOP refuses motion
    assert r.torque_on().ok and r.torque
    # max_step_mm is a runaway backstop, not the workspace bound: it must be wide enough that ANY target
    # already inside the AABB is reachable in one commanded move, or legal picks fail as "exceeds max_step"
    lo_a, hi_a = cfg.aabb_min_mm, cfg.aabb_max_mm
    diag = math.hypot(hi_a[0] - lo_a[0], hi_a[1] - lo_a[1])
    assert cfg.max_step_mm >= diag, (f"max_step_mm {cfg.max_step_mm} is smaller than the workspace's own "
                                     f"XY diagonal {diag:.0f} mm: reachable targets will be refused")
    tiny = dataclasses.replace(cfg, max_step_mm=100.0)  # ... but it does still bound a huge jump
    rt = MockRobot(tiny)
    rt.home()
    bad_step = rt.move_to(300.0, 150.0, 60.0)
    assert not bad_step.ok and "XY step" in bad_step.message, bad_step
    # it must not depend on how deep the grasp goes, or a bigger negative trim would silently shrink the
    # reachable table (this used to reject far drop points)
    r.home()
    far = (300.0, 150.0)
    assert r.move_to(*far, grasp_z_mm(base)).ok, "a far target at grasp height must be reachable"
    deep2 = dataclasses.replace(base, z_trim_mm=-40.0)
    rd = MockRobot(deep2)
    rd.home()
    assert rd.move_to(*far, grasp_z_mm(deep2)).ok, "a deep trim must not shrink the reachable table"
    assert r.move_to(275, 0, 60).ok
    try:  # far beyond the arm's reach -> the FK(IK) sanity check must reject (ik_table has no AABB gate)
        r.ik_table(r._read_joints(), 600.0, 0.0, 60.0, 0.0)
        raise AssertionError("IK sanity accepted an unreachable point")
    except (SafetyError, _robot.SafetyError) as e:
        assert "IK sanity" in str(e), e
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
