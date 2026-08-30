"""Teleoperated overhead-camera calibration -> sortbot/calib/calib.json (no ArUco tags needed).

The follower gripper holds a coloured target (default: the green ball). A background thread teleoperates the
follower from the leader arm (~30 Hz, leader.get_action() -> follower joints). The human moves the target around
the workspace and CAPTUREs samples: each sample pairs the target's overhead pixel centroid with the FK base-frame
xyz. After >= 4 samples a homography H (px -> base mm) is refitted on every capture so the residuals are visible
live. FINISH persists H; from then on the table frame IS the base frame in xy (x fwd, y left, mm) and
table_T_base is identity apart from the z offset from the TOUCH-TABLE step (FK z of the fingertip on the table).

    ./run.sh -m sortbot.calibrate                    # leader + follower + overhead cam, HUD on :8765 (buttons) + keys
    ./run.sh -m sortbot.calibrate --target orange    # or hsv:lo_h,lo_s,lo_v,hi_h,hi_s,hi_v, or --sample U V (from a frame)
    ./run.sh -m sortbot.calibrate --mock             # MockRobot + virtual leader + rendered frames; asserts < 1 mm
    ./run.sh -m sortbot.calibrate --mode aruco       # legacy: ArUco mat + Kabsch rigid transform (sortbot.calibrate_aruco)

Keys (terminal): space = capture, u = undo, t = touch table, enter = finish, q = cancel. HUD: same as buttons.
Height: H is exact for centroids at plane_z_mm (mean target-centre height of the samples); taller objects project
slightly outward from the camera nadir, lower ones inward.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from sortbot import config as cfgmod
from sortbot.calibration import (ColorTarget, _ip, detect_target, fit_px_to_mm, load_calib_dict,
                                 save_ball_calib)

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TELEOP_HZ, DETECT_EVERY = 30.0, 3  # detection runs every DETECT_EVERY teleop ticks


# ---------------------------------------------------------------- rigs (follower + camera)


class RobotRig:
    """Follower side for sortbot.robot._KinematicBase (SO101Robot or MockRobot). frame_fn -> overhead RGB."""

    def __init__(self, robot, frame_fn=None):
        self.robot = robot
        self.frame_fn = frame_fn or (lambda: robot.capture("overhead"))

    def read_joints(self) -> np.ndarray:
        return self.robot._read_joints()

    def write_joints(self, q: np.ndarray) -> None:
        self.robot._write_joints(np.asarray(q, float))

    def fk_base_mm(self, q: np.ndarray) -> np.ndarray:
        return self.robot.kin.forward_kinematics(np.asarray(q, float))[:3, 3] * 1000.0

    def frame(self) -> np.ndarray:
        return self.frame_fn()


class FakeRig(RobotRig):
    """--mock: MockRobot follower, overhead frames rendered with the target at the *true* projection of the FK
    xy (hidden ground-truth H_true: base mm -> px) on top of `background` (or a dark mat)."""

    def __init__(self, robot, H_true_mm_to_px: np.ndarray, background=None, z_offset_true_mm: float = 25.0,
                 target: ColorTarget | None = None):
        super().__init__(robot)
        self.H_true, self.z_off_true, self.bg = np.asarray(H_true_mm_to_px, float), z_offset_true_mm, background
        self.color = (40, 220, 60) if target is None or target.name == "green" else (255, 150, 20)
        self.w, self.h = robot.cfg.overhead_cam.width, robot.cfg.overhead_cam.height

    def frame(self) -> np.ndarray:
        img = self.bg() if callable(self.bg) else (self.bg.copy() if self.bg is not None else np.full((self.h, self.w, 3), 30, np.uint8))
        p = self.fk_base_mm(self.read_joints())
        u, v = cv2.perspectiveTransform(np.array([[[p[0], p[1]]]]), self.H_true).ravel()
        cv2.circle(img, _ip((u, v)), 16, self.color, -1)
        return img


class VirtualLeader:
    """Scripted 'human' for --mock: get_action() returns the joints of the current scripted pose (base-frame
    IK on the follower); drive(session) walks the poses, capturing each, and finishes."""

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

    def drive(self, ctrl: "CalibController", dwell_s: float = 0.15) -> None:
        """Script the human against the controller (so HUD status / on_done fire exactly as with real triggers)."""
        self.goto(self.touch, dwell_s)
        print("  " + ctrl.trigger("touch_table")["message"])
        for q in self.poses:
            self.goto(q, dwell_s)
            print("  " + ctrl.trigger("capture")["message"])
        print("  " + ctrl.trigger("finish")["message"])

    def disconnect(self) -> None:
        pass


def open_leader(cfg: cfgmod.Config):
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    ld = SO101Leader(SO101LeaderConfig(port=cfg.leader_port, id=cfg.leader_id, calibration_dir=cfg.robot_calibration_dir))
    ld.connect()
    return ld


# ---------------------------------------------------------------- session


class CalibSession:
    """One teleoperated calibration run. Thread-safe; all triggers return {ok, message, data}."""

    def __init__(self, rig: RobotRig, leader, target: ColorTarget, out: Path, min_spacing_mm: float = 15.0,
                 z_offset_mm: float | None = None, on_frame=None):
        self.rig, self.leader, self.target, self.out = rig, leader, target, Path(out)
        self.min_spacing, self.on_frame = min_spacing_mm, on_frame
        self.z_off = z_offset_mm  # None until touch_table() (or a previous calib.json value is passed in)
        self.samples: list[dict] = []  # {px:(u,v), base_mm:(x,y,z)}
        self.H = self.res = None
        self.det = self.fk = None
        self.frame_rgb: np.ndarray | None = None
        self.state, self.message = "idle", ""
        self._lock, self._stop = threading.RLock(), threading.Event()
        self._thread: threading.Thread | None = None

    # ---- teleop loop ----
    def start(self) -> None:
        self.state, self.message = "running", "teleop on: move the target, then capture"
        self._thread = threading.Thread(target=self._loop, daemon=True, name="calib-teleop")
        self._thread.start()

    def _loop(self) -> None:
        tick, dt = 0, 1.0 / TELEOP_HZ
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                a = self.leader.get_action()
                with self._lock:
                    self.rig.write_joints(np.array([float(a[f"{m}.pos"]) for m in MOTORS]))
                    if tick % DETECT_EVERY == 0:
                        self._observe()
            except Exception as e:  # noqa: BLE001
                self.state, self.message = "error", f"teleop: {e}"
                break
            tick += 1
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))

    def _observe(self) -> tuple[np.ndarray, tuple | None, np.ndarray]:
        self.frame_rgb = self.rig.frame()
        self.det = detect_target(self.frame_rgb, self.target)
        self.fk = self.rig.fk_base_mm(self.rig.read_joints())
        if self.on_frame:
            self.on_frame(self.annotated())
        return self.frame_rgb, self.det, self.fk

    def _stop_teleop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    # ---- triggers ----
    def capture(self) -> dict:
        with self._lock:
            if self.state != "running":
                return _r(False, f"not running ({self.state})")
            _, det, fk = self._observe()
            if det is None:
                return _r(False, "target not detected; adjust colour (click it in the HUD) or lighting")
            for s in self.samples:
                if np.linalg.norm(np.array(s["base_mm"][:2]) - fk[:2]) < self.min_spacing:
                    return _r(False, f"too close to an existing sample (< {self.min_spacing:.0f} mm); move further")
            self.samples.append({"px": (det[0], det[1]), "base_mm": tuple(float(v) for v in fk)})
            self._refit()
            n = len(self.samples)
            self.message = f"sample {n}: px=({det[0]:.0f},{det[1]:.0f}) base=({fk[0]:.0f},{fk[1]:.0f},{fk[2]:.0f}) mm" + (
                f"  residual mean {self.res.mean():.2f} max {self.res.max():.2f} mm" if self.res is not None else f"  ({max(0, 4 - n)} more for a fit)")
            return _r(True, self.message, {"n": n})

    def undo(self) -> dict:
        with self._lock:
            if not self.samples:
                return _r(False, "no samples to undo")
            self.samples.pop()
            self._refit()
            self.message = f"undo -> {len(self.samples)} samples"
            return _r(True, self.message, {"n": len(self.samples)})

    def touch_table(self) -> dict:
        with self._lock:
            fk = self.rig.fk_base_mm(self.rig.read_joints())
            self.z_off = -float(fk[2])
            self.message = f"table plane: base_link origin is {self.z_off:.1f} mm above the table"
            return _r(True, self.message, {"z_offset_mm": self.z_off})

    def finish(self) -> dict:
        with self._lock:
            if self.state != "running":
                return _r(False, f"not running ({self.state})")
            if len(self.samples) < 4:
                return _r(False, f"need >= 4 samples, have {len(self.samples)}")
            if self.z_off is None:
                print("[calibrate] no touch-table step: assuming base_link origin is on the table plane (z offset 0)")
                self.z_off = 0.0
            self._refit()
            if self.H is None:
                return _r(False, self.message or "fit failed; capture more spread-out samples")
            px, mm = self._arrays()
            plane_z = float(mm[:, 2].mean() + self.z_off)
            save_ball_calib(self.out, self.H, px, mm[:, :2], self.res, plane_z, self.target, self.z_off, method="teleop")
            self.state = "fitted"
            self.message = f"fitted {len(px)} pts: residual mean {self.res.mean():.2f} max {self.res.max():.2f} mm, plane z {plane_z:.0f} mm -> {self.out}"
            self._stop_teleop()
            return _r(True, self.message, {"residuals_mm": self.res.tolist(), "plane_z_mm": plane_z})

    def cancel(self) -> dict:
        with self._lock:
            self.state, self.message = "cancelled", "cancelled (nothing written)"
            self._stop_teleop()
            return _r(True, self.message)

    def sample(self, u: float, v: float) -> dict:
        """Pick the target colour from the latest frame at (u, v)."""
        with self._lock:
            frame = self.frame_rgb if self.frame_rgb is not None else self.rig.frame()
            self.target = ColorTarget.from_sample(frame, u, v)
            self.det = det = detect_target(frame, self.target)
            self.message = f"target {self.target.name} hsv {self.target.hsv_lo.tolist()}..{self.target.hsv_hi.tolist()}: " + (
                f"detected at ({det[0]:.0f},{det[1]:.0f}) r={det[2]:.0f}" if det else "not detected")
            return _r(det is not None, self.message, {"target": self.target.to_dict(), "det": det})

    # ---- fit / status ----
    def _arrays(self):
        return np.array([s["px"] for s in self.samples], float), np.array([s["base_mm"] for s in self.samples], float)

    def _refit(self) -> None:
        self.H = self.res = None
        if len(self.samples) >= 4:
            px, mm = self._arrays()
            try:
                self.H, self.res = fit_px_to_mm(px, mm[:, :2])
            except Exception as e:  # noqa: BLE001
                self.message = f"fit failed: {e}"

    def status(self) -> dict:
        return {"state": self.state, "message": self.message, "n": len(self.samples),
                "fk_mm": None if self.fk is None else [round(float(v), 1) for v in self.fk],
                "det": None if self.det is None else [round(float(v), 1) for v in self.det],
                "residual_mean_mm": None if self.res is None else round(float(self.res.mean()), 2),
                "residual_max_mm": None if self.res is None else round(float(self.res.max()), 2),
                "z_offset_mm": self.z_off, "target": self.target.to_dict(),
                "samples": [[round(v, 1) for v in (*s["px"], *s["base_mm"])] for s in self.samples]}

    def annotated(self) -> np.ndarray:
        img = self.frame_rgb.copy()
        for s in self.samples:
            cv2.drawMarker(img, _ip(s["px"]), (255, 200, 0), cv2.MARKER_TILTED_CROSS, 10, 1)
        if self.det:
            cv2.circle(img, _ip(self.det), int(self.det[2]), (0, 255, 0), 2)
        txt = f"{self.state} n={len(self.samples)}" + (f" res {self.res.mean():.1f}/{self.res.max():.1f}mm" if self.res is not None else "")
        cv2.putText(img, txt, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        return img


def _r(ok: bool, message: str, data=None) -> dict:
    return {"ok": ok, "message": message, "data": data}


# ---------------------------------------------------------------- controller (HUD actions, shared by main + CLI)


class CalibController:
    """Owns at most one CalibSession; registers HUD actions (group 'calibration') and the 'calibration' state
    source. rig_fn/leader_fn are lazy so main.py can open the leader only when a session starts."""

    def __init__(self, cfg: cfgmod.Config, rig_fn, leader_fn, target: ColorTarget, out: Path | None = None,
                 on_done=None, latest_frame=None, driver=None):
        self.cfg, self.rig_fn, self.leader_fn, self.target = cfg, rig_fn, leader_fn, target
        self.out, self.on_done, self.latest_frame, self.driver = Path(out or cfg.calib_file), on_done, latest_frame, driver
        self.session: CalibSession | None = None
        self.hud = None

    @property
    def active(self) -> bool:
        return self.session is not None and self.session.state == "running"

    def start(self) -> dict:
        if self.active:
            return _r(False, "calibration already running")
        try:
            rig, leader = self.rig_fn(), self.leader_fn()
        except Exception as e:  # noqa: BLE001
            return _r(False, f"cannot start: {e}")
        prev = load_calib_dict(self.out)
        z_prev = prev["base_z_offset_mm"] if prev.get("method") else None
        on_frame = (lambda img: self.hud.update(img, None, {})) if self.hud else None
        self.session = CalibSession(rig, leader, self.target, self.out, self.cfg.calib_min_spacing_mm, z_prev, on_frame)
        self.session.start()
        if self.driver:  # --mock: virtual leader scripts the human
            threading.Thread(target=self.driver, args=(leader, self), daemon=True, name="calib-driver").start()
        return _r(True, "calibration started" + (f" (reusing z offset {z_prev:.1f} mm; press touch-table to re-measure)" if z_prev is not None else ""))

    def _end(self, r: dict) -> dict:
        if r["ok"] and self.session is not None and self.session.state != "running":
            try:
                self.session.leader.disconnect()
            except Exception:  # noqa: BLE001
                pass
            if self.on_done:
                self.on_done(self.session)
        return r

    def trigger(self, fn_name: str) -> dict:
        """capture | undo | touch_table | finish | cancel on the current session."""
        if self.session is None:
            return _r(False, "no calibration session; press Start")
        r = getattr(self.session, fn_name)()
        return self._end(r) if fn_name in ("finish", "cancel") else r

    def sample(self, u: float, v: float) -> dict:
        if self.session is not None:
            r = self.session.sample(u, v)
            self.target = self.session.target
            return r
        frame = self.latest_frame() if self.latest_frame else None
        if frame is None:
            return _r(False, "no overhead frame yet")
        self.target = ColorTarget.from_sample(frame, u, v)
        det = detect_target(frame, self.target)
        return _r(det is not None, f"target hsv {self.target.hsv_lo.tolist()}..{self.target.hsv_hi.tolist()}: " +
                  (f"detected at ({det[0]:.0f},{det[1]:.0f}) r={det[2]:.0f}" if det else "not detected"),
                  {"target": self.target.to_dict(), "det": det})

    def status(self) -> dict:
        if self.session is None:
            return {"state": "idle", "message": "press Start (gripper must hold the target)", "n": 0, "target": self.target.to_dict()}
        return self.session.status()

    def register(self, hud) -> None:
        self.hud = hud
        g = "calibration"
        hud.register("calib_start", self.start, "Start calibration", g,
                     help="Begin the teleoperated calibration: the leader arm drives the follower while you capture samples (target ball in the gripper first).")
        hud.register("calib_touch", lambda: self.trigger("touch_table"), "Touch table", g,
                     help="With the fingertip resting on the tabletop, record the table height (once per calibration).")
        hud.register("calib_capture", lambda: self.trigger("capture"), "Capture", g,
                     help="Capture a sample: pairs the target's pixel position with the arm's FK position (spacebar). Need 4+, spread out.")
        hud.register("calib_undo", lambda: self.trigger("undo"), "Undo", g,
                     help="Drop the last captured sample.")
        hud.register("calib_finish", lambda: self.trigger("finish"), "Finish", g,
                     help="Fit the homography from the samples (4+, well spread over the mat) and save calib.json.")
        hud.register("calib_cancel", lambda: self.trigger("cancel"), "Cancel", g,
                     help="Abandon the calibration session; nothing is written.")
        hud.register("calib_sample", lambda u, v: self.sample(float(u), float(v)), None, g,  # click-to-pick (no button)
                     help="Click the overhead image: picks the target colour at that pixel and shows the detection circle.")
        hud.add_state_source("calibration", self.status)


# ---------------------------------------------------------------- keyboard + mock helpers


class KeyReader(threading.Thread):
    """space=capture, u=undo, t=touch table, enter=finish, q=cancel. Raw tty if available, else line mode."""

    def __init__(self, ctrl: CalibController):
        super().__init__(daemon=True, name="calib-keys")
        self.ctrl = ctrl

    def run(self) -> None:
        keys = {" ": "capture", "u": "undo", "t": "touch_table", "\r": "finish", "\n": "finish", "q": "cancel"}
        try:
            import termios, tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            try:
                while self.ctrl.active:
                    k = sys.stdin.read(1)
                    if k in keys:
                        print(self.ctrl.trigger(keys[k])["message"])
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except (ImportError, OSError, termios.error):  # not a tty: one command per line
            for line in sys.stdin:
                k = (line.strip()[:1] or "\n").lower()
                if k in keys:
                    print(self.ctrl.trigger(keys[k])["message"])
                if not self.ctrl.active:
                    break


MOCK_POSES_MM = [(180, -120, 40), (180, 120, 40), (290, 80, 40), (290, -80, 40),  # a quad first: 4th sample fits
                 (180, 0, 40), (240, -120, 40), (240, 0, 45), (240, 120, 40), (210, 60, 60)]
MOCK_Z_OFF_TRUE = 25.0


def mock_rig(cfg: cfgmod.Config, robot=None, H_mm_to_px: np.ndarray | None = None, background=None,
             target: ColorTarget | None = None) -> tuple[FakeRig, VirtualLeader]:
    """MockRobot follower + FakeRig frames + VirtualLeader. Default H_true: 2 px/mm, x fwd = up, y left = left,
    plus a little perspective so a true homography (not an affinity) is being recovered."""
    from sortbot.robot import MockRobot

    robot = robot or MockRobot(cfg)
    if H_mm_to_px is None:
        w, h = cfg.overhead_cam.width, cfg.overhead_cam.height
        H_mm_to_px = np.array([[0.0, -2.0, w / 2], [-2.0, 0.0, h / 2 + 2 * 250], [1e-4, 5e-5, 1.0]]) @ np.eye(3)
    rig = FakeRig(robot, H_mm_to_px, background, MOCK_Z_OFF_TRUE, target)
    return rig, VirtualLeader(robot, MOCK_POSES_MM, -MOCK_Z_OFF_TRUE)


def _print_table(s: CalibSession) -> None:
    print(f"{'#':>2} {'u':>6} {'v':>6} {'base x':>7} {'base y':>7} {'base z':>7} {'res mm':>7}")
    for i, (smp, r) in enumerate(zip(s.samples, s.res), 1):
        u, v = smp["px"]; x, y, z = smp["base_mm"]
        print(f"{i:>2} {u:>6.1f} {v:>6.1f} {x:>7.1f} {y:>7.1f} {z:>7.1f} {r:>7.2f}")
    print(f"residuals: mean {s.res.mean():.2f} mm  max {s.res.max():.2f} mm  |  z offset {s.z_off:.1f} mm")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["teleop", "ball", "aruco"], default="teleop", help="ball = teleop (alias)")
    ap.add_argument("--mock", action="store_true", help="no hardware: MockRobot + virtual leader + rendered frames")
    ap.add_argument("--target", default=None, help="green | orange | hsv:lo_h,lo_s,lo_v,hi_h,hi_s,hi_v (default: config)")
    ap.add_argument("--sample", type=float, nargs=2, metavar=("U", "V"), help="pick the target colour at this overhead pixel")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--hud-port", type=int, default=0, help="HUD port (default config); 0 = config")
    ap.add_argument("--no-hud", action="store_true")
    a, rest = ap.parse_known_args(argv)
    if a.mode == "aruco":
        from sortbot import calibrate_aruco
        return calibrate_aruco.main(rest + (["--mock"] if a.mock else []) + (["--out", a.out] if a.out else []))

    cfg = cfgmod.load()
    target = ColorTarget.parse(a.target or cfg.calib_target)
    out = Path(a.out or cfg.calib_file)
    if a.mock:
        rig, leader = mock_rig(cfg, target=target)
        rig_fn, leader_fn, driver = (lambda: rig), (lambda: leader), (lambda ld, c: ld.drive(c))
    else:
        from sortbot.robot import SO101Robot
        robot = SO101Robot(cfg)
        rig = RobotRig(robot)
        rig_fn, leader_fn, driver = (lambda: rig), (lambda: open_leader(cfg)), None
    if a.sample:
        target = ColorTarget.from_sample(rig.frame(), *a.sample)
        print(f"sampled target: {target.to_dict()}")
    done = threading.Event()
    ctrl = CalibController(cfg, rig_fn, leader_fn, target, out, on_done=lambda s: done.set(), driver=driver)
    hud = None
    if not a.no_hud:
        from sortbot.hud import HUD
        hud = HUD(port=a.hud_port or cfg.hud_port)
        ctrl.register(hud)
        try:
            hud.start()
        except Exception as e:  # noqa: BLE001
            print(f"[calibrate] HUD unavailable ({e}); keyboard only")
            hud = None
    print(ctrl.start()["message"])
    if not a.mock:
        print("keys: space=capture  u=undo  t=touch table  enter=finish  q=cancel" + ("  (or use the HUD buttons)" if hud else ""))
        KeyReader(ctrl).start()
    done.wait()
    s = ctrl.session
    if hud:
        hud.stop()
    if s.state != "fitted":
        print(s.message)
        return 1
    _print_table(s)
    print(s.message)
    if a.mock:
        px, _ = s._arrays()
        truth = cv2.perspectiveTransform(px.reshape(-1, 1, 2), np.linalg.inv(rig.H_true)).reshape(-1, 2)
        fit = cv2.perspectiveTransform(px.reshape(-1, 1, 2), s.H).reshape(-1, 2)
        e = np.linalg.norm(fit - truth, axis=1).max()
        print(f"[mock] max error vs hidden ground truth {e:.3f} mm, z offset err {abs(s.z_off - MOCK_Z_OFF_TRUE):.2f} mm")
        assert s.res.max() < 1.0 and e < 1.0 and abs(s.z_off - MOCK_Z_OFF_TRUE) < 1.0, "mock calibration did not recover ground truth"
        print("calibrate mock selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
