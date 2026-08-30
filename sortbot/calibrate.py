"""Teleoperated overhead-camera calibration -> sortbot/calib/calib.json (no ArUco tags needed).

The follower gripper holds a coloured target (default: the green ball). A background thread teleoperates the
follower from the leader arm (~30 Hz, leader.get_action() -> follower joints). The human moves the target around
the workspace and CAPTUREs samples: each sample pairs the target's overhead pixel centroid with the FK base-frame
xyz. After >= 4 samples a homography H (px -> base mm) is refitted on every capture so the residuals are visible
live. FINISH persists H; from then on the table frame IS the base frame in xy (x fwd, y left, mm) and
table_T_base is identity apart from the z offset from the TOUCH-TABLE step (FK z of the fingertip on the table).

    ./run.sh -m sortbot.calibrate                    # leader + follower + overhead cam, HUD on :8765 (buttons) + keys
    ./run.sh -m sortbot.calibrate --target orange    # or hsv:lo_h,lo_s,lo_v,hi_h,hi_s,hi_v, or --sample U V (from a frame)
    ./run.sh -m sortbot.calibrate --mode aruco       # legacy: ArUco mat + Kabsch rigid transform (sortbot.calibrate_aruco)
    ./run.sh -m sortbot.calibrate --selftest         # no hardware: scripted run against sortbot/testing.py fixtures

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
from sortbot.calibration import (ColorTarget, _ip, calib_summary_file, collinearity_ratio, coverage_pct,
                                 coverage_verdict, detect_target, fit_px_to_mm, load_calib_dict,
                                 sample_hull_px, save_ball_calib)

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


def open_leader(cfg: cfgmod.Config):
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    ld = SO101Leader(SO101LeaderConfig(port=cfg.leader_port, id=cfg.leader_id, calibration_dir=cfg.robot_calibration_dir))
    ld.connect()
    return ld


# ---------------------------------------------------------------- session


class CalibSession:
    """One teleoperated calibration run. Thread-safe; all triggers return {ok, message, data}."""

    def __init__(self, rig: RobotRig, leader, target: ColorTarget, out: Path, min_spacing_mm: float = 15.0,
                 z_offset_mm: float | None = None, on_frame=None, bus_lock=None):
        self.rig, self.leader, self.target, self.out = rig, leader, target, Path(out)
        self.min_spacing, self.on_frame = min_spacing_mm, on_frame
        self.z_off = z_offset_mm  # None until touch_table() (or a previous calib.json value is passed in)
        self.samples: list[dict] = []  # {px:(u,v), base_mm:(x,y,z)}
        self.H = self.res = None
        self.det = self.fk = None
        self.frame_rgb: np.ndarray | None = None
        self.frame_wh: tuple[int, int] | None = None  # for sample-coverage feedback
        self.state, self.message = "idle", ""
        self._lock, self._stop = threading.RLock(), threading.Event()
        # Serializes ALL motor-bus / follower access. Share the caller's robot lock so no other thread
        # (e.g. the HUD /state poller) sync-reads the serial port mid teleop tick -> feetech "Port is in use".
        # Ordering: _bus is only ever taken inside _lock (or alone), never the other way around.
        self._bus = bus_lock or threading.RLock()
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
                with self._lock:
                    with self._bus:
                        a = self.leader.get_action()
                        self.rig.write_joints(np.array([float(a[f"{m}.pos"]) for m in MOTORS]))
                    if tick % DETECT_EVERY == 0:
                        self._observe()
            except Exception as e:  # noqa: BLE001
                self.state, self.message = "error", f"teleop: {e}"
                break
            tick += 1
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))

    def _observe(self) -> tuple[np.ndarray, tuple | None, np.ndarray]:
        with self._bus:  # frame may come through the follower; joints always do
            self.frame_rgb = self.rig.frame()
            self.fk = self.rig.fk_base_mm(self.rig.read_joints())
        self.frame_wh = (self.frame_rgb.shape[1], self.frame_rgb.shape[0])
        self.det = detect_target(self.frame_rgb, self.target)
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
            with self._bus:
                fk = self.rig.fk_base_mm(self.rig.read_joints())
            self.z_off = -float(fk[2])
            self.message = f"table plane: base_link origin is {self.z_off:.1f} mm above the table"
            return _r(True, self.message, {"z_offset_mm": self.z_off})

    def finish(self, force: bool = False) -> dict:
        with self._lock:
            if self.state != "running":
                return _r(False, f"not running ({self.state})")
            if len(self.samples) < 4:
                return _r(False, f"need >= 4 samples, have {len(self.samples)}")
            px_now = [s["px"] for s in self.samples]
            cov = coverage_pct(px_now, self.frame_wh)
            ratio = collinearity_ratio(px_now)
            if not force and (cov < 10.0 or ratio < 0.15):
                # a clustered/near-collinear fit LOOKS fine in residual but extrapolates garbage off-hull
                sig = (len(self.samples), None if self.z_off is None else round(float(self.z_off), 3))
                if getattr(self, "_finish_blocked_sig", None) == sig:
                    force = True  # a second Finish with nothing changed is the operator's override
                else:
                    self._finish_blocked_sig = sig
                    why = ("samples are nearly on one line" if ratio < 0.15
                           else f"samples cover only {cov:.0f}% of the camera view")
                    self.message = (f"not saved: {why} — the fit would be way off everywhere else. "
                                    f"Capture more spread-out samples (aim for the corners), or press "
                                    f"Finish again to save anyway.")
                    return _r(False, self.message, {"force_needed": True, "coverage_pct": round(cov, 1),
                                                    "collinearity": round(ratio, 3)})
            if self.z_off is None:
                print("[calibrate] no touch-table step: assuming base_link origin is on the table plane (z offset 0)")
                self.z_off = 0.0
            self._refit()
            if self.H is None:
                return _r(False, self.message or "fit failed; capture more spread-out samples")
            px, mm = self._arrays()
            plane_z = float(mm[:, 2].mean() + self.z_off)
            save_ball_calib(self.out, self.H, px, mm[:, :2], self.res, plane_z, self.target, self.z_off,
                            method="teleop", frame_wh=self.frame_wh)
            self.state = "fitted"
            self.message = (f"fitted {len(px)} pts: residual mean {self.res.mean():.2f} max {self.res.max():.2f} mm, "
                            f"coverage {cov:.0f}%, plane z {plane_z:.0f} mm -> {self.out} (previous kept as .bak)")
            self._stop_teleop()
            return _r(True, self.message, {"residuals_mm": self.res.tolist(), "plane_z_mm": plane_z,
                                           "coverage_pct": round(cov, 1)})

    def cancel(self) -> dict:
        with self._lock:
            self.state, self.message = "cancelled", "cancelled (nothing written)"
            self._stop_teleop()
            return _r(True, self.message)

    def sample(self, u: float, v: float) -> dict:
        """Pick the target colour from the latest frame at (u, v). A click that clearly sampled the table /
        floor (or detects an implausibly large blob) is rejected and the previous target is kept."""
        with self._lock:
            if self.frame_rgb is not None:
                frame = self.frame_rgb
            else:
                with self._bus:
                    frame = self.rig.frame()
            cand = ColorTarget.from_sample(frame, u, v)
            det = detect_target(frame, cand)
            err = _target_sanity(frame, u, v, det)
            if err:
                self.message = err
                return _r(False, err, {"target": self.target.to_dict(), "det": None})
            self.target, self.det = cand, det
            self.message = f"target {cand.name} hsv {cand.hsv_lo.tolist()}..{cand.hsv_hi.tolist()}: " + (
                f"detected at ({det[0]:.0f},{det[1]:.0f}) r={det[2]:.0f}" if det else "not detected")
            return _r(det is not None, self.message, {"target": cand.to_dict(), "det": det})

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
        px = [s["px"] for s in self.samples]
        cov = coverage_pct(px, self.frame_wh)
        res = None if self.res is None else [round(float(r), 2) for r in self.res]
        worst_i = None if not res else int(np.argmax(self.res))
        med = None if not res else float(np.median(self.res))
        outliers = [] if not res else [i for i, r in enumerate(self.res)
                                       if med is not None and med > 1e-6 and r > 3.0 * med]
        return {"state": self.state, "message": self.message, "n": len(self.samples),
                "fk_mm": None if self.fk is None else [round(float(v), 1) for v in self.fk],
                "det": None if self.det is None else [round(float(v), 1) for v in self.det],
                "residual_mean_mm": None if self.res is None else round(float(self.res.mean()), 2),
                "residual_max_mm": None if self.res is None else round(float(self.res.max()), 2),
                "residuals_mm": res, "worst_i": worst_i, "outlier_i": outliers,
                "coverage_pct": round(cov, 1), "coverage_verdict": coverage_verdict(cov),
                "z_offset_mm": self.z_off, "target": self.target.to_dict(),
                "samples": [[round(v, 1) for v in (*s["px"], *s["base_mm"])] for s in self.samples]}

    def annotated(self) -> np.ndarray:
        """Live calibration view: every sample as a numbered dot (worst residual in red), the samples'
        convex hull, the detection circle and a status line with coverage — so it is visible at a glance
        where has and has NOT been sampled. As soon as a fit exists (4+ samples) the cm grid of the
        CURRENT fit is drawn live, so it is obvious while calibrating whether the projection lines up
        with the table (it refits on every capture)."""
        img = self.frame_rgb.copy()
        if self.H is not None:
            from sortbot import perception  # grid under the sample markers: how the current fit looks
            try:
                img = perception.render_overlay(img, self.H, [], None, [])
            except Exception as e:  # noqa: BLE001 - a degenerate interim fit must not kill the teleop view
                cv2.putText(img, f"grid preview unavailable: {e}", (8, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 80), 1, cv2.LINE_AA)
        px = [s["px"] for s in self.samples]
        hull = sample_hull_px(px)
        if hull is not None:
            cv2.polylines(img, [hull.round().astype(np.int32).reshape(-1, 1, 2)], True, (255, 200, 0), 1, cv2.LINE_AA)
        worst_i = None if self.res is None or not len(self.res) else int(np.argmax(self.res))
        for i, s in enumerate(self.samples):
            col = (255, 60, 60) if i == worst_i and self.res is not None and self.res[worst_i] > 3.0 else (255, 200, 0)
            cv2.circle(img, _ip(s["px"]), 5, col, -1)
            cv2.putText(img, str(i + 1), (_ip(s["px"])[0] + 6, _ip(s["px"])[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if self.det:
            cv2.circle(img, _ip(self.det), int(self.det[2]), (0, 255, 0), 2)
        cov = coverage_pct(px, self.frame_wh)
        txt = f"{self.state} n={len(self.samples)} coverage {cov:.0f}%" + (
            f" res {self.res.mean():.1f}/{self.res.max():.1f}mm" if self.res is not None else "")
        cv2.putText(img, txt, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        return img


def _r(ok: bool, message: str, data=None) -> dict:
    return {"ok": ok, "message": message, "data": data}


def _target_sanity(frame_rgb: np.ndarray, u: float, v: float, det) -> str | None:
    """None if the sampled colour looks like a compact coloured target, else a rejection message.
    Guards against clicks that landed on the bare table/floor: a low-saturation window matches half the
    frame and detect_target then returns a giant blob (seen live: r=146 px on a 640 px frame)."""
    h, w = frame_rgb.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    patch = frame_rgb[max(0, vi - 6):min(h, vi + 7), max(0, ui - 6):min(w, ui + 7)]
    hsv = cv2.cvtColor(patch.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    if int(np.median(hsv[:, 1])) < 60:
        return "that doesn't look like a compact coloured target (gray/brown surface?) -- click directly on the ball"
    if det is not None and det[2] > 0.12 * w:
        return (f"that doesn't look like a compact coloured target -- click directly on the ball "
                f"(detected radius {det[2]:.0f}px is too big)")
    return None


# ---------------------------------------------------------------- controller (HUD actions, shared by main + CLI)


class CalibController:
    """Owns at most one CalibSession; registers HUD actions (group 'calibration') and the 'calibration' state
    source. rig_fn/leader_fn are lazy so main.py can open the leader only when a session starts."""

    def __init__(self, cfg: cfgmod.Config, rig_fn, leader_fn, target: ColorTarget, out: Path | None = None,
                 on_done=None, latest_frame=None, driver=None, bus_lock=None):
        self.cfg, self.rig_fn, self.leader_fn, self.target = cfg, rig_fn, leader_fn, target
        self.out, self.on_done, self.latest_frame, self.driver = Path(out or cfg.calib_file), on_done, latest_frame, driver
        self.bus_lock = bus_lock  # shared motor-bus lock (main.py passes Session.robot_lock)
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
        self.session = CalibSession(rig, leader, self.target, self.out, self.cfg.calib_min_spacing_mm, z_prev,
                                    on_frame, bus_lock=self.bus_lock)
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

    def trigger(self, fn_name: str, **kw) -> dict:
        """capture | undo | touch_table | finish | cancel on the current session."""
        if self.session is None:
            return _r(False, "no calibration session; press Start")
        r = getattr(self.session, fn_name)(**kw)
        return self._end(r) if fn_name in ("finish", "cancel") else r

    def sample(self, u: float, v: float) -> dict:
        if self.session is not None:
            r = self.session.sample(u, v)
            self.target = self.session.target
            return r
        frame = self.latest_frame() if self.latest_frame else None
        if frame is None:
            return _r(False, "no overhead frame yet")
        cand = ColorTarget.from_sample(frame, u, v)
        det = detect_target(frame, cand)
        err = _target_sanity(frame, u, v, det)
        if err:  # keep the previous target: a bad click must not poison the calibration
            return _r(False, err, {"target": self.target.to_dict(), "det": None})
        self.target = cand
        return _r(det is not None, f"target hsv {cand.hsv_lo.tolist()}..{cand.hsv_hi.tolist()}: " +
                  (f"detected at ({det[0]:.0f},{det[1]:.0f}) r={det[2]:.0f}" if det else "not detected"),
                  {"target": cand.to_dict(), "det": det})

    def status(self) -> dict:
        if self.session is None:
            return {"state": "idle", "message": "press Start (gripper must hold the target)", "n": 0,
                    "target": self.target.to_dict(), "loaded": self._loaded_summary()}
        st = self.session.status()
        if self.session.state != "running":
            st["loaded"] = self._loaded_summary()
        return st

    def _loaded_summary(self) -> str:
        """One line about the SAVED calibration (persists in calib.json and auto-loads on startup) --
        shown in the Setup tab so nobody re-calibrates just because they doubt it stuck. Cached on mtime."""
        try:
            mt = self.out.stat().st_mtime
        except OSError:
            mt = None
        cache = getattr(self, "_sum_cache", None)
        if cache is None or cache[0] != mt:
            s = calib_summary_file(self.out) if mt is not None else None
            self._sum_cache = (mt, s or "no saved calibration yet — run the calibration steps below")
        return self._sum_cache[1]

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
        hud.register("calib_finish", lambda force=False: self.trigger("finish", force=bool(force)), "Finish", g,
                     help="Fit the homography from the samples (4+, spread over the WHOLE camera view) and save "
                          "calib.json (previous file kept as .bak). Clustered or collinear samples block the save; "
                          "press Finish again to save anyway.")
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


def _print_table(s: CalibSession) -> None:
    print(f"{'#':>2} {'u':>6} {'v':>6} {'base x':>7} {'base y':>7} {'base z':>7} {'res mm':>7}")
    for i, (smp, r) in enumerate(zip(s.samples, s.res), 1):
        u, v = smp["px"]; x, y, z = smp["base_mm"]
        print(f"{i:>2} {u:>6.1f} {v:>6.1f} {x:>7.1f} {y:>7.1f} {z:>7.1f} {r:>7.2f}")
    print(f"residuals: mean {s.res.mean():.2f} mm  max {s.res.max():.2f} mm  |  z offset {s.z_off:.1f} mm")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["teleop", "ball", "aruco"], default="teleop", help="ball = teleop (alias)")
    ap.add_argument("--selftest", action="store_true",
                    help="no hardware: scripted run against the sortbot/testing.py fixtures (writes a temp file)")
    ap.add_argument("--target", default=None, help="green | orange | hsv:lo_h,lo_s,lo_v,hi_h,hi_s,hi_v (default: config)")
    ap.add_argument("--sample", type=float, nargs=2, metavar=("U", "V"), help="pick the target colour at this overhead pixel")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--hud-port", type=int, default=0, help="HUD port (default config); 0 = config")
    ap.add_argument("--no-hud", action="store_true")
    a, rest = ap.parse_known_args(argv)
    if a.mode == "aruco":
        from sortbot import calibrate_aruco
        return calibrate_aruco.main(rest + (["--selftest"] if a.selftest else []) + (["--out", a.out] if a.out else []))

    cfg = cfgmod.load()
    target = ColorTarget.parse(a.target or cfg.calib_target)
    if a.selftest:
        import tempfile
        from sortbot.testing import mock_rig  # test fixtures; only reachable from this selftest
        out = Path(a.out or Path(tempfile.mkdtemp()) / "calib_selftest.json")  # NEVER the real calib.json
        rig, leader = mock_rig(cfg, target=target)
        rig_fn, leader_fn, driver = (lambda: rig), (lambda: leader), (lambda ld, c: ld.drive(c))
    else:
        out = Path(a.out or cfg.calib_file)
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
    if not a.selftest:
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
    if a.selftest:
        from sortbot.testing import MOCK_Z_OFF_TRUE
        px, _ = s._arrays()
        truth = cv2.perspectiveTransform(px.reshape(-1, 1, 2), np.linalg.inv(rig.H_true)).reshape(-1, 2)
        fit = cv2.perspectiveTransform(px.reshape(-1, 1, 2), s.H).reshape(-1, 2)
        e = np.linalg.norm(fit - truth, axis=1).max()
        print(f"[selftest] max error vs hidden ground truth {e:.3f} mm, z offset err {abs(s.z_off - MOCK_Z_OFF_TRUE):.2f} mm")
        assert s.res.max() < 1.0 and e < 1.0 and abs(s.z_off - MOCK_Z_OFF_TRUE) < 1.0, "selftest calibration did not recover ground truth"
        print("calibrate selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
