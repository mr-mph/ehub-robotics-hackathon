"""Legacy ArUco-mat calibration (`sortbot.calibrate --mode aruco`) -> calib.json with a rigid table_T_base.

Drives the arm to N points, reads FK base-frame xyz (meters) and the table-mm position of a ping-pong ball held
in the gripper as seen by the overhead camera (HSV detect with parallax correction, or manual entry), then solves
the rigid table_T_base with Kabsch. Requires the 4 ArUco tags from config.yaml to be visible.

    ./run.sh -m sortbot.calibrate --mode aruco --mock             # synthetic rig, writes calib.json
    ./run.sh -m sortbot.calibrate --mode aruco --cam-height 700    # real hardware
    ./run.sh -m sortbot.calibrate --mode aruco --manual            # type table mm instead of ball detection

Parallax: the ball centre sits at the commanded z, so the nadir offset is scaled by (H - z)/H. Vertical offset:
measured first by touching the fingertip to the table (or --base-z-offset-mm to skip).
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from sortbot import config as cfgmod
from sortbot.calibration import (ORANGE, TableHomography, ball_table_xy, base_to_table, detect_target,
                                 render_mat, residuals_mm, save_calib, solve_rigid)

DEFAULT_POINTS = [(200, -120, 120), (200, 120, 120), (320, 120, 120), (320, -120, 120),
                  (260, 0, 160), (380, 0, 120), (260, -60, 90), (260, 60, 90)]


def make_fk(cfg: cfgmod.Config):
    from lerobot.model.kinematics import RobotKinematics

    k = RobotKinematics(str(cfg.urdf), "gripper_frame_link")
    return lambda joints_deg: k.forward_kinematics(np.asarray(joints_deg, float))[:3, 3]


class _FakeRig:
    """--mock: move_to(x, y, z) executes in the *base* frame (identity prior); the hidden transform says where
    that really is on the table, and the overhead frame is rendered with the ball there."""

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


def measure_base_z_offset(a, fake, robot, fk) -> float:
    if a.base_z_offset_mm is not None:
        return float(a.base_z_offset_mm)
    if fake is not None:
        return -fake.touch_z_base_m() * 1000.0
    zs = []
    while len(zs) < 2:
        print(f"touch point {len(zs) + 1}/2: torque off / jog the arm so the fingertip rests on the table,")
        s = input("  then Enter to read FK ('d' = done if >= 1 point, 's' = skip -> assume 0): ").strip().lower()
        if s == "s":
            return 0.0
        if s == "d" and zs:
            break
        zs.append(fk(robot.get_joints_deg())[2] * 1000.0)
        print(f"  FK z at touch = {zs[-1]:.1f} mm")
    return -float(np.mean(zs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--cam-height", type=float, default=700.0)
    ap.add_argument("--cam-xy", type=float, nargs=2, default=None)
    ap.add_argument("--ball-radius", type=float, default=20.0)
    ap.add_argument("--points", type=str, default=None, help="x,y,z;x,y,z;... table mm")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--base-z-offset-mm", type=float, default=None)
    a = ap.parse_args(argv)

    cfg = cfgmod.load()
    tags = np.array(list(cfg.aruco_tags_mm.values()))
    cam_xy = np.array(a.cam_xy if a.cam_xy else tags.mean(0))
    points = [tuple(map(float, p.split(","))) for p in a.points.split(";")] if a.points else DEFAULT_POINTS
    out = a.out or cfg.calib_file

    fake = None
    if a.mock:
        fake = robot = _FakeRig(cfg, a.cam_height, cam_xy, a.ball_radius)
        fk, frame_fn = None, fake.frame
    else:
        from sortbot.robot import SO101Robot

        robot = SO101Robot(cfg)
        fk = make_fk(cfg)
        frame_fn = lambda: robot.capture("overhead")  # noqa: E731

    homog = TableHomography(cfg, mode="aruco")
    z_off = measure_base_z_offset(a, fake, robot, fk)
    print(f"base_link origin is {z_off:.1f} mm above the table")
    base_m, table_mm = [], []
    for i, (x, y, z) in enumerate(points):
        print(f"[{i + 1}/{len(points)}] move_to({x:.0f}, {y:.0f}, {z:.0f})")
        robot.move_to(x, y, z)
        if not a.mock and input("  settle, then Enter (or 's' to skip): ").strip().lower() == "s":
            continue
        frame = frame_fn()
        if not homog.update(frame):
            print("  no homography (tags not visible) - skipping point")
            continue
        p_base = fake.fk_base_m() if fake else fk(robot.get_joints_deg())
        tz = z + z_off
        xy = None
        if not a.manual:
            det = detect_target(frame, ORANGE)
            if det:
                xy = ball_table_xy(homog, det[0], det[1], tz, a.cam_height, tuple(cam_xy))
                print(f"  ball px=({det[0]:.0f},{det[1]:.0f}) r={det[2]:.0f} -> table ({xy[0]:.1f}, {xy[1]:.1f}) mm")
        if xy is None and not a.mock:
            s = input("  table x,y mm of the ball (blank = skip): ").strip()
            if s:
                xy = tuple(map(float, s.split(",")))
        if xy is None:
            print("  skipped")
            continue
        base_m.append(p_base)
        table_mm.append((xy[0], xy[1], tz))
    if len(base_m) < 3:
        print("need >= 3 points", file=sys.stderr)
        return 1
    base_m, table_mm = np.array(base_m), np.array(table_mm, float)
    T = solve_rigid(base_m * 1000.0, table_mm)
    res = residuals_mm(T, base_m, table_mm)
    for p, q, r in zip(base_m, table_mm, res):
        print(f"  base {np.round(p, 4)} -> table {np.round(base_to_table(T, p), 1)}  target {q}  res {r:.2f} mm")
    print(f"residuals: mean {res.mean():.2f} mm  max {res.max():.2f} mm")
    save_calib(out, T, base_m, table_mm, {"method": "aruco", "cam_height_mm": a.cam_height, "cam_xy_mm": cam_xy.tolist(),
                                            "ball_radius_mm": a.ball_radius, "base_z_offset_mm": z_off})
    print(f"wrote {out}")
    if fake is not None:
        e_rot, e_t = np.abs(T[:3, :3] - fake.T_true[:3, :3]).max(), np.abs(T[:3, 3] - fake.T_true[:3, 3]).max()
        print(f"[mock] rotation err {e_rot:.4f}, translation err {e_t:.2f} mm")
        assert e_rot < 0.005 and e_t < 1.0 and res.max() < 2.0, "mock calibration did not recover ground truth"
        print("calibrate aruco mock selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
