"""Overhead overlay rendering: the VLM does the seeing, this module only draws what the VLM (and the
human) need to read positions off the image — a labelled grid, the zone outlines and the EE marker —
plus the px<->mm coordinate helpers.

homography H (3x3) maps overhead pixel (u, v, 1) -> table-frame (x_mm, y_mm, 1); produced by
sortbot.calibration (teleop-fitted H from calib.json, or ArUco tags). There is no object detector here:
the VLM reads object positions straight off the labelled grid in the overlay.

UNIT BOUNDARY: all geometry in this module is table-frame MILLIMETERS (matching the robot, config.yaml
and calib.json), but every label painted onto the overlay is CENTIMETERS — the VLM-facing surface is cm
(see sortbot.vlm). The /10 conversions in render_overlay are that boundary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from sortbot.config import Config
from sortbot.types import Pose, Zone

# ---------------------------------------------------------------- coordinate helpers


def px_to_mm(H: np.ndarray, pts_px) -> np.ndarray:
    p = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def mm_to_px(H: np.ndarray, pts_mm) -> np.ndarray:
    p = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.linalg.inv(np.asarray(H, dtype=np.float64))).reshape(-1, 2)


def mat_polygon_mm(config: Config) -> list[tuple[float, float]]:
    """Mat outline from the ArUco tag positions (TL, TR, BR, BL order in config)."""
    return [config.aruco_tags_mm[i] for i in sorted(config.aruco_tags_mm)]


# ---------------------------------------------------------------- overlay renderer

_BGR = dict(grid=(120, 120, 120), tick=(60, 235, 255), zone=(0, 200, 255), ee=(255, 80, 255),
            text=(255, 255, 255), axis=(80, 220, 120), hull=(200, 160, 60))


def _put(img, text, org, col, scale=0.42, thick=1):
    # dark halo first so labels stay readable on any background (the VLM must be able to read them)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def render_overlay(
    frame: np.ndarray,
    homography: np.ndarray,
    zones: list[Zone],
    ee_pose: Pose | None = None,
    rules: list[str] = (),
    grid_mm: float = 50.0,
    calib_region_px: np.ndarray | None = None,
    calib_samples_px=None,
) -> np.ndarray:
    """frame is RGB; returns an RGB copy annotated with a cm-labelled grid, zone outlines + names + drop
    markers, the EE marker and the rules legend. calib_samples_px: optional Nx2 of the calibration's own
    sample pixels, drawn as small anchors -- the grid is pinned there and interpolated everywhere else, so
    they show at a glance whether a mismatch means 'bad fit' or 'far from any anchor'.
    calib_region_px: optional Nx2 polygon (overhead px) of the area covered by the calibration samples — drawn as a subtle outline so it is visible where the
    px->cm mapping is trustworthy."""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    corners = px_to_mm(homography, [(0, 0), (w, 0), (w, h), (0, h)])
    (xmin, ymin), (xmax, ymax) = corners.min(0), corners.max(0)
    line = lambda a, b, col, t=1: cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), col, t, cv2.LINE_AA)
    # grid lines every grid_mm, tick labels in CM at BOTH ends of every line (dense on purpose:
    # the VLM reads object coordinates off these labels), placed at the line's own endpoints so they
    # stay next to their line in any camera orientation
    def tick(a, b, lbl):
        for ex, ey in (a, b):
            _put(img, lbl, (int(np.clip(ex + 3, 2, w - 46)), int(np.clip(ey - 4, 60, h - 6))), _BGR["tick"])

    for x in np.arange(np.floor(xmin / grid_mm) * grid_mm, xmax + 1, grid_mm):
        a, b = mm_to_px(homography, [(x, ymin), (x, ymax)])
        line(a, b, _BGR["grid"])
        tick(a, b, f"x{x / 10.0:g}")  # mm -> cm label (unit boundary)
    for y in np.arange(np.floor(ymin / grid_mm) * grid_mm, ymax + 1, grid_mm):
        a, b = mm_to_px(homography, [(xmin, y), (xmax, y)])
        line(a, b, _BGR["grid"])
        tick(a, b, f"y{y / 10.0:g}")  # mm -> cm label (unit boundary)
    # axis-direction arrows + origin marker (origin = robot base, usually off-frame below the workspace)
    cx_mm, cy_mm = (xmin + xmax) / 2, (ymin + ymax) / 2
    o = mm_to_px(homography, [(cx_mm, cy_mm)])[0]
    for dx, dy, name in ((grid_mm, 0.0, "+x fwd"), (0.0, grid_mm, "+y left")):
        tip = mm_to_px(homography, [(cx_mm + dx, cy_mm + dy)])[0]
        cv2.arrowedLine(img, tuple(map(int, o)), tuple(map(int, tip)), _BGR["axis"], 2, cv2.LINE_AA, tipLength=0.25)
        _put(img, name, (int(tip[0]) + 4, int(tip[1]) - 4), _BGR["axis"])
    og = mm_to_px(homography, [(0.0, 0.0)])[0]
    if 0 <= og[0] < w and 0 <= og[1] < h:
        cv2.drawMarker(img, tuple(map(int, og)), _BGR["axis"], cv2.MARKER_TILTED_CROSS, 16, 2)
        _put(img, "base (0,0)", (int(og[0]) + 8, int(og[1]) - 6), _BGR["axis"])
    if calib_samples_px is not None and len(calib_samples_px):
        for i, (su, sv) in enumerate(np.asarray(calib_samples_px, float).reshape(-1, 2), 1):
            cv2.drawMarker(img, (int(round(su)), int(round(sv))), _BGR["hull"], cv2.MARKER_DIAMOND, 9, 1)
            _put(img, f"c{i}", (int(round(su)) + 6, int(round(sv)) + 12), _BGR["hull"], 0.34)
    if calib_region_px is not None and len(calib_region_px) >= 3:
        pts = np.asarray(calib_region_px, np.float64).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, _BGR["hull"], 1, cv2.LINE_AA)
        _put(img, "calibrated area", (int(pts[:, 0, 0].min()) + 4, int(pts[:, 0, 1].min()) + 14), _BGR["hull"], 0.38)
    for z in zones:
        pts = mm_to_px(homography, z.polygon_mm).round().astype(np.int32)
        cv2.polylines(img, [pts], True, _BGR["zone"], 2, cv2.LINE_AA)
        d = mm_to_px(homography, [z.drop_point_mm])[0].astype(int)
        cv2.drawMarker(img, tuple(d), _BGR["zone"], cv2.MARKER_TILTED_CROSS, 12, 1)
        _put(img, z.name, (pts[:, 0].min() + 4, pts[:, 1].min() + 16), _BGR["zone"], 0.5)
    if ee_pose is not None:
        ex, ey = mm_to_px(homography, [(ee_pose.x, ee_pose.y)])[0].astype(int)
        cv2.drawMarker(img, (ex, ey), _BGR["ee"], cv2.MARKER_CROSS, 24, 2)
        _put(img, f"EE z={ee_pose.z / 10.0:.1f}cm", (ex + 12, ey - 8), _BGR["ee"], 0.45)  # mm -> cm label
    legend = [f"grid {grid_mm / 10.0:g} cm | coordinates in cm: x fwd, y left, origin at robot base",
              "yellow = zone (X = drop point)  magenta = EE"] + [f"rule: {r}" for r in rules]
    for i, t in enumerate(legend):
        _put(img, t, (6, 16 + 15 * i), _BGR["text"])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- selftest


def synth_homography(config: Config, w: int, h: int, margin: int = 60) -> np.ndarray:
    """Overhead cam looking down, mat filling the frame minus a margin; x fwd = up in image, y left = left."""
    mat = mat_polygon_mm(config)
    dst = np.float32(mat)
    src = np.float32([(margin, margin), (w - margin, margin), (w - margin, h - margin), (margin, h - margin)])
    return cv2.getPerspectiveTransform(src, dst)


def _selftest(out_png: Path) -> None:
    from sortbot import config as cfgmod

    cfg = cfgmod.load()
    w, h = cfg.overhead_cam.width, cfg.overhead_cam.height
    H = synth_homography(cfg, w, h)

    # px <-> mm round trip
    pts_mm = [(200.0, -120.0), (300.0, 0.0), (380.0, 150.0)]
    back = px_to_mm(H, mm_to_px(H, pts_mm))
    assert np.allclose(back, pts_mm, atol=0.5), back

    rng = np.random.default_rng(0)
    img = rng.integers(20, 40, (h, w, 3), dtype=np.uint8)  # dark matte mat with noise
    for (x, y), col in {(380.0, 150.0): (220, 40, 40), (300.0, 0.0): (40, 200, 60),
                        (200.0, -120.0): (240, 220, 50)}.items():
        cx, cy = mm_to_px(H, [(x, y)])[0].astype(int)
        cv2.rectangle(img, (cx - 22, cy - 15), (cx + 22, cy + 15), col, -1)

    hull = mm_to_px(H, [(160, -180), (400, -180), (400, 180), (160, 180)])
    ov = render_overlay(img, H, cfg.zones, Pose(*cfg.home.__dict__.values()),
                        ["put white things in LEFT"], calib_region_px=hull)
    assert ov.shape == img.shape and not np.array_equal(ov, img)
    # rendering must not depend on optional inputs
    ov2 = render_overlay(img, H, [], None, [])
    assert ov2.shape == img.shape
    # the grid tick labels really are painted (bright cyan-ish pixels along the bottom label row)
    strip = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)[h - 20:h, :, :]
    assert (strip[..., 1].astype(int) > 180).any(), "no tick labels rendered along the bottom edge"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    print(f"overlay (cm grid + zones + EE + calibrated-area outline) -> {out_png}\nselftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("out") / "selftest_overlay.png"))
    a = ap.parse_args()
    if a.selftest:
        _selftest(Path(a.out))
    else:
        ap.print_help()
