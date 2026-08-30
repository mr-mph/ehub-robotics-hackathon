"""Overhead overlay rendering: the VLM does the seeing, this module only draws what the VLM (and the
human) need to read positions off the image — a labelled grid and the EE marker —
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
import threading
from pathlib import Path

import cv2
import numpy as np

from sortbot.calibration import expand_hull
from sortbot.config import Config
from sortbot.types import Pose

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

_BGR = dict(grid=(120, 120, 120), tick=(60, 235, 255), ee=(255, 80, 255),
            text=(255, 255, 255), axis=(80, 220, 120), hull=(200, 160, 60))


def _put(img, text, org, col, scale=0.42, thick=1):
    # dark halo first so labels stay readable on any background (the VLM must be able to read them)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


class Overlay:
    """Renders the STATIC annotation layer once and reuses it.

    Everything on the overlay except the end-effector marker is a function of the calibration and the
    config, not of the picture: the grid, its cm tick labels, the axis arrows, the calibration anchors and
    the legend. Re-drawing them per frame made the overlay shimmer and shift while nothing had actually
    changed. The layer is cached against a key built from exactly those inputs, so it is recomputed only
    when the homography, the frame size, the grid spacing or the calibration anchors change -- i.e. only
    when the configuration does, never mid-run for no reason.

    Only SPATIAL things are drawn: the grid and its cm tick labels, the axis arrows, the base marker, the
    calibration anchors and the sampled-area outline. Text that merely *describes* the scene (the legend,
    and the RULES in force) is fed to the model as TEXT instead -- see sortbot.vlm._state_text.
    """

    def __init__(self, grid_mm: float = 50.0):
        self.grid_mm = grid_mm
        self._key = self._layer = self._mask = None
        self._lock = threading.Lock()  # the loop thread and the preview thread share one instance

    @staticmethod
    def _bytes(a) -> bytes | None:
        return None if a is None or not len(a) else np.asarray(a, float).round(2).tobytes()

    def _key_for(self, hw, H, region, samples) -> tuple:
        return (hw, np.asarray(H, float).round(9).tobytes(), self.grid_mm,
                self._bytes(region), self._bytes(samples))

    def _static(self, hw, H, region, samples):
        """(layer_bgr, mask) of the static annotations -- built only when the key changes."""
        key = self._key_for(hw, H, region, samples)
        if key == self._key:
            return self._layer, self._mask
        h, w = hw
        img = np.zeros((h, w, 3), np.uint8)
        grid = np.zeros((h, w, 3), np.uint8)  # grid lines go here first, so they can be dimmed off-hull
        corners = px_to_mm(H, [(0, 0), (w, 0), (w, h), (0, h)])
        (xmin, ymin), (xmax, ymax) = corners.min(0), corners.max(0)
        line = lambda a, b, col, t=1: cv2.line(grid, tuple(map(int, a)), tuple(map(int, b)), col, t, cv2.LINE_AA)
        g = self.grid_mm

        # grid lines every grid_mm, tick labels in CM at BOTH ends of every line (dense on purpose: the
        # VLM reads object coordinates off these labels), placed at each line's own endpoints
        def tick(a, b, lbl):
            for ex, ey in (a, b):
                _put(img, lbl, (int(np.clip(ex + 3, 2, w - 46)), int(np.clip(ey - 4, 60, h - 6))), _BGR["tick"])

        for x in np.arange(np.floor(xmin / g) * g, xmax + 1, g):
            a, b = mm_to_px(H, [(x, ymin), (x, ymax)])
            line(a, b, _BGR["grid"])
            tick(a, b, f"x{x / 10.0:g}")  # mm -> cm label (unit boundary)
        for y in np.arange(np.floor(ymin / g) * g, ymax + 1, g):
            a, b = mm_to_px(H, [(xmin, y), (xmax, y)])
            line(a, b, _BGR["grid"])
            tick(a, b, f"y{y / 10.0:g}")  # mm -> cm label (unit boundary)
        # Outside the calibrated hull the mapping is EXTRAPOLATED and not to be trusted -- dim the grid
        # there so "the grid looks off" is immediately attributable to being outside the sampled area
        # rather than to a broken fit.
        if region is not None and len(region) >= 3:
            inside = np.zeros((h, w), np.uint8)
            cv2.fillPoly(inside, [expand_hull(np.asarray(region, float)).round().astype(np.int32).reshape(-1, 1, 2)], 255)
            grid[inside == 0] = (grid[inside == 0] * 0.35).astype(np.uint8)
        img |= grid
        # axis-direction arrows + origin marker (origin = robot base, usually off-frame)
        cx_mm, cy_mm = (xmin + xmax) / 2, (ymin + ymax) / 2
        o = mm_to_px(H, [(cx_mm, cy_mm)])[0]
        for dx, dy, name in ((g, 0.0, "+x fwd"), (0.0, g, "+y left")):
            tip = mm_to_px(H, [(cx_mm + dx, cy_mm + dy)])[0]
            cv2.arrowedLine(img, tuple(map(int, o)), tuple(map(int, tip)), _BGR["axis"], 2, cv2.LINE_AA, tipLength=0.25)
            _put(img, name, (int(tip[0]) + 4, int(tip[1]) - 4), _BGR["axis"])
        og = mm_to_px(H, [(0.0, 0.0)])[0]
        if 0 <= og[0] < w and 0 <= og[1] < h:
            cv2.drawMarker(img, tuple(map(int, og)), _BGR["axis"], cv2.MARKER_TILTED_CROSS, 16, 2)
            _put(img, "base (0,0)", (int(og[0]) + 8, int(og[1]) - 6), _BGR["axis"])
        if samples is not None and len(samples):
            for i, (su, sv) in enumerate(np.asarray(samples, float).reshape(-1, 2), 1):
                cv2.drawMarker(img, (int(round(su)), int(round(sv))), _BGR["hull"], cv2.MARKER_DIAMOND, 9, 1)
                _put(img, f"c{i}", (int(round(su)) + 6, int(round(sv)) + 12), _BGR["hull"], 0.34)
        if region is not None and len(region) >= 3:
            pts = np.asarray(region, np.float64).round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, _BGR["hull"], 1, cv2.LINE_AA)
            _put(img, "calibrated area", (int(pts[:, 0, 0].min()) + 4, int(pts[:, 0, 1].min()) + 14), _BGR["hull"], 0.38)
        # Nothing else is drawn. Everything that is not a SPATIAL annotation -- what the grid spacing is,
        # what the colours mean, the sorting instructions in force -- is prose, and prose belongs in the
        # TEXT half of the prompt (sortbot.vlm._state_text + SYSTEM_PROMPT), not painted over the
        # photograph. It used to burn a block of writing into the top-left corner of every single frame,
        # which hid whatever was underneath from both the model and the operator, and left the model
        # reading its instructions off a picture of them.
        self._layer, self._mask, self._key = img, img.any(axis=2), key
        return self._layer, self._mask

    def render(self, frame: np.ndarray, homography: np.ndarray, ee_pose: Pose | None = None,
               rules: list[str] = (), calib_region_px=None, calib_samples_px=None) -> np.ndarray:
        """frame is RGB; returns an RGB copy with the cached static layer composited plus the live EE marker.

        `rules` is accepted and IGNORED (callers still pass it): rules reach the model as text, never as
        pixels. It is not part of the cache key either, so editing a rule no longer rebuilds the layer."""
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with self._lock:
            layer, mask = self._static(img.shape[:2], homography, calib_region_px, calib_samples_px)
            layer, mask = layer.copy(), mask.copy()
        img[mask] = layer[mask]
        if ee_pose is not None:  # the ONLY per-frame element: a mark, not a caption
            ex, ey = mm_to_px(homography, [(ee_pose.x, ee_pose.y)])[0].astype(int)
            cv2.drawMarker(img, (ex, ey), _BGR["ee"], cv2.MARKER_CROSS, 24, 2)
            # No caption. The gripper's x, y and z are NUMBERS, and numbers go to the model as text
            # (vlm._state_text's "EE pose (table cm)" line) -- printing the height beside the cross only
            # covered up the table next to the gripper, which is exactly where the object being picked is.
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


_OVERLAY = Overlay()  # shared: the loop and the preview thread must draw the identical static layer


def render_overlay(frame: np.ndarray, homography: np.ndarray, ee_pose: Pose | None = None,
                   rules: list[str] = (), grid_mm: float = 50.0,
                   calib_region_px=None, calib_samples_px=None) -> np.ndarray:
    """Module-level convenience wrapper around the shared cached Overlay (see Overlay for what is drawn)."""
    if grid_mm != _OVERLAY.grid_mm:
        _OVERLAY.grid_mm = grid_mm
    return _OVERLAY.render(frame, homography, ee_pose, rules, calib_region_px, calib_samples_px)


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
    ov = Overlay()
    out = ov.render(img, H, Pose(*cfg.home.__dict__.values()), ["put white things in LEFT"],
                    calib_region_px=hull, calib_samples_px=hull)
    assert out.shape == img.shape and not np.array_equal(out, img)

    # the static layer is CACHED: same inputs must not rebuild it, and the same frame must render
    # byte-identical (the overlay may not shimmer or drift while nothing has changed)
    key0 = ov._key
    again = ov.render(img, H, Pose(*cfg.home.__dict__.values()), ["put white things in LEFT"],
                      calib_region_px=hull, calib_samples_px=hull)
    assert ov._key is key0 or ov._key == key0, "static layer rebuilt for identical inputs"
    assert np.array_equal(out, again), "overlay changed between identical renders"
    # RULES ARE TEXT, NOT PIXELS: no legend/rule block is painted on the frame, so changing (or dropping)
    # the rules cannot change a single pixel and cannot rebuild the cached layer
    no_rules = ov.render(img, H, Pose(*cfg.home.__dict__.values()), [],
                         calib_region_px=hull, calib_samples_px=hull)
    assert np.array_equal(out, no_rules), "rules text is still being drawn on the overlay"
    other = ov.render(img, H, Pose(*cfg.home.__dict__.values()), ["a completely different rule"],
                      calib_region_px=hull, calib_samples_px=hull)
    assert np.array_equal(out, other) and ov._key == key0, "the rules still affect the overlay layer"
    # ... and it DOES rebuild when the configuration changes
    ov.grid_mm = 100.0
    ov.render(img, H, None, [], calib_region_px=hull, calib_samples_px=hull)
    assert ov._key != key0, "static layer not rebuilt after the grid spacing changed"
    ov.grid_mm = 50.0

    # only the EE marker may differ between two frames that share a configuration
    a = ov.render(img, H, Pose(200.0, 50.0, 100.0, 0.0), [])
    b = ov.render(img, H, Pose(260.0, -50.0, 100.0, 0.0), [])
    assert not np.array_equal(a, b), "EE marker did not move"

    # no zone graphics anywhere: rendering must not depend on config zones at all
    import inspect
    assert "zone" not in inspect.signature(render_overlay).parameters, "zones are back on the overlay"

    out2 = render_overlay(img, H)  # optional args only
    assert out2.shape == img.shape
    strip = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)[h - 20:h, :, :]
    assert (strip[..., 1].astype(int) > 180).any(), "no tick labels rendered along the bottom edge"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"overlay (cached cm grid + anchors + EE, no zones) -> {out_png}\nselftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("out") / "selftest_overlay.png"))
    a = ap.parse_args()
    if a.selftest:
        _selftest(Path(a.out))
    else:
        ap.print_help()
