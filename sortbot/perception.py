"""Overhead perception: classical blob detector on a dark mat + overlay renderer.

homography H (3x3) maps overhead pixel (u, v, 1) -> table-frame (x_mm, y_mm, 1). Produced by
sortbot.calib from the ArUco tags in config; here it is just an input.
Deviation note: `filled_zones` (names of zones already containing sorted objects) is an explicit
parameter so the caller decides what "already filled" means.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from sortbot.config import Config
from sortbot.types import DetectedObject, Pose, Zone

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


def roi_polygons_mm(config: Config, method: str = "aruco") -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    """(search polygon, polygons to blank out). aruco: the mat between the tags minus the tags themselves;
    ball/teleop (no tags): the workspace AABB xy from config, nothing masked."""
    if method == "aruco":
        r = config.aruco_tag_size_mm * 0.75  # the ROI edge runs through the tag centres: blank out each tag
        tags = [[(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)] for x, y in config.aruco_tags_mm.values()]
        return mat_polygon_mm(config), tags
    (x0, y0, _), (x1, y1, _) = config.aabb_min_mm, config.aabb_max_mm
    return [(x1, y1), (x1, y0), (x0, y0), (x0, y1)], []


# ---------------------------------------------------------------- detectors


@dataclass
class RawBlob:
    bbox_px: tuple[int, int, int, int]
    area_px: float
    centroid_px: tuple[float, float]
    color_hint: str


class Detector(Protocol):
    def detect(self, rgb: np.ndarray, roi_mask: np.ndarray) -> list[RawBlob]: ...


def _color_hint(rgb: np.ndarray, mask: np.ndarray) -> str:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = (float(np.median(hsv[..., i][mask > 0])) for i in range(3))
    if v < 60:
        return "black"
    if s < 50:
        return "white" if v > 170 else "gray"
    for name, hi in (("red", 10), ("orange", 22), ("yellow", 35), ("green", 85), ("cyan", 100), ("blue", 130), ("purple", 155), ("pink", 170)):
        if h < hi:
            return name
    return "red"


class ClassicalDetector:
    """Anything brighter / more saturated than the dark matte mat is an object."""

    def __init__(self, min_area_px: int = 300, max_area_px: int = 60_000, v_thresh: int = 70, s_thresh: int = 90, kernel: int = 5):
        self.min_area, self.max_area = min_area_px, max_area_px
        self.v_thresh, self.s_thresh, self.kernel = v_thresh, s_thresh, kernel

    def detect(self, rgb: np.ndarray, roi_mask: np.ndarray) -> list[RawBlob]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        fg = ((hsv[..., 2] > self.v_thresh) | (hsv[..., 1] > self.s_thresh)).astype(np.uint8) * 255
        fg &= roi_mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel, self.kernel))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area):
                continue
            x, y, w, h = cv2.boundingRect(c)
            m = cv2.moments(c)
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            cm = np.zeros(rgb.shape[:2], np.uint8)
            cv2.drawContours(cm, [c], -1, 255, -1)
            out.append(RawBlob((x, y, x + w, y + h), float(area), (cx, cy), _color_hint(rgb, cm)))
        return out


class OpenVocabDetector:
    """Stub for OWL-ViT / GroundingDINO. Same interface; loads model lazily on first use."""

    def __init__(self, queries: list[str], model_id: str = "google/owlvit-base-patch32"):
        self.queries, self.model_id = queries, model_id

    def detect(self, rgb: np.ndarray, roi_mask: np.ndarray) -> list[RawBlob]:
        raise NotImplementedError("OpenVocabDetector: model loading not wired up yet")


# ---------------------------------------------------------------- public API


def _poly_mask(shape, H, polys_mm) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for poly in polys_mm:
        cv2.fillPoly(m, [mm_to_px(H, poly).round().astype(np.int32)], 255)
    return m


def detect_objects(
    overhead_rgb: np.ndarray,
    homography: np.ndarray,
    config: Config,
    detector: Detector | None = None,
    filled_zones: list[str] = (),
    method: str = "aruco",
) -> list[DetectedObject]:
    """method: which homography source produced H ('aruco' | 'ball' | 'teleop') -> ROI choice, see roi_polygons_mm."""
    detector = detector or ClassicalDetector()
    search, blank = roi_polygons_mm(config, method)
    roi = _poly_mask(overhead_rgb.shape, homography, [search])
    if blank:
        roi &= cv2.bitwise_not(_poly_mask(overhead_rgb.shape, homography, blank))
    skip = [z.polygon_mm for z in config.zones if z.name.upper() in {n.upper() for n in filled_zones}]
    if skip:
        roi &= cv2.bitwise_not(_poly_mask(overhead_rgb.shape, homography, skip))
    blobs = detector.detect(overhead_rgb, roi)
    blobs.sort(key=lambda b: (round(b.centroid_px[0] / 40), b.centroid_px[1]))  # left-to-right, stable
    objs = []
    for i, b in enumerate(blobs, 1):
        x, y = px_to_mm(homography, [b.centroid_px])[0]
        objs.append(DetectedObject(i, (float(x), float(y)), b.bbox_px, b.area_px, b.color_hint))
    return objs


_BGR = dict(grid=(90, 90, 90), zone=(0, 200, 255), obj=(0, 255, 0), ee=(255, 80, 255), text=(255, 255, 255))


def render_overlay(
    frame: np.ndarray,
    homography: np.ndarray,
    objects: list[DetectedObject],
    zones: list[Zone],
    ee_pose: Pose | None = None,
    rules: list[str] = (),
    grid_mm: float = 50.0,
) -> np.ndarray:
    """frame is RGB; returns an RGB copy with annotations."""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    corners = px_to_mm(homography, [(0, 0), (w, 0), (w, h), (0, h)])
    (xmin, ymin), (xmax, ymax) = corners.min(0), corners.max(0)
    line = lambda a, b, col, t=1: cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), col, t, cv2.LINE_AA)
    for x in np.arange(np.floor(xmin / grid_mm) * grid_mm, xmax + 1, grid_mm):
        a, b = mm_to_px(homography, [(x, ymin), (x, ymax)])
        line(a, b, _BGR["grid"])
        cv2.putText(img, f"x{x:.0f}", (int(a[0]) + 2, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _BGR["grid"], 1)
    for y in np.arange(np.floor(ymin / grid_mm) * grid_mm, ymax + 1, grid_mm):
        a, b = mm_to_px(homography, [(xmin, y), (xmax, y)])
        line(a, b, _BGR["grid"])
        cv2.putText(img, f"y{y:.0f}", (4, int(a[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _BGR["grid"], 1)
    for z in zones:
        pts = mm_to_px(homography, z.polygon_mm).round().astype(np.int32)
        cv2.polylines(img, [pts], True, _BGR["zone"], 2, cv2.LINE_AA)
        d = mm_to_px(homography, [z.drop_point_mm])[0].astype(int)
        cv2.drawMarker(img, tuple(d), _BGR["zone"], cv2.MARKER_TILTED_CROSS, 12, 1)
        cv2.putText(img, z.name, (pts[:, 0].min() + 4, pts[:, 1].min() + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BGR["zone"], 1, cv2.LINE_AA)
    for o in objects:
        x0, y0, x1, y1 = o.bbox_px
        cv2.rectangle(img, (x0, y0), (x1, y1), _BGR["obj"], 2)
        cx, cy = mm_to_px(homography, [o.centroid_mm])[0].astype(int)
        cv2.circle(img, (cx, cy), 3, _BGR["obj"], -1)
        cv2.putText(img, f"#{o.id}", (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _BGR["obj"], 2, cv2.LINE_AA)
        cv2.putText(img, f"({o.centroid_mm[0]:.0f},{o.centroid_mm[1]:.0f}) {o.color_hint}", (x0, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _BGR["obj"], 1, cv2.LINE_AA)
    if ee_pose is not None:
        ex, ey = mm_to_px(homography, [(ee_pose.x, ee_pose.y)])[0].astype(int)
        cv2.drawMarker(img, (ex, ey), _BGR["ee"], cv2.MARKER_CROSS, 24, 2)
        cv2.putText(img, f"EE z={ee_pose.z:.0f}", (ex + 12, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _BGR["ee"], 1, cv2.LINE_AA)
    legend = [f"grid {grid_mm:.0f}mm | table frame mm: x fwd, y left", "green #n = object  yellow = zone  magenta = EE"] + [f"rule: {r}" for r in rules]
    for i, t in enumerate(legend):
        cv2.putText(img, t, (6, 16 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _BGR["text"], 1, cv2.LINE_AA)
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
    rng = np.random.default_rng(0)
    img = rng.integers(20, 40, (h, w, 3), dtype=np.uint8)  # dark matte mat with noise
    truth = {(380.0, 150.0): (220, 40, 40), (300.0, 0.0): (40, 200, 60), (200.0, -120.0): (240, 220, 50), (250.0, 100.0): (255, 255, 255)}
    for (x, y), col in truth.items():
        cx, cy = mm_to_px(H, [(x, y)])[0].astype(int)
        cv2.rectangle(img, (cx - 22, cy - 15), (cx + 22, cy + 15), col, -1)
    # distractors: outside the mat, a tiny speck, and white ArUco-sized squares centred on each tag position
    cv2.rectangle(img, (5, 5, ), (40, 40), (255, 255, 255), -1)
    for tx, ty in cfg.aruco_tags_mm.values():
        s = cfg.aruco_tag_size_mm / 2
        cv2.fillPoly(img, [mm_to_px(H, [(tx - s, ty - s), (tx + s, ty - s), (tx + s, ty + s), (tx - s, ty + s)]).round().astype(np.int32)], (255, 255, 255))
    cv2.circle(img, (w // 2, h // 2 + 100), 3, (255, 255, 255), -1)

    objs = detect_objects(img, H, cfg)
    assert len(objs) == len(truth), [o.centroid_mm for o in objs]
    for o in objs:
        err = min(np.hypot(o.centroid_mm[0] - tx, o.centroid_mm[1] - ty) for tx, ty in truth)
        assert err < 3.0, (o, err)
    assert [o.id for o in objs] == list(range(1, len(objs) + 1))
    assert {o.color_hint for o in objs} == {"red", "green", "yellow", "white"}, [o.color_hint for o in objs]

    filled = detect_objects(img, H, cfg, filled_zones=["SENSORS"])
    assert len(filled) == 3, filled  # the green one at (300,0) is inside SENSORS
    ball = detect_objects(img, H, cfg, method="ball")  # no tag masking: the 4 white tag squares become objects
    assert len(ball) == len(truth) + 4, [o.centroid_mm for o in ball]

    ov = render_overlay(img, H, objs, cfg.zones, Pose(*cfg.home.__dict__.values()), ["put white things in WIRES"])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    print(f"detected {len(objs)}: " + ", ".join(f"#{o.id} {o.color_hint} @({o.centroid_mm[0]:.1f},{o.centroid_mm[1]:.1f})" for o in objs))
    print(f"overlay -> {out_png}\nselftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("out") / "selftest_overlay.png"))
    a = ap.parse_args()
    if a.selftest:
        _selftest(Path(a.out))
    else:
        ap.print_help()
