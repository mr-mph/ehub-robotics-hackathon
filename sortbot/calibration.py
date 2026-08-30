"""Calibration: image<->table-mm homography (ball-fitted or ArUco), colour-target detector, table_T_base.

Ball mode (default, no tags): the gripper holds a coloured target; at each grid point the overhead pixel centroid
and the FK base-frame xy are paired and a single homography H_px_to_mm is fitted (RANSAC). The table frame then IS
the base frame in xy (x fwd, y left, mm); table_T_base is identity apart from the z offset from the fingertip touch
step. H is exact for centroids at `plane_z_mm` (the target-centre height during calibration); taller objects
project slightly outward from the camera nadir, lower ones inward.

table_T_base is a 4x4 that maps base-frame FK positions (meters) to table-frame mm:
    p_table_mm = R @ (p_base_m * 1000) + t
Camera height / xy for ball parallax are not in config.yaml (scaffold contract), so they are CLI args in
sortbot/calibrate.py and stored in calib.json.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sortbot import config as cfgmod


# ---------------------------------------------------------------- sample-coverage helpers
# A homography fitted from samples that cover only a small patch of the camera view extrapolates
# wildly everywhere else ("the projected plane looks way off"). These helpers quantify coverage,
# feed the live calibration UI, gate Finish, and guard runtime conversions outside the sampled area.


def sample_hull_px(px) -> np.ndarray | None:
    """Convex hull (Nx2 float, overhead px) of the sample pixels; None with < 3 points."""
    px = np.asarray(px, np.float32).reshape(-1, 2)
    if len(px) < 3:
        return None
    return cv2.convexHull(px).reshape(-1, 2).astype(float)


def coverage_pct(px, frame_wh) -> float:
    """Convex-hull area of the sample pixels as a percentage of the frame area."""
    hull = sample_hull_px(px)
    if hull is None or not frame_wh:
        return 0.0
    return 100.0 * cv2.contourArea(hull.astype(np.float32)) / float(frame_wh[0] * frame_wh[1])


def coverage_verdict(pct: float) -> str:
    if pct < 15.0:
        return "samples are clustered — spread them to the corners of the camera view"
    return "ok — wider is better" if pct <= 40.0 else "good coverage"


def collinearity_ratio(px) -> float:
    """2nd/1st singular value of the centred sample pixels: ~0 = the samples lie on one line
    (the fit LOOKS fine in residual but is garbage off that line)."""
    px = np.asarray(px, float).reshape(-1, 2)
    if len(px) < 3:
        return 0.0
    s = np.linalg.svd(px - px.mean(0), compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 1e-9 else 0.0


def expand_hull(hull: np.ndarray, margin: float = 0.2) -> np.ndarray:
    """Hull grown about its centroid by `margin` (20% default) — the runtime trust region."""
    c = hull.mean(0)
    return c + (np.asarray(hull, float) - c) * (1.0 + margin)


def in_calibrated_region(hull_px: np.ndarray | None, u: float, v: float, margin: float = 0.2) -> bool:
    """Is overhead pixel (u, v) inside the calibrated sample hull grown by `margin`? True when no
    hull is known (old calib.json / mock H): the guard only fires with real coverage data."""
    if hull_px is None:
        return True
    big = expand_hull(hull_px, margin).astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(big, (float(u), float(v)), False) >= 0


def calib_summary(d: dict) -> str | None:
    """One human line about the saved calibration, or None when there is no fitted H."""
    if d.get("H_px_to_mm") is None:
        return None
    pts = (d.get("points") or {}).get("px") or []
    res = np.asarray(d.get("residuals_mm") or [], float)
    parts = [f"calibration loaded: {len(pts)} points"]
    if res.size:
        used = int((res <= 5.0).sum())
        parts.append(f"residual mean {res.mean():.1f} / max {res.max():.1f} mm")
        if used < 8:
            # 8 DOF = 4 pairs: at 4-7 usable points the fit passes through them almost exactly whatever
            # the data says, so a tiny residual is arithmetic, not accuracy. Say so instead of implying 
            parts.append(f"only ~{used} usable for an 8-DOF fit (residuals prove little -- capture 8+)")
    if d.get("frame_wh"):
        parts.append(f"coverage {coverage_pct(pts, d['frame_wh']):.0f}%")
    if d.get("saved_at"):
        age = max(0.0, time.time() - float(d["saved_at"]))
        ago = f"{age / 60:.0f} min" if age < 3600 else (f"{age / 3600:.0f} h" if age < 86400 else f"{age / 86400:.0f} d")
        parts.append(f"saved {ago} ago")
    return ", ".join(parts)


def calib_summary_file(path: Path | None = None) -> str | None:
    """calib_summary straight from calib.json; None when absent/unfitted."""
    try:
        return calib_summary(load_calib_dict(path))
    except Exception:  # noqa: BLE001
        return None

# ---------------------------------------------------------------- ArUco homography


class TableHomography:
    """Image px <-> table mm. mode: 'ball' = fixed H from calib.json; 'aruco' = 4 corner tags (last good H is
    kept if tags drop out); 'auto' = fixed H if present, tags override for frames where all 4 are seen."""

    def __init__(self, cfg: cfgmod.Config | None = None, mode: str | None = None):
        cfg = cfg or cfgmod.load()
        self.cfg = cfg
        self.mode = mode or cfg.calib_mode
        assert self.mode in ("ball", "aruco", "auto"), self.mode
        self.tags_mm = cfg.aruco_tags_mm
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco_dict))
        self._det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        self.H: np.ndarray | None = None  # px -> mm
        self.H_inv: np.ndarray | None = None
        self.H_fixed: np.ndarray | None = None
        self.region_px: np.ndarray | None = None  # sample hull: where the fitted H is trustworthy
        self.samples_px: np.ndarray | None = None  # the fit's own anchor points (drawn on the overlay)
        self.summary: str | None = None  # human line about the loaded calibration (or None)
        self.last_centers_px: dict[int, tuple[float, float]] = {}
        self.method = "aruco"  # which source produced self.H
        if self.mode != "aruco":
            self.reload()

    def reload(self, path: Path | None = None) -> bool:
        """(Re)load the fitted H from calib.json (ball mode). Returns True if one was found. Also loads
        the sample hull (`region_px`, the area where the fit is trustworthy) and logs a summary so it is
        obvious the saved calibration persisted and was picked up."""
        d = load_calib_dict(path or self.cfg.calib_file)
        self.H_fixed = d.get("H_px_to_mm")
        pts_px = (d.get("points") or {}).get("px") or []
        self.region_px = sample_hull_px(pts_px)
        self.samples_px = np.asarray(pts_px, float).reshape(-1, 2) if pts_px else None
        self.summary = calib_summary(d)
        if self.summary:
            print(f"[calibration] {self.summary}")
        if self.H_fixed is not None and self.mode != "aruco":
            self._set(self.H_fixed, "ball")
        return self.H_fixed is not None

    def _set(self, H: np.ndarray, method: str) -> None:
        self.H, self.H_inv, self.method = np.asarray(H, float), np.linalg.inv(H), method

    def detect(self, frame_rgb: np.ndarray) -> dict[int, tuple[float, float]]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY) if frame_rgb.ndim == 3 else frame_rgb
        corners, ids, _ = self._det.detectMarkers(gray)
        out = {}
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                if int(i) in self.tags_mm:
                    out[int(i)] = tuple(c[0].mean(axis=0).tolist())
        return out

    def update(self, frame_rgb: np.ndarray) -> bool:
        """Returns True if H is usable. ball: fixed H. aruco: re-solve when all 4 tags are visible (else cached).
        auto: tags when all 4 are seen, otherwise the fixed H."""
        if self.mode == "ball":
            return self.H is not None
        centers = self.detect(frame_rgb)
        self.last_centers_px = centers
        if len(centers) >= 4:
            ids = sorted(centers)
            H, _ = cv2.findHomography(np.float32([centers[i] for i in ids]), np.float32([self.tags_mm[i] for i in ids]))
            if H is not None:
                self._set(H, "aruco")
        elif self.mode == "auto" and self.H_fixed is not None:
            self._set(self.H_fixed, "ball")
        return self.H is not None

    @property
    def ready(self) -> bool:
        return self.H is not None

    def px_to_mm(self, u: float, v: float) -> tuple[float, float]:
        return _apply(self.H, u, v)

    def mm_to_px(self, x: float, y: float) -> tuple[float, float]:
        return _apply(self.H_inv, x, y)

    def draw_grid(self, frame_rgb: np.ndarray, spacing_mm: float = 50.0) -> np.ndarray:
        """Overlay the table grid (mm) with axis labels on a copy of the frame."""
        img = frame_rgb.copy()
        if not self.ready:
            cv2.putText(img, "NO HOMOGRAPHY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            return img
        pts = self.tags_mm.values() if self.method == "aruco" else [self.cfg.aabb_min_mm, self.cfg.aabb_max_mm]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        col = (0, 200, 255)
        for x in np.arange(np.ceil(x0 / spacing_mm) * spacing_mm, x1 + 1e-6, spacing_mm):
            a, b = self.mm_to_px(x, y0), self.mm_to_px(x, y1)
            cv2.line(img, _ip(a), _ip(b), col, 1)
            cv2.putText(img, f"x{int(x)}", _ip(b), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        for y in np.arange(np.ceil(y0 / spacing_mm) * spacing_mm, y1 + 1e-6, spacing_mm):
            a, b = self.mm_to_px(x0, y), self.mm_to_px(x1, y)
            cv2.line(img, _ip(a), _ip(b), col, 1)
            cv2.putText(img, f"y{int(y)}", _ip(a), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        for i, c in self.last_centers_px.items():
            cv2.circle(img, _ip(c), 6, (0, 255, 0), 2)
            cv2.putText(img, f"id{i}", (_ip(c)[0] + 8, _ip(c)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img


def _apply(H: np.ndarray | None, a: float, b: float) -> tuple[float, float]:
    if H is None:
        raise RuntimeError("homography not solved yet")
    p = H @ np.array([a, b, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def _ip(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def render_mat(tags_mm: dict[int, tuple[float, float]], aruco_dict: str, size=(1280, 720), px_per_mm=2.0,
               origin_px=(640, 360), center_mm=(270.0, 0.0), tag_px=80) -> tuple[np.ndarray, callable]:
    """Synthetic overhead view of the mat (white bg, 4 markers). Returns (rgb, mm->px mapping)."""
    # Camera looks down: image u grows with -y (left is left), image v grows with -x (forward is up).
    def mm_to_px(x, y):
        return origin_px[0] - (y - center_mm[1]) * px_per_mm, origin_px[1] - (x - center_mm[0]) * px_per_mm

    img = np.full((size[1], size[0]), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict))
    for i, (x, y) in tags_mm.items():
        m = cv2.aruco.generateImageMarker(d, i, tag_px)
        m = cv2.copyMakeBorder(m, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        u, v = _ip(mm_to_px(x, y))
        h = m.shape[0] // 2
        img[v - h:v - h + m.shape[0], u - h:u - h + m.shape[1]] = m
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), mm_to_px


# ---------------------------------------------------------------- colour target detector


@dataclass
class ColorTarget:
    """HSV window (OpenCV H 0..179). If hsv_lo[0] > hsv_hi[0] the hue range wraps (reds)."""
    hsv_lo: np.ndarray
    hsv_hi: np.ndarray
    name: str = "custom"

    def __post_init__(self):
        self.hsv_lo, self.hsv_hi = np.asarray(self.hsv_lo, np.uint8), np.asarray(self.hsv_hi, np.uint8)

    def mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        lo, hi = self.hsv_lo.copy(), self.hsv_hi.copy()
        if lo[0] <= hi[0]:
            return cv2.inRange(hsv, lo, hi)
        a, b = lo.copy(), hi.copy()
        a[0], b[0] = 0, hi[0]
        lo2, hi2 = lo.copy(), hi.copy()
        hi2[0] = 179
        return cv2.inRange(hsv, a, b) | cv2.inRange(hsv, lo2, hi2)

    def to_dict(self) -> dict:
        return {"name": self.name, "hsv_lo": self.hsv_lo.tolist(), "hsv_hi": self.hsv_hi.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ColorTarget":
        return cls(d["hsv_lo"], d["hsv_hi"], d.get("name", "custom"))

    @classmethod
    def from_sample(cls, frame_rgb: np.ndarray, u: float, v: float, radius_px: int = 6,
                    h_tol: int = 12, s_tol: int = 70, v_tol: int = 90) -> "ColorTarget":
        """Tolerant window around the median HSV of the patch at (u, v); hue wraps for reds."""
        h, w = frame_rgb.shape[:2]
        u, v = int(round(u)), int(round(v))
        patch = frame_rgb[max(0, v - radius_px):min(h, v + radius_px + 1), max(0, u - radius_px):min(w, u + radius_px + 1)]
        hsv = cv2.cvtColor(patch.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        hm, sm, vm = (int(np.median(hsv[:, i])) for i in range(3))
        # a genuinely colourful target keeps a saturation floor so the window can't swallow gray surfaces
        lo_s = max(40, sm - s_tol) if sm >= 100 else max(0, sm - s_tol)
        lo = ((hm - h_tol) % 180, lo_s, max(0, vm - v_tol))
        hi = ((hm + h_tol) % 180, min(255, sm + s_tol), min(255, vm + v_tol))
        return cls(lo, hi, f"sampled@{u},{v}")

    @classmethod
    def parse(cls, spec: str) -> "ColorTarget":
        """'green' | 'orange' | 'hsv:lo_h,lo_s,lo_v,hi_h,hi_s,hi_v'."""
        if spec.startswith("hsv:"):
            v = [int(x) for x in spec[4:].split(",")]
            assert len(v) == 6, spec
            return cls(v[:3], v[3:], "custom")
        return {"green": GREEN_BALL, "orange": ORANGE}[spec.lower()]


# Green preset: bright saturated green (the table's yellow-green can sits at H~40 S~90 and is excluded).
GREEN_BALL = ColorTarget((45, 100, 70), (85, 255, 255), "green")
ORANGE = ColorTarget((15, 80, 120), (40, 255, 255), "orange")  # orange/yellow ping-pong ball
BALL_HSV_LO, BALL_HSV_HI = tuple(ORANGE.hsv_lo.tolist()), tuple(ORANGE.hsv_hi.tolist())


def detect_target(frame_rgb: np.ndarray, target: ColorTarget, min_r_px: int = 5) -> tuple[float, float, float] | None:
    """Blob in the target's HSV window -> (u, v, r_px) of its min enclosing circle, or None. Among blobs at least ~a
    sixth the size of the largest, the roundest wins (a ball beats a bigger same-coloured box/cable)."""
    mask = cv2.morphologyEx(target.mask(frame_rgb), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = [(cv2.contourArea(c), c) for c in cnts]
    if not cands:
        return None
    amax = max(a for a, _ in cands)
    big = [(a, c) for a, c in cands if a >= 0.15 * amax and a > 0]
    a, c = max(big, key=lambda ac: ac[0] / (np.pi * cv2.minEnclosingCircle(ac[1])[1] ** 2 + 1e-9))
    (u, v), r = cv2.minEnclosingCircle(c)
    return (float(u), float(v), float(r)) if r >= min_r_px else None


def detect_ball(frame_rgb: np.ndarray, hsv_lo=BALL_HSV_LO, hsv_hi=BALL_HSV_HI, min_r_px=5) -> tuple[float, float, float] | None:
    return detect_target(frame_rgb, ColorTarget(hsv_lo, hsv_hi), min_r_px)


def fit_px_to_mm(px, mm, ransac_thresh_mm: float = 5.0,
                 min_pts_for_homography: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Overhead px -> table mm from paired points (Nx2 each). Returns (H, residuals_mm, inliers, model).

    MODEL CHOICE MATTERS MORE THAN THE FIT. A full homography has 8 degrees of freedom, so from 4 points
    it is an exact interpolation of whatever noise those 4 points carry -- residual 0.0 and a projection
    that shears and converges wildly a few centimetres away (the classic "the grid looks way off" result).
    An affine fit has 6, is over-determined from 4 points, and cannot invent perspective. For an overhead
    camera looking near-straight down the true perspective term is tiny, so below `min_pts_for_homography`
    points affine is both stabler AND closer to the truth. With 8+ points the full homography is justified
    and is used.
    """
    px, mm = np.asarray(px, np.float64).reshape(-1, 2), np.asarray(mm, np.float64).reshape(-1, 2)
    if len(px) < 4:
        raise ValueError(f"need >= 4 points for a homography, got {len(px)}")
    if len(px) >= min_pts_for_homography:
        H, mask = cv2.findHomography(np.float32(px), np.float32(mm), cv2.RANSAC, ransac_thresh_mm)
        model = "homography"
    else:
        A, mask = cv2.estimateAffine2D(np.float32(px), np.float32(mm), method=cv2.RANSAC,
                                       ransacReprojThreshold=ransac_thresh_mm)
        H = None if A is None else np.vstack([A, [0.0, 0.0, 1.0]])
        model = "affine"
        print(f"[calibration] {len(px)} points: fitting a 6-DOF affine (a homography needs "
              f"{min_pts_for_homography}+ to be meaningful)")
    if H is None:
        raise RuntimeError("fit failed (degenerate points?)")
    proj = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
    inliers = np.ones(len(px), bool) if mask is None else mask.ravel().astype(bool)
    return H, np.linalg.norm(proj - mm, axis=1), inliers, model


def ball_table_xy(homog: TableHomography, u: float, v: float, ball_radius_mm: float,
                  cam_height_mm: float, cam_xy_mm: tuple[float, float]) -> tuple[float, float]:
    """Parallax correction: the ray through the ball centre (at z=r) hits the table plane further from the
    camera nadir than the ball actually is. Scale the nadir-relative offset by (H - r) / H."""
    px, py = homog.px_to_mm(u, v)  # where the ray meets the table plane
    cx, cy = cam_xy_mm
    s = (cam_height_mm - ball_radius_mm) / cam_height_mm
    return cx + (px - cx) * s, cy + (py - cy) * s


# ---------------------------------------------------------------- rigid transform


def solve_rigid(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Kabsch: 4x4 T with dst ~= R @ src + t (no scale). src/dst: Nx3."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    ms, md = src.mean(0), dst.mean(0)
    Hm = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(Hm)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, md - R @ ms
    return T


def base_to_table(T: np.ndarray, xyz_m) -> np.ndarray:
    """FK base-frame meters -> table-frame mm."""
    return T[:3, :3] @ (np.asarray(xyz_m, float) * 1000.0) + T[:3, 3]


def table_to_base(T: np.ndarray, xyz_mm) -> np.ndarray:
    """Table-frame mm -> base-frame meters."""
    return (T[:3, :3].T @ (np.asarray(xyz_mm, float) - T[:3, 3])) / 1000.0


def residuals_mm(T: np.ndarray, base_m: np.ndarray, table_mm: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.array([base_to_table(T, p) for p in base_m]) - table_mm, axis=1)


def identity_calib() -> np.ndarray:
    """Fallback when calib.json is missing: base_link origin == table origin, axes aligned."""
    return np.eye(4)


def save_calib(path: Path, T: np.ndarray, base_m=None, table_mm=None, extra: dict | None = None) -> None:
    d = {"table_T_base": np.asarray(T).tolist(), "units": "p_table_mm = R @ (p_base_m*1000) + t"}
    if base_m is not None and table_mm is not None:
        d["points"] = {"base_m": np.asarray(base_m).tolist(), "table_mm": np.asarray(table_mm).tolist()}
        d["residuals_mm"] = residuals_mm(T, np.asarray(base_m), np.asarray(table_mm)).tolist()
    d.update(extra or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2))


def save_ball_calib(path: Path, H: np.ndarray, px, mm, residuals, plane_z_mm: float, target: ColorTarget,
                    base_z_offset_mm: float, method: str = "teleop", frame_wh=None,
                    saved_at: float | None = None) -> None:
    """calib.json for ball mode: table_T_base = identity xy + z offset; H_px_to_mm maps overhead px -> table mm.
    Writing only ever happens here (on Finish); the previous file is kept as calib.json.bak so a bad
    calibration run can never destroy a good one."""
    path = Path(path)
    if path.exists():  # timestamped safety net: one Finish must not clobber the only good calibration
        path.with_suffix(path.suffix + ".bak").write_bytes(path.read_bytes())
    T = np.eye(4)
    T[2, 3] = base_z_offset_mm
    save_calib(path, T, extra={
        "method": method, "H_px_to_mm": np.asarray(H).tolist(), "plane_z_mm": float(plane_z_mm),
        "target": target.to_dict(), "base_z_offset_mm": float(base_z_offset_mm),
        "points": {"px": np.asarray(px).tolist(), "table_mm": np.asarray(mm).tolist()},
        "residuals_mm": np.asarray(residuals).tolist(),
        "frame_wh": None if frame_wh is None else [int(frame_wh[0]), int(frame_wh[1])],
        "saved_at": float(saved_at if saved_at is not None else time.time()),
        "note": "H exact for object centroids at plane_z_mm; taller objects project slightly outward from the camera nadir"})


def load_calib_dict(path: Path | None = None) -> dict:
    """Whole calib.json (tolerates old aruco-era files): keys table_T_base (4x4), H_px_to_mm (3x3 or None),
    plane_z_mm, target (ColorTarget or None), method ('ball'|'aruco'), base_z_offset_mm."""
    path = Path(path or cfgmod.load().calib_file)
    d = json.loads(path.read_text()) if path.exists() else {}
    H = d.get("H_px_to_mm")
    return {**d, "table_T_base": np.array(d.get("table_T_base", np.eye(4)), float),
            "H_px_to_mm": None if H is None else np.array(H, float).reshape(3, 3),
            "plane_z_mm": d.get("plane_z_mm"), "method": d.get("method", "aruco" if d else None),
            "target": ColorTarget.from_dict(d["target"]) if d.get("target") else None,
            "base_z_offset_mm": float(d.get("base_z_offset_mm", d.get("table_T_base", np.eye(4))[2][3] if d else 0.0))}


def load_calib(path: Path | None = None) -> np.ndarray:
    """table_T_base from calib.json, or identity (with a warning) if absent."""
    path = Path(path or cfgmod.load().calib_file)
    if not path.exists():
        print(f"[calibration] {path} missing; using identity table_T_base")
        return identity_calib()
    return np.array(json.loads(path.read_text())["table_T_base"], float)


# ---------------------------------------------------------------- selftest


def _selftest() -> None:
    import tempfile

    cfg = cfgmod.load()
    # isolate from the machine's real calib.json: in auto mode a fitted H would silently take over
    # whenever a tag drops out, and this synthetic scene has nothing to do with that H
    cfg.calib_file = Path(tempfile.mkdtemp()) / "no_calib.json"
    img, truth = render_mat(cfg.aruco_tags_mm, cfg.aruco_dict)
    h = TableHomography(cfg)
    assert h.update(img), f"tags found: {h.last_centers_px}"
    assert len(h.last_centers_px) == 4
    for x, y in [(270.0, 0.0), (200.0, 100.0), (380.0, -150.0), (275.0, 140.0)]:
        u, v = truth(x, y)
        rx, ry = h.px_to_mm(u, v)
        assert abs(rx - x) < 1.0 and abs(ry - y) < 1.0, (x, y, rx, ry)
        bu, bv = h.mm_to_px(x, y)
        assert abs(bu - u) < 1.5 and abs(bv - v) < 1.5, (u, v, bu, bv)
    # cached H survives a tag dropping out
    blank = img.copy()
    u, v = _ip(truth(*cfg.aruco_tags_mm[0]))
    blank[v - 60:v + 60, u - 60:u + 60] = 255
    assert h.update(blank) and len(h.last_centers_px) == 3
    grid = h.draw_grid(img)
    assert grid.shape == img.shape and not np.array_equal(grid, img)

    # ball: render an orange disc at a known mm position, detect, parallax-correct
    cam_h, cam_xy, r_mm = 700.0, (270.0, 0.0), 20.0
    ball_xy = (350.0, -100.0)
    s = (cam_h - r_mm) / cam_h
    plane_xy = (cam_xy[0] + (ball_xy[0] - cam_xy[0]) / s, cam_xy[1] + (ball_xy[1] - cam_xy[1]) / s)
    bimg = img.copy()
    cv2.circle(bimg, _ip(truth(*plane_xy)), 30, (255, 150, 20), -1)
    det = detect_ball(bimg)
    assert det is not None
    bx, by = ball_table_xy(h, det[0], det[1], r_mm, cam_h, cam_xy)
    assert abs(bx - ball_xy[0]) < 1.5 and abs(by - ball_xy[1]) < 1.5, (bx, by)

    # colour targets: sampling, hue wrap, detect_target, homography fit
    red = np.zeros((120, 160, 3), np.uint8)
    cv2.circle(red, (100, 60), 14, (230, 20, 30), -1)
    t = ColorTarget.from_sample(red, 100, 60)
    assert t.hsv_lo[0] > t.hsv_hi[0], t  # red wraps around H=0
    d = detect_target(red, t)
    assert d and abs(d[0] - 100) < 1 and abs(d[1] - 60) < 1 and 12 < d[2] < 16, d
    g = np.full((240, 320, 3), 30, np.uint8)
    cv2.circle(g, (200, 100), 16, (40, 220, 60), -1)
    cv2.rectangle(g, (20, 150), (80, 200), (40, 220, 60), -1)  # a bigger green box must not steal the detection
    dg = detect_target(g, GREEN_BALL)
    assert dg and abs(dg[0] - 200) < 1 and abs(dg[1] - 100) < 1 and detect_target(g, ORANGE) is None, dg
    assert ColorTarget.parse("hsv:1,2,3,4,5,6").hsv_hi.tolist() == [4, 5, 6] and ColorTarget.parse("orange") is ORANGE
    assert ColorTarget.from_dict(GREEN_BALL.to_dict()).hsv_lo.tolist() == GREEN_BALL.hsv_lo.tolist()
    Ht = np.array([[-0.5, 0.01, 400.0], [0.02, -0.52, 190.0], [1e-5, 2e-5, 1.0]])
    pxs = np.random.default_rng(1).uniform([50, 50], [600, 400], (9, 2))
    mms = cv2.perspectiveTransform(pxs.reshape(-1, 1, 2), Ht).reshape(-1, 2)
    Hf, res, inl, model = fit_px_to_mm(pxs, mms + np.random.default_rng(2).normal(0, 0.2, mms.shape))
    assert model == 'homography', model  # 9 points -> full 8-DOF fit
    assert inl.all(), inl  # clean synthetic data: nothing may be silently dropped
    chk = np.array([[100.0, 100.0], [500.0, 350.0]]).reshape(-1, 1, 2)
    assert res.max() < 1.0 and np.allclose(cv2.perspectiveTransform(chk, Hf), cv2.perspectiveTransform(chk, Ht), atol=1.0), (res, Hf)
    bad = mms.copy()
    bad[2] += 40.0  # one wrecked sample must be REPORTED as an outlier, not quietly averaged in
    _, res_b, inl_b, _ = fit_px_to_mm(pxs, bad)
    assert not inl_b[2] and inl_b.sum() >= 6 and res_b[2] > 20.0, (inl_b, res_b)
    # few points -> affine (6 DOF), which cannot invent the wild perspective an 8-DOF fit would
    Ha, _, _, model_a = fit_px_to_mm(pxs[:5], mms[:5])
    assert model_a == "affine" and np.allclose(Ha[2], [0, 0, 1]), (model_a, Ha[2])
    chk5 = np.array([[120.0, 120.0], [520.0, 360.0]]).reshape(-1, 1, 2)
    assert np.allclose(cv2.perspectiveTransform(chk5, Ha), cv2.perspectiveTransform(chk5, Ht), atol=12.0)
    try:
        fit_px_to_mm(pxs[:3], mms[:3])
        raise AssertionError("accepted 3 points")
    except ValueError:
        pass
    # ball-mode TableHomography round trip via calib.json
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "calib.json"
    save_ball_calib(tmp, Hf, pxs, mms, res, 38.0, GREEN_BALL, 12.0)
    dd = load_calib_dict(tmp)
    assert dd["method"] == "teleop" and dd["plane_z_mm"] == 38.0 and dd["target"].name == "green" and dd["base_z_offset_mm"] == 12.0
    assert np.allclose(load_calib(tmp), np.diag([1, 1, 1, 1.0]) + np.array([[0, 0, 0, 0], [0] * 4, [0, 0, 0, 12.0], [0] * 4]))
    old = Path(tempfile.mkdtemp()) / "old.json"
    save_calib(old, np.eye(4))
    assert load_calib_dict(old)["H_px_to_mm"] is None and load_calib_dict(old)["method"] == "aruco"

    # coverage / persistence helpers: hull, coverage, collinearity, trust region, summary, .bak backup
    save_ball_calib(tmp, Hf, pxs, mms, res, 38.0, GREEN_BALL, 12.0, frame_wh=(640, 480))
    assert tmp.with_name("calib.json.bak").exists(), "previous calib.json not backed up on save"
    dd = load_calib_dict(tmp)
    assert dd["frame_wh"] == [640, 480] and dd["saved_at"] > 0
    summ = calib_summary(dd)
    assert summ and "9 points" in summ and "coverage" in summ and "saved" in summ, summ
    assert calib_summary({"H_px_to_mm": None}) is None and calib_summary_file(tmp) == summ
    hull = sample_hull_px(pxs)
    assert hull is not None and len(hull) >= 3
    assert in_calibrated_region(hull, *pxs.mean(0))          # centre of the sampled area
    assert not in_calibrated_region(hull, -200.0, -200.0)    # far outside the sampled area
    assert in_calibrated_region(None, -200.0, -200.0)        # no hull recorded -> guard disabled
    pct = coverage_pct(pxs, (640, 480))
    assert 10.0 < pct <= 100.0, pct
    assert "clustered" in coverage_verdict(5.0) and "good" in coverage_verdict(50.0)
    line_px = np.array([[10.0 * i, 5.0 * i + 50] for i in range(6)])
    assert collinearity_ratio(line_px) < 0.01 < collinearity_ratio(pxs)
    hb2 = TableHomography(cfgmod.load(), mode="ball")
    hb2.cfg.calib_file = tmp
    assert hb2.reload(tmp) and hb2.region_px is not None and hb2.summary == summ
    for mode in ("ball", "auto"):
        cfg_b = cfgmod.load(); cfg_b.calib_file = tmp
        hb = TableHomography(cfg_b, mode=mode)
        assert hb.ready and hb.update(g) and hb.method == "ball"
        u, v = pxs[0]
        assert np.allclose(hb.px_to_mm(u, v), mms[0], atol=1.0)
        assert hb.draw_grid(g).shape == g.shape
    hb = TableHomography(cfg_b, mode="auto")
    assert hb.update(img) and hb.method == "aruco"  # tags win in auto
    assert hb.update(g) and hb.method == "ball"  # ..and fall back to the fitted H when they vanish
    assert not TableHomography(cfg_b, mode="aruco").ready

    # rigid transform recovery with noise
    rng = np.random.default_rng(0)
    ang = np.deg2rad(7.0)
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]])
    t = np.array([12.0, -5.0, 30.0])
    base_m = rng.uniform([0.15, -0.2, 0.05], [0.4, 0.2, 0.2], (8, 3))
    table_mm = (R @ (base_m * 1000).T).T + t + rng.normal(0, 0.5, (8, 3))
    T = solve_rigid(base_m * 1000, table_mm)
    assert np.allclose(T[:3, :3], R, atol=2e-3) and np.allclose(T[:3, 3], t, atol=1.5), T
    assert residuals_mm(T, base_m, table_mm).max() < 2.0
    p = np.array([0.3, 0.1, 0.1])
    assert np.allclose(table_to_base(T, base_to_table(T, p)), p)
    print("calibration selftest OK")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Live ArUco mat check (writes nothing; use sortbot.calibrate for calib.json).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-target", default=None, metavar="SPEC",
                    help="grab one overhead frame (cv2), run detect_target with SPEC (green|orange|hsv:..), write sortbot/out/target_check.png")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.check_target:
        cfg = cfgmod.load()
        cap = cv2.VideoCapture(cfg.overhead_cam.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.overhead_cam.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.overhead_cam.height)
        for _ in range(8):
            ok, bgr = cap.read()
        assert ok, "camera read failed"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = ColorTarget.parse(a.check_target)
        det = detect_target(rgb, t)
        if det:
            cv2.circle(bgr, _ip(det), int(det[2]), (0, 255, 0), 2)
            cv2.putText(bgr, f"{t.name} ({det[0]:.0f},{det[1]:.0f}) r={det[2]:.0f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out = Path(__file__).with_name("out") / "target_check.png"
        cv2.imwrite(str(out), bgr)
        print(f"target {t.name}: {'found ' + str(tuple(round(x) for x in det)) if det else 'NOT found'} -> {out}")
    else:
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        cfg = cfgmod.load()
        c = cfg.overhead_cam
        cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=c.index, fps=c.fps, width=c.width, height=c.height))
        cam.connect()
        h = TableHomography(cfg)
        while True:
            frame = cam.read()
            h.update(frame)
            cv2.imshow("mat", cv2.cvtColor(h.draw_grid(frame), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
