"""Shared data contracts. Table frame = mm, origin at robot base on tabletop, x fwd, y left, z up."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Pose:
    x: float
    y: float
    z: float
    roll_deg: float = 0.0  # wrist_roll; gripper always points straight down


@dataclass
class DetectedObject:
    id: int
    centroid_mm: tuple[float, float]
    bbox_px: tuple[int, int, int, int]  # x0, y0, x1, y1 in overhead image
    area_px: float
    color_hint: str
    label: str | None = None


@dataclass
class Zone:
    name: str
    polygon_mm: list[tuple[float, float]]
    drop_point_mm: tuple[float, float]

    def contains(self, x: float, y: float) -> bool:
        inside = False
        pts = self.polygon_mm
        for i in range(len(pts)):
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
            if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
                inside = not inside
        return inside


@dataclass
class WorldState:
    objects: list[DetectedObject] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    ee_pose: Pose = field(default_factory=lambda: Pose(0, 0, 0))
    gripper_open: bool = True
    holding: int | None = None
    rules: list[str] = field(default_factory=list)


@dataclass
class Command:
    tool: str  # pick | place_in_zone | place_at | done | say | move_to | open | close | turn_to
    args: dict = field(default_factory=dict)


@dataclass
class ExecResult:
    ok: bool
    message: str = ""


@runtime_checkable
class RobotAPI(Protocol):
    """Implemented by sortbot.robot (real + mock). All coordinates are table-frame mm."""

    def home(self) -> ExecResult: ...
    def open_gripper(self) -> ExecResult: ...
    def close_gripper(self) -> ExecResult: ...
    def move_to(self, x: float, y: float, z: float) -> ExecResult: ...
    def turn_to(self, deg: float) -> ExecResult: ...
    def get_ee_pose(self) -> Pose: ...
    def get_joints_deg(self) -> np.ndarray: ...
    def capture(self, name: str) -> np.ndarray: ...  # 'overhead' | 'wrist' -> RGB HxWx3
    def pick(self, obj: DetectedObject) -> ExecResult: ...
    def place_at(self, x: float, y: float) -> ExecResult: ...
