"""Config loaded from sortbot/config.yaml. Paths are resolved relative to the repo root."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from sortbot.types import Pose, Zone

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = Path(__file__).with_name("config.yaml")


@dataclass
class CameraCfg:
    index: int
    width: int
    height: int
    fps: int = 30


@dataclass
class Config:
    robot_port: str
    robot_id: str
    robot_calibration_dir: Path | None
    urdf: Path
    overhead_cam: CameraCfg
    wrist_cam: CameraCfg
    table_z_mm: float
    gripper_clearance_mm: float
    travel_z_mm: float
    grasp_z_mm: float
    max_step_mm: float
    aabb_min_mm: tuple[float, float, float]
    aabb_max_mm: tuple[float, float, float]
    home: Pose
    aruco_dict: str
    aruco_tag_size_mm: float
    aruco_tags_mm: dict[int, tuple[float, float]]
    zones: list[Zone]
    openai_model: str
    elevenlabs_voice_id: str
    hud_port: int
    calib_file: Path
    leader_port: str = "/dev/tty.usbmodem5B790176171"
    leader_id: str = "my_awesome_leader_arm"
    calib_mode: str = "auto"  # ball | aruco | auto
    calib_target: str = "green"  # green | orange | hsv:lo_h,lo_s,lo_v,hi_h,hi_s,hi_v
    calib_ball_radius_mm: float = 20.0
    calib_min_spacing_mm: float = 15.0
    tts_model: str = "eleven_turbo_v2_5"   # voice.tts_model
    stt_model: str = "scribe_v2"           # voice.stt_model
    source_path: Path = DEFAULT_YAML       # yaml file this config was loaded from (set_model persists here)
    raw: dict = field(default_factory=dict, repr=False)

    def zone(self, name: str) -> Zone | None:
        return next((z for z in self.zones if z.name.upper() == name.upper()), None)


def _rect_zone(name: str, d: dict) -> Zone:
    (x0, y0), (x1, y1) = d["rect"]
    return Zone(name, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], tuple(d["drop"]))


def load(path: str | Path = DEFAULT_YAML) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    r, c, w, a = raw["robot"], raw["cameras"], raw["workspace"], raw["aruco"]
    cal, ld = raw.get("calibration") or {}, raw.get("leader") or {}
    return Config(
        robot_port=r["port"],
        robot_id=r["id"],
        robot_calibration_dir=(REPO_ROOT / r["calibration_dir"]) if r.get("calibration_dir") else None,
        urdf=REPO_ROOT / r["urdf"],
        overhead_cam=CameraCfg(**c["overhead"]),
        wrist_cam=CameraCfg(**c["wrist"]),
        table_z_mm=float(w["table_z_mm"]),
        gripper_clearance_mm=float(w["gripper_clearance_mm"]),
        travel_z_mm=float(w["travel_z_mm"]),
        grasp_z_mm=float(w["grasp_z_mm"]),
        max_step_mm=float(w["max_step_mm"]),
        aabb_min_mm=tuple(w["aabb_mm"]["min"]),
        aabb_max_mm=tuple(w["aabb_mm"]["max"]),
        home=Pose(**w["home"]),
        aruco_dict=a["dict"],
        aruco_tag_size_mm=float(a["tag_size_mm"]),
        aruco_tags_mm={int(k): tuple(v) for k, v in a["tags"].items()},
        zones=[_rect_zone(n, d) for n, d in raw["zones"].items()],
        openai_model=raw["vlm"]["model"],
        elevenlabs_voice_id=raw["voice"]["elevenlabs_voice_id"],
        hud_port=int(raw["hud"]["port"]),
        calib_file=REPO_ROOT / raw["calib_file"],
        leader_port=str(ld.get("port", Config.leader_port)),
        leader_id=str(ld.get("id", Config.leader_id)),
        calib_mode=str(cal.get("mode", "auto")),
        calib_target=str(cal.get("target", "green")),
        calib_ball_radius_mm=float(cal.get("ball_radius_mm", 20.0)),
        calib_min_spacing_mm=float(cal.get("min_sample_spacing_mm", 15.0)),
        tts_model=str(raw["voice"].get("tts_model", Config.tts_model)),
        stt_model=str(raw["voice"].get("stt_model", Config.stt_model)),
        source_path=Path(path),
        raw=raw,
    )


if __name__ == "__main__":
    cfg = load()
    assert cfg.urdf.exists(), cfg.urdf
    assert cfg.zone("wires") and cfg.zone("WIRES").contains(*cfg.zone("WIRES").drop_point_mm)
    assert cfg.calib_mode in ("ball", "aruco", "auto") and cfg.leader_port
    print(cfg)
    print("selftest OK")
