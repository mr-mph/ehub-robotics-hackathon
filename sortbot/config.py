"""Config loaded from sortbot/config.yaml. Paths are resolved relative to the repo root."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from sortbot.types import Pose

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
    tool_offset_mm: tuple[float, float, float]
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
    # workspace.z_trim_mm: negative lowers the plane the gripper descends to (see robot.grasp_z_mm).
    # Every z floor in the app is derived from it, so one number moves them all together.
    z_trim_mm: float = 0.0
    z_floor_mm: float = -150.0   # workspace.z_floor_mm: absolute floor for any commanded z (a backstop)
    tts_model: str = "eleven_turbo_v2_5"   # voice.tts_model
    stt_model: str = "scribe_v2"           # voice.stt_model
    # voice.listen_on_start: start hands-free Listening at boot (user's explicit choice; the mic then
    # hears the whole room until toggled off -- the MIC LIVE chip shows while it is on).
    listen_on_start: bool = True
    # vlm.chat_model / vlm.verify_model: the small fast models behind the conversational Chat worker and
    # the pre-grasp alignment check. Empty falls back to chat_model, then to openai_model.
    chat_model: str = ""
    verify_model: str = ""
    chat_effort: str = "low"               # vlm.chat_effort: reasoning effort for chat + verify
    # grasp:  the claw must NEVER close without checking BOTH cameras first (see main.Loop._verify_grasp)
    grasp_verify: bool = True
    grasp_max_correction_cm: float = 2.0
    grasp_max_retries: int = 2
    grasp_min_confidence: float = 0.5
    source_path: Path = DEFAULT_YAML       # yaml file this config was loaded from (set_model persists here)
    raw: dict = field(default_factory=dict, repr=False)




def load(path: str | Path = DEFAULT_YAML) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    r, c, w, a = raw["robot"], raw["cameras"], raw["workspace"], raw["aruco"]
    cal, ld = raw.get("calibration") or {}, raw.get("leader") or {}
    vl, gr = raw.get("vlm") or {}, raw.get("grasp") or {}
    chat_model = str(vl.get("chat_model") or "") or str(vl["model"])
    return Config(
        robot_port=r["port"],
        robot_id=r["id"],
        robot_calibration_dir=(REPO_ROOT / r["calibration_dir"]) if r.get("calibration_dir") else None,
        urdf=REPO_ROOT / r["urdf"],
        overhead_cam=CameraCfg(**c["overhead"]),
        wrist_cam=CameraCfg(**c["wrist"]),
        table_z_mm=float(w["table_z_mm"]),
        tool_offset_mm=tuple(float(v) for v in w.get("tool_offset_mm", (0.0, 0.0, 0.0))),
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
        z_trim_mm=float(w.get("z_trim_mm", Config.z_trim_mm)),
        z_floor_mm=float(w.get("z_floor_mm", Config.z_floor_mm)),
        tts_model=str(raw["voice"].get("tts_model", Config.tts_model)),
        stt_model=str(raw["voice"].get("stt_model", Config.stt_model)),
        listen_on_start=bool(raw["voice"].get("listen_on_start", Config.listen_on_start)),
        chat_model=chat_model,
        verify_model=str(vl.get("verify_model") or "") or chat_model,
        chat_effort=str(vl.get("chat_effort", Config.chat_effort)),
        grasp_verify=bool(gr.get("verify", True)),
        grasp_max_correction_cm=float(gr.get("max_correction_cm", Config.grasp_max_correction_cm)),
        grasp_max_retries=int(gr.get("max_retries", Config.grasp_max_retries)),
        grasp_min_confidence=float(gr.get("min_confidence", Config.grasp_min_confidence)),
        source_path=Path(path),
        raw=raw,
    )


if __name__ == "__main__":
    cfg = load()
    assert cfg.urdf.exists(), cfg.urdf
    assert cfg.calib_mode in ("ball", "aruco", "auto") and cfg.leader_port
    assert cfg.chat_model and cfg.verify_model, (cfg.chat_model, cfg.verify_model)
    assert cfg.grasp_verify is True, "grasp.verify must default to true"
    assert cfg.grasp_max_correction_cm > 0 and cfg.grasp_max_retries >= 0
    from sortbot import robot as _rb
    assert -_rb.Z_TRIM_LIMIT_MM <= cfg.z_trim_mm <= _rb.Z_TRIM_LIMIT_MM, cfg.z_trim_mm
    assert cfg.z_floor_mm < cfg.table_z_mm, (cfg.z_floor_mm, cfg.table_z_mm)
    print(f"grasp height {_rb.grasp_z_mm(cfg):.1f} mm (trim {cfg.z_trim_mm:+.1f} mm, "
          f"absolute floor {_rb.hard_floor_mm(cfg):.1f} mm)")
    print(cfg)
    print("selftest OK")
