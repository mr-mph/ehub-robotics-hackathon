"""End-to-end run with the test doubles injected straight into the Loop (no mock mode exists in
the app; the doubles live in sortbot/testing.py): `python -m sortbot.tests.test_e2e` or pytest."""
from __future__ import annotations

import io
import json
import logging
import socket
import tempfile
from pathlib import Path

from sortbot import config as cfgmod
from sortbot import main as m
from sortbot.testing import MockRobot, MockVLM, SimScene
from sortbot.voice import RulesStore, VoiceIO


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_e2e(with_hud: bool = True) -> None:
    logging.disable(logging.WARNING)
    cfg = cfgmod.load()
    scene = SimScene(MockRobot(cfg), cfg)
    n_objects = len(scene.blobs)
    voice = VoiceIO(stdin=io.StringIO("put white things on the left\n"), force_text=True)
    voice.start()
    hud = None
    if with_hud:
        from sortbot.hud import HUD
        hud = HUD(port=_free_port())
        hud.start()
    rules_path = Path(tempfile.mkdtemp()) / "rules.json"
    loop = m.Loop(cfg, scene, MockVLM(), voice, hud, m.Homography(cfg, scene.H), RulesStore(rules_path), max_steps=12)
    try:
        result = loop.run()
    finally:
        voice.stop()
        if hud:
            hud.stop()
    assert result.startswith("done"), result
    assert loop.placed == n_objects, (loop.placed, n_objects)
    assert loop.holding is None and scene.held is None
    for xy, _ in scene.blobs:  # every blob now sits inside some zone
        assert any(z.contains(*xy) for z in cfg.zones), xy
    assert json.loads(rules_path.read_text()) == ["put white things on the left"]
    assert [h["tool"] for h in loop.history].count("pick") == n_objects
    # a rejected command is fed back as a history entry, not raised
    world = m.WorldState(objects=[], zones=cfg.zones)
    assert loop.validate(m.Command("pick", {"id": 99}), world)
    assert loop.validate(m.Command("place_at", {"x_mm": 900, "y_mm": 0}), world)
    print(f"e2e OK: {result}, placed {loop.placed}/{n_objects}, hud={'on' if hud else 'off'}")


if __name__ == "__main__":
    test_e2e()
