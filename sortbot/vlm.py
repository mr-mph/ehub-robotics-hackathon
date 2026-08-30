"""VLM planner: one tool call per step via the OpenAI Responses API.

plan_step(overhead_overlay_png, wrist_png, world, history) -> Command. Images are PNG bytes.
history = list of dicts {"tool": str, "args": dict, "result": str} of prior steps (last 10 sent).
Selftest: `python -m sortbot.vlm --selftest` (mock); `--live` makes one real call with a synthetic image.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from typing import Iterable

from sortbot.types import Command, DetectedObject, WorldState, Zone

TOOLS: list[dict] = [
    ("pick", "Pick up object <id> from the numbered object list.", {"id": {"type": "integer"}}),
    ("place_in_zone", "Place the held object at the drop point of the named zone.", {"zone": {"type": "string"}}),
    ("place_at", "Place the held object at a table-frame point (mm).",
     {"x_mm": {"type": "number"}, "y_mm": {"type": "number"}}),
    ("open_gripper", "Low-level recovery: open the gripper.", {}),
    ("close_gripper", "Low-level recovery: close the gripper.", {}),
    ("move_to", "Low-level recovery: move the end effector to table-frame x,y,z (mm).",
     {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}),
    ("turn_to", "Low-level recovery: rotate the wrist roll to <deg>.", {"deg": {"type": "number"}}),
    ("say", "Speak a short message to the human. Use sparingly (questions, ambiguity, finish).",
     {"text": {"type": "string"}}),
    ("done", "Finish: nothing sensible left to sort. Give a one-line summary.", {"summary": {"type": "string"}}),
]
TOOLS = [
    {"type": "function", "name": n, "description": d, "strict": True,
     "parameters": {"type": "object", "properties": p, "required": list(p), "additionalProperties": False}}
    for n, d, p in TOOLS
]
# Tool name -> Command.tool (types.Command uses open/close for the gripper tools).
TOOL_TO_CMD = {"open_gripper": "open", "close_gripper": "close"}

# Rough $ / 1M tokens (input, output); longest-prefix match. Good enough for a per-call HUD estimate.
PRICES_PER_MTOK = {
    "gpt-5-nano": (0.05, 0.40), "gpt-5-mini": (0.25, 2.00), "gpt-5": (1.25, 10.00),
    "gpt-4.1-nano": (0.10, 0.40), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00),
    "o3-pro": (20.00, 80.00), "o3": (2.00, 8.00), "o4-mini": (1.10, 4.40),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Rough per-call cost from the price table; None for unknown models."""
    for prefix in sorted(PRICES_PER_MTOK, key=len, reverse=True):
        if model.startswith(prefix):
            i, o = PRICES_PER_MTOK[prefix]
            return (input_tokens * i + output_tokens * o) / 1e6
    return None

SYSTEM_PROMPT = """You are the planner for a small tabletop robot arm that sorts objects into zones.
Each step you see an overhead photo with numbered candidate objects (and zone outlines) and a wrist-camera photo,
plus a text state block. Decide the single best next action and call exactly one tool.
Policy: sort however makes sense — group similar objects (by type, colour, size) into the named zones; if a zone name
obviously matches an object (e.g. wires -> WIRES) use it. Pick only objects that are not already in a sensible zone.
If holding an object, place it. When everything is sorted, call done.
Use say() sparingly: only for genuine ambiguity, a question to the human, or a final summary.
RULES from the human override everything else and must always be respected.
Prefer pick / place_in_zone / place_at; use move_to / open_gripper / close_gripper / turn_to only to recover
from a failure. Never invent object ids not in the list."""


def _state_text(world: WorldState, history: list) -> str:
    p = world.ee_pose
    lines = [f"EE pose (table mm): x={p.x:.0f} y={p.y:.0f} z={p.z:.0f} roll={p.roll_deg:.0f}",
             f"table_z=0  gripper_open={world.gripper_open}  holding={world.holding}",
             "OBJECTS (id: centroid mm, colour, label):"]
    lines += [f"  {o.id}: ({o.centroid_mm[0]:.0f}, {o.centroid_mm[1]:.0f}) {o.color_hint}"
              + (f" {o.label}" if o.label else "") for o in world.objects] or ["  (none detected)"]
    lines.append("ZONES (name: drop point mm):")
    lines += [f"  {z.name}: ({z.drop_point_mm[0]:.0f}, {z.drop_point_mm[1]:.0f})" for z in world.zones]
    lines.append("RULES:")
    lines += [f"  - {r}" for r in world.rules] or ["  (none)"]
    lines.append("HISTORY (most recent last):")
    lines += [f"  {h.get('tool')}({json.dumps(h.get('args', {}))}) -> {h.get('result', '')}"
              for h in history[-10:]] or ["  (none)"]
    return "\n".join(lines)


def _img(png: bytes) -> dict:
    return {"type": "input_image", "detail": "high",
            "image_url": "data:image/png;base64," + base64.b64encode(png).decode()}


class VLM:
    def __init__(self, model: str | None = None, client=None):
        if model is None:
            from sortbot import config
            model = config.load().openai_model
        self.model = model
        self.last_latency_ms: int | None = None
        self.last_usage: dict | None = None
        self.last_cost_usd: float | None = None
        if client is None:
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
            client = OpenAI()
        self.client = client

    def plan_step(self, overhead_overlay_png: bytes, wrist_png: bytes, world: WorldState, history: list) -> Command:
        content = [{"type": "input_text", "text": "Overhead camera (numbered objects, zones):"}, _img(overhead_overlay_png),
                   {"type": "input_text", "text": "Wrist camera:"}, _img(wrist_png),
                   {"type": "input_text", "text": _state_text(world, history)}]
        t0 = time.time()
        resp = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": content}],
            tools=TOOLS,
            tool_choice="required",
            parallel_tool_calls=False,
        )
        self.last_latency_ms = int((time.time() - t0) * 1000)
        u = getattr(resp, "usage", None)
        if u is not None:
            self.last_usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                               "total_tokens": u.total_tokens}
            self.last_cost_usd = estimate_cost_usd(self.model, u.input_tokens, u.output_tokens)
        calls = [o for o in resp.output if getattr(o, "type", "") == "function_call"]
        if not calls:
            raise RuntimeError(f"VLM returned no tool call: {resp.output_text!r}")
        c = calls[0]
        return Command(TOOL_TO_CMD.get(c.name, c.name), json.loads(c.arguments or "{}"))


def _synthetic_png(w: int = 320, h: int = 240) -> bytes:
    import cv2
    import numpy as np
    img = np.full((h, w, 3), 200, np.uint8)
    cv2.rectangle(img, (40, 60), (100, 120), (0, 0, 255), -1)
    cv2.putText(img, "1", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(img, (200, 100), (260, 160), (255, 0, 0), -1)
    cv2.putText(img, "2", (210, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _demo_world() -> WorldState:
    from sortbot import config
    cfg = config.load()
    return WorldState(
        objects=[DetectedObject(1, (250.0, 120.0), (40, 60, 100, 120), 3600, "red", "wire"),
                 DetectedObject(2, (300.0, -50.0), (200, 100, 260, 160), 3600, "blue", "sensor")],
        zones=cfg.zones, ee_pose=cfg.home, rules=["put red things in WIRES"])


def _selftest() -> None:
    from sortbot.testing import MockVLM  # test fixture; only reachable from this selftest

    world, png = _demo_world(), _synthetic_png()
    vlm, seq, hist = MockVLM(), [], []
    for _ in range(10):
        cmd = vlm.plan_step(png, png, world, hist)
        seq.append(cmd.tool)
        hist.append({"tool": cmd.tool, "args": cmd.args, "result": "ok"})
        if cmd.tool == "pick":
            world.holding = cmd.args["id"]
        elif cmd.tool == "place_in_zone":
            world.objects = [o for o in world.objects if o.id != world.holding]
            world.holding = None
        elif cmd.tool == "done":
            break
    assert seq == ["pick", "place_in_zone", "pick", "place_in_zone", "done"], seq
    assert all(t["name"] in {"pick", "place_in_zone", "place_at", "open_gripper", "close_gripper",
                             "move_to", "turn_to", "say", "done"} for t in TOOLS)
    assert "RULES" in _state_text(world, hist) and "put red things" in _state_text(world, hist)
    c = estimate_cost_usd("gpt-5", 10_000, 1_000)
    assert c is not None and abs(c - 0.0225) < 1e-9, c
    assert estimate_cost_usd("gpt-5-mini-2025", 1000, 0) == 0.00025
    assert estimate_cost_usd("mock", 1, 1) is None
    print("selftest OK:", seq)


def _live() -> None:
    vlm = VLM()
    try:
        cmd = vlm.plan_step(_synthetic_png(), _synthetic_png(), _demo_world(), [])
    except Exception as e:  # model rejected -> list vision models
        print(f"model {vlm.model!r} failed: {e}")
        names = sorted(m.id for m in vlm.client.models.list())
        print("available:", [n for n in names if n.startswith(("gpt-5", "gpt-4o", "gpt-4.1", "o"))])
        raise
    print("live tool call:", cmd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.live:
        _selftest()
    if a.live:
        if not os.environ.get("OPENAI_API_KEY"):
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
        if os.environ.get("OPENAI_API_KEY"):
            _live()
        else:
            print("no OPENAI_API_KEY; skipping live")
