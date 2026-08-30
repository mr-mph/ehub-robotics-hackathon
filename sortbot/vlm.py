"""VLM planner: one tool call per step via the OpenAI Responses API. The VLM does the SEEING itself:
it gets the overhead photo with a cm-labelled grid overlay (plus the wrist photo) and emits
coordinate-based tools — there is no object detector and no numbered object list anywhere.

UNIT BOUNDARY: everything the VLM sees and emits is table-frame CENTIMETERS (x forward, y left,
origin at the robot base, table z=0). Internally sortbot works in millimeters (robot, config.yaml,
calib.json, safety envelope); sortbot.main.Loop converts each Command's *_cm args to mm in exactly
one place (Loop._args_mm). _state_text's /10 conversions below are the mm->cm half of that boundary.

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

from sortbot.types import Command, WorldState

TOOLS: list[dict] = [
    ("pick_at", "Pick up the object whose centre is at table-frame (x, y) in CENTIMETERS. Read the "
                "position off the labelled grid in the overhead image.",
     {"x_cm": {"type": "number"}, "y_cm": {"type": "number"}}),
    ("place_at", "Place the held object at a table-frame point (cm).",
     {"x_cm": {"type": "number"}, "y_cm": {"type": "number"}}),
    ("open_gripper", "Low-level recovery: open the gripper.", {}),
    ("close_gripper", "Low-level recovery: close the gripper.", {}),
    ("move_to", "Low-level recovery: move the end effector to table-frame x, y, z (cm).",
     {"x_cm": {"type": "number"}, "y_cm": {"type": "number"}, "z_cm": {"type": "number"}}),
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

SYSTEM_PROMPT = """You are the planner for a small tabletop robot arm that tidies things on a table.
Each step you see an overhead photo overlaid with a labelled coordinate grid, a wrist-camera photo, and a text state block. YOU do the seeing: look at the photos, decide what is on the
table and where, and call exactly one tool.
COORDINATES: everything you see and emit is CENTIMETERS in the table frame — x forward from the robot
base, y left, z up, table at z=0, origin at the base. Read positions straight off the grid: the overlay
lines are labelled in cm at both ends (x.. and y..) and the grid spacing is stated in the legend. Aim
pick_at at the centre of an object. Every coordinate you send is validated against a safety envelope;
an out-of-reach or unsafe command is not executed and comes back as a FAILED tool result in the history
— read it and react (adjust the coordinate, or pick something else).
Policy: follow the human's GOAL and RULES exactly when given; if no task is given, group similar items
(by type, colour, size) into tidy clusters on the table, choosing free table space for each group and
placing every member of a group at nearby coordinates. Ignore the robot arm itself and shadows, and
leave anything that is already where it belongs. If holding something, place it. When everything is
sorted, call done.
Use say() sparingly: only for genuine ambiguity, a question to the human, or a final summary.
RULES from the human override everything else and must always be respected.
Prefer pick_at / place_at; use move_to / open_gripper / close_gripper / turn_to only to
recover from a failure."""


def _state_text(world: WorldState, history: list, workspace_mm=None) -> str:
    # mm -> cm for everything the VLM reads (the unit boundary; see the module docstring)
    p = world.ee_pose
    lo, hi = workspace_mm or ((120.0, -220.0, 0.0), (420.0, 220.0, 250.0))
    lines = [f"EE pose (table cm): x={p.x / 10:.1f} y={p.y / 10:.1f} z={p.z / 10:.1f} roll={p.roll_deg:.0f}deg",
             f"table z=0  gripper_open={world.gripper_open}  holding={world.holding or 'nothing'}",
             f"REACHABLE AREA (cm): x {lo[0] / 10:g}..{hi[0] / 10:g}, y {lo[1] / 10:g}..{hi[1] / 10:g}"]
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

    def plan_step(self, overhead_overlay_png: bytes, wrist_png: bytes, world: WorldState, history: list,
                  workspace_mm=None) -> Command:
        content = [{"type": "input_text", "text": "Overhead camera (cm grid):"}, _img(overhead_overlay_png),
                   {"type": "input_text", "text": "Wrist camera:"}, _img(wrist_png),
                   {"type": "input_text", "text": _state_text(world, history, workspace_mm)}]
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
    cv2.rectangle(img, (200, 100), (260, 160), (255, 0, 0), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _demo_world() -> WorldState:
    from sortbot import config
    cfg = config.load()
    return WorldState(ee_pose=cfg.home, rules=["put red things on the left of the table"])


def _selftest() -> None:
    from sortbot.testing import MockVLM  # test fixture; only reachable from this selftest

    world, png = _demo_world(), _synthetic_png()
    vlm = MockVLM(targets_cm=[(25.0, 12.0), (30.0, -5.0)])
    seq, hist = [], []
    for _ in range(10):
        cmd = vlm.plan_step(png, png, world, hist)
        seq.append(cmd.tool)
        hist.append({"tool": cmd.tool, "args": cmd.args, "result": "ok"})
        if cmd.tool == "pick_at":
            world.holding = f"object picked at ({cmd.args['x_cm']:.1f}, {cmd.args['y_cm']:.1f}) cm"
        elif cmd.tool == "place_at":
            world.holding = None
        elif cmd.tool == "done":
            break
    assert seq == ["pick_at", "place_at", "pick_at", "place_at", "done"], seq
    assert {t["name"] for t in TOOLS} == {"pick_at", "place_at", "open_gripper", "close_gripper",
                                          "move_to", "turn_to", "say", "done"}, [t["name"] for t in TOOLS]
    for gone in ("pick", "place_in_zone"):
        assert not any(t["name"] == gone for t in TOOLS), f"{gone} must be gone"
    st = _state_text(world, hist)
    assert "RULES" in st and "put red things" in st
    assert "cm" in st and "zone" not in st.lower(), st  # coordinates only; zones are gone
    low = SYSTEM_PROMPT.lower()
    for banned in ("numbered", "object id", "object list", "detected object", "detector", "zone"):
        assert banned not in low, f"detector-era phrase {banned!r} back in the system prompt"
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
