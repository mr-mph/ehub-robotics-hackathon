"""VLM planner: one tool call per step via the OpenAI Responses API. The VLM does the SEEING itself:
it gets the overhead photo with a cm-labelled grid overlay (plus the wrist photo) and emits
coordinate-based tools — there is no object detector and no numbered object list anywhere.

UNIT BOUNDARY: everything the VLM sees and emits is table-frame CENTIMETERS (x forward, y left,
origin at the robot base, table z=0). Internally sortbot works in millimeters (robot, config.yaml,
calib.json, safety envelope); sortbot.main.Loop converts each Command's *_cm args to mm in exactly
one place (Loop._args_mm). _state_text's /10 conversions below are the mm->cm half of that boundary.

plan_step(overhead_overlay_png, wrist_png, world, history) -> Command. Images are PNG bytes.
history = list of dicts {"tool": str, "args": dict, "result": str} of prior steps (last 10 sent).

Two more calls share this object (and its client) but run on the SMALL FAST model (config vlm.chat_model /
vlm.verify_model), because both are on a human-visible latency path:
  chat(heard, context, overhead_jpeg, wrist_jpeg)  -> {reply, rules, hints, urgent}  (the "luna-chat" worker)
  verify_grasp(overhead_jpeg, wrist_jpeg, x_cm, y_cm) -> {aligned, dx_cm, dy_cm, reason, confidence}
Both are strict structured outputs, low reasoning effort and a short max output.

Selftest: `python -m sortbot.vlm --selftest` (mock); `--live` makes one real call with a synthetic image
(`--live-chat` times one real chat call).
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
    ("turn_to", "Set the wrist angle: rotate the wrist roll to an ABSOLUTE angle in degrees "
                "(-90..90, 0 = jaws square to the table's x axis). Use it to line the jaws up with a "
                "long or narrow object before picking it.",
     {"deg": {"type": "number"}}),
    ("turn_by", "Adjust the wrist angle: rotate the wrist roll BY <deg> degrees relative to where it is "
                "now (positive = counter-clockwise seen from above). Small nudges to square up on an "
                "object; the resulting angle must stay within -90..90.",
     {"deg": {"type": "number"}}),
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
Nothing is written on the photos except the grid, its cm tick labels and the spatial markers: the RULES,
the state and what the overlay's colours mean are all given to you as TEXT below the images. Do not look
for instructions in the picture.
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
CONVERSATION: you are NOT the one talking to the human. A separate fast chat worker ("Luna") answers them
in real time while you work, and distils what they say into the RULES and the `(human)` hints you see in the
state block. So use say() SPARINGLY -- only a final summary or a question the hints cannot answer; anything
conversational is already handled and a say() of yours only delays the next step.
GRASPING: pick_at never closes the claw blind -- it descends, checks the overhead AND wrist views that the
jaws are centred on the object, nudges itself if not, and ABORTS with "alignment not confirmed after N
tries: ..." rather than closing on nothing. Treat that failure as "I could not see it well enough there":
re-read the photos and try a corrected coordinate, or pick something else.
RULES from the human override everything else and must always be respected.
Prefer pick_at / place_at. turn_to (absolute wrist angle) and turn_by (relative nudge) are there to square
the jaws up with a long or narrow object before picking it; use move_to / open_gripper / close_gripper only
to recover from a failure."""


CHAT_SYSTEM = """You are Luna, the voice of a small tabletop robot arm that tidies objects on a table.
You are the CONVERSATION channel, not the planner: a separate thread is doing the actual sorting right now
and keeps working while you talk. Never claim to be moving the arm yourself, and never stall the human.
Answer in ONE short spoken sentence (at most about 20 words, plain text -- it is read aloud, so no markdown,
no lists, no coordinates unless asked). Be warm and concrete: say what you can actually see in the photos
and what the state block says you are doing.
You also translate the human into instructions for the planner:
  rules  -- persistent policy they want to stick ("red things go on the left", "never touch the mug").
            Write each as one short imperative sentence. Only when they clearly mean it to last.
  hints  -- one-shot nudges for the very next step ("the blue one is behind the cup", "skip that pile").
  urgent -- "stop" if they want everything to stop, "pause" if they want it to hold, else "none".
Return empty rules and hints when nothing needs to change. NEVER invent an instruction they did not give."""

CHAT_SCHEMA = {
    "type": "object",
    "properties": {"reply": {"type": "string"},
                   "rules": {"type": "array", "items": {"type": "string"}},
                   "hints": {"type": "array", "items": {"type": "string"}},
                   "urgent": {"type": "string", "enum": ["none", "stop", "pause"]}},
    "required": ["reply", "rules", "hints", "urgent"], "additionalProperties": False,
}

VERIFY_SYSTEM = """You are the grasp-alignment checker for a small tabletop robot arm. The gripper has just
descended to grasp height over an object and the jaws are still OPEN -- nothing has been grasped yet. You get
two photos: OVERHEAD (the whole table, with a grid labelled in table centimetres, x forward from the robot
base, y to its left) and WRIST (a camera on the gripper looking straight down between the jaws).
Answer exactly one question: is the centre point between the jaws directly over the object it is about to
pick up?
  aligned    -- true ONLY if closing the jaws right now would grab that object cleanly.
  dx_cm/dy_cm-- how far the GRIPPER still has to move, in table centimetres, to be centred on the object:
                dx = forward (away from the robot base), dy = to the robot's left. Use 0 for both when
                aligned. Keep it small and honest -- these are centimetres, not pixels.
  reason     -- one short clause: what you see and which way it is off.
  confidence -- 0..1. If the wrist view is dark, blurred, empty or you cannot tell which object is meant,
                answer aligned=false with LOW confidence instead of guessing. A wrong "aligned" makes the
                arm close on nothing (or on the wrong thing), which is far worse than one more check."""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"aligned": {"type": "boolean"}, "dx_cm": {"type": "number"}, "dy_cm": {"type": "number"},
                   "reason": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["aligned", "dx_cm", "dy_cm", "reason", "confidence"], "additionalProperties": False,
}

# Models that take the Responses API `reasoning` parameter (gpt-4.x and older do not).
_REASONING_FAMILIES = ("gpt-5", "o1", "o3", "o4")


def _state_text(world: WorldState, history: list, workspace_mm=None, grid_cm: float = 5.0) -> str:
    # mm -> cm for everything the VLM reads (the unit boundary; see the module docstring)
    p = world.ee_pose
    lo, hi = workspace_mm or ((120.0, -220.0, 0.0), (420.0, 220.0, 250.0))
    # What the overlay MEANS is prose, so it is said here instead of being painted over the photo:
    # nothing is written on the image except the grid, its cm tick labels and the spatial markers.
    lines = [f"OVERLAY KEY: grid lines every {grid_cm:g} cm, each labelled in cm at both ends (x.. forward, "
             f"y.. left); magenta cross = the gripper; thin outline = the calibrated camera area and a "
             f"dimmed grid outside it means positions read there are unreliable.",
             f"EE pose (table cm): x={p.x / 10:.1f} y={p.y / 10:.1f} z={p.z / 10:.1f} roll={p.roll_deg:.0f}deg",
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


def _img_jpeg(jpg: bytes, detail: str = "low") -> dict:
    """Small cached JPEG (chat/verify): the fast path never re-encodes a full-resolution PNG."""
    return {"type": "input_image", "detail": detail,
            "image_url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}


class VLM:
    """One OpenAI client, three call sites: the PLANNER (`model`, one tool call per step), the CHAT worker
    (`chat_model`, a spoken reply + directives) and the pre-grasp CHECK (`verify_model`). chat/verify share
    the small fast model because a human is waiting on both."""

    def __init__(self, model: str | None = None, client=None, chat_model: str | None = None,
                 verify_model: str | None = None, chat_effort: str = "low"):
        if model is None:
            from sortbot import config
            model = config.load().openai_model
        self.model = model
        self.chat_model = chat_model or model
        self.verify_model = verify_model or self.chat_model
        self.chat_effort = chat_effort or "low"
        self.last_latency_ms: int | None = None
        self.last_usage: dict | None = None
        self.last_cost_usd: float | None = None
        self.last_chat_latency_ms: int | None = None
        self.last_verify_latency_ms: int | None = None
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

    # ------------------------------------------------------------------ fast structured calls
    def _structured(self, model: str, system: str, content: list, schema: dict, name: str,
                    max_output_tokens: int) -> dict:
        kw: dict = {}
        if model.startswith(_REASONING_FAMILIES):  # gpt-4.x rejects `reasoning` outright
            kw["reasoning"] = {"effort": self.chat_effort}
        resp = self.client.responses.create(
            model=model, instructions=system, input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
            max_output_tokens=max_output_tokens, **kw)
        txt = (resp.output_text or "").strip()
        if not txt:
            raise RuntimeError(f"{name}: empty response (model {model!r})")
        return json.loads(txt)

    def chat(self, heard: str, context: str, overhead_jpeg: bytes | None = None,
             wrist_jpeg: bytes | None = None) -> dict:
        """ONE fast call for the conversation channel -> {reply, rules, hints, urgent}.
        Images are the LATEST CACHED small JPEGs handed in by the caller; this never captures anything."""
        content: list = []
        if overhead_jpeg:
            content += [{"type": "input_text", "text": "Overhead camera (latest):"}, _img_jpeg(overhead_jpeg)]
        if wrist_jpeg:
            content += [{"type": "input_text", "text": "Wrist camera (latest):"}, _img_jpeg(wrist_jpeg)]
        content.append({"type": "input_text", "text": f"{context}\n\nThe human just said: {heard}"})
        t0 = time.time()
        d = self._structured(self.chat_model, CHAT_SYSTEM, content, CHAT_SCHEMA, "luna_reply", 400)
        self.last_chat_latency_ms = int((time.time() - t0) * 1000)
        return {"reply": str(d.get("reply", "")).strip(),
                "rules": [str(r).strip() for r in (d.get("rules") or []) if str(r).strip()],
                "hints": [str(h).strip() for h in (d.get("hints") or []) if str(h).strip()],
                "urgent": d.get("urgent") if d.get("urgent") in ("stop", "pause") else "none"}

    def verify_grasp(self, overhead_jpeg: bytes, wrist_jpeg: bytes, x_cm: float, y_cm: float,
                     attempt: int = 1) -> dict:
        """ONE fast call: are the jaws centred on the object at table (x_cm, y_cm)?
        -> {aligned, dx_cm, dy_cm, reason, confidence}. Called with the gripper already at grasp height
        and still OPEN; the caller closes only on an aligned verdict."""
        content = [
            {"type": "input_text", "text": "OVERHEAD camera (cm grid, x forward / y left):"},
            _img_jpeg(overhead_jpeg, "high"),
            {"type": "input_text", "text": "WRIST camera (between the jaws, looking down):"},
            _img_jpeg(wrist_jpeg, "high"),
            {"type": "input_text",
             "text": (f"The gripper is at table ({x_cm:.1f}, {y_cm:.1f}) cm, lowered to grasp height, jaws "
                      f"OPEN, about to close on the object there. Check {attempt} of this grasp. Is the "
                      f"jaw centre directly over that object?")},
        ]
        t0 = time.time()
        d = self._structured(self.verify_model, VERIFY_SYSTEM, content, VERIFY_SCHEMA, "grasp_check", 400)
        self.last_verify_latency_ms = int((time.time() - t0) * 1000)
        return {"aligned": bool(d.get("aligned")), "dx_cm": float(d.get("dx_cm", 0.0)),
                "dy_cm": float(d.get("dy_cm", 0.0)), "reason": str(d.get("reason", "")).strip(),
                "confidence": max(0.0, min(1.0, float(d.get("confidence", 0.0))))}


def _synthetic_png(w: int = 320, h: int = 240) -> bytes:
    import cv2
    import numpy as np
    img = np.full((h, w, 3), 200, np.uint8)
    cv2.rectangle(img, (40, 60), (100, 120), (0, 0, 255), -1)
    cv2.rectangle(img, (200, 100), (260, 160), (255, 0, 0), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _synthetic_jpeg(w: int = 320, h: int = 240) -> bytes:
    import cv2
    import numpy as np
    img = np.full((h, w, 3), 200, np.uint8)
    cv2.rectangle(img, (40, 60), (100, 120), (0, 0, 255), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
                                          "move_to", "turn_to", "turn_by", "say", "done"}, [t["name"] for t in TOOLS]
    for gone in ("pick", "place_in_zone"):
        assert not any(t["name"] == gone for t in TOOLS), f"{gone} must be gone"
    st = _state_text(world, hist)
    assert "RULES" in st and "put red things" in st
    # the overlay legend is TEXT now, not pixels (sortbot.perception draws no legend/rules block)
    assert "OVERLAY KEY" in st and "grid lines every 5 cm" in st, st
    assert "written on the photos" in SYSTEM_PROMPT
    import sortbot.perception as _pc
    import inspect as _i
    src = _i.getsource(_pc.Overlay._static)
    assert "rule: " not in src and '_BGR["text"]' not in src, "the overlay still paints the rules on the frame"
    assert "cm" in st and "zone" not in st.lower(), st  # coordinates only; zones are gone
    low = SYSTEM_PROMPT.lower()
    for banned in ("numbered", "object id", "object list", "detected object", "detector", "zone"):
        assert banned not in low, f"detector-era phrase {banned!r} back in the system prompt"
    # the planner must be told about the conversation channel and the pre-grasp check
    for frag in ("CONVERSATION", "chat worker", "say() SPARINGLY", "alignment not confirmed"):
        assert frag in SYSTEM_PROMPT, frag
    assert CHAT_SCHEMA["required"] == ["reply", "rules", "hints", "urgent"]
    assert CHAT_SCHEMA["properties"]["urgent"]["enum"] == ["none", "stop", "pause"]
    assert VERIFY_SCHEMA["required"] == ["aligned", "dx_cm", "dy_cm", "reason", "confidence"]
    assert all(s["additionalProperties"] is False for s in (CHAT_SCHEMA, VERIFY_SCHEMA))
    v = MockVLM(chat_script=[{"reply": "hi", "rules": ["r"], "hints": [], "urgent": "none"}],
                verify_script=[{"aligned": True, "dx_cm": 0.0, "dy_cm": 0.0, "reason": "centred",
                                "confidence": 0.9}])
    assert v.chat("hello luna", "ctx")["rules"] == ["r"]
    assert v.verify_grasp(b"o", b"w", 25.0, 12.0)["aligned"] is True
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


def _live_chat() -> None:
    """One REAL chat call on the configured fast model, timed (what the human waits for)."""
    from sortbot import config
    cfg = config.load()
    vlm = VLM(cfg.openai_model, chat_model=cfg.chat_model, verify_model=cfg.verify_model,
              chat_effort=cfg.chat_effort)
    jpg = _synthetic_jpeg()
    t0 = time.time()
    d = vlm.chat("Luna, what are you up to? And from now on put the red things on the left.",
                 "phase=running step=3/40 holding=nothing task=(none) rules: (none)", jpg, jpg)
    print(f"chat model {vlm.chat_model} effort={vlm.chat_effort} "
          f"{int((time.time() - t0) * 1000)} ms (api {vlm.last_chat_latency_ms} ms)")
    print("  reply :", d["reply"])
    print("  rules :", d["rules"], " hints:", d["hints"], " urgent:", d["urgent"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--live-chat", action="store_true", help="one real chat call on vlm.chat_model, timed")
    a = ap.parse_args()
    if a.selftest or not (a.live or a.live_chat):
        _selftest()
    if a.live or a.live_chat:
        if not os.environ.get("OPENAI_API_KEY"):
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
        if os.environ.get("OPENAI_API_KEY"):
            if a.live:
                _live()
            if a.live_chat:
                _live_chat()
        else:
            print("no OPENAI_API_KEY; skipping live")
