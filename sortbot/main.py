"""Server-first entry point: `python -m sortbot.main` starts the HUD immediately with NOTHING connected;
the real devices are connected from the Setup tab (RUN group: connect_robot / connect_cameras / connect_vlm).

Devices (any combination; "cameras + no robot yet" gives a live preview without an arm):
  robot: the SO101 follower arm    cams: overhead + wrist OpenCV    vlm: the OpenAI planner (needs OPENAI_API_KEY)

The Loop is startable/pausable/stoppable from the page repeatedly without restarting the process.
SafetyError / rejected commands are fed back to the VLM as the tool result, never raised. torque_off is the
E-STOP: cuts motor torque (bus.disable_torque()) and pauses the loop. HUD calibration: "Start calibration"
runs immediately when idle, else at the next step boundary (loop pauses).

Tests inject doubles through Session(..., factories=...) (see sortbot/testing.py); no mock is reachable
from this CLI or the page.

MULTI-QUEUE CONVERSATION ARCHITECTURE (feature: talk to "Luna" while she keeps working)
--------------------------------------------------------------------------------------
One loop doing perception -> one planner call -> one tool cannot also hold a conversation: a reply that
costs a whole `say` tool call lands ten seconds late and a step behind. So talking and acting are split
across FOUR queues and two threads, and the action loop NEVER waits on the conversation:

  q_heard      voice.VoiceIO._q      everything the human says (mic PTT / the Listening toggle / the HUD
                                     text box / say_to_bot). Produced by voice.py, exactly as before.
                CONSUMER: the Chat worker (or, when a Loop is built without a Session -- tests, selftests
                -- the Loop's own legacy classifier path, so nothing is ever silently dropped).

  q_directives main.DirectiveQueue   what the conversation decided the ACTION loop must obey:
                                     kind "rule"  -> persisted into RulesStore (rides every planner prompt)
                                     kind "hint"  -> one-shot "(human) ..." for the next planner prompt
                                     kind "cmd"   -> a bare open/close/home the loop runs under the bus lock
                                     kind "stop"  -> end the run at the step boundary
                PRODUCER: the Chat worker.  CONSUMER: Loop.drain_inputs(), at the SAME drain point voice
                corrections used to be read -- a non-blocking drain(), never a wait.

  q_say        voice.VoiceIO._speak_q ONE TTS worker, so the bot never talks over itself. A conversational
                                     reply is queued with priority=True, which drops the stale backlog so
                                     the fresh reply wins over queued planner chatter.

  q_log        main.DecisionLog       unchanged: every decision/event for the HUD.

THREAD "luna-chat" (main.ChatWorker): drains q_heard at 20 Hz and answers immediately.
  * URGENT PATH FIRST, no LLM: voice.urgent_kind() (regex) catches stop/pause/E-STOP-shaped speech and
    sets the existing Control events THEN AND THERE. A conversational reply must never delay a stop.
    voice.bare_command() likewise short-circuits "open"/"close"/"home" straight into q_directives.
  * Everything else: ONE fast call (vlm.chat, config vlm.chat_model, low effort, short max output) with the
    LATEST CACHED overhead+wrist JPEGs (<= 512 px, published by the preview/loop threads) plus a text
    snapshot of phase/step/holding/last tool call/rules/task -> {reply, rules, hints, urgent}.
    The reply goes to q_say, the rules/hints to q_directives.
  * IT NEVER TOUCHES THE ROBOT OR THE BUS and never captures a frame of its own: it reads cached state
    only, so it cannot violate the bus invariant below (and holds no lock anything else waits on).

THREADING INVARIANT -- the Feetech serial bus: EXACTLY ONE THREAD MAY TOUCH THE MOTOR BUS AT A TIME.
Every bus access in the app goes through Session.robot_lock (shared by the Loop thread, the HUD /state
poller, the HUD ROBOT actions and the calibration teleop thread); any component that cannot get the lock
serves cached data instead of reading (see Session._robot_state). The Loop never reads the bus just to
draw the HUD: it caches the pose from its last locked robot call and hands the cached value to hud_update.
The preview thread never touches the robot at all (cameras only -- they are opened directly by the Session,
not through the follower). The one deliberate exception is torque_off (E-STOP), which jumps the queue if
the lock cannot be acquired within 1 s. Set SORTBOT_BUS_ASSERT=1 to wrap the connected robot in a proxy
that raises on any bus-touching call made without the lock held (the test suite runs with it on;
SORTBOT_BUS_ASSERT=warn logs loudly instead of raising).
"""
from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from sortbot import config as cfgmod
from sortbot import perception
from sortbot.types import Command, ExecResult, Pose, RobotAPI, WorldState
from sortbot.models import PROVIDERS, ModelRegistry, yaml_set
from sortbot.vlm import VLM
from sortbot.voice import RulesStore, VoiceIO, bare_command, classify, urgent_kind

log = logging.getLogger("sortbot.main")

HIGH_LEVEL = {"pick_at", "place_at", "done", "say"}
LOW_LEVEL = {"move_to", "open", "close", "turn_to"}

# ============================== UNIT BOUNDARY (cm <-> mm) ==============================
# The VLM surface (tool args, overlay grid labels, prompt state, history, rejection messages) is
# CENTIMETERS; everything internal (robot, config.yaml, calib.json, safety envelope) is
# MILLIMETERS. _mm_args() below is the ONLY place VLM centimeters become internal millimeters
# (sortbot.vlm._state_text and sortbot.perception's grid labels are the mm -> cm half).
_CM_KEYS = {"x_cm": "x", "y_cm": "y", "z_cm": "z"}


def _mm_args(args: dict) -> dict:
    """{'x_cm': 25.0, ...} -> {'x': 250.0, ...} (mm). Non-coordinate args pass through unchanged."""
    return {_CM_KEYS.get(k, k): (float(v) * 10.0 if k in _CM_KEYS else v) for k, v in args.items()}


class CamRig:
    """Any robot + real cameras: capture() reads OpenCV cams, everything else is the robot."""

    def __init__(self, robot: RobotAPI, cams):
        self.robot, self.cams = robot, cams

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def capture(self, name: str) -> np.ndarray:
        return self.cams.read(name)


class Homography:
    """Uniform px->mm H source: fixed (mock) or TableHomography (real: fitted calib.json H and/or ArUco tags)."""

    def __init__(self, cfg: cfgmod.Config, fixed: np.ndarray | None = None):
        self.cfg, self.fixed = cfg, fixed
        self.tracker = None if fixed is not None else __import__("sortbot.calibration", fromlist=["x"]).TableHomography(cfg)
        self._lk = threading.Lock()  # update() mutates the tracker; preview + HUD threads both call it

    @property
    def method(self) -> str:
        return "ball" if self.tracker is None else self.tracker.method

    @property
    def samples_px(self) -> np.ndarray | None:
        """The calibration's own anchor pixels (drawn on the overlay so a mismatch is diagnosable)."""
        return None if self.tracker is None else getattr(self.tracker, "samples_px", None)

    @property
    def region_px(self) -> np.ndarray | None:
        """Convex hull (overhead px) of the calibration samples: where px<->mm is trustworthy.
        None for a fixed/injected H (tests) or an old calib.json without stored points."""
        return None if self.tracker is None else getattr(self.tracker, "region_px", None)

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        if self.fixed is not None:
            return self.fixed
        with self._lk:
            return self.tracker.H if self.tracker.update(frame) else None

    def reload(self, path: Path | None = None) -> None:
        """After a calibration: pick up the new H (real) or the fitted one from `path` (mock)."""
        if self.tracker is not None:
            with self._lk:
                self.tracker.reload(path)
        elif path is not None:
            H = __import__("sortbot.calibration", fromlist=["x"]).load_calib_dict(path)["H_px_to_mm"]
            if H is not None:
                self.fixed = H


class Control:
    """Pause/resume/stop/single-step gate shared between the Loop thread and the HUD actions."""

    def __init__(self):
        self.stop_ev, self.pause_ev = threading.Event(), threading.Event()
        self._steps, self._slock = 0, threading.Lock()
        self.phase = "idle"  # idle|running|paused|done|stopped|error

    def grant_steps(self, n: int = 1) -> None:
        with self._slock:
            self._steps += n

    def gate(self) -> bool:
        """Blocks while paused (unless a step token is granted); False = stop requested."""
        while not self.stop_ev.is_set():
            if not self.pause_ev.is_set():
                self.phase = "running"
                return True
            with self._slock:
                if self._steps > 0:
                    self._steps -= 1
                    self.phase = "running"
                    return True
            self.phase = "paused"
            time.sleep(0.05)
        return False


def png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def small_jpeg(rgb: np.ndarray | None, max_w: int = 512, quality: int = 72) -> bytes | None:
    """Downscale to <= max_w and JPEG-encode. Every latency-sensitive VLM call (chat, grasp check) sends
    these, not full-resolution PNGs: the upload dominates the round trip."""
    if rgb is None:
        return None
    try:
        h, w = rgb.shape[:2]
        if w > max_w:
            rgb = cv2.resize(rgb, (max_w, max(1, round(h * max_w / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None
    except Exception as e:  # noqa: BLE001
        log.debug("small_jpeg: %s", e)
        return None


class DirectiveQueue:
    """q_directives (see the MULTI-QUEUE block): what the conversation decided the ACTION loop must obey.
    kind: "rule" (persistent) | "hint" (one-shot) | "cmd" (bare open/close/home) | "stop".
    drain() is non-blocking on purpose -- the action loop must never wait on the conversation."""

    KINDS = ("rule", "hint", "cmd", "stop")

    def __init__(self):
        self._q: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def put(self, kind: str, text: str = "") -> None:
        assert kind in self.KINDS, kind
        with self._lock:
            self._q.append({"kind": kind, "text": str(text).strip(), "t": round(time.time(), 2)})

    def drain(self) -> list[dict]:
        with self._lock:
            out, self._q = list(self._q), collections.deque()
        return out

    def peek(self) -> list[dict]:
        with self._lock:
            return list(self._q)


class Transcript:
    """Ring buffer of the conversation for the HUD panel: {t, who: 'you'|'luna', text}, oldest first."""

    def __init__(self, maxlen: int = 60):
        self._d: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._n = 0

    def add(self, who: str, text: str) -> None:
        text = str(text).strip()
        if not text:
            return
        with self._lock:
            self._n += 1
            self._d.append({"i": self._n, "who": who, "t": round(time.time(), 2), "text": text})

    def entries(self) -> list[dict]:
        with self._lock:
            return list(self._d)

    def clear(self) -> int:
        with self._lock:
            n = len(self._d)
            self._d.clear()
            return n


class ChatWorker:
    """Thread "luna-chat": the CONVERSATION channel. Drains q_heard immediately, answers in one fast LLM
    call, and turns what it learns into q_directives for the action loop. See the MULTI-QUEUE block above.

    IT NEVER TOUCHES THE ROBOT OR THE BUS, and it never captures a frame: `frames_fn` hands it the latest
    CACHED overhead/wrist JPEGs that the preview and loop threads already publish."""

    POLL_S = 0.05
    ACKS = {"stop": "Stopping.", "pause": "Pausing."}

    def __init__(self, voice: VoiceIO, directives: DirectiveQueue, transcript: Transcript,
                 vlm_fn, ctx_fn, frames_fn, urgent_fn=None, log_fn=None):
        self.voice, self.directives, self.transcript = voice, directives, transcript
        self.vlm_fn, self.ctx_fn, self.frames_fn = vlm_fn, ctx_fn, frames_fn
        self.urgent_fn, self.log_fn = urgent_fn, log_fn
        self.last_latency_ms: int | None = None
        self.last_error = ""
        self.busy = False
        self.replies = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="luna-chat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    # -- work
    def _loop(self) -> None:
        while not self._stop.is_set():
            heard = self.voice.drain()
            if not heard:
                time.sleep(self.POLL_S)
                continue
            try:
                self.handle(heard)
            except Exception as e:  # noqa: BLE001 - the conversation must never kill its own thread
                self.last_error = str(e)
                log.warning("chat worker: %s", e)

    def handle(self, heard: list[str] | str) -> dict | None:
        """One turn. Utterances that piled up are merged so the reply is to the LATEST thing said, not to
        a backlog. Returns the chat result (tests call this synchronously)."""
        lines = [heard] if isinstance(heard, str) else list(heard)
        lines = [s.strip() for s in lines if s and s.strip()]
        if not lines:
            return None
        text = " ".join(lines)
        self.transcript.add("you", text)

        # ---- URGENT PATH: regex only, no LLM. A stop must never wait for a model. ----
        kind = urgent_kind(text)
        if kind != "none":
            self._urgent(kind, text)
            return {"reply": self.ACKS[kind], "rules": [], "hints": [], "urgent": kind}

        # ---- bare "open" / "close" / "home": the loop can just do it, no model needed ----
        cmd = bare_command(text)
        if cmd is not None:
            self.directives.put("cmd", cmd)
            self._record("chat", {"heard": text}, f"command queued for the loop: {cmd}")
            return {"reply": "", "rules": [], "hints": [], "urgent": "none"}

        vlm = self.vlm_fn()
        if vlm is None or not hasattr(vlm, "chat"):  # no model connected: fall back to the old classifier
            it = classify(text)
            self.directives.put("rule" if it.kind == "rule" else "hint", it.text)
            self._record("chat", {"heard": text}, f"no chat model; kept as {it.kind}: {it.text}", ok=False)
            return None

        ov, wr = self.frames_fn()
        self.busy = True
        t0 = time.time()
        try:
            d = vlm.chat(text, self.ctx_fn(), ov, wr)
        finally:
            self.busy = False
        self.last_latency_ms = int((time.time() - t0) * 1000)
        self.replies += 1

        if d.get("urgent") in ("stop", "pause"):  # the model heard a stop the regex did not
            self._urgent(d["urgent"], text, speak=False)
        reply = str(d.get("reply", "")).strip()
        if reply:
            self.transcript.add("luna", reply)
            self.voice.speak(reply, priority=True)  # q_say: a fresh reply pre-empts stale chatter
        for r in d.get("rules") or []:
            self.directives.put("rule", r)
        for h in d.get("hints") or []:
            self.directives.put("hint", h)
        self._record("chat", {"heard": text}, f"{reply or '(no reply)'}"
                     + (f"  [rules: {d.get('rules')}]" if d.get("rules") else "")
                     + (f"  [hints: {d.get('hints')}]" if d.get("hints") else ""),
                     say=reply, latency_ms=self.last_latency_ms)
        return d

    def _urgent(self, kind: str, text: str, speak: bool = True) -> None:
        """Take effect IMMEDIATELY: set the control events, then talk about it."""
        if self.urgent_fn is not None:
            try:
                self.urgent_fn(kind)
            except Exception as e:  # noqa: BLE001
                log.warning("urgent %s: %s", kind, e)
        if kind == "stop":
            self.directives.put("stop", text)
        ack = self.ACKS[kind]
        if speak:
            self.transcript.add("luna", ack)
            self.voice.speak(ack, priority=True)
        self._record("chat", {"heard": text}, f"URGENT {kind} (regex pre-filter, no model call)",
                     ok=False, say=ack if speak else "")

    def _record(self, tool, args, result, ok=True, say="", latency_ms=None) -> None:
        if self.log_fn is not None:
            try:
                self.log_fn(tool, args, result, ok, say, latency_ms)
            except Exception as e:  # noqa: BLE001
                log.debug("chat log: %s", e)


class DecisionLog:
    """Ring buffer (last 200) of decisions/events for the HUD LOG tab, served newest-first at GET /log.
    Entries: {i, step, t, tool, args, result, ok, say, latency_ms, thumb_b64 (160px overlay jpeg)}."""

    def __init__(self, maxlen: int = 200):
        self._d: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._n = 0

    def add(self, tool: str, args: dict, result: str, ok: bool = True, step: int = 0, say: str = "",
            latency_ms: int | None = None, frame: np.ndarray | None = None) -> None:
        thumb = None
        if frame is not None:
            try:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (160, max(1, round(h * 160 / w))), interpolation=cv2.INTER_AREA)
                okj, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                if okj:
                    thumb = base64.b64encode(buf).decode()
            except Exception as e:  # noqa: BLE001
                log.debug("log thumb: %s", e)
        with self._lock:
            self._n += 1
            self._d.append({"i": self._n, "step": int(step), "t": round(time.time(), 2), "tool": tool,
                            "args": args, "result": result, "ok": bool(ok), "say": say,
                            "latency_ms": latency_ms, "thumb_b64": thumb})

    def entries(self) -> list[dict]:
        """Newest first."""
        with self._lock:
            return list(self._d)[::-1]

    def clear(self) -> int:
        with self._lock:
            n = len(self._d)
            self._d.clear()
            return n


class Loop:
    def __init__(self, cfg, robot, vlm, voice, hud, homography, rules: RulesStore, max_steps: int,
                 task: str = "", ctl: Control | None = None, lock: threading.RLock | None = None,
                 step_delay_s: float = 0.0, directives: "DirectiveQueue | None" = None):
        self.cfg, self.robot, self.vlm, self.voice, self.hud = cfg, robot, vlm, voice, hud
        self.homog, self.rules, self.max_steps = homography, rules, max_steps
        self.task, self.ctl, self.lock = task, ctl, lock or threading.RLock()
        self.step_delay_s = step_delay_s
        # q_directives: set by the Session, and then the Chat worker owns q_heard (see the MULTI-QUEUE
        # block). None (a Loop built directly, e.g. tests/selftests) = no chat worker, so this loop keeps
        # draining q_heard itself through the old classifier and nothing is dropped.
        self.directives = directives
        self.last_verdict: dict | None = None  # last pre-grasp alignment verdict (HUD + decision log)
        self.step = 0
        self.history: list[dict] = []
        self.hints: list[str] = []
        self.holding: str | None = None  # description of the last pick ("object picked at (x, y) cm"), or None
        self.placed = 0
        self.last_say = ""
        self.calib = None  # CalibController (see attach_calibration / Session)
        self._calib_requested = threading.Event()
        self.last_overhead: np.ndarray | None = None
        self.frame_sink = None  # optional callable(overhead) so the Session sees the latest raw frame
        self.dlog: DecisionLog | None = None
        self._H: np.ndarray | None = None  # homography of the current step (extrapolation guard)
        self._overlay: np.ndarray | None = None  # overlay at decision time (log thumbnails)
        self._t0: float | None = None
        self._pose_cache: Pose | None = None  # last pose read under the bus lock; hud_update NEVER reads the bus

    # ---- bus access (see the module THREADING INVARIANT) ----
    def _read_pose(self) -> Pose:
        """One bus read under the shared lock; refreshes the cached pose hud_update serves."""
        with self.lock:
            self._pose_cache = self.robot.get_ee_pose()
        return self._pose_cache

    def _refresh_pose_locked(self) -> None:
        """Caller must hold self.lock. Best-effort: a cache refresh must never mask the real error."""
        try:
            self._pose_cache = self.robot.get_ee_pose()
        except Exception as e:  # noqa: BLE001
            log.debug("pose cache refresh: %s", e)

    # ---- conversation -> action (q_directives; see the MULTI-QUEUE block at the top) ----
    def drain_inputs(self) -> bool:
        """The loop's ONE drain point for everything the human said. Non-blocking: the action loop never
        waits on the conversation. Returns False if a stop was requested."""
        keep_going = True
        for d in (self.directives.drain() if self.directives is not None else []):
            kind, text = d["kind"], d["text"]
            log.info("directive %s: %s", kind, text)
            if kind == "rule":
                self.rules.append(text)
            elif kind == "hint":
                self.hints.append(f"(human) {text}")
            elif kind == "cmd":
                self._run_bare(text)
            elif kind == "stop":
                keep_going = False
        if self.directives is None:  # no chat worker: legacy classifier path, this loop owns q_heard
            keep_going = self._drain_voice_legacy() and keep_going
        return keep_going

    def _drain_voice_legacy(self) -> bool:
        """Pre-chat-worker path, still used when a Loop is built without a Session (tests, selftests)."""
        for text in self.voice.drain():
            it = classify(text)
            log.info("voice %s: %s", it.kind, it.text)
            if it.kind == "rule":
                self.rules.append(it.text)
            elif it.kind == "action":
                w = it.text.split()[0]
                if w in ("stop", "halt", "freeze"):
                    return False
                # only short bare commands act immediately; "drop the red one in LEFT" is a hint for the VLM
                cmd = bare_command(it.text)
                if cmd is not None:
                    self._run_bare(cmd)
                else:
                    self.hints.append(f"(human) {it.text}")
            else:
                self.hints.append(f"(human) {it.text}")
        return True

    def _run_bare(self, cmd: str) -> None:
        """open / close / home, executed here on the LOOP thread -- the conversation must never touch the
        bus itself (module invariant); every access goes through the shared lock."""
        fns = {"open": lambda r: r.open_gripper(), "close": lambda r: r.close_gripper(),
               "home": lambda r: r.home()}
        fn = fns.get(cmd)
        if fn is None:
            return
        with self.lock:
            r = fn(self.robot)
            self._refresh_pose_locked()
        self.record(cmd, {}, r)
        if cmd == "open":
            self.holding = None

    def record(self, tool: str, args: dict, r: ExecResult) -> None:
        self.history.append({"tool": tool, "args": args, "result": ("ok: " if r.ok else "FAILED: ") + r.message})
        log.info("%s(%s) -> %s", tool, args, self.history[-1]["result"])
        if self.dlog is not None:
            self.dlog.add(tool, args, self.history[-1]["result"], ok=r.ok, step=self.step,
                          say=self.last_say if tool == "say" else "",
                          latency_ms=int((time.time() - self._t0) * 1000) if self._t0 else None,
                          frame=self._overlay)

    # ---- validate + execute ----
    def _extrapolation_err(self, x_mm: float, y_mm: float) -> str | None:
        """A homography is only trustworthy where it was sampled: refuse xy targets whose overhead pixel
        falls outside the calibrated sample hull (+20% margin). No hull (mock H / old calib.json) = no guard."""
        region = self.homog.region_px
        if region is None or self._H is None:
            return None
        from sortbot.calibration import in_calibrated_region
        u, v = perception.mm_to_px(self._H, [(x_mm, y_mm)])[0]
        if in_calibrated_region(region, u, v):
            return None
        return (f"({x_mm / 10:g}, {y_mm / 10:g}) cm is outside the calibrated camera area (the outlined "
                f"region on the overhead image) — positions read there are unreliable; recalibrate with "
                f"wider coverage to work there")

    def validate(self, cmd: Command, world: WorldState) -> str | None:
        """Coordinate tools are validated against the workspace AABB; rejection messages are written in
        CENTIMETERS (the VLM's units -- it must be able to react to them)."""
        a, t = cmd.args, cmd.tool
        if t not in HIGH_LEVEL | LOW_LEVEL:
            return f"unknown tool {t!r}"
        lo, hi = self.cfg.aabb_min_mm, self.cfg.aabb_max_mm
        if t in ("pick_at", "place_at", "move_to"):
            if t == "pick_at" and self.holding is not None:
                return f"already holding ({self.holding}); place it first"
            if t == "place_at" and self.holding is None:
                return "not holding anything"
            m = _mm_args(a)
            x, y = m.get("x"), m.get("y")
            if x is None or y is None:
                return "missing x_cm/y_cm"
            if not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]):
                return (f"({x / 10:g}, {y / 10:g}) cm is outside the reachable workspace "
                        f"x [{lo[0] / 10:g}, {hi[0] / 10:g}] cm, y [{lo[1] / 10:g}, {hi[1] / 10:g}] cm")
            err = self._extrapolation_err(x, y)
            if err and t != "move_to":  # xy targets must lie where the camera calibration is trustworthy
                return err
            if t == "move_to":
                from sortbot.robot import grasp_z_mm
                zlo = min(lo[2], grasp_z_mm(self.cfg))  # workspace.z_trim_mm can lower the floor below the AABB
                z = m.get("z")
                if z is None or not (zlo <= z <= hi[2]):
                    return f"z must be within [{zlo / 10:g}, {hi[2] / 10:g}] cm"
        return None

    def execute(self, cmd: Command, world: WorldState) -> ExecResult:
        a, t, r = _mm_args(cmd.args), cmd.tool, self.robot  # mm from here down (see the UNIT BOUNDARY)
        if t == "pick_at":
            res = self.pick_verified(a["x"], a["y"])
            if res.ok:
                self.holding = f"object picked at ({a['x'] / 10:g}, {a['y'] / 10:g}) cm"
                # the alignment verdict rides along so the planner (and the log) see WHY this was allowed
                note = f" -- {res.message}" if res.message and res.message != "pick" else ""
                return ExecResult(True, self.holding.replace("object picked", "picked the object") + note)
            return res
        if t == "place_at":
            res = r.place_at(a["x"], a["y"])
            if res.ok:
                self.holding, self.placed = None, self.placed + 1
                return ExecResult(True, f"placed at ({a['x'] / 10:g}, {a['y'] / 10:g}) cm")
            return res
        if t == "move_to":
            res = r.move_to(a["x"], a["y"], a["z"])
            if res.ok:
                return ExecResult(True, f"at ({a['x'] / 10:g}, {a['y'] / 10:g}, {a['z'] / 10:g}) cm")
            return res
        if t == "open":
            res = r.open_gripper()
            self.holding = None
            return res
        if t == "close":
            # the claw NEVER closes on an object unseen: even the low-level recovery tool is checked
            # against BOTH cameras first (see pick_verified / verify_alignment)
            p = self._pose_cache or self._read_pose()
            ok, msg, _, _ = self.verify_alignment(p.x, p.y, p.z, what="close")
            if not ok:
                return ExecResult(False, msg)
            return r.close_gripper()
        if t == "turn_to":
            return r.turn_to(float(a["deg"]))
        if t == "say":
            self.last_say = str(a.get("text", ""))
            self.voice.speak(self.last_say)
            return ExecResult(True, "said")
        return ExecResult(True, str(a.get("summary", "")))

    # ---- pre-grasp verification (BOTH cameras, every close) ----
    def _grasp_frames(self, x_mm: float, y_mm: float) -> tuple[np.ndarray, np.ndarray]:
        """Fresh overhead + wrist through the CAMERAS (CamRig.capture -> the Cameras object, never the
        motor bus), with the overhead annotated exactly like the planner sees it plus a marker on the
        target so the checker knows which object is meant."""
        ov, wr = self.robot.capture("overhead"), self.robot.capture("wrist")
        if self._H is not None:
            try:
                ov = perception.render_overlay(ov, self._H, self._pose_cache, self.rules.list(),
                                               calib_region_px=self.homog.region_px,
                                               calib_samples_px=self.homog.samples_px)
                u, v = perception.mm_to_px(self._H, [(x_mm, y_mm)])[0]
                u, v = int(round(float(u))), int(round(float(v)))
                cv2.drawMarker(ov, (u, v), (255, 80, 255), cv2.MARKER_CROSS, 26, 2)
                cv2.circle(ov, (u, v), 16, (255, 80, 255), 2)
            except Exception as e:  # noqa: BLE001 - a missing annotation must not block the check
                log.debug("verify annotation: %s", e)
        return ov, wr

    @staticmethod
    def _verify_thumb(ov: np.ndarray, wr: np.ndarray) -> np.ndarray | None:
        """Side-by-side overhead|wrist for the decision log, so the human can see WHY a grasp was refused."""
        try:
            h = 200
            a = cv2.resize(ov, (max(1, round(ov.shape[1] * h / ov.shape[0])), h), interpolation=cv2.INTER_AREA)
            b = cv2.resize(wr, (max(1, round(wr.shape[1] * h / wr.shape[0])), h), interpolation=cv2.INTER_AREA)
            return np.hstack([a, b])
        except Exception as e:  # noqa: BLE001
            log.debug("verify thumb: %s", e)
            return None

    def verify_alignment(self, x_mm: float, y_mm: float, z_mm: float, what: str = "pick"):
        """HARD REQUIREMENT: the gripper never closes without checking the overhead AND wrist views first.

        With the jaws open at grasp height over (x, y), take BOTH camera views and make ONE structured VLM
        call (the fast verify model) -> {aligned, dx_cm, dy_cm, reason, confidence}. Not aligned ->
        a BOUNDED correction (clamped to grasp.max_correction_cm, through the normal safety envelope) and
        re-check, up to grasp.max_retries times. Still not aligned -> DO NOT CLOSE.

        Returns (aligned, message, x_mm, y_mm) with the final xy; the message is what the planner sees.
        Caller must already hold the bus lock (this runs inside Loop.execute's locked section)."""
        cfg = self.cfg
        x, y = float(x_mm), float(y_mm)
        import sortbot.robot as _rb
        if _rb.large_trim(cfg):
            log.warning("grasp depth trim is %+.1f mm (grasp height %.1f mm) -- that is a LARGE trim; "
                        "check the arm clears the table before running",
                        _rb.z_trim_mm(cfg), _rb.grasp_z_mm(cfg))
        if not cfg.grasp_verify:
            self.last_verdict = {"aligned": None, "reason": "grasp.verify is OFF in config.yaml",
                                 "confidence": None, "tries": 0, "t": round(time.time(), 2),
                                 "x_cm": round(x / 10, 1), "y_cm": round(y / 10, 1), "what": what}
            return True, "alignment check DISABLED (grasp.verify=false)", x, y
        tries = max(1, int(cfg.grasp_max_retries) + 1)
        lo, hi = cfg.aabb_min_mm, cfg.aabb_max_mm
        last = "no verdict"
        for i in range(1, tries + 1):
            ov, wr = self._grasp_frames(x, y)
            try:
                v = self.vlm.verify_grasp(small_jpeg(ov), small_jpeg(wr), x / 10.0, y / 10.0, attempt=i)
            except Exception as e:  # noqa: BLE001 - a failed check is "not verified", never "go ahead"
                log.error("grasp check failed: %s", e)
                v = {"aligned": False, "dx_cm": 0.0, "dy_cm": 0.0, "confidence": 0.0,
                     "reason": f"alignment check errored ({e})"}
            conf, reason = float(v.get("confidence", 0.0)), str(v.get("reason", ""))
            good = bool(v.get("aligned")) and conf >= cfg.grasp_min_confidence
            verdict = {"aligned": bool(v.get("aligned")), "confidence": round(conf, 2),
                       "dx_cm": round(float(v.get("dx_cm", 0.0)), 2), "dy_cm": round(float(v.get("dy_cm", 0.0)), 2),
                       "reason": reason, "try": i, "tries": tries, "accepted": good, "what": what,
                       "x_cm": round(x / 10, 1), "y_cm": round(y / 10, 1), "t": round(time.time(), 2),
                       "latency_ms": getattr(self.vlm, "last_verify_latency_ms", None)}
            self.last_verdict = verdict
            if self.dlog is not None:
                self.dlog.add("verify_grasp", {"x_cm": verdict["x_cm"], "y_cm": verdict["y_cm"], "try": i},
                              ("aligned" if good else "NOT aligned") +
                              f" (confidence {conf:.2f}, dx {verdict['dx_cm']:+g} dy {verdict['dy_cm']:+g} cm): {reason}",
                              ok=good, step=self.step, latency_ms=verdict["latency_ms"],
                              frame=self._verify_thumb(ov, wr))
            if good:
                return True, f"alignment confirmed on check {i}/{tries} (confidence {conf:.2f}): {reason}", x, y
            last = (f"{reason or 'not aligned'} (confidence {conf:.2f}"
                    + (", low" if bool(v.get("aligned")) and conf < cfg.grasp_min_confidence else "") + ")")
            if i == tries:
                break
            # BOUNDED correction, clamped and pushed through the normal safety envelope
            mc = float(cfg.grasp_max_correction_cm)
            dx = max(-mc, min(mc, float(v.get("dx_cm", 0.0)))) * 10.0
            dy = max(-mc, min(mc, float(v.get("dy_cm", 0.0)))) * 10.0
            nx = min(max(x + dx, lo[0]), hi[0])
            ny = min(max(y + dy, lo[1]), hi[1])
            if abs(nx - x) < 0.5 and abs(ny - y) < 0.5:
                log.info("grasp check %d/%d: no usable correction, looking again", i, tries)
                continue
            mr = self.robot.move_to(nx, ny, z_mm)
            self._refresh_pose_locked()
            if not mr.ok:
                return False, (f"alignment not confirmed after {i} tries: {last}; the correction to "
                               f"({nx / 10:g}, {ny / 10:g}) cm was rejected ({mr.message})"), x, y
            log.info("grasp correction %d/%d: %+.1f, %+.1f mm -> (%.0f, %.0f)", i, tries, nx - x, ny - y, nx, ny)
            x, y = nx, ny
        return False, f"alignment not confirmed after {tries} tries: {last}", x, y

    def pick_verified(self, x_mm: float, y_mm: float) -> ExecResult:
        """robot.pick(), with the alignment check wedged between "descended" and "close". Never closes the
        claw unless BOTH cameras agree the jaws are on the object; otherwise it retreats, jaws still open,
        and returns a FAILED result the planner can re-plan from. Runs under the bus lock (Loop.execute)."""
        cfg = self.cfg
        import sortbot.robot as _rb
        if _rb.large_trim(cfg):
            log.warning("grasp depth trim is %+.1f mm (grasp height %.1f mm) -- that is a LARGE trim; "
                        "check the arm clears the table before running",
                        _rb.z_trim_mm(cfg), _rb.grasp_z_mm(cfg))
        if not cfg.grasp_verify:
            return self.robot.pick(x_mm, y_mm)
        from sortbot.robot import grasp_z_mm
        zg = grasp_z_mm(cfg)  # the ONE grasp-height source (workspace.z_trim_mm trims it)
        # 1. get there with the jaws OPEN -- nothing has been grasped yet
        for step in (lambda: self.robot.open_gripper(), lambda: self.robot.move_to(x_mm, y_mm, zg)):
            r = step()
            self._refresh_pose_locked()
            if not r.ok:
                return ExecResult(False, f"pick failed: {r.message}")
        # 2. BOTH cameras must agree the jaws are on the object (with bounded corrections)
        ok, msg, x, y = self.verify_alignment(x_mm, y_mm, zg, what="pick")
        if not ok:
            self.robot.move_to(x, y, cfg.travel_z_mm)  # retreat with the jaws still OPEN
            self._refresh_pose_locked()
            return ExecResult(False, msg)
        # 3. only now the normal grasp (open -> descend -> CLOSE -> lift) at the verified coordinate
        res = self.robot.pick(x, y)
        self._refresh_pose_locked()
        return ExecResult(True, msg) if res.ok else res

    # ---- calibration from the HUD ----
    def attach_calibration(self, ctrl, mock_out: Path | None = None) -> None:
        """Register the calibration actions; Start is deferred to the next step boundary so the teleop session never
        fights an in-flight motion. mock_out: where the mock session writes (never the real calib.json)."""
        self.calib, self._calib_out = ctrl, mock_out
        if self.hud is None:
            return
        ctrl.register(self.hud)
        self.hud.register("calib_start", self._request_calib, "Start calibration", "calibration",
                          help="Begin the teleoperated camera calibration at the next step boundary (the loop pauses).")

    def _request_calib(self) -> dict:
        if self.calib.active or self._calib_requested.is_set():
            return {"ok": False, "message": "calibration already running/requested", "data": None}
        self._calib_requested.set()
        return {"ok": True, "message": "calibration will start at the next step boundary (loop pauses)", "data": None}

    def _calib_done(self, session) -> None:
        if session.state == "fitted":
            self.homog.reload(getattr(self, "_calib_out", None))
        log.info("calibration %s: %s", session.state, session.message)

    def _maybe_calibrate(self) -> None:
        if self.calib is None or not self._calib_requested.is_set():
            return
        self._calib_requested.clear()
        r = self.calib.start()
        log.info("calibration: %s", r["message"])
        while self.calib.active:
            time.sleep(0.2)

    # ---- main ----
    def run(self) -> str:
        from sortbot.robot import SafetyError

        while self.step < self.max_steps:
            if self.ctl is not None and not self.ctl.gate():
                return "stopped"
            self.step += 1
            step = self.step
            t0 = time.time()
            self._t0 = t0
            self._maybe_calibrate()
            with self.lock:
                hr = self.robot.home()
                self._refresh_pose_locked()
            if not hr.ok:  # routine homing is not worth a history slot; failures are
                self.record("home", {}, hr)
            if not self.drain_inputs():  # before capture so voice actions cannot stale the frame
                return "stopped by voice"
            overhead, wrist = self.robot.capture("overhead"), self.robot.capture("wrist")
            self.last_overhead = overhead
            if self.frame_sink:
                self.frame_sink(overhead)
            H = self.homog.update(overhead)
            if H is None:
                log.warning("no homography (no calib.json H and ArUco tags not visible); run calibration")
                self.hud_update(overhead, wrist, step, t0, "no homography - calibrate")
                time.sleep(0.5)
                continue
            self._H = H
            rules = ([f"GOAL: {self.task}"] if self.task else []) + self.rules.list() + self.hints
            world = WorldState(self._read_pose(), self.robot.gripper_open, self.holding, rules)
            overlay = perception.render_overlay(overhead, H, world.ee_pose, rules,
                                                calib_region_px=self.homog.region_px,
                                                calib_samples_px=self.homog.samples_px)
            self._overlay = overlay
            self.hud_update(overlay, wrist, step, t0, "planning")
            try:
                cmd = self.vlm.plan_step(png(overlay), png(wrist), world, self.history,
                                         workspace_mm=(self.cfg.aabb_min_mm, self.cfg.aabb_max_mm))
            except Exception as e:  # noqa: BLE001
                log.error("VLM failed: %s", e)
                self.history.append({"tool": "vlm", "args": {}, "result": f"error {e}"})
                if self.dlog is not None:
                    self.dlog.add("vlm", {}, f"error {e}", ok=False, step=step, frame=overlay,
                                  latency_ms=int((time.time() - t0) * 1000))
                continue
            err = self.validate(cmd, world)
            if err:
                self.record(cmd.tool, cmd.args, ExecResult(False, f"rejected: {err}"))
                continue
            try:
                with self.lock:
                    try:
                        res = self.execute(cmd, world)
                    finally:
                        self._refresh_pose_locked()
            except SafetyError as e:
                res = ExecResult(False, f"safety: {e}")
            self.record(cmd.tool, dict(cmd.args), res)
            self.hud_update(overlay, wrist, step, t0, f"{cmd.tool} -> {res.message}")
            if cmd.tool == "done":
                log.info("VLM called done() at step %d: %s", step, res.message)
                return f"done (the model decided it had finished): {res.message}"
            if self.step_delay_s:
                time.sleep(self.step_delay_s)
        return (f"stopped: step budget reached ({self.max_steps} steps). Nothing failed — "
                f"raise Max steps in the Run tab and press Start to continue.")

    def hud_update(self, overlay, wrist, step, t0, status) -> None:
        if self.hud is None:
            return
        # cached pose only: hud_update must NEVER read the bus (it runs outside the locked motion sections
        # and would race the /state poller's locked read -> feetech "Port is in use!" / serial garbage)
        self.hud.update(overlay, wrist, dict(
            step=step, status=status, ee_pose=self._pose_cache, holding=self.holding,
            gripper_open=self.robot.gripper_open, last_call=self.history[-1] if self.history else None,
            say=self.last_say, rules=self.rules.list() + self.hints, latency_ms=int((time.time() - t0) * 1000)))


class BusAssertRobot:
    """Debug proxy (env SORTBOT_BUS_ASSERT): asserts the shared bus lock is held on every bus-touching
    RobotAPI call, so a future unlocked bus reader fails loudly instead of intermittently corrupting the
    Feetech serial port. SORTBOT_BUS_ASSERT=warn logs an error and continues; any other truthy value raises.
    torque_off/_set_torque are deliberately unguarded: the E-STOP is allowed to jump the lock queue."""

    BUS_METHODS = frozenset({"home", "open_gripper", "close_gripper", "move_to", "turn_to",
                             "get_ee_pose", "get_joints_deg", "pick", "place_at", "torque_on",
                             # low-level entry points too: the calibration rig calls these directly
                             "_read_joints", "_write_joints", "capture", "disconnect"})

    def __init__(self, robot, lock, strict: bool = True):
        object.__setattr__(self, "_robot", robot)
        object.__setattr__(self, "_lock", lock)
        object.__setattr__(self, "_strict", strict)

    def __getattr__(self, name):
        attr = getattr(self._robot, name)
        if name in self.BUS_METHODS and callable(attr):
            lock, strict = self._lock, self._strict

            def guarded(*a, **kw):
                if not lock._is_owned():  # noqa: SLF001 - RLock owner check, CPython guarantees it
                    msg = (f"BUS LOCK VIOLATION: {name}() called without holding the robot bus lock "
                           f"(thread {threading.current_thread().name}); see the sortbot.main invariant")
                    log.error(msg)
                    if strict:
                        raise AssertionError(msg)
                return attr(*a, **kw)

            return guarded
        return attr

    def __setattr__(self, name, value):  # forward e.g. robot.torque / robot.table_T_base
        setattr(self._robot, name, value)


def _maybe_bus_assert(robot, lock):
    flag = os.environ.get("SORTBOT_BUS_ASSERT", "").strip().lower()
    if not flag or flag in ("0", "false", "off", "no"):
        return robot
    return BusAssertRobot(robot, lock, strict=flag != "warn")


# ---- default device factories (the ONLY things the app can connect; tests override via Session(factories=...))


def _real_robot(session: "Session"):
    from sortbot.robot import SO101Robot
    return SO101Robot(session.cfg, with_cameras=False)


def _real_cams(session: "Session"):
    from sortbot.robot import Cameras
    return Cameras(session.cfg)


def _real_vlm(session: "Session"):
    from dotenv import load_dotenv
    load_dotenv(cfgmod.REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing -- put it in .env at the repo root")
    c = session.cfg
    return VLM(c.openai_model, chat_model=c.chat_model, verify_model=c.verify_model,
               chat_effort=c.chat_effort)


def _real_homography(session: "Session"):
    return Homography(session.cfg)


def _real_calib(session: "Session"):
    """-> (CalibController, out_path). out_path None = the controller writes the real calib.json."""
    from sortbot import calibrate as cal

    target = cal.ColorTarget.parse(session.cfg.calib_target)
    rig = cal.RobotRig(session.robot, lambda: session._grab_frame("overhead"))
    ctrl = cal.CalibController(session.cfg, lambda: rig, lambda: cal.open_leader(session.cfg), target,
                               session.cfg.calib_file, session._calib_done, lambda: session.last_overhead,
                               bus_lock=session.robot_lock)
    return ctrl, None


DEFAULT_FACTORIES = {"robot": _real_robot, "cams": _real_cams, "vlm": _real_vlm,
                     "homography": _real_homography, "calib": _real_calib}


class Session:
    """Owns the devices (robot / cams / vlm), one Loop thread at a time, and the RUN + ROBOT HUD actions.
    Everything is reachable over POST /action/*; nothing requires the terminal.

    `factories` is the test-injection seam: a dict overriding any of DEFAULT_FACTORIES so the test suite can
    hand in doubles (sortbot.testing.session_factories()); the app itself always uses the real devices."""

    JOG_AXES = ("x", "y", "z", "roll")

    def __init__(self, cfg: cfgmod.Config, hud, voice: VoiceIO, rules: RulesStore,
                 max_steps: int = 200, step_delay_s: float = 0.0, factories: dict | None = None):
        self.cfg, self.hud, self.voice, self.rules = cfg, hud, voice, rules
        self.factories = {**DEFAULT_FACTORIES, **(factories or {})}
        self.robot = None          # SO101Robot (RobotAPI)
        self.cams = None           # robot_mod.Cameras
        self.vlm = None            # vlm.VLM
        self.homog: Homography | None = None  # persistent px->mm source, created lazily
        self.loop: Loop | None = None
        self.ctl: Control | None = None
        self.thread: threading.Thread | None = None
        self.max_steps, self.task = max_steps, ""
        self.step_delay_s = step_delay_s
        self.last_error, self.result = "", ""
        self.robot_lock = threading.RLock()
        self._robot_cache: dict = {}
        self.calib, self._calib_out = None, None
        self.models = ModelRegistry()
        self.dlog = DecisionLog()
        self._overlay_hold: tuple[float, np.ndarray] | None = None  # (until_ts, overlay) held briefly while idle
        self._calib_flag: tuple[float, bool] | None = None  # (calib.json mtime, has fitted H) cache
        self.last_overhead: np.ndarray | None = None
        self.last_wrist: np.ndarray | None = None
        # --- conversation channel (see the MULTI-QUEUE block at the top of this module) ---
        self.directives = DirectiveQueue()
        self.transcript = Transcript()
        self.chat = ChatWorker(self.voice, self.directives, self.transcript, lambda: self.vlm,
                               self._chat_context, self._chat_frames, self._chat_urgent, self._chat_log)
        self._shutdown = threading.Event()
        import sortbot.robot as _rb
        if _rb.large_trim(cfg):
            log.warning("grasp depth trim is %+.1f mm (grasp height %.1f mm) -- that is a LARGE trim; "
                        "check the arm clears the table before running",
                        _rb.z_trim_mm(cfg), _rb.grasp_z_mm(cfg))
        if not cfg.grasp_verify:
            log.warning("!" * 78)
            log.warning("!! grasp.verify is FALSE in %s -- the claw will CLOSE WITHOUT CHECKING the",
                        cfg.source_path)
            log.warning("!! overhead and wrist views that the jaws are on the object. This is unsafe;")
            log.warning("!! set grasp.verify: true unless you are deliberately debugging the arm.")
            log.warning("!" * 78)
        self._register()
        self.chat.start()
        threading.Thread(target=self._preview_loop, daemon=True, name="preview").start()

    # ------------------------------------------------ registration
    def _register(self) -> None:
        if self.hud is None:
            return
        h = self.hud
        h.register("connect_robot", self.connect_robot, "Connect robot", "run",
                   help="Connect the SO-101 follower arm (connect=false disconnects). Torque comes on; nothing moves until you act.")
        h.register("connect_cameras", self.connect_cameras, "Connect cameras", "run",
                   help="Open the overhead + wrist cameras (connect=false disconnects). Works with no robot, for a live preview.")
        h.register("connect_vlm", self.connect_vlm, "Connect vision model", "run",
                   help="Connect the OpenAI planner (needs OPENAI_API_KEY in .env; connect=false disconnects).")
        h.register("start", self.start, "Start", "run",
                   help="Start a sorting run with the connected devices (needs robot + cams + VLM).")
        h.register("pause", self.pause, "Pause", "run",
                   help="Pause the run; the current step finishes first.")
        h.register("resume", self.resume, "Resume", "run",
                   help="Continue a paused run (refused while torque is off after an E-STOP).")
        h.register("stop", self.stop, "Stop", "run",
                   help="End the run at the next step boundary; devices stay connected for the next Start.")
        h.register("step_once", self.step_once, "Step once", "run",
                   help="Run exactly one step; from idle the run is started paused first.")
        h.register("set_max_steps", self.set_max_steps, "Set max steps", "run",
                   help="Cap the number of steps for this and future runs (n >= 1).")
        h.register("set_task", self.set_task, "Set task", "run",
                   help="Free-text goal sent to the VLM as GOAL: ... (empty = group similar things sensibly).")
        h.register("home", lambda: self._robot_act(lambda r: r.home()), "Home", "robot", params=[],
                   help="Move the arm to its HOME pose above the table.")
        h.register("open_gripper", lambda: self._robot_act(lambda r: r.open_gripper()), "Open gripper", "robot", params=[],
                   help="Open the gripper (drops whatever it is holding).")
        h.register("close_gripper", lambda: self._robot_act(lambda r: r.close_gripper()), "Close gripper", "robot", params=[],
                   help="Close the gripper.")
        h.register("jog", self.jog, "Jog", "robot",
                   help="Nudge the arm along one axis (axis: x|y|z|roll; delta in mm, degrees for roll), through the safety envelope.")
        h.register("goto", self.goto, "Go to", "robot",
                   help="Move the gripper to table-frame (x, y, z) mm, through the safety envelope.")
        h.register("torque_off", self.torque_off, "E-STOP (torque off)", "robot", params=[],
                   help="E-STOP: cut motor torque immediately and pause the run; all motion is refused until torque_on.")
        h.register("set_z_trim", self.set_z_trim, "Set grasp depth trim", "robot",
                   help="Grasp depth trim: negative lowers the plane the gripper descends to. Increase (more "
                        "negative) if it stops short of the table, decrease if it presses into it.")
        h.register("torque_on", self.torque_on, "Torque on", "robot", params=[],
                   help="Re-enable motor torque after an E-STOP.")
        h.register("say_to_bot", self.say_to_bot, "Send to bot", "voice",
                   help="Send a typed correction through the voice classifier (rule / hint / immediate command), exactly like speech.")
        h.register("say_to_robot", self.say_to_bot, None, "voice",
                   help="Alias of say_to_bot, kept for older pages/scripts.")
        h.register("transcribe", self.transcribe, None, "voice",
                   help="Push-to-talk: POST base64 audio (webm/opus and friends); ElevenLabs STT transcribes it and the text goes through say_to_bot.")
        h.register("speak", self.speak, "Speak (TTS test)", "voice",
                   help="Say the text aloud via ElevenLabs TTS with the current voice/model (plays on the server machine).")
        h.register("mic_on", lambda: self._voice_event("mic_on", self.voice.mic_on()), "Listening ON", "voice",
                   help="Start hands-free listening: the mic transcribes continuously until toggled off. A MIC LIVE chip shows while on.")
        h.register("mic_off", lambda: self._voice_event("mic_off", self.voice.mic_off()), "Listening OFF", "voice",
                   help="Stop hands-free listening. Push-to-talk and the text box keep working.")
        h.register("add_rule", self.add_rule, "Add rule", "rules",
                   help="Append a persistent rule; rules ride along with every VLM prompt and survive restarts.")
        h.register("delete_rule", self.delete_rule, None, "rules",
                   help="Delete the rule at 0-based index i (the x button in the list).")
        h.register("move_rule", self.move_rule, None, "rules",
                   help="Move rule i one slot up or down (dir: up|down); earlier rules read as higher priority.")
        h.register("clear_hints", self.clear_hints, "Clear hints", "rules",
                   help="Drop the one-shot (human) hints accumulated during the current run; persisted rules are kept.")
        h.register("get_models", self.get_models, "Refresh models", "models",
                   help="List selectable OpenAI / ElevenLabs models and voices (OpenAI listing cached 5 min).")
        h.register("set_model", self.set_model, None, "models",
                   help="Hot-swap a model (provider: openai|elevenlabs_tts|elevenlabs_stt|elevenlabs_voice) on the live clients and persist it to config.yaml.")
        h.register("px_to_mm", self.px_to_mm, None, "perception",
                   help="Convert an overhead pixel (u, v) to table-frame mm via the current homography -- click the overhead image to read a position off it.")
        h.register("log_clear", self.log_clear, "Clear log", "log", params=[],
                   help="Empty the decision log (ring buffer of the last 200 decisions/events, served at GET /log).")
        h.register("clear_chat", self.clear_chat, "Clear conversation", "voice", params=[],
                   help="Empty the conversation transcript panel (what you said / what Luna said). Rules and hints are kept.")
        h.add_state_source("run", self._run_state)
        h.add_state_source("robot", self._robot_state)
        h.add_state_source("voice", self._voice_state)
        h.add_state_source("rules", self._rules_state)
        h.add_state_source("vlm", self._vlm_state)
        h.add_state_source("perception", self._perception_state)
        h.add_state_source("chat", self._chat_state)
        h.add_state_source("grasp", self._grasp_state)
        h.add_route("/log", self.dlog.entries)
        self._register_calib_placeholders()

    def _register_calib_placeholders(self) -> None:
        if self.hud is None:
            return
        no = lambda **kw: {"ok": False, "message": "no robot connected; connect it in Setup first", "data": None}  # noqa: E731
        for name, label in (("calib_start", "Start calibration"), ("calib_touch", "Touch table"),
                            ("calib_capture", "Capture"), ("calib_undo", "Undo"),
                            ("calib_finish", "Finish"), ("calib_cancel", "Cancel"), ("calib_sample", None)):
            self.hud.register(name, no, label, "calibration", params=[],
                              help="Calibration needs a robot: connect it in Setup first.")

        def idle_state() -> dict:
            try:
                from sortbot.calibration import calib_summary_file
                loaded = calib_summary_file(self.cfg.calib_file) or "no saved calibration yet"
            except Exception:  # noqa: BLE001
                loaded = None
            return {"state": "idle", "n": 0, "message": "connect a robot first", "loaded": loaded}

        self.hud.add_state_source("calibration", idle_state)

    # ------------------------------------------------ mode / devices
    def _running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def connect_robot(self, connect: bool = True) -> dict:
        """Connect (or, with connect=false, disconnect) the follower arm."""
        return self._set_device("robot", connect)

    def connect_cameras(self, connect: bool = True) -> dict:
        return self._set_device("cams", connect)

    def connect_vlm(self, connect: bool = True) -> dict:
        return self._set_device("vlm", connect)

    def _set_device(self, key: str, connect) -> dict:
        """(Dis)connect one device via its factory; errors are reported in the response, never raised."""
        if self._running():
            return {"ok": False, "message": "stop the run before connecting or disconnecting devices", "data": None}
        calib = self.calib
        if calib is not None and calib.active:
            return {"ok": False, "message": "finish or cancel the calibration before changing devices", "data": None}
        want = str(connect).lower() not in ("false", "0", "off", "no", "")
        self._disconnect(key)
        ok, msg = True, f"{key} disconnected"
        if want:
            try:
                dev = self.factories[key](self)
                if key == "robot":
                    dev = _maybe_bus_assert(dev, self.robot_lock)  # SORTBOT_BUS_ASSERT: catch unlocked bus calls
                setattr(self, key, dev)
                if key == "robot":
                    self._attach_calib()
                msg = f"{key} connected"
            except Exception as e:  # noqa: BLE001
                ok = False
                self.last_error = msg = f"{key}: {e}"
        name = "connect_cameras" if key == "cams" else f"connect_{key}"
        self.dlog.add(name, {"connect": want}, msg, ok=ok, step=self.loop.step if self.loop is not None else 0)
        return {"ok": ok, "message": msg, "data": {"connected": self._connected()}}

    def _disconnect(self, key: str) -> None:
        dev = getattr(self, key)
        if dev is not None and hasattr(dev, "disconnect"):
            try:
                if key == "robot":  # disconnect touches the bus: don't race the /state poller's locked read
                    with self.robot_lock:
                        dev.disconnect()
                else:
                    dev.disconnect()
            except Exception as e:  # noqa: BLE001
                log.warning("%s disconnect: %s", key, e)
        setattr(self, key, None)
        if key == "robot":
            self.calib = None
            self._register_calib_placeholders()

    def _connected(self) -> dict:
        return {"robot": self.robot is not None, "cams": self.cams is not None, "vlm": self.vlm is not None}

    # ------------------------------------------------ calibration
    def _attach_calib(self) -> None:
        self.calib, self._calib_out = self.factories["calib"](self)
        if self.hud is not None:
            self.calib.register(self.hud)
            self.hud.register("calib_start", self._calib_start, "Start calibration", "calibration", params=[],
                              help="Begin the teleoperated camera calibration (put the target ball in the gripper first); "
                                   "mid-run it starts at the next step boundary.")

    def _calib_start(self) -> dict:
        if self.calib is None:
            return {"ok": False, "message": "no robot connected", "data": None}
        # start immediately only when the Loop is genuinely parked at a step boundary (phase 'paused',
        # set inside Control.gate) -- pause_ev alone is set the instant Pause is pressed, while the
        # current step (and its motion) is still in flight
        if self._running() and self.ctl is not None and self.ctl.phase != "paused":
            if self.calib.active or (self.loop and self.loop._calib_requested.is_set()):
                return {"ok": False, "message": "calibration already running/requested", "data": None}
            self.loop._calib_requested.set()
            return {"ok": True, "message": "calibration will start at the next step boundary (loop pauses)", "data": None}
        return self.calib.start()

    def _calib_done(self, session) -> None:
        if session.state == "fitted":
            if self._calib_out is not None:  # test fixture wrote a side file, never the real calib.json
                if self.homog is not None:
                    self.homog.reload(self._calib_out)
            else:  # real: the z offset may have changed too
                import sortbot.robot as robot_mod
                with self.robot_lock:  # atomic pair: a /state FK between the two writes would be torn
                    self.robot.table_T_base = robot_mod.load_calib(self.cfg.calib_file)
                    self.robot.base_T_table = np.linalg.inv(self.robot.table_T_base)
                if self.homog is not None:
                    self.homog.reload()
        log.info("calibration %s: %s", session.state, session.message)

    def _grab_frame(self, name: str) -> np.ndarray:
        if self.cams is not None:
            return self.cams.read(name)
        raise RuntimeError("no cameras connected (connect_cameras)")

    # ------------------------------------------------ conversation channel ("luna-chat")
    def _chat_context(self) -> str:
        """Cached state ONLY -- this runs on the chat thread, which never reads the bus."""
        run, loop = self._run_state(), self.loop
        lines = [f"You are Luna. phase={run['phase']} step={run['step']}/{run['max_steps']} "
                 f"devices={'+'.join(k for k, v in run['connected'].items() if v) or 'none'}",
                 f"task: {self.task or '(none given - group similar things sensibly)'}",
                 f"holding: {(loop.holding if loop is not None else None) or 'nothing'}"]
        if loop is not None and loop.history:
            h = loop.history[-1]
            lines.append(f"last thing the planner did: {h['tool']}({json.dumps(h.get('args', {}))}) -> {h['result']}")
        if loop is not None and loop.last_verdict:
            v = loop.last_verdict
            lines.append(f"last grasp check: {'aligned' if v.get('accepted') else 'NOT aligned'} - {v.get('reason', '')}")
        rules = self.rules.list()
        lines.append("rules in force: " + ("; ".join(rules) if rules else "(none)"))
        pend = self.directives.peek()
        if pend:
            lines.append("already queued for the planner: " + "; ".join(f"{d['kind']}:{d['text']}" for d in pend))
        return "\n".join(lines)

    def _chat_frames(self) -> tuple[bytes | None, bytes | None]:
        """The LATEST CACHED frames the preview/loop threads published -- never a capture of its own."""
        return small_jpeg(self.last_overhead), small_jpeg(self.last_wrist)

    def _chat_urgent(self, kind: str) -> None:
        """Urgent speech, applied IMMEDIATELY on the chat thread: only the Control events, never the bus."""
        if self.ctl is None:
            return
        self.ctl.pause_ev.set()
        if kind == "stop":
            self.ctl.stop_ev.set()

    def _chat_log(self, tool, args, result, ok=True, say="", latency_ms=None) -> None:
        self.dlog.add(tool, args, result, ok=ok, say=say, latency_ms=latency_ms,
                      step=self.loop.step if self.loop is not None else 0)

    def _chat_state(self) -> dict:
        v = self.vlm
        return {"transcript": self.transcript.entries(), "thinking": self.chat.busy,
                "alive": self.chat.alive, "replies": self.chat.replies,
                "last_latency_ms": self.chat.last_latency_ms, "last_error": self.chat.last_error,
                "model": getattr(v, "chat_model", None) if v is not None else None,
                "directives": self.directives.peek()}

    def _grasp_state(self) -> dict:
        """The last pre-grasp alignment verdict, so the page can show WHY a grasp was refused."""
        import sortbot.robot as robot_mod
        return {"verify": bool(self.cfg.grasp_verify), "max_correction_cm": self.cfg.grasp_max_correction_cm,
                "z_trim_mm": round(float(self.cfg.z_trim_mm), 2),
                "grasp_z_cm": round(robot_mod.grasp_z_mm(self.cfg) / 10.0, 2),
                "grasp_z_mm": round(robot_mod.grasp_z_mm(self.cfg), 1),
                "hard_floor_mm": round(robot_mod.hard_floor_mm(self.cfg), 1),
                "large_trim": robot_mod.large_trim(self.cfg),
                "max_retries": self.cfg.grasp_max_retries, "min_confidence": self.cfg.grasp_min_confidence,
                "verify_model": getattr(self.vlm, "verify_model", None) if self.vlm is not None else None,
                "last": self.loop.last_verdict if self.loop is not None else None}

    def clear_chat(self) -> dict:
        n = self.transcript.clear()
        return {"ok": True, "message": f"cleared {n} conversation line(s)", "data": None}

    # ------------------------------------------------ RUN actions
    def start(self, paused: bool = False) -> dict:
        if self._running():
            return {"ok": False, "message": "already running; stop first", "data": None}
        missing = [k for k, ok in self._connected().items() if not ok]
        if missing:
            return {"ok": False, "message": f"not connected: {', '.join(missing)} -- connect the devices in Setup", "data": None}
        for dev in (self.robot, self.cams, self.vlm):
            if hasattr(dev, "reset"):  # a device that supports it starts every run fresh (test doubles do)
                dev.reset()
        if self.homog is None:
            self.homog = self.factories["homography"](self)
        rig = CamRig(self.robot, self.cams)
        self.ctl = Control()
        if paused:
            self.ctl.pause_ev.set()
        self.loop = Loop(self.cfg, rig, self.vlm, self.voice, self.hud, self.homog, self.rules,
                         self.max_steps, task=self.task, ctl=self.ctl, lock=self.robot_lock,
                         step_delay_s=self.step_delay_s, directives=self.directives)
        self.loop.calib, self.loop._calib_out = self.calib, self._calib_out
        self.loop.frame_sink = lambda f: setattr(self, "last_overhead", f)
        self.loop.dlog = self.dlog
        self.result = ""
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="sortbot-loop")
        self.thread.start()
        return {"ok": True, "message": "run started" + (" (paused)" if paused else ""), "data": None}

    def _run_loop(self) -> None:
        try:
            self.result = self.loop.run()
            self.ctl.phase = "stopped" if self.ctl.stop_ev.is_set() else "done"
        except Exception as e:  # noqa: BLE001
            log.exception("loop crashed")
            self.last_error = str(e)
            self.result = f"error: {e}"
            self.ctl.phase = "error"
        log.info("finished: %s (placed %d)", self.result, self.loop.placed)

    def pause(self) -> dict:
        if not self._running():
            return {"ok": False, "message": "not running", "data": None}
        self.ctl.pause_ev.set()
        return {"ok": True, "message": "pausing (current step finishes first)", "data": None}

    def resume(self) -> dict:
        if not self._running():
            return {"ok": False, "message": "not running (use start)", "data": None}
        if self.robot is not None and not getattr(self.robot, "torque", True):
            return {"ok": False, "message": "torque is off (E-STOP); press torque_on first", "data": None}
        self.ctl.pause_ev.clear()
        return {"ok": True, "message": "resumed", "data": None}

    def stop(self) -> dict:
        if not self._running():
            return {"ok": False, "message": "not running", "data": None}
        self.ctl.stop_ev.set()
        return {"ok": True, "message": "stop requested (takes effect at the step boundary)", "data": None}

    def step_once(self) -> dict:
        if self._running():
            if not self.ctl.pause_ev.is_set():
                return {"ok": False, "message": "already running; pause first to single-step", "data": None}
            self.ctl.grant_steps(1)
            return {"ok": True, "message": "stepping once", "data": None}
        r = self.start(paused=True)
        if not r["ok"]:
            return r
        self.ctl.grant_steps(1)
        return {"ok": True, "message": "started paused; stepping once", "data": None}

    def set_max_steps(self, n: int) -> dict:
        n = int(n)
        if n < 1:
            return {"ok": False, "message": "n must be >= 1", "data": None}
        self.max_steps = n
        if self.loop is not None:
            self.loop.max_steps = n
        return {"ok": True, "message": f"max_steps = {n}", "data": None}

    def set_task(self, text: str = "") -> dict:
        self.task = str(text).strip()
        if self.loop is not None:
            self.loop.task = self.task
        return {"ok": True, "message": f"task: {self.task or '(default: sort sensibly)'}", "data": None}

    def say_to_bot(self, text: str) -> dict:
        """Same path as spoken corrections: classifier -> rule / immediate command / VLM hint."""
        text = str(text).strip()
        if not text:
            return {"ok": False, "message": "empty text", "data": None}
        it = classify(text)
        if it.kind == "rule" and not self._running():
            self.rules.append(it.text)  # nothing drains the queue while idle; persist now so the page shows it
            self.dlog.add("voice", {"text": text}, f"rule added: {it.text}",
                          step=self.loop.step if self.loop is not None else 0)
            return {"ok": True, "message": f"rule added: {it.text}", "data": {"kind": "rule"}}
        self.voice.push(text)
        kind = {"rule": "rule", "action": "command", "unknown": "hint"}[it.kind]
        when = "" if self._running() else " (queued until the next run)"
        self.dlog.add("voice", {"text": text}, f"heard as {kind}: {it.text}{when}",
                      step=self.loop.step if self.loop is not None else 0)
        return {"ok": True, "message": f"heard as {kind}: {it.text}{when}", "data": {"kind": it.kind}}

    def transcribe(self, audio_b64: str, mime: str = "audio/webm") -> dict:
        """HUD push-to-talk: base64 audio -> ElevenLabs STT -> say_to_bot."""
        try:
            data = base64.b64decode(str(audio_b64))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": f"bad base64 audio: {e}", "data": None}
        if not data:
            return {"ok": False, "message": "empty audio", "data": None}
        text = self.voice.transcribe_bytes(data, str(mime))  # raises without a key -> reported by the HUD wrapper
        if not text:
            return {"ok": False, "message": "no speech recognized", "data": None}
        r = self.say_to_bot(text)
        return {"ok": True, "message": f'"{text}" -> {r["message"]}',
                "data": {"text": text, "kind": (r.get("data") or {}).get("kind")}}

    def speak(self, text: str) -> dict:
        text = str(text).strip()
        if not text:
            return {"ok": False, "message": "empty text", "data": None}
        self.voice.speak(text)
        if self.voice._client is None:
            return {"ok": True, "message": "no ELEVENLABS_API_KEY: logged to the server console only", "data": None}
        return {"ok": True, "message": f"speaking ({self.voice.tts_model}, voice {self.voice.voice_id})", "data": None}

    # ------------------------------------------------ RULES actions
    def add_rule(self, text: str) -> dict:
        text = str(text).strip()
        if not text:
            return {"ok": False, "message": "empty rule", "data": None}
        self.rules.append(text)
        return {"ok": True, "message": f"rule added: {text}", "data": {"rules": self.rules.list()}}

    def delete_rule(self, i: int) -> dict:
        r = self.rules.delete(int(i))
        if r is None:
            return {"ok": False, "message": f"no rule at index {i}", "data": None}
        return {"ok": True, "message": f"deleted rule {i}: {r}", "data": {"rules": self.rules.list()}}

    def move_rule(self, i: int, dir: str = "up") -> dict:
        d = str(dir).lower()
        if d not in ("up", "down"):
            return {"ok": False, "message": "dir must be up|down", "data": None}
        if not self.rules.move(int(i), -1 if d == "up" else 1):
            return {"ok": False, "message": f"cannot move rule {i} {d}", "data": None}
        return {"ok": True, "message": f"moved rule {i} {d}", "data": {"rules": self.rules.list()}}

    def clear_hints(self) -> dict:
        n = len(self.loop.hints) if self.loop is not None else 0
        if self.loop is not None:
            self.loop.hints = []
        return {"ok": True, "message": f"cleared {n} hint(s)", "data": None}

    # ------------------------------------------------ MODELS actions
    def _model_current(self) -> dict:
        return {"openai": self.cfg.openai_model, "openai_chat": self.cfg.chat_model,
                "openai_verify": self.cfg.verify_model, "elevenlabs_tts": self.voice.tts_model,
                "elevenlabs_stt": self.voice.stt_model, "elevenlabs_voice": self.voice.voice_id}

    def get_models(self) -> dict:
        d = self.models.get(self._model_current())
        msg = "; ".join(d["notes"]) or f"{len(d['openai'])} openai models, {len(d['elevenlabs']['voices'])} voices"
        return {"ok": True, "message": msg, "data": d}

    def set_model(self, provider: str, value: str) -> dict:
        """Hot-swap on the live objects (VLM / VoiceIO) + persist to config.yaml (comments preserved)."""
        provider, value = str(provider).lower(), str(value).strip()
        if provider not in PROVIDERS:
            return {"ok": False, "message": f"provider must be one of {'|'.join(PROVIDERS)}", "data": None}
        if not value:
            return {"ok": False, "message": "empty value", "data": None}
        if provider == "openai":
            self.cfg.openai_model = value
            if isinstance(self.vlm, VLM):
                self.vlm.model = value
            sec, key = "vlm", "model"
        elif provider == "openai_chat":
            self.cfg.chat_model = value
            if isinstance(self.vlm, VLM):
                self.vlm.chat_model = value
            sec, key = "vlm", "chat_model"
        elif provider == "openai_verify":
            self.cfg.verify_model = value
            if isinstance(self.vlm, VLM):
                self.vlm.verify_model = value
            sec, key = "vlm", "verify_model"
        elif provider == "elevenlabs_tts":
            self.voice.tts_model = self.cfg.tts_model = value
            sec, key = "voice", "tts_model"
        elif provider == "elevenlabs_stt":
            self.voice.stt_model = self.cfg.stt_model = value
            sec, key = "voice", "stt_model"
        else:
            self.voice.voice_id = self.cfg.elevenlabs_voice_id = value
            sec, key = "voice", "elevenlabs_voice_id"
        try:
            yaml_set(self.cfg.source_path, sec, key, value)
            persisted = f"persisted to {self.cfg.source_path.name}"
        except Exception as e:  # noqa: BLE001
            persisted = f"NOT persisted: {e}"
        return {"ok": True, "message": f"{provider} = {value} (live; {persisted})",
                "data": {"current": self._model_current()}}

    # ------------------------------------------------ PERCEPTION + LOG actions
    def _current_H(self) -> np.ndarray | None:
        """Homography for the latest overhead frame (built lazily from the factory)."""
        if self.last_overhead is None:
            return None
        if self.homog is None:
            self.homog = self.factories["homography"](self)
        return self.homog.update(self.last_overhead)

    def _refresh_overlay(self) -> None:
        """Re-render the grid overlay on the latest overhead frame (no robot step); held for a few
        seconds so a config change is visible immediately while idle."""
        frame = self.last_overhead
        if frame is None:
            return
        H = self._current_H()
        if H is None:
            return
        overlay = perception.render_overlay(frame, H, None, self.rules.list(),
                                            calib_region_px=self.homog.region_px if self.homog else None,
                                            calib_samples_px=self.homog.samples_px if self.homog else None)
        self._overlay_hold = (time.time() + 5.0, overlay)
        if self.hud is not None and not self._running():
            self.hud.update(overlay, None, {})

    def px_to_mm(self, u: float, v: float) -> dict:
        H = self._current_H()
        if H is None:
            return {"ok": False, "message": "no homography (connect cameras / calibrate first)", "data": None}
        region = self.homog.region_px if self.homog is not None else None
        if region is not None:
            from sortbot.calibration import in_calibrated_region
            if not in_calibrated_region(region, float(u), float(v)):
                return {"ok": False, "message": "that point is outside the calibrated area — positions there "
                                                "are unreliable; recalibrate with wider coverage", "data": None}
        x, y = perception.px_to_mm(H, [(float(u), float(v))])[0]
        return {"ok": True, "message": f"({u}, {v}) px -> ({x:.1f}, {y:.1f}) mm",
                "data": {"x": round(float(x), 1), "y": round(float(y), 1)}}

    def log_clear(self) -> dict:
        n = self.dlog.clear()
        return {"ok": True, "message": f"cleared {n} log entries", "data": None}

    def _calibrated(self) -> bool:
        """Is a px->mm homography available for the current cams? (fixed/injected H: always; real: fitted H
        in calib.json). Drives the page's 'Not calibrated' banner + first-run checklist; cached on mtime."""
        if self.homog is not None and self.homog.fixed is not None:
            return True
        p = self.cfg.calib_file
        try:
            mt = p.stat().st_mtime
        except OSError:
            return False
        if self._calib_flag is None or self._calib_flag[0] != mt:
            try:
                from sortbot.calibration import load_calib_dict
                self._calib_flag = (mt, load_calib_dict(p)["H_px_to_mm"] is not None)
            except Exception:  # noqa: BLE001
                self._calib_flag = (mt, False)
        return self._calib_flag[1]

    def _perception_state(self) -> dict:
        return {"calibrated": self._calibrated()}

    # ------------------------------------------------ ROBOT actions
    def _robot_act(self, fn, allow_while_running: bool = False) -> dict:
        if self.robot is None:
            return {"ok": False, "message": "no robot connected (connect_robot)", "data": None}
        if not allow_while_running and self._running() and not self.ctl.pause_ev.is_set():
            return {"ok": False, "message": "loop is running; pause or stop first", "data": None}
        if not self.robot_lock.acquire(timeout=2.0):
            return {"ok": False, "message": "robot busy (finishing a motion); retry", "data": None}
        try:
            r = fn(self.robot)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": str(e), "data": None}
        finally:
            self.robot_lock.release()
        return {"ok": r.ok, "message": r.message, "data": None}

    def jog(self, axis: str, delta: float = 10.0) -> dict:
        """delta: mm for x|y|z, degrees for roll. Goes through the normal safety envelope."""
        axis = str(axis).lower()
        if axis not in self.JOG_AXES:
            return {"ok": False, "message": f"axis must be one of {'|'.join(self.JOG_AXES)}", "data": None}
        d = float(delta)

        def do(r):
            p = r.get_ee_pose()
            if axis == "roll":
                return r.turn_to(p.roll_deg + d)
            x, y, z = p.x + d * (axis == "x"), p.y + d * (axis == "y"), p.z + d * (axis == "z")
            return r.move_to(x, y, z)

        return self._robot_act(do)

    def goto(self, x: float, y: float, z: float) -> dict:
        return self._robot_act(lambda r: r.move_to(float(x), float(y), float(z)))

    def set_z_trim(self, mm: float) -> dict:
        """workspace.z_trim_mm: shift the plane the arm works to, live and persisted. Negative = lower.
        Applied to the live Config object every robot/loop call reads, so no restart is needed."""
        import sortbot.robot as robot_mod
        try:
            v = round(float(mm), 2)
        except (TypeError, ValueError):
            return {"ok": False, "message": f"mm must be a number, got {mm!r}", "data": None}
        lim = robot_mod.Z_TRIM_LIMIT_MM
        if not -lim <= v <= lim:
            return {"ok": False, "message": f"trim must be within [-{lim:g}, {lim:g}] mm", "data": None}
        self.cfg.z_trim_mm = v  # one shared Config: the robot, the Loop and the envelope all read it
        for obj in (self.robot, self.loop):
            c = getattr(obj, "cfg", None)
            if c is not None and c is not self.cfg:
                c.z_trim_mm = v
        try:
            yaml_set(self.cfg.source_path, "workspace", "z_trim_mm", f"{v:g}")
            persisted = f"persisted to {self.cfg.source_path.name}"
        except Exception as e:  # noqa: BLE001
            persisted = f"NOT persisted: {e}"
        zg = robot_mod.grasp_z_mm(self.cfg)
        warn = (f" LARGE TRIM ({v:+g} mm): check the arm clears the table before running."
                if robot_mod.large_trim(self.cfg) else "")
        if warn:
            log.warning("grasp depth trim %+g mm is large -- check the arm clears the table", v)
        self.dlog.add("set_z_trim", {"mm": v}, f"grasp depth trim {v:+g} mm -> grasp height {zg:.1f} mm{warn}",
                      ok=not warn, step=self.loop.step if self.loop is not None else 0)
        return {"ok": True, "message": f"grasp depth trim {v:+g} mm -> the gripper now descends to "
                                       f"{zg / 10:.2f} cm ({zg:.1f} mm); {persisted}.{warn}",
                "data": {"z_trim_mm": v, "grasp_z_mm": round(zg, 1), "grasp_z_cm": round(zg / 10.0, 2),
                         "large_trim": bool(warn), "hard_floor_mm": round(robot_mod.hard_floor_mm(self.cfg), 1)}}

    def torque_off(self) -> dict:
        """E-STOP: flag first (any in-flight motion aborts within one tick), pause the loop, then cut bus torque."""
        if self.robot is None:
            return {"ok": False, "message": "no robot connected", "data": None}
        self.robot.torque = False
        if self.ctl is not None:
            self.ctl.pause_ev.set()
        got = self.robot_lock.acquire(timeout=1.0)
        try:
            self.robot._set_torque(False)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": f"torque off (flag set) but bus call failed: {e}", "data": None}
        finally:
            if got:
                self.robot_lock.release()
        self.dlog.add("torque_off", {}, "TORQUE OFF (E-STOP); loop paused", ok=False,  # ok=False -> red in the log
                      step=self.loop.step if self.loop is not None else 0)
        return {"ok": True, "message": "TORQUE OFF (E-STOP); loop paused", "data": None}

    def torque_on(self) -> dict:
        r = self._robot_act(lambda r: r.torque_on())
        if r["ok"]:
            self.dlog.add("torque_on", {}, "torque on", step=self.loop.step if self.loop is not None else 0)
        return r

    def _voice_event(self, tool: str, msg: str) -> dict:
        self.dlog.add(tool, {}, msg, step=self.loop.step if self.loop is not None else 0)
        return {"ok": True, "message": msg}

    # ------------------------------------------------ state
    def _run_state(self) -> dict:
        return {"phase": self.ctl.phase if self.ctl is not None else "idle",
                "step": self.loop.step if self.loop is not None else 0,
                "max_steps": self.loop.max_steps if self.loop is not None else self.max_steps,
                "task": self.task, "last_error": self.last_error, "result": self.result,
                "connected": self._connected()}

    def _voice_state(self) -> dict:
        v = self.voice
        # "queue" = everything heard but not yet acted on by the loop: q_heard not yet taken by the chat
        # worker, plus the q_directives it has already distilled and handed on.
        return {"mode": v.mode, "listening": v.listening,
                "queue": v.peek() + [d["text"] for d in self.directives.peek() if d["text"]],
                "say_queue": v.pending_say(), "last_said": v.last_said,
                "last_transcript": v.last_transcript,
                "tts_model": v.tts_model, "stt_model": v.stt_model, "voice_id": v.voice_id}

    def _rules_state(self) -> dict:
        return {"list": self.rules.list(), "hints": list(self.loop.hints) if self.loop is not None else []}

    def _vlm_state(self) -> dict | None:
        v = self.vlm
        if v is None:
            return None
        return {"model": getattr(v, "model", "?"), "last_latency_ms": getattr(v, "last_latency_ms", None),
                "last_cost_usd": getattr(v, "last_cost_usd", None), "last_usage": getattr(v, "last_usage", None)}

    def _robot_state(self) -> dict | None:
        r = self.robot
        if r is None:
            return None
        # While a calibration teleop session owns the bus, never sync-read the serial port from here
        # (feetech: "Port is in use!"); serve the cached last pose instead.
        calib = self.calib  # local: _disconnect can null the attribute between the two reads
        calib_busy = calib is not None and calib.active
        if not calib_busy and self.robot_lock.acquire(timeout=0.2):
            try:
                p = r.get_ee_pose()
                self._robot_cache = {"ee_pose": {"x": round(p.x, 1), "y": round(p.y, 1), "z": round(p.z, 1),
                                                "roll_deg": round(p.roll_deg, 1)},
                                     "joints_deg": [round(float(v), 1) for v in r.get_joints_deg()]}
            except Exception as e:  # noqa: BLE001
                self._robot_cache.setdefault("ee_pose", None)
                self._robot_cache["error"] = str(e)
            finally:
                self.robot_lock.release()
        return {**self._robot_cache, "gripper_open": r.gripper_open, "torque": bool(getattr(r, "torque", True)),
                "holding": self.loop.holding if self.loop is not None else None}

    # ------------------------------------------------ idle preview + teardown
    def _preview_loop(self) -> None:
        """Streams camera frames to the HUD continuously -- while idle AND mid-run -- so the page never
        shows a stale image (the Loop only pushes a few frames per step, and a VLM call can take tens of
        seconds). Only an active calibration session owns the stream (it pushes annotated frames itself).
        This thread NEVER touches the robot: cameras only (they are opened directly by the Session, not
        through the follower -- the serial bus belongs to the lock holders, see the module invariant)."""
        while not self._shutdown.is_set():
            try:
                calib = self.calib  # local: the attribute can be nulled by a disconnect mid-check
                calib_busy = calib is not None and calib.active
                if self.hud is not None and not calib_busy and self.cams is not None:
                    ov, wr = self.cams.read("overhead"), self.cams.read("wrist")
                    self.last_overhead, self.last_wrist = ov, wr
                    hold = self._overlay_hold
                    if hold is not None and time.time() < hold[0]:
                        show = hold[1]
                    else:
                        show = ov
                        # draw the grid overlay on the live frame; mid-run reuse the Loop's homography
                        # and cached pose (plain attribute reads -- no tracker mutation, no bus)
                        if self._running() and self.loop is not None:
                            H, pose = self.loop._H, self.loop._pose_cache
                        else:
                            H, pose = self._current_H(), None
                        if H is not None:
                            try:
                                show = perception.render_overlay(
                                    ov, H, pose, self.rules.list(),
                                    calib_region_px=self.homog.region_px if self.homog is not None else None,
                                    calib_samples_px=self.homog.samples_px if self.homog is not None else None)
                            except Exception as e:  # noqa: BLE001
                                log.debug("preview overlay: %s", e)
                    self.hud.update(show, wr, {})
            except Exception as e:  # noqa: BLE001
                log.debug("preview: %s", e)
            time.sleep(0.4)

    def shutdown(self) -> None:
        self._shutdown.set()
        self.chat.stop()
        if self.ctl is not None:
            self.ctl.stop_ev.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        for key, dev in (("cams", self.cams), ("robot", self.robot)):
            if dev is not None and hasattr(dev, "disconnect"):
                try:
                    if key == "robot":  # bus toucher: serialize with any straggling /state read
                        with self.robot_lock:
                            dev.disconnect()
                    else:
                        dev.disconnect()
                except Exception:  # noqa: BLE001
                    pass


def serve(args, factories: dict | None = None) -> Session:
    """Build the HUD + Session (used by main() and the HTTP tests; factories is the test-injection seam)."""
    cfg = cfgmod.load(args.config) if args.config else cfgmod.load()
    from sortbot.hud import HUD
    hud = HUD(port=args.hud_port or cfg.hud_port)
    hud.start()
    voice = VoiceIO(cfg.elevenlabs_voice_id, stdin=io.StringIO("") if args.no_voice else None,
                    force_text=True, tts_model=cfg.tts_model, stt_model=cfg.stt_model)  # mic only via explicit mic_on toggle
    voice.start()
    rules = RulesStore(args.rules_file) if args.rules_file else RulesStore()
    return Session(cfg, hud, voice, rules, max_steps=args.max_steps, factories=factories)


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--no-voice", action="store_true", help="do not read stdin for corrections")
    ap.add_argument("--hud-port", type=int, default=0, help="0 = the port from config.yaml (default 8765)")
    ap.add_argument("--rules-file", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    session = serve(args)
    try:
        while True:  # server-first: keep serving; everything happens from the page
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        session.shutdown()
        session.voice.stop()
        if session.hud is not None:
            session.hud.stop()
    log.info("finished: %s", session.result)
    return session.result


if __name__ == "__main__":
    main()
