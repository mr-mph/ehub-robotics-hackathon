"""The conversational channel: q_heard -> "luna-chat" -> q_say + q_directives -> the ACTION loop.

Covers the contract in the MULTI-QUEUE block at the top of sortbot/main.py:
  * a chat exchange produces a RULE that the action loop then obeys (and a one-shot hint with it);
  * the URGENT path (stop / pause) takes effect IMMEDIATELY, from the regex pre-filter, with NO model call
    -- a conversational reply can never delay a stop;
  * bare "open"/"close"/"home" short-circuit into q_directives and are executed by the LOOP thread (the
    conversation never touches the bus -- SORTBOT_BUS_ASSERT=1 is armed here and would raise if it did);
  * the reply is spoken with priority, pre-empting stale queued chatter, through the single TTS worker;
  * with no chat model connected the old classifier still catches everything (nothing is dropped).

`python -m sortbot.tests.test_chat` or pytest.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import socket
import tempfile
import threading
import time
import urllib.request
from argparse import Namespace
from pathlib import Path

os.environ["SORTBOT_BUS_ASSERT"] = "1"  # the chat worker must never make a bus call

from sortbot import main as m
from sortbot import testing
from sortbot.testing import MockVLM
from sortbot.voice import VoiceIO, bare_command, urgent_kind


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _mk(base):
    def get(path):
        with urllib.request.urlopen(base + path, timeout=5) as r:
            return json.load(r)

    def post(name, body=None):
        req = urllib.request.Request(
            base + "/action/" + name,
            data=json.dumps(body or {}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)

    return get, post


def _wait(fn, timeout=20.0, what=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError(f"timeout waiting for {what}")


class BoomChatVLM(MockVLM):
    """A planner double whose chat() must never be called (the urgent/bare paths run without a model)."""

    def chat(self, heard, context, overhead_jpeg=None, wrist_jpeg=None):
        raise AssertionError(
            f"the chat model was called for {heard!r} -- the regex pre-filter must handle it"
        )


def test_directive_queue() -> None:
    q = m.DirectiveQueue()
    q.put("rule", "reds on the left")
    q.put("hint", "the blue one is behind the cup")
    assert [d["text"] for d in q.peek()] == [
        "reds on the left",
        "the blue one is behind the cup",
    ]
    got = q.drain()
    assert (
        [d["kind"] for d in got] == ["rule", "hint"]
        and q.drain() == []
        and q.peek() == []
    )
    t = m.Transcript(maxlen=3)
    for i in range(5):
        t.add("you" if i % 2 == 0 else "luna", f"line {i}")
    assert [e["text"] for e in t.entries()] == [
        "line 2",
        "line 3",
        "line 4",
    ], t.entries()
    assert [e["who"] for e in t.entries()] == ["you", "luna", "you"], t.entries()
    assert t.clear() == 3 and t.entries() == []


def test_push_dedupe() -> None:
    """q_heard: push-to-talk and the Listening stream can hear the same sentence -- it must arrive once."""
    v = VoiceIO(force_text=True)
    v.push("put the red ones on the left")
    v.push("Put the red ones on the left  ")  # same sentence, different casing/spacing, moments later
    assert v.peek() == ["put the red ones on the left"], v.peek()
    v.push("and the blue ones on the right")
    assert len(v.peek()) == 2, v.peek()
    v._last_push = (v._last_push[0], 0.0)  # pretend the window has passed
    v.push("and the blue ones on the right")
    assert len(v.peek()) == 3, "a genuine repeat after the window must still get through"


def test_speak_priority() -> None:
    """q_say: ONE worker (the bot never talks over itself) and a fresh reply drops the stale backlog."""
    v = VoiceIO(force_text=True)
    v._client = object()  # pretend TTS is configured...
    v._player = "ffplay"  # ...and playable, without ever synthesizing anything
    v._speak_q = queue.Queue()
    hold = (
        threading.Event()
    )  # a live stand-in for the TTS worker, so nothing drains q_say here
    v._speak_thread = threading.Thread(target=hold.wait, daemon=True)
    v._speak_thread.start()
    try:
        v.speak("planner line one")
        v.speak("planner line two")
        assert v.pending_say() == [
            "planner line one",
            "planner line two",
        ], v.pending_say()
        v.speak("fresh answer to the human", priority=True)
        assert v.pending_say() == ["fresh answer to the human"], v.pending_say()
        assert v.last_said == "fresh answer to the human"
    finally:
        hold.set()
        v._speak_thread.join(timeout=2)


def test_chat_worker() -> None:
    logging.disable(logging.WARNING)
    port = _free_port()
    tmp = Path(tempfile.mkdtemp())
    reply = "This is a test message."
    script = [
        {
            "reply": reply,
            "rules": ["put red things on the left side of the table"],
            "hints": ["start with the red block"],
            "urgent": "none",
        }
    ]
    vlms: list = []

    def make_vlm():
        vlms.append(MockVLM(chat_script=script))
        return vlms[-1]

    args = Namespace(
        max_steps=6,
        no_voice=True,
        hud_port=port,
        rules_file=str(tmp / "rules.json"),
        config=None,
    )
    session = m.serve(args, factories=testing.session_factories(make_vlm=make_vlm))
    session.step_delay_s = 0.05
    get, post = _mk(f"http://127.0.0.1:{port}")
    try:
        assert session.chat.alive, "the luna-chat thread must be running from the start"
        assert (
            post("connect_robot")["ok"]
            and post("connect_cameras")["ok"]
            and post("connect_vlm")["ok"]
        )
        vlm = vlms[0]

        # the chat worker reads the frames the PREVIEW thread caches -- wait for the first pair
        _wait(
            lambda: session.last_overhead is not None
            and session.last_wrist is not None,
            what="cached preview frames",
        )

        # --- 1. a chat exchange -> a rule the ACTION loop then obeys -------------------------------
        assert post("start", {"paused": True})["ok"]
        _wait(lambda: get("/state")["run"]["phase"] == "paused", what="run paused")
        t0 = time.time()
        r = post("say_to_bot", {"text": "hey Luna, what are you up to right now?"})
        assert r["ok"], r
        # the worker answers WITHOUT waiting for the loop: transcript + spoken reply + queued directives
        _wait(
            lambda: any(
                e["who"] == "luna" for e in get("/state")["chat"]["transcript"]
            ),
            what="Luna replied",
        )
        answered_in = time.time() - t0
        assert (
            answered_in < 3.0
        ), f"chat reply took {answered_in:.2f}s -- the loop must not be in the path"
        st = get("/state")
        tr = st["chat"]["transcript"]
        assert [e["who"] for e in tr] == ["you", "luna"], tr
        assert tr[0]["text"].startswith("hey Luna") and tr[1]["text"] == reply, tr
        assert (
            st["chat"]["model"] == "mock"
            and st["chat"]["replies"] == 1
            and st["chat"]["alive"]
        )
        assert session.voice.last_said == reply, "the reply must go out on q_say"
        assert (
            vlm.chat_calls and vlm.chat_calls[0]["images"] == 2
        ), vlm.chat_calls  # cached overhead + wrist
        ctx = vlm.chat_calls[0]["context"]
        assert "phase=" in ctx and "holding:" in ctx and "rules in force" in ctx, ctx
        kinds = {d["kind"] for d in session.directives.peek()}
        assert kinds == {"rule", "hint"}, session.directives.peek()
        assert (
            "put red things on the left side of the table"
            in get("/state")["voice"]["queue"]
        )
        assert (
            "put red things on the left side of the table"
            not in get("/state")["rules"]["list"]
        )

        # the loop drains q_directives at its normal drain point and obeys
        assert post("step_once")["ok"]
        _wait(
            lambda: "put red things on the left side of the table"
            in get("/state")["rules"]["list"],
            what="the action loop obeyed the rule",
        )
        assert "(human) start with the red block" in get("/state")["rules"]["hints"]
        assert session.directives.peek() == [], "directives must be consumed once"
        assert Path(
            tmp / "rules.json"
        ).exists() and "put red things on the left side of the table" in json.loads(
            Path(tmp / "rules.json").read_text()
        ), "the rule must persist"

        # --- 2. URGENT: a stop takes effect immediately, with NO model call ------------------------
        session.vlm = BoomChatVLM()  # any chat() call from here on is a test failure
        assert post("resume")["ok"]
        _wait(lambda: get("/state")["run"]["phase"] == "running", what="running again")
        t0 = time.time()
        assert post("say_to_bot", {"text": "Luna, stop!"})["ok"]
        _wait(
            lambda: session.ctl.stop_ev.is_set(),
            timeout=3.0,
            what="stop_ev set by the urgent path",
        )
        assert time.time() - t0 < 1.0, "the urgent path must not wait on anything"
        assert session.ctl.pause_ev.is_set(), "a stop pauses the gate too"
        # the control events are set BEFORE the acknowledgement is spoken (that is the point), so the
        # spoken ack lands a moment later
        _wait(lambda: session.voice.last_said == "Stopping.", timeout=3.0, what="the stop acknowledged")
        _wait(lambda: "Stopping." in [e["text"] for e in get("/state")["chat"]["transcript"]],
              timeout=3.0, what="the stop in the transcript")
        _wait(
            lambda: get("/state")["run"]["phase"] in ("stopped", "done"),
            what="run stopped",
        )
        assert not session.chat.last_error, session.chat.last_error

        # --- 2b. URGENT ON AN INTERIM TRANSCRIPT: fires mid-sentence, before endpointing ---
        # (session.vlm is still BoomChatVLM: any model call here fails the test)
        session.ctl.stop_ev.clear()
        session.ctl.pause_ev.clear()
        session.chat._last_ack = None
        before = len(session.transcript.entries())
        n0 = session.chat.urgent_from_partials
        session.voice._on_partial("luna, sto")        # not recognisable yet -> nothing happens
        assert not session.ctl.stop_ev.is_set() and session.chat.urgent_from_partials == n0
        t0 = time.time()
        session.voice._on_partial("luna, stop")       # the word lands -> stop, right now
        assert session.ctl.stop_ev.is_set(), "an interim 'stop' did not fire the stop path"
        assert time.time() - t0 < 0.5, "the interim urgent path must be synchronous"
        assert session.chat.urgent_from_partials == n0 + 1
        assert session.voice.last_said == "Stopping."
        session.voice._on_partial("luna, stop right now please")  # same utterance -> fires once
        assert session.chat.urgent_from_partials == n0 + 1, "the urgent fired twice for one utterance"
        # no transcript line from the partial: the endpointed sentence adds it, in the right order
        assert len(session.transcript.entries()) == before, session.transcript.entries()[before:]
        # ... and when the endpointed sentence arrives, it is NOT acknowledged out loud a second time
        session.voice.speak("something else")
        session.chat.handle("luna, stop!")
        assert session.ctl.stop_ev.is_set()
        assert session.voice.last_said == "something else", "the stop was acknowledged twice"
        said = [e["text"] for e in session.transcript.entries()[before:]]
        assert said.count("Stopping.") == 1 and any(s.startswith("luna, stop") for s in said), said

        # a pause is urgent too, and is also regex-only
        session.ctl.stop_ev.clear()
        session.chat.handle("hold on a second")  # synchronous: no race here
        assert session.ctl.pause_ev.is_set() and not session.ctl.stop_ev.is_set()
        assert session.voice.last_said == "Pausing."

        # --- 3. bare commands never need a model, and the LOOP (not the chat thread) runs them -----
        assert post("say_to_bot", {"text": "open"})["ok"]
        _wait(
            lambda: any(
                d["kind"] == "cmd" and d["text"] == "open"
                for d in session.directives.peek()
            ),
            what="bare command queued as a directive",
        )
        assert (
            "open" in get("/state")["voice"]["queue"]
        )  # still "queued for the loop" on the page
        assert post("start")[
            "ok"
        ]  # a fresh run drains it -- under the bus lock, on the loop thread
        _wait(
            lambda: any(e["tool"] == "open" for e in get("/log")),
            what="the loop executed the bare command",
        )
        post("stop")  # may already have finished on its own
        _wait(
            lambda: get("/state")["run"]["phase"] in ("stopped", "done"),
            what="run over",
        )

        # --- 4. no chat model connected: the old classifier still catches everything ---------------
        assert post("connect_vlm", {"connect": False})["ok"]
        session.chat.handle("screws go in the right bin please")
        assert any(
            d["kind"] == "rule" for d in session.directives.peek()
        ), session.directives.peek()

        # --- 5. the conversation never touched the bus (SORTBOT_BUS_ASSERT=1 is armed) -------------
        bad = [
            e for e in get("/log") if "BUS LOCK VIOLATION" in str(e.get("result", ""))
        ]
        assert not bad, bad[:2]
        assert not session.chat.last_error, session.chat.last_error
        assert post("clear_chat")["ok"] and get("/state")["chat"]["transcript"] == []
    finally:
        session.shutdown()
        session.voice.stop()
        if session.hud is not None:
            session.hud.stop()
    assert not session.chat.alive, "the chat worker must stop with the session"
    print(
        f"chat OK: rule obeyed by the loop, urgent stop in <1s with no model call, bare command via "
        f"q_directives, priority speech, no bus access from luna-chat (reply in {answered_in * 1000:.0f} ms)"
    )


def test_chat() -> None:
    # the regex pre-filter is the thing standing between a slow model and an E-STOP: cover it here too
    assert urgent_kind("STOP") == "stop" and urgent_kind("pause please") == "pause"
    assert urgent_kind("what are you doing") == "none"
    assert bare_command("open") == "open" and bare_command("what is open") is None
    test_directive_queue()
    test_push_dedupe()
    test_speak_priority()
    test_chat_worker()


if __name__ == "__main__":
    test_chat()
