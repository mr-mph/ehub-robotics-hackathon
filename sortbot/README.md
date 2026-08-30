# sortbot

VLM-driven pick-and-place on an SO101 arm: tidy the things on the table into groups.
The app is task-agnostic — your spoken task and RULES say what goes where; with no task it groups similar items
sensibly. There are no zones: the VLM chooses coordinates.

## Architecture

```
 you speak/type ─> q_heard ─> "luna-chat" thread ─> q_say  ──> voice.py (ONE TTS worker)
 (voice.py)                   (ONE fast VLM call)  └─> q_directives ─┐   rules / hints / open|close|home / stop
                                                                     v
 overhead cam ─┐              ┌────────────── main.py ACTION loop ───┴──┐
 wrist cam  ───┴─> perception.py ─> vlm.py ─> robot.py ─> SO101
                   (cm-grid overlay, (sees the   (safety      ^
                    EE marker)       photos, one envelope, IK) └─ verify_grasp: BOTH cameras must agree
                        │             tool call)  │               the jaws are on the object before ANY close
                        └── hud.py <──────────────┘   (browser HUD, port 8765)
 calibration.py: overhead px <-> table mm (teleop-fitted H, or ArUco mat) + table_T_base -> calib/calib.json
 calibrate.py:   teleoperated calibration session (leader arm + coloured target in the gripper), HUD buttons + keys
```

**Two threads, four queues.** The action loop (perception -> one planner call -> one tool) is far too slow
to hold a conversation, so talking and acting are split. `q_heard` (everything you say) is drained by the
**`luna-chat`** thread, which answers in ONE fast call (`vlm.chat_model`) within a second or so and pushes the
spoken reply to `q_say` and any instructions it distilled to `q_directives`; the action loop drains
`q_directives` at its existing drain point and never waits on the conversation. Stop/pause-shaped speech is
caught by a **regex pre-filter before any model call**, so an E-STOP can never queue behind a chat reply. The
chat worker reads only cached state and the latest cached camera JPEGs -- it never captures, and never touches
the robot or the bus. Full contract: the MULTI-QUEUE block at the top of `main.py`.

There is NO object detector: the VLM does the seeing itself. Loop: `home()` -> capture overhead + wrist ->
composite the cached cm-grid overlay -> VLM reads the photos and emits ONE coordinate-based tool call
(`pick_at(x_cm,y_cm)`, `place_at(x_cm,y_cm)`, `say(text)`, `done`; wrist angle
`turn_to(deg)` absolute / `turn_by(deg)` relative; low-level
`move_to/open_gripper/close_gripper` for recovery) -> validate against the workspace envelope
(rejections go back to the VLM as FAILED tool results) -> execute -> repeat. Voice corrections drain at the
top of each iteration into a persistent RULES list sent with every prompt.

**Units**: the VLM-facing surface (tool args, overlay grid labels, prompt state, history) is CENTIMETERS;
everything internal (robot, config.yaml, calib.json, safety envelope) stays MILLIMETERS. The cm->mm
conversion happens in exactly one place (`main._mm_args`; `vlm._state_text` + the overlay labels are the
mm->cm half). Recalibrating is never needed after unit changes — calib.json is untouched.

**Grasping**: the claw NEVER closes on an object unchecked. `pick_at` descends with the jaws open, then
`Loop.verify_alignment` takes BOTH camera views and makes one structured call on the fast `vlm.verify_model`
-> `{aligned, dx_cm, dy_cm, reason, confidence}`. Not aligned (or aligned below `grasp.min_confidence`) ->
a correction clamped to `grasp.max_correction_cm` through the normal safety envelope, then another check, up
to `grasp.max_retries` more times. Still not aligned -> the arm retreats with the jaws still OPEN and the
planner gets `FAILED: alignment not confirmed after N tries: <reason>` to re-plan from. The low-level
`close_gripper` tool goes through the same check. Every verdict, with a side-by-side overhead|wrist
thumbnail, lands in the decision log; the last one shows in the Operate tab.

**Threading**: exactly one thread may touch the Feetech serial bus at a time — every bus access goes
through `Session.robot_lock`, and components that cannot get the lock serve cached state (see the invariant
at the top of `main.py`; `SORTBOT_BUS_ASSERT=1` arms a proxy that fails loudly on unlocked bus calls).

## Modules / owners

| file | owner | role |
|---|---|---|
| `types.py`, `config.py`, `config.yaml` | scaffold | shared contracts (do not edit) |
| `robot.py` | robot | `RobotAPI` real SO101; safety envelope, lift-translate-descend, IK sanity |
| `calibration.py` | calib | colour-target detector, homography px->mm (fitted or ArUco), `table_T_base`, calib.json I/O |
| `calibrate.py`, `calibrate_aruco.py` | calib | teleop calibration session/controller (HUD actions), legacy ArUco+Kabsch flow |
| `perception.py` | perception | overlay render (cached cm-grid layer + EE marker) + px<->mm helpers |
| `vlm.py` | vlm | planner prompt + coordinate tool schema (cm) -> `Command`; plus the two FAST structured calls on the small model: `chat()` (Luna's reply + directives) and `verify_grasp()` (both-cameras alignment) |
| `voice.py` | voice | ElevenLabs/mic in, keyboard fallback, `q_heard` + `q_say` (one TTS worker, priority pre-emption), the `urgent_kind`/`bare_command` regex pre-filter, `transcribe_bytes` for HUD push-to-talk |
| `models.py` | models | selectable OpenAI/ElevenLabs model listings (cached 5 min) + `yaml_set` config.yaml persistence |
| `hud.py` | hud | FastAPI status page + generic action registry (`register(name, fn, label, group, params, help)` -> `POST /action/{name}`; `help` is served in `GET /actions` for tooltips) |
| `main.py` | main | server-first Session (device connects, RUN/ROBOT actions, E-STOP) + the loop + the queues/`ChatWorker` |
| `testing.py` | — | test fixtures (MockRobot, MockVLM with scriptable `verify_script` / `chat_script`, SimScene, FakeRig, VirtualLeader, `session_factories()`) — not reachable from the app; imported only by `tests/*` and `--selftest` blocks |

## Frames

* **Table frame**: mm, origin = robot base projected onto tabletop, x forward, y left, z up, table z=0.
* **Base frame** (FK/IK): meters, `base_link` of the URDF. `table_T_base` (rigid, from calibration) converts.
* End effector always points straight down; only `wrist_roll` changes -- `turn_to(deg)` sets it absolutely,
  `turn_by(deg)` nudges it relatively (both clamp to -90..90). Use them to square the jaws up with a long
  or narrow object before picking it.
* Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.

## Running

```
./run.sh -m sortbot.main                                     # SERVER-FIRST (the only entry point): HUD on :8765 with nothing
                                                             #   connected; connect robot/cams/vlm in Setup, then Start
./run.sh -m sortbot.config                                   # print config (selftest)
./run.sh -m sortbot.tests.test_e2e                           # e2e test, doubles injected into the Loop (HUD on a random port)
./run.sh -m sortbot.tests.test_hud_actions                   # HTTP tests for every /action endpoint (injected doubles, random port)
./run.sh -m sortbot.calibrate                                # hardware: teleoperated target calibration -> calib/calib.json (see below)
./run.sh -m sortbot.calibrate --mode aruco                   # legacy ArUco mat + Kabsch rigid transform
./run.sh -m sortbot.calibration --check-target green         # grab one overhead frame, detect the preset -> out/target_check.png
./run.sh -m sortbot.<module> --selftest                      # per-module smoke test (calibrate: scripted no-hardware run -> temp file)
```

`main` flags: `--max-steps N` (default 40), `--hud-port P`, `--no-voice`, `--rules-file PATH` (default
`sortbot/calib/rules.json`, persistent across runs; delete it to forget rules), `--config PATH`. There is no
mock/sim mode in the app: tests inject the doubles from `sortbot/testing.py` through `Session(factories=...)`.

## HUD (browser page, everything works without the terminal)

Layout (redesigned around what an operator is trying to do): sticky header with connection dots
(robot / cams / vlm), a plain-words status banner ("Not connected -- connect the devices in Setup",
"Sorting -- step 7/30 ..."; clicking it jumps to the relevant tab), a pulsing red **MIC LIVE** chip whenever the
microphone is listening, and the red **E-STOP** (always visible, key `e`). Below: the overhead stream as the hero
(~65% width) with the wrist cam as a picture-in-picture -- **drag it anywhere on the stream, cycle its size
(18/26/38%) with the arrows button, or hide it with x / the `w` key** (a small "wrist" button brings it back;
position, size and hidden state are remembered per browser) -- and four intent tabs on the right:

* **SETUP** -- one Connect/Disconnect row per device (Robot / Cameras / Vision model, with live status and
  clear connect errors), then the calibration flow with a numbered, state-driven how-to
  (current step highlighted, live residuals, collinear-samples warning, when-to-recalibrate note).
* **OPERATE** -- task text, big Start/Pause/Resume/Stop (+ Step once / max steps), corrections (text box,
  push-to-talk, and the **Listening** toggle mapped to `mic_on`/`mic_off` -- the mic NEVER runs unless that
  toggle is on, and it is off on every start), and the rules editor.
* **TUNE** -- model dropdowns (planner / chat / grasp-check / TTS / STT / voice) and the **grasp depth** trim.
* **DEBUG** -- manual arm control (home / gripper / jog pad / go-to / torque), the decision log, and a raw
  list of every registered action.

A dismissible 3-step first-run checklist (1 Connect devices -> 2 Calibrate -> 3 Start sorting, current step
highlighted, steps link to their tabs) sits under the header and disappears once a run starts. Disabled controls
say why in one line ("needs a robot connected -- connect it in Setup") instead of being greyed out silently.
The "?" button in the tab bar reveals every control's one-line `help` text (all registered actions carry
`help=`, also served as title tooltips). Buttons show a pending spinner then a tick / inline error (no
alert()s); the bot's say() lines and safety rejections pop up as toasts. **Demo** (header button) is a
fullscreen judging view -- stream + status + last decision + rules ticker, no controls (Esc exits). Keys:
`e` = E-STOP, `p` = pause/resume, `w` = hide/show the wrist PiP, `space` = capture while calibrating. Groups still come dynamically from
`GET /actions`: known groups map onto the four tabs, and any new group/action renders as generic button rows
(unknown groups land in Debug). `GET /state` `perception.calibrated` reports whether a px->mm homography exists
for the current cams (fitted H in calib.json) -- it drives the "Not calibrated" banner and
checklist step 2. While no run is active a preview thread keeps the camera streams live.

Devices (`connect_robot` / `connect_cameras` / `connect_vlm`, RUN group; `connect=false` disconnects; connect
errors are reported in the response, never raised):

* robot: the SO101 follower arm
* cams: overhead + wrist OpenCV cameras, opened directly (not through the follower)
* vlm: the OpenAI planner (fails with a clear message if `OPENAI_API_KEY` is missing)

Any combination works, e.g. **cameras with no robot yet** for a live preview with no arm. Device changes are
refused while a run is active (Stop first) and while a calibration session is active (Finish/Cancel first) --
disconnecting the robot mid-calibration would orphan the teleop thread against a discarded robot.

RUN group: `start` / `pause` / `resume` / `stop` / `step_once` (from idle: starts paused and runs exactly one
step) / `set_max_steps(n)` / `set_task(text)` (free-text goal, e.g. "sort it however makes sense" -- prepended
to the VLM prompt as `GOAL: ...`). Runs are restartable from the page without restarting the process.
`GET /state` carries `run: {phase, step, max_steps, task, last_error, result, connected: {robot, cams, vlm}}`.

ROBOT group (all through the normal safety envelope; refused while the loop is running -- pause first):
`home`, `open_gripper`, `close_gripper`, `jog(axis: x|y|z|roll, delta)` (mm, or degrees for roll),
`goto(x, y, z)`, `set_wrist(deg)` (absolute wrist angle, -90..90; 0 = jaws square to the table x axis) and
`adjust_wrist(delta)` (turn BY delta degrees from where it is now) -- both also on the Debug tab with
±5/±15° nudges and a live readout, `set_z_trim(mm)` (grasp depth, see below), `torque_on`, and `torque_off` = **E-STOP** (always visible in the header):
`bus.disable_torque()` plus a flag; every motion raises/returns a torque error until
`torque_on`, and the loop is paused. `GET /state` carries `robot: {ee_pose, joints_deg, gripper_open, holding,
torque, z_trim_mm, grasp_z_cm}`.

### Grasp depth (the gripper stops short of the table / presses into it)

`workspace.z_trim_mm` (default 0, range ±150 mm) shifts the plane the arm works to. **Negative lowers it.**
If picks close on air just above the object, the assumed table plane is too high -- trim down; if the arm
leans on the table, trim up. One number moves the commanded grasp z, the pre-place z and the safety
envelope's z floor together (`robot.grasp_z_mm` is the single source; nothing else derives a z floor).
`workspace.z_floor_mm` (default -150 mm) is the absolute backstop against a typo, not a policy limit --
lower it if your table really is deeper. Live from the HUD **Tune > Grasp depth**: a typed field plus
-10/-2/+2/+10 mm nudges, the resulting grasp height in cm, and an amber warning past ±40 mm (large trims
are allowed, but check the arm clears the table first). `set_z_trim` persists to `config.yaml`, so it
survives a restart. A deep trim makes every grasp further from HOME: if moves start coming back as
"step of N mm exceeds max_step" the message says how much of that the trim added -- reduce the trim or
raise `workspace.max_step_mm`.

VOICE group: `say_to_bot(text)` (alias `say_to_robot`) sends a typed correction through the voice classifier --
rule-shaped sentences are persisted to RULES immediately (or by the loop while running), short bare commands
(`open`, `home`, `stop`, ...) execute at the next step, anything else becomes a one-shot `(human)` hint for the
VLM. **Push-to-talk**: hold the mic button; the page records with MediaRecorder (webm/opus) and POSTs the clip to
`transcribe(audio_b64, mime)`, which runs ElevenLabs STT (`voice.stt_model`, default `scribe_v2`) and feeds the
transcript through `say_to_bot`; the transcript is shown on the page and in `/state` (`voice.last_transcript`).
A note appears if the browser has no microphone access. `speak(text)` is a TTS test (plays on the server machine).
`GET /state` carries `voice: {mode, queue, last_transcript, tts_model, stt_model, voice_id}`.

RULES group: `add_rule(text)`, `delete_rule(i)`, `move_rule(i, dir: up|down)` and `clear_hints()` manage the
persistent RULES list (same RulesStore the voice path uses; survives restarts via `--rules-file`) and the
one-shot hints of the current run. The rules tab renders the list with per-rule up/down/delete controls;
`GET /state` carries `rules: {list, hints}`.

PERCEPTION group (overlay-only -- the VLM does the seeing, and there are no detector or zone controls):
`px_to_mm(u, v)` converts an overhead pixel to table-frame mm via the current homography (click the overhead
image to read a position off it). `GET /state` carries `perception: {calibrated}`.

LOG group: a ring buffer of the last 200 decisions/events -- every loop tool call (with a 160px overlay jpeg
thumbnail at decision time, latency and the say text), plus voice events, mode changes and
torque on/off -- served newest-first at `GET /log` as
`{i, step, t, tool, args, result, ok, say, latency_ms, thumb_b64}`. The log tab renders a scrolling list with
thumbnails; rejections / safety errors / E-STOP (`ok: false`, `FAILED:`/`rejected:`/`safety:` results) are red.
`log_clear()` empties it.

CHAT group (VOICE actions): everything you say reaches `q_heard` the same way as before; the `luna-chat`
worker answers it. `GET /state` carries `chat: {transcript, thinking, alive, replies, model,
last_latency_ms, directives}` (the Operate transcript panel) and `voice.queue` now shows both the undrained
utterances and the directives already distilled for the loop. `clear_chat()` empties the transcript.
`GET /state` `grasp: {verify, max_correction_cm, max_retries, min_confidence, verify_model, z_trim_mm,
grasp_z_cm, large_trim, last: {aligned, accepted, dx_cm, dy_cm, reason, confidence, try, tries, x_cm, y_cm}}`.

MODELS group: `get_models()` lists what is selectable -- OpenAI vision-capable ids from `models.list()`
(gpt-5 / gpt-4.1 / gpt-4o / o3 / o4 families, cached 5 min, graceful when the key is missing) and the
ElevenLabs TTS/STT model ids plus the first ~30 voices from `voices.search()`. `set_model(provider, value)`
with `provider: openai|openai_chat|openai_verify|elevenlabs_tts|elevenlabs_stt|elevenlabs_voice` hot-swaps the
live VLM/VoiceIO **and** persists to `config.yaml` (`vlm.model`, `vlm.chat_model`, `vlm.verify_model`,
`voice.tts_model`, `voice.stt_model`, `voice.elevenlabs_voice_id`)
via a minimal text edit that preserves comments. The models tab shows one dropdown per provider; beside the
OpenAI one the last VLM call's latency and a rough per-call $ estimate (from token usage and a small built-in
price table) are shown, also served as `/state` `vlm: {model, last_latency_ms, last_cost_usd, last_usage}`.

Grasp config (`config.yaml`): `grasp: {verify: true, max_correction_cm: 2.0, max_retries: 2,
min_confidence: 0.5}`. `verify: false` is possible but logs a loud multi-line warning at startup and shows
the check as OFF in the HUD -- the default is and should stay `true`.

Loop behaviour: every iteration starts with `home()`, so objects and drop points must be within `max_step_mm`
of HOME (with the default config: x 160-300 mm, |y| <= 160 mm). Rejected/unsafe commands are returned to the VLM
as `FAILED: ...` history entries, never raised. Voice input: "rule"-shaped sentences are persisted to RULES,
`stop` ends the run, `open`/`close`/`home` execute directly, anything else is passed to the VLM as a `(human) ...` hint.
What counts as "already sorted" is the VLM's call: the prompt tells it to leave things that are already where
they belong.

## Calibration (default: teleop / ball mode, no ArUco tags)

Put the calibration target (the green ball; or anything with a distinct colour) in the follower's gripper, then
`./run.sh -m sortbot.calibrate` (leader + follower + overhead cam; HUD on :8765) or press **Start calibration** in
the HUD of a running `main` with the robot connected. A background thread teleoperates the follower from the leader arm (config
`leader:`); the sorting loop pauses at its next step boundary. Then:

1. **Pick the target colour** (optional): click the target in the HUD overhead image. The server samples a tolerant
   HSV window (`ColorTarget.from_sample`, hue wraps for reds), runs the detector and draws the detection circle; a click that
   sampled a gray/brown surface or detects an implausibly large blob is rejected and the previous target kept.
   Presets: `--target green|orange|hsv:lo_h,lo_s,lo_v,hi_h,hi_s,hi_v` or `--sample U V`; default from
   `config.yaml calibration.target`. Among same-coloured blobs the roundest large one wins.
2. **Touch table** (once): rest the fingertip on the table and press *Touch table* / `t`. FK z there is the table
   plane -> `base_z_offset_mm` (the only non-identity part of `table_T_base`). Skipped = 0 (or the previous value).
3. **Capture** (`space`) at **8 or more** well-spread positions (>= 15 mm apart, `min_sample_spacing_mm`),
   with the target **resting on the table**, the **gripper pointing straight down**, and the arm **let go of
   and settled** — a capture taken while the arm is still drifting is rejected (the frame and the FK reading
   would describe different poses; this was the source of the 15-20 mm outliers seen in practice).
   Why 8: a homography has **8 degrees of freedom**, so 4-5 point pairs are threaded through *exactly*
   whatever the data says — a sub-millimetre residual there is arithmetic, not accuracy, and the fit is free
   to bend anywhere else. Each sample pairs the target's pixel centroid with the FK base xyz; after 4 the
   homography refits on every capture and the live view shows numbered sample dots, the sample hull, a
   **coverage %**, the height/tilt spread, and the **fitted cm grid** so you can see the projection against
   the real table. Samples RANSAC excluded from the fit are ringed in red with their error — ***Drop worst
   sample*** removes the worst one and refits; *Undo* (`u`) drops the last.
4. **Set table height** (`t`) — **optional**. It measures only the table height (how deep the gripper may go
   to grasp), by resting the fingertip on the tabletop. The captures give the x/y mapping and say nothing
   about height — *but if you captured them with the target on the table, Finish reads the height straight
   off them*, so you can skip this step entirely. (That is why it looks redundant: in the normal workflow it
   is.)
5. **Finish** (`enter`): RANSAC fit `H_px_to_mm`, per-point residual table, written to `calib/calib.json`
   together with `plane_z_mm`, `target`, `method: "teleop"`, `points`, `residuals_mm`, `frame_wh` and
   `saved_at`; the previous file is kept as `calib.json.bak`. Finish **refuses a fit it can tell is bad** and
   says which of these is wrong — too few usable points for 8 DOF, < 10% coverage, near-collinear samples,
   varying target height, varying gripper tilt, or samples excluded by RANSAC — press Finish again (or pass
   `force`) to save anyway. *Cancel* (`q`) writes nothing. The leader is released and the loop resumes with
   the reloaded homography.

**Fit model**: below 8 usable points the fit is a **6-DOF affine**, not an 8-DOF homography. A homography from
4 points is an exact interpolation of those 4 points' noise, which shears and converges wildly a few centimetres
away — the classic "the grid looks way off". Affine is over-determined from 4 points and cannot invent
perspective, and for a near-nadir overhead camera the true perspective term is tiny. With 8+ points the full
homography is used. `GET /state calibration.model` reports which.

**Tool point**: every xyz in the app — FK, IK, the safety envelope and the calibration pairing — refers to the
grasp point between the jaws, i.e. the URDF `gripper_frame_link` plus `workspace.tool_offset_mm`. That dummy
frame sits ~7.8 mm off the jaw centreline and ~3.7 mm behind the fingertips; since the offset rotates with
`shoulder_pan` / `wrist_roll` it lands in a different table direction at every arm pose, so leaving it in makes
picks miss by up to 8 mm *and* injects a calibration error no homography can absorb. Set it to `[0, 0, 0]` for
the raw URDF frame (and recalibrate).

The calibration **persists**: it is written only on Finish, auto-loads on startup (a summary line —
points / residuals / coverage / age — is logged and shown in the Setup tab), and nothing else ever
overwrites it. The overlay itself is a **cached static layer**: the grid, its cm tick labels, the axis arrows and the
calibration anchors are functions of the calibration and the config, not of the picture, so they are
rendered once and composited unchanged onto every frame -- the overlay only changes when the homography,
frame size, grid spacing or anchors change (i.e. when the configuration does), never frame to frame.
**Only spatial things are drawn.** There is no legend and no rules block on the image: what the grid
spacing is, what the colours mean and the RULES in force are given to the VLM as **text** (an
`OVERLAY KEY:` line plus the `RULES:` block in `vlm._state_text`), not painted over the photograph -- a
text block in the corner hid whatever was under it from the model and from you, and made the model read
its instructions off a picture of them. `render_overlay(..., rules=...)` still accepts the argument and
ignores it. The end-effector cross is the one live element. At runtime the calibration's own anchor points are drawn on the overlay (small diamonds, `c1`, `c2`, ...)
next to the sampled-area outline -- the grid is pinned at those anchors and interpolated everywhere else, so
a visible mismatch tells you immediately whether the fit is bad or you are simply far from any anchor.
Coordinates outside the sampled area (+20% margin) are refused ("outside the calibrated area — recalibrate with wider coverage"),
because the homography is only trustworthy where it was sampled.

After that the **table frame is the base frame** in xy (x fwd, y left, mm). `H` is exact for object centroids at
`plane_z_mm`; taller objects project slightly outward from the camera nadir, lower ones inward (a few mm at most).
`calibration.mode` in config: `ball` = always the fitted H, `aruco` = tags only, `auto` (default) = fitted H, with the
4 ArUco tags overriding per frame when all are visible.

**Recalibrate** whenever the overhead camera or the robot base is moved/bumped, the camera resolution changes, or
the HUD overlay grid stops lining up with the table. Old `calib.json` files without the new keys
still load (rigid `table_T_base` only, no fixed H).

Model config (`config.yaml`): `vlm.model` is the planner; `vlm.chat_model` (default `gpt-5.4-mini`) is
Luna's conversational voice and `vlm.verify_model` (default the same) the pre-grasp checker -- both are on a
human-visible latency path, so they run a small fast model at `vlm.chat_effort: low` with a short max output
and images downscaled to 512 px. All three are swappable from the HUD Tune tab.

Keys: `OPENAI_API_KEY` (and optional `ELEVENLABS_API_KEY`) in `.env` at repo root. Without the ElevenLabs key,
push-to-talk/`speak` report the missing key in their response instead of failing silently.
