# sortbot

VLM-driven pick-and-sort on an SO101 arm: sort the things on the table into the LEFT / MIDDLE / RIGHT zones.
The app is task-agnostic — your spoken task and RULES say what goes where; with no task it groups similar items sensibly.

## Architecture

```
 voice.py ──(corrections -> RULES)──┐
                                    v
 overhead cam ─┐              ┌── main.py loop ──┐
 wrist cam  ───┴─> perception.py ─> vlm.py ─> robot.py ─> SO101
                   (cm-grid overlay, (sees the   (safety
                    zones, EE)        photos, one envelope, IK)
                        │             tool call)  │
                        └── hud.py <──────────────┘   (browser HUD, port 8765)
 calibration.py: overhead px <-> table mm (teleop-fitted H, or ArUco mat) + table_T_base -> calib/calib.json
 calibrate.py:   teleoperated calibration session (leader arm + coloured target in the gripper), HUD buttons + keys
```

There is NO object detector: the VLM does the seeing itself. Loop: `home()` -> capture overhead + wrist ->
render the cm-labelled grid overlay -> VLM reads the photos and emits ONE coordinate-based tool call
(`pick_at(x_cm,y_cm)`, `place_in_zone(zone)`, `place_at(x_cm,y_cm)`, `say(text)`, `done`; low-level
`move_to/open_gripper/close_gripper/turn_to` for recovery) -> validate against the workspace envelope
(rejections go back to the VLM as FAILED tool results) -> execute -> repeat. Voice corrections drain at the
top of each iteration into a persistent RULES list sent with every prompt.

**Units**: the VLM-facing surface (tool args, overlay grid labels, prompt state, history) is CENTIMETERS;
everything internal (robot, config.yaml, calib.json, safety envelope, zones) stays MILLIMETERS. The cm->mm
conversion happens in exactly one place (`main._mm_args`; `vlm._state_text` + the overlay labels are the
mm->cm half). Recalibrating is never needed after unit changes — calib.json is untouched.

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
| `perception.py` | perception | overlay render (cm-labelled grid, zones, EE marker) + px<->mm helpers |
| `vlm.py` | vlm | prompt + coordinate tool schema (cm), OpenAI call -> `Command` |
| `voice.py` | voice | ElevenLabs/mic in, keyboard fallback, queue of corrections, `transcribe_bytes` for HUD push-to-talk |
| `models.py` | models | selectable OpenAI/ElevenLabs model listings (cached 5 min) + `yaml_set` config.yaml persistence |
| `hud.py` | hud | FastAPI status page + generic action registry (`register(name, fn, label, group, params, help)` -> `POST /action/{name}`; `help` is served in `GET /actions` for tooltips) |
| `main.py` | main | server-first Session (device connects, RUN/ROBOT actions, E-STOP) + the loop |
| `testing.py` | — | test fixtures (MockRobot, MockVLM, SimScene, FakeRig, VirtualLeader, `session_factories()`) — not reachable from the app; imported only by `tests/*` and `--selftest` blocks |

## Frames

* **Table frame**: mm, origin = robot base projected onto tabletop, x forward, y left, z up, table z=0.
* **Base frame** (FK/IK): meters, `base_link` of the URDF. `table_T_base` (rigid, from calibration) converts.
* End effector always points straight down; only `wrist_roll` changes via `turn_to(deg)`.
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
(~65% width) with the wrist cam picture-in-picture on its corner, and four intent tabs on the right:

* **SETUP** -- one Connect/Disconnect row per device (Robot / Cameras / Vision model, with live status and
  clear connect errors), then the calibration flow with a numbered, state-driven how-to
  (current step highlighted, live residuals, collinear-samples warning, when-to-recalibrate note).
* **OPERATE** -- task text, big Start/Pause/Resume/Stop (+ Step once / max steps), corrections (text box,
  push-to-talk, and the **Listening** toggle mapped to `mic_on`/`mic_off` -- the mic NEVER runs unless that
  toggle is on, and it is off on every start), and the rules editor.
* **TUNE** -- model dropdowns and zone drop points.
* **DEBUG** -- manual arm control (home / gripper / jog pad / go-to / torque), the decision log, and a raw
  list of every registered action.

A dismissible 3-step first-run checklist (1 Connect devices -> 2 Calibrate -> 3 Start sorting, current step
highlighted, steps link to their tabs) sits under the header and disappears once a run starts. Disabled controls
say why in one line ("needs a robot connected -- connect it in Setup") instead of being greyed out silently.
The "?" button in the tab bar reveals every control's one-line `help` text (all registered actions carry
`help=`, also served as title tooltips). Buttons show a pending spinner then a tick / inline error (no
alert()s); the bot's say() lines and safety rejections pop up as toasts. **Demo** (header button) is a
fullscreen judging view -- stream + status + last decision + rules ticker, no controls (Esc exits). Keys:
`e` = E-STOP, `p` = pause/resume, `space` = capture while calibrating. Groups still come dynamically from
`GET /actions`: known groups map onto the four tabs, and any new group/action renders as generic button rows
(unknown groups land in Debug). `GET /state` `perception.calibrated` reports whether a px->mm homography exists
for the current cams (fitted H in calib.json) -- it drives the "Not calibrated" banner and
checklist step 2. While no run is active a preview thread keeps the camera streams live.

Devices (`connect_robot` / `connect_cameras` / `connect_vlm`, RUN group; `connect=false` disconnects; connect
errors are reported in the response, never raised):

* robot: the SO101 follower arm
* cams: overhead + wrist OpenCV cameras, opened directly (not through the follower)
* vlm: the OpenAI planner (fails with a clear message if `OPENAI_API_KEY` is missing)

Any combination works, e.g. **cameras with no robot yet** to tune perception with no arm. Device changes are
refused while a run is active (Stop first) and while a calibration session is active (Finish/Cancel first) --
disconnecting the robot mid-calibration would orphan the teleop thread against a discarded robot.

RUN group: `start` / `pause` / `resume` / `stop` / `step_once` (from idle: starts paused and runs exactly one
step) / `set_max_steps(n)` / `set_task(text)` (free-text goal, e.g. "sort it however makes sense" -- prepended
to the VLM prompt as `GOAL: ...`). Runs are restartable from the page without restarting the process.
`GET /state` carries `run: {phase, step, max_steps, task, last_error, result, connected: {robot, cams, vlm}}`.

ROBOT group (all through the normal safety envelope; refused while the loop is running -- pause first):
`home`, `open_gripper`, `close_gripper`, `jog(axis: x|y|z|roll, delta)` (mm, or degrees for roll),
`goto(x, y, z)`, `torque_on`, and `torque_off` = **E-STOP** (always visible in the header):
`bus.disable_torque()` plus a flag; every motion raises/returns a torque error until
`torque_on`, and the loop is paused. `GET /state` carries `robot: {ee_pose, joints_deg, gripper_open, holding,
torque}`.

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

PERCEPTION group (overlay-only -- the VLM does the seeing, there are no detector controls):
`set_zone_drop(name, x, y)` moves a zone's drop point (table mm, validated against the workspace) and persists it
into `config.yaml zones` (rect untouched, comments preserved); on the page press a zone's **set drop** then click
the overhead image -- the click is converted px -> mm via the `px_to_mm(u, v)` action. `GET /state` carries
`perception: {calibrated, zones: [{name, drop, rect}]}`.

LOG group: a ring buffer of the last 200 decisions/events -- every loop tool call (with a 160px overlay jpeg
thumbnail at decision time, latency and the say text), plus voice events, mode changes, zone-drop moves and
torque on/off -- served newest-first at `GET /log` as
`{i, step, t, tool, args, result, ok, say, latency_ms, thumb_b64}`. The log tab renders a scrolling list with
thumbnails; rejections / safety errors / E-STOP (`ok: false`, `FAILED:`/`rejected:`/`safety:` results) are red.
`log_clear()` empties it.

MODELS group: `get_models()` lists what is selectable -- OpenAI vision-capable ids from `models.list()`
(gpt-5 / gpt-4.1 / gpt-4o / o3 / o4 families, cached 5 min, graceful when the key is missing) and the
ElevenLabs TTS/STT model ids plus the first ~30 voices from `voices.search()`. `set_model(provider, value)`
with `provider: openai|elevenlabs_tts|elevenlabs_stt|elevenlabs_voice` hot-swaps the live VLM/VoiceIO **and**
persists to `config.yaml` (`vlm.model`, `voice.tts_model`, `voice.stt_model`, `voice.elevenlabs_voice_id`)
via a minimal text edit that preserves comments. The models tab shows one dropdown per provider; beside the
OpenAI one the last VLM call's latency and a rough per-call $ estimate (from token usage and a small built-in
price table) are shown, also served as `/state` `vlm: {model, last_latency_ms, last_cost_usd, last_usage}`.

Loop behaviour: every iteration starts with `home()`, so objects and drop points must be within `max_step_mm`
of HOME (with the default config: x 160-300 mm, |y| <= 160 mm). Rejected/unsafe commands are returned to the VLM
as `FAILED: ...` history entries, never raised. Voice input: "rule"-shaped sentences are persisted to RULES,
`stop` ends the run, `open`/`close`/`home` execute directly, anything else is passed to the VLM as a `(human) ...` hint.
What counts as "already sorted" is the VLM's call: the prompt tells it to leave things that already sit in a
sensible zone alone.

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
3. **Capture** (`space`) at >= 4 well-spread positions (>= 15 mm apart, `min_sample_spacing_mm`); ~8 over the whole
   pick area is good. Each sample pairs the target's pixel centroid with the FK base xyz; after 4 samples the
   homography is refitted on every capture and the residuals are shown live. *Undo* (`u`) drops the last sample.
4. **Finish** (`enter`): RANSAC fit `H_px_to_mm`, per-point residual table, written to `calib/calib.json` together
   with `plane_z_mm` (mean target-centre height), `target`, `method: "teleop"`, `points`, `residuals_mm`.
   *Cancel* (`q`) writes nothing. The leader is released and the loop resumes with the reloaded homography.

After that the **table frame is the base frame** in xy (x fwd, y left, mm). `H` is exact for object centroids at
`plane_z_mm`; taller objects project slightly outward from the camera nadir, lower ones inward (a few mm at most).
`calibration.mode` in config: `ball` = always the fitted H, `aruco` = tags only, `auto` (default) = fitted H, with the
4 ArUco tags overriding per frame when all are visible.

**Recalibrate** whenever the overhead camera or the robot base is moved/bumped, the camera resolution changes, or
the HUD overlay grid / zone outlines stop lining up with the table. Old `calib.json` files without the new keys
still load (rigid `table_T_base` only, no fixed H).

Keys: `OPENAI_API_KEY` (and optional `ELEVENLABS_API_KEY`) in `.env` at repo root. Without the ElevenLabs key,
push-to-talk/`speak` report the missing key in their response instead of failing silently.
