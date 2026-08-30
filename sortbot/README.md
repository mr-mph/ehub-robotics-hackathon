# sortbot

VLM-driven pick-and-sort on an SO101 arm: sort parts into WIRES / SENSORS / ACTUATORS zones.

## Architecture

```
 voice.py ──(corrections -> RULES)──┐
                                    v
 overhead cam ─┐              ┌── main.py loop ──┐
 wrist cam  ───┴─> perception.py ─> vlm.py ─> robot.py ─> SO101
                   (numbered objs,   (one tool   (safety
                    table-frame mm)   call)       envelope, IK)
                        │                │
                        └── hud.py <─────┘   (browser HUD, port 8765)
 calibration.py: overhead px <-> table mm (teleop-fitted H, or ArUco mat) + table_T_base -> calib/calib.json
 calibrate.py:   teleoperated calibration session (leader arm + coloured target in the gripper), HUD buttons + keys
```

Loop: `home()` -> capture overhead + wrist -> overlay numbered candidates -> VLM emits ONE tool call
(`pick(id)`, `place_in_zone(zone)`, `place_at(x,y)`, `say(text)`, `done`; low-level `move_to/open/close/turn_to`
for recovery) -> validate against safety envelope -> execute -> repeat. Voice corrections drain at the top of
each iteration into a persistent RULES list sent with every prompt.

## Modules / owners

| file | owner | role |
|---|---|---|
| `types.py`, `config.py`, `config.yaml` | scaffold | shared contracts (do not edit) |
| `robot.py` | robot | `RobotAPI` real + mock; safety envelope, lift-translate-descend, IK sanity |
| `calibration.py` | calib | colour-target detector, homography px->mm (fitted or ArUco), `table_T_base`, calib.json I/O |
| `calibrate.py`, `calibrate_aruco.py` | calib | teleop calibration session/controller (HUD actions), legacy ArUco+Kabsch flow |
| `perception.py` | perception | segment candidates, `DetectedObject`s, overlay render |
| `vlm.py` | vlm | prompt + tool schema, OpenAI call -> `Command` |
| `voice.py` | voice | ElevenLabs/mic in, keyboard fallback, queue of corrections |
| `hud.py` | hud | FastAPI status page + generic action registry (`register(name, fn, label, group)` -> `POST /action/{name}`) |
| `main.py` | main | the loop |

## Frames

* **Table frame**: mm, origin = robot base projected onto tabletop, x forward, y left, z up, table z=0.
* **Base frame** (FK/IK): meters, `base_link` of the URDF. `table_T_base` (rigid, from calibration) converts.
* End effector always points straight down; only `wrist_roll` changes via `turn_to(deg)`.
* Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.

## Running

```
./run.sh -m sortbot.config                                   # print config (selftest)
./run.sh -m sortbot.main --mock                              # no hardware: MockRobot + synthetic blobs + MockVLM, HUD on :8765,
                                                             #   type corrections on stdin ("put white things in wires", "stop")
./run.sh -m sortbot.main --mock --live-vlm                   # same, but real OpenAI planning (needs OPENAI_API_KEY)
./run.sh -m sortbot.main --mock --no-hud --no-voice --max-steps 12   # headless
./run.sh -m sortbot.tests.test_e2e_mock                      # e2e mock test (HUD on a random port)
./run.sh -m sortbot.calibrate                                # hardware: teleoperated target calibration -> calib/calib.json (see below)
./run.sh -m sortbot.calibrate --mock                         # same flow with MockRobot + virtual leader (selftest, < 1 mm)
./run.sh -m sortbot.calibrate --mode aruco                   # legacy ArUco mat + Kabsch rigid transform
./run.sh -m sortbot.calibration --check-target green         # grab one overhead frame, detect the preset -> out/target_check.png
./run.sh -m sortbot.main --real                              # SO101 + overhead/wrist cams + OpenAI (+ ElevenLabs mic if key set)
./run.sh -m sortbot.<module> --selftest                      # per-module smoke test
```

`main` flags: `--max-steps N` (default 40), `--no-hud`, `--hud-port P`, `--no-voice`, `--rules-file PATH`
(default `sortbot/calib/rules.json`, persistent across runs; delete it to forget rules), `--config PATH`.

Loop behaviour: every iteration starts with `home()`, so objects and drop points must be within `max_step_mm`
of HOME (with the default config: x 160-300 mm, |y| <= 160 mm). Rejected/unsafe commands are returned to the VLM
as `FAILED: ...` history entries, never raised. Voice input: "rule"-shaped sentences are persisted to RULES,
`stop` ends the run, `open`/`close`/`home` execute directly, anything else is passed to the VLM as a `(human) ...` hint.
Zones that have received a placement are excluded from detection (`filled_zones`), so objects already inside a
zone at start are treated as sorted.

## Calibration (default: teleop / ball mode, no ArUco tags)

Put the calibration target (the green ball; or anything with a distinct colour) in the follower's gripper, then
`./run.sh -m sortbot.calibrate` (leader + follower + overhead cam; HUD on :8765) or press **Start calibration** in
the HUD of a running `main --real`. A background thread teleoperates the follower from the leader arm (config
`leader:`); the sorting loop pauses at its next step boundary. Then:

1. **Pick the target colour** (optional): click the target in the HUD overhead image. The server samples a tolerant
   HSV window (`ColorTarget.from_sample`, hue wraps for reds), runs the detector and draws the detection circle.
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
4 ArUco tags overriding per frame when all are visible. `detect_objects` uses the mat between the tags as ROI in
aruco mode and the workspace AABB xy in ball mode.

**Recalibrate** whenever the overhead camera or the robot base is moved/bumped, the camera resolution changes, or
the HUD overlay grid / zone outlines stop lining up with the table. Old `calib.json` files without the new keys
still load (rigid `table_T_base` only, no fixed H).

Keys: `OPENAI_API_KEY` (and optional `ELEVENLABS_API_KEY`) in `.env` at repo root.
