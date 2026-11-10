# Real SO-100 arm: what we know

Facts measured on the physical arm (2026-10-19). Read this before
touching `src/real_manual.py` or writing the real-robot backend.

**Docs source:** always check the official LeRobot docs, not memory:
- Cheat sheet: https://huggingface.co/docs/lerobot/main/en/cheat-sheet
- SO-100: https://huggingface.co/docs/lerobot/main/en/so100
- Installation: https://huggingface.co/docs/lerobot/main/en/installation

## Hardware

- One **SO-100 follower** arm (gripper with jaws). No leader arm in use.
- Clamped to the table. Controller board on the back, USB-C to the Mac, separate power supply.
- Port: `/dev/tty.usbmodem575E0032501` (new controller board with a CH343 USB chip, 2026-10-28; same servos, calibration unchanged and verified). Old board: `/dev/tty.usbmodem5A680112971`. `/dev/cu.DenonDHT-S217` is an unrelated speaker.
- 6x Feetech **STS3215** (model number 777), IDs 1-6, 1,000,000 baud. Already configured,
  so `lerobot-setup-motors` is not needed.

| ID | Motor |
|---|---|
| 1 | shoulder_pan |
| 2 | shoulder_lift |
| 3 | elbow_flex |
| 4 | wrist_flex |
| 5 | wrist_roll |
| 6 | gripper |

## Software

- `lerobot[feetech,hardware]` 0.6.1 in the project venv. Run its tools with `uv run`
  (e.g. `uv run lerobot-find-port`); they aren't on the global PATH.
- **Python is pinned to 3.12** (`.python-version`). LeRobot's CLI parser (draccus) crashes on
  3.14 with `TypeError: str | None is not callable`.

## Calibration

- Robot id **`my_follower`**. Every LeRobot command must use `--robot.type=so100_follower --robot.id=my_follower`.
- File: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json`
- Calibration also writes homing offsets and min/max position limits **into the motors' EEPROM**,
  so the motors themselves refuse to go past the end stops.
- Recalibrate:
  `uv run lerobot-calibrate --robot.type=so100_follower --robot.port=/dev/tty.usbmodem5A680112971 --robot.id=my_follower`
  (middle position, Enter, then move every joint except wrist_roll end to end, including base and gripper, Enter).

### Units and limits

With `use_degrees=True` (default), LeRobot maps raw 0-4095 to degrees with
**0 deg = midpoint between the calibrated min and max** (not a physical "home" pose).
So the arm joints come out symmetric. Gripper is always 0 (closed) to 100 (open).

| Joint | Raw min-max | Degrees |
|---|---|---|
| shoulder_pan | 659-3456 | -122.9 to +122.9 |
| shoulder_lift | 983-3284 | -101.1 to +101.1 |
| elbow_flex | 777-2993 | -97.4 to +97.4 |
| wrist_flex | 840-3293 | -107.8 to +107.8 |
| wrist_roll | 0-4095 (full turn) | -180 to +180 |
| gripper | 1576-3056 | 0 to 100 |

### Directions

Viewed **from behind the arm** (at the controller board, looking where the gripper points),
tested with +20 deg per joint:

| Joint | Positive means |
|---|---|
| shoulder_pan | turns **right** |
| shoulder_lift | upper arm tilts **forward / down toward the table** |
| elbow_flex | forearm **down** in the test pose; really: **folds the elbow shut** (+97 = fully folded, rest). Negative opens it; straight is near the negative end (~-85, estimated) |
| wrist_flex | gripper tilts **down** |
| wrist_roll | twists **counterclockwise** |
| gripper | **opens** (0 closed, 100 open) |

### Poses (`poses.json`, calibrated degrees)

Recorded by hand with `free` / `save` in `real_manual.py`. Separate from the 3D model's `POSES`
in `src/robot.py`; the two frames don't match.

- `rest`: pan -4.9, lift -100.7 (fully back), elbow 97.1 (fully folded), wrist_flex 81.5, roll 0, gripper 2.4

### Geometry landmarks (from the URDF, not measured)

The URDF joint ranges match the calibrated ranges (lift 200 vs 202 deg, elbow 180 vs 194,
wrist 212 vs 215) and the folded rest pose sits at the same end stops, so
calibrated = URDF + offset (lift -101.1, elbow +97.4, wrist about +35, wrist sign unverified).
Forward kinematics gives:

| Shape | URDF | Calibrated (this arm) |
|---|---|---|
| upper arm vertical | lift 89 | shoulder_lift about -12 |
| forearm straight | elbow -164 | elbow_flex about -67 |
| gripper in line with forearm | wrist -21.5 | wrist_flex about 0 to +15 |

Cross-check: the LLM sent shoulder_lift +80 for "reach for the stars" and the arm went
horizontal, which fits vertical = -12 (+80 = 92 deg forward). The earlier guess of -50 was wrong.

## Observations

- **shoulder_lift 0 drives the arm into the table.** All-zero is not a usable pose. Upright is
  roughly shoulder_lift -12 (see Geometry landmarks).
- Warning `Relative goal position magnitude had to be clamped to be safe` = the arm was blocked
  (table) while the commanded target kept moving; `max_relative_target` clipped it, but the motor
  was still pushing. Motivation for blocking detection.
- Speed feels too quick with LeRobot defaults: it sets motor `Acceleration` to 254 (no ramp) on
  every connect, for snappy teleoperation.

## Motion pipeline (`src/real_manual.py`)

1. Duration = longest joint distance x 1.5 / `MAX_SPEED` (40 deg/s), at least `MIN_MOVE_SECONDS` (0.4 s).
2. Every `TICK_SECONDS` (0.02 s) send the smoothstep-interpolated target for all moving joints
   (slow-fast-slow; peak speed = `MAX_SPEED`).
3. LeRobot `send_action` converts degrees to raw; `max_relative_target=10` rejects jumps > 10 deg.
4. Motor `Acceleration` is set to 50 after connect (softer ramp inside the servo; resets to 254
   on the next LeRobot connect or power cycle).

Commands: `joint <name> <deg>`, `<name> <delta>`, `gripper open|closed`, `free`, `hold`,
`save <name>`, `pose <name>`, `poses`, `status` (shows min/max), `quit` (asks to hold the arm,
then releases torque). Up/down arrows recall commands. `hold` writes Goal_Position = present
position before enabling torque so the arm doesn't snap back.

## Servo feedback available (STS3215 registers)

`Present_Position`, `Present_Velocity`, `Present_Load` (% of max torque), `Present_Current`,
`Present_Temperature`, `Present_Voltage`, `Moving`, `Status`. Built-in protection:
`Protection_Current`, `Overload_Torque`, `Protective_Torque`, `Torque_Limit`. LeRobot only
lowers torque/current limits for the gripper (50%).

## Voice control of the real arm (built, not yet tried on hardware)

Set `ROBOT=real` in `.env.local`, close `real_manual.py` (only one program can hold the port),
then `lk agent console`. Without `ROBOT=real` the agent keeps using `FakeRobot` + Rerun.

- `src/real_arm.py` (`RealArm`): same glide as `real_manual.py`, plus
  - validation against calibrated limits minus `LIMIT_MARGIN` (3 deg) before anything moves
  - blocking detection: joint > `LAG_LIMIT` (8 deg) behind its commanded target AND moved < 0.5 deg
    since the previous check, for 3 checks in a row (~0.2 s). Lag alone is not enough: lifting
    against gravity lags while still moving (first hardware test: false "blocked" on elbow_flex)
    -> stop, hold, report "blocked" (gripper excluded: it may be holding something)
  - `stop()`: ends the running move within one tick and holds
  - `disable_torque_on_disconnect=False`: the arm keeps holding when the agent exits
  - every move returns a plain-language result (reached / blocked / stopped / rejected)
- `src/real_assistant.py` (`RealArmAssistant`): no hard-coded gestures. Instructions describe
  each joint (meaning, direction, safe range) and saved poses; the LLM composes motions.
  Tools: `move_joints`, `run_motion` (keyframes + repeat), `go_to_pose` (clamped into the
  safe range, since hand-recorded poses sit at end stops), `get_status`, `stop`.
- Stop paths: `agent.py` matches `stop`/`stopp...` in live (interim) transcripts and calls
  `arm.stop()` without waiting for the LLM; a genuine interruption while a motion tool runs
  also stops the arm; the LLM has a `stop` tool. "halt" is left to the LLM because it is a
  common German filler word.
- **Blocking detection and obstacle memory are OFF** (`DETECT_BLOCKING = False` in `real_arm.py`):
  stalls were often the arm's own weight, not an obstacle, and the user stops real collisions by
  voice. Tests switch it on. Without it, a motor pressing on something keeps pushing until "stop".
- **Session obstacle memory** (only with `DETECT_BLOCKING = True`) (`Obstacle` in `real_arm.py`, in memory only, gone on restart):
  every "blocked" result stores the arm configuration, the stuck joint and its travel
  direction. Forbidden region = that joint at/beyond the blocked angle (minus 3 deg) in that
  direction while every other arm joint is within 10 deg of where it was. `move` and
  `run_motion` check the whole joint-space path in 2 deg steps before moving and reject paths
  into a known obstacle with the reason. When the arm already sits at an obstacle's edge,
  only going deeper is forbidden, so it can back off. `get_status` lists learned obstacles.
- `real_manual.py` has the same blocking detection (no obstacle memory).
- Tests: `tests/test_real_arm.py` (simulated bus, plus an LLM test that "Wink mal" becomes a
  `run_motion` call).

## Open

- Load-based blocking detection (`Present_Load`): measure a threshold on the arm first.
- Geometry landmarks are computed from the URDF, not measured on the arm; verify if moves look off.
- Rerun/3D model stays with `FakeRobot` only.

## Dream-1 (Dream Labs) as the hands-on planner (built 2026-10-28, not yet run on hardware)

Voice -> `do_task("pick up the cpu")` in `RealArmAssistant` -> `src/dream_task.py` -> `RealArm.follow()`.

- Wrist camera: USB "USB 2.0 Camera", OpenCV index 0, 1280x720 @ 30 fps. Mounted turned (probably 90 deg):
  `WRIST_CAMERA_ROTATE=90` in `.env.local` turns it upright in software. It should see the fingertips at
  the bottom edge and the work area in front of them.
- Dream-1 runs as "other_embodiment" (route B in their docs): it reads the camera and the task, ignores
  the joints, and answers TCP poses anchored on the TCP pose we send. Overhead and left wrist get black
  frames. API key in `~/.dreamlabs/credentials`. One paid call per 48-row chunk (2.4 s).
- `src/so100_kinematics.py`: FK/IK from the URDF, and calibrated deg <-> URDF rad:
  `urdf = SIGN * calibrated + OFFSET`. Pan and roll signs are flipped (from the Directions table);
  wrist_flex sign checked by FK (+ tilts down in both). Roll offset unknown (0).
- **TCP**: wrist_flex axle to the point between the closed fingertips = 16 cm (measured by the user).
  Sideways offset not measured, assumed 0. Fingers along -y of the URDF gripper link.
- `RealArm.follow()`: every target validated against the safe range, speed capped at 25 deg/s
  (gripper 100/s), holds on stale camera (> 0.2 s), stop, rejection, or planner error.
- Round trip to Dream-1 is 2-4 s, longer than a chunk: when the arm held still during the call, the
  chunk plays from row 0 on arrival instead of skipping rows already "past".
- **Check before the first real run**: `uv run python src/check_mapping.py` (read-only, torque off)
  mirrors the arm in Rerun while you move it by hand. Every joint must move the same way in the model.
