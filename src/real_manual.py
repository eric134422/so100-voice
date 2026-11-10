"""Control the real SO-100 one joint at a time by typing.

Run with: uv run python src/real_manual.py

Commands:
  joint <name> <deg>   absolute target, e.g. "joint shoulder_pan 30"
                       0 deg = midpoint between the calibrated min and max
  <name> <delta>       relative move, e.g. "shoulder_pan 5" or "elbow_flex -5"
  gripper open|closed  fully open or close the gripper
  free                 motors off, place the arm by hand
  hold                 motors on, hold where the arm is now
  save <name>          save the current position as a pose, e.g. "save standup"
  pose <name>          glide to a saved pose (turns motors on first)
  poses                list saved poses
  status               print current positions and each joint's min/max
  quit                 release the motors (hold the arm, it goes limp)

Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
The gripper uses 0 (closed) to 100 (open) instead of degrees.
Up/down arrows recall earlier commands.

Moves ease in and out at up to MAX_SPEED, so even a big absolute target glides
instead of jumping.
"""

import json
import os
import readline  # noqa: F401  (enables arrow-key history for input())
import time
from pathlib import Path

from dotenv import load_dotenv
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

load_dotenv(".env.local")
PORT = os.getenv("ROBOT_PORT", "/dev/tty.usbmodem5A680112971")
ROBOT_ID = os.getenv(
    "ROBOT_ID", "my_follower"
)  # must match the --robot.id used for calibration
# Real-arm poses, in calibrated degrees. Separate from the 3D model's POSES in robot.py.
POSES_PATH = Path(__file__).resolve().parent.parent / "poses.json"
MAX_SPEED = 40.0  # degrees per second at the fastest point of a move
MIN_MOVE_SECONDS = 0.4  # even tiny moves ease in and out over at least this long
TICK_SECONDS = 0.02
# Motor-side acceleration ramp (0-254). LeRobot sets 254 (no ramp) for snappy
# teleoperation; a lower value softens starts and stops in the motor itself.
MOTOR_ACCELERATION = 50
LAG_LIMIT = 8.0  # degrees behind the commanded target that count as blocked
GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 0.0
MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def positions(robot: SO100Follower) -> dict[str, float]:
    obs = robot.get_observation()
    return {m: obs[f"{m}.pos"] for m in MOTORS}


def limits(robot: SO100Follower) -> dict[str, tuple[float, float]]:
    """Each joint's range in the units send_action uses, from the calibration file.

    LeRobot's 0 deg is the midpoint between the calibrated min and max, so the
    arm joints come out symmetric. The gripper is always 0 (closed) to 100 (open).
    """
    result = {}
    for name, cal in robot.calibration.items():
        if name == "gripper":
            result[name] = (0.0, 100.0)
        else:
            half = (cal.range_max - cal.range_min) / 2 * 360 / 4095
            result[name] = (-half, half)
    return result


def print_status(robot: SO100Follower) -> None:
    ranges = limits(robot)
    print(f"  {'joint':<14} {'now':>7}   {'min':>7} {'max':>7}")
    for name, pos in positions(robot).items():
        low, high = ranges[name]
        unit = " " if name == "gripper" else "°"
        print(f"  {name:<14} {pos:7.1f}{unit}  {low:7.1f} {high:7.1f}")


def move_to(robot: SO100Follower, targets: dict[str, float]) -> None:
    """Glide one or more joints to their targets together: slow start, top speed
    in the middle, slow stop. All joints start and finish at the same time.

    Goals beyond the calibrated range are clamped by the motors' position
    limits, so a joint simply stops at its end stop.
    """
    current = positions(robot)
    start = {name: current[name] for name in targets}
    longest = max(abs(targets[n] - start[n]) for n in targets)
    duration = max(longest * 1.5 / MAX_SPEED, MIN_MOVE_SECONDS)
    over = dict.fromkeys(targets, 0)
    last = dict(start)
    tick = 0
    t0 = time.monotonic()
    while (elapsed := time.monotonic() - t0) < duration:
        s = elapsed / duration
        s = s * s * (3 - 2 * s)  # smoothstep: peak speed is 1.5x the average
        commanded = {n: start[n] + (targets[n] - start[n]) * s for n in targets}
        robot.send_action({f"{n}.pos": v for n, v in commanded.items()})
        time.sleep(TICK_SECONDS)

        # Blocking detection: a joint that stays LAG_LIMIT behind AND stops moving
        # is pushing against something (usually the table). Lagging while still
        # moving is just load, e.g. lifting against gravity. The gripper may hold
        # objects.
        tick += 1
        if elapsed > 0.25 and tick % 3 == 0:
            now = positions(robot)
            for n in targets:
                lagging = abs(commanded[n] - now[n]) > LAG_LIMIT
                stalled = abs(now[n] - last[n]) < 0.5
                over[n] = over[n] + 1 if lagging and stalled else 0
                last[n] = now[n]
            blocked = [n for n in targets if n != "gripper" and over[n] >= 3]
            if blocked:
                robot.bus.sync_write("Goal_Position", now)
                print(
                    f"  BLOCKED: {blocked[0]} stuck at {now[blocked[0]]:.1f}, holding"
                )
                return
    robot.send_action({f"{n}.pos": t for n, t in targets.items()})
    time.sleep(0.3)  # let the motors settle before reporting
    now = positions(robot)
    for n, t in targets.items():
        print(f"  {n} at {now[n]:.1f} (target {t:.1f})")


def load_poses() -> dict[str, dict[str, float]]:
    return json.loads(POSES_PATH.read_text()) if POSES_PATH.exists() else {}


def save_pose(robot: SO100Follower, name: str) -> None:
    poses = load_poses()
    poses[name] = {m: round(v, 1) for m, v in positions(robot).items()}
    POSES_PATH.write_text(json.dumps(poses, indent=2) + "\n")
    print(f"  saved '{name}' to {POSES_PATH.name}")


def free(robot: SO100Follower) -> None:
    robot.bus.disable_torque()
    print("  motors OFF: hold the arm and place it by hand")


def hold(robot: SO100Follower) -> None:
    # Set the goal to where the arm is now before powering on, otherwise the
    # motors would snap back to the last commanded position.
    robot.bus.sync_write("Goal_Position", positions(robot))
    robot.bus.enable_torque()
    print("  motors ON: holding position")


def parse(words: list[str], current: dict[str, float]) -> dict[str, float]:
    match words:
        case ["gripper", "open"]:
            return {"gripper": GRIPPER_OPEN}
        case ["gripper", "closed" | "close"]:
            return {"gripper": GRIPPER_CLOSED}
        case ["joint", name, deg] if name in MOTORS:
            return {name: float(deg)}
        case [name, delta] if name in MOTORS:
            return {name: current[name] + float(delta)}
    raise ValueError


def main() -> None:
    robot = SO100Follower(
        SO100FollowerConfig(
            port=PORT,
            id=ROBOT_ID,
            # LeRobot's own safety net: no single command jumps further than this
            max_relative_target=10.0,
        )
    )
    robot.connect(calibrate=False)
    for motor in MOTORS:
        robot.bus.write("Acceleration", motor, MOTOR_ACCELERATION)
    print("Connected. Motors are ON and holding position.")
    print(__doc__.split("Commands:")[1])
    print_status(robot)

    try:
        while True:
            words = input("> ").strip().split()
            if words == ["quit"]:
                break
            match words:
                case ["status"]:
                    print_status(robot)
                    continue
                case ["free"]:
                    free(robot)
                    continue
                case ["hold"]:
                    hold(robot)
                    continue
                case ["save", name]:
                    save_pose(robot, name)
                    continue
                case ["poses"]:
                    print("  " + (", ".join(load_poses()) or "no poses saved yet"))
                    continue
                case ["pose", name]:
                    if name not in load_poses():
                        print(f"  no pose '{name}', saved: {', '.join(load_poses())}")
                        continue
                    hold(robot)
                    move_to(robot, load_poses()[name])
                    continue
            try:
                targets = parse(words, positions(robot))
            except ValueError:
                print(
                    "  usage: joint <name> <deg> | <name> <delta> | gripper open|closed"
                    " | free | hold | save <name> | pose <name> | poses | status | quit"
                )
                continue
            move_to(robot, targets)
    finally:
        input("Hold the arm, then press ENTER to release the motors...")
        robot.disconnect()


if __name__ == "__main__":
    main()
