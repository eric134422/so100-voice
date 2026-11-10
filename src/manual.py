"""Type commands to move the fake robot in the Rerun viewer, no voice needed.

Run with: uv run python src/manual.py

Commands:
  pose <name>            home, rest, reach_forward, look_up
  joint <name> <deg>     e.g. joint shoulder_pan 45
  gripper open|closed
  status
  quit
"""

import asyncio
import logging

from robot import JOINT_LIMITS, JOINTS, POSES, FakeRobot


async def handle(robot: FakeRobot, words: list[str]) -> None:
    match words:
        case ["pose", name] if name in POSES:
            await robot.move_joints(POSES[name])
        case ["joint", name, deg] if name in JOINTS:
            angles = list(robot.status()["joints"].values())
            angles[JOINTS.index(name)] = float(deg)
            await robot.move_joints(tuple(angles))
        case ["gripper", ("open" | "closed") as state]:
            await robot.set_gripper(state)
        case ["status"]:
            for name, deg in robot.status()["joints"].items():
                low, high = JOINT_LIMITS[name]
                print(f"  {name:<14} {deg:7.1f}°   (safe {low:.0f} to {high:.0f})")
            print(f"  {'gripper':<14} {robot.status()['gripper']}")
        case _:
            print(__doc__.split("Commands:")[1])


async def main() -> None:
    robot = FakeRobot(visualize=True)
    print(__doc__.split("Commands:")[1])
    while True:
        line = await asyncio.to_thread(input, "> ")
        words = line.strip().split()
        if words in (["quit"], ["exit"]):
            break
        try:
            await handle(robot, words)
        except ValueError as e:
            print(f"  rejected: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
