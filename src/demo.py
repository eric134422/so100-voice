"""Cycle the fake robot through every pose in the Rerun viewer, no voice needed.

Run with: uv run python src/demo.py
"""

import asyncio
import logging

from robot import POSES, FakeRobot


async def main() -> None:
    robot = FakeRobot(visualize=True)
    for pose in POSES:
        await robot.move_joints(POSES[pose])
        await asyncio.sleep(0.5)
    await robot.set_gripper("open")
    await robot.set_gripper("closed")
    await robot.move_joints(POSES["rest"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
