"""Real SO-100 backend for the voice agent.

Same motion pipeline as real_manual.py (smoothstep glide, softer motor
acceleration), plus what a voice-controlled arm needs:
- validation against the calibrated limits before anything moves
- blocking detection: a joint that falls too far behind its target stops the move
- stop(): ends any running move at once and holds the arm where it is
- a plain-language result for every move, so the LLM knows what happened

Angles are LeRobot's calibrated degrees (0 = midpoint of the calibrated range),
not the 3D model's angles in robot.py. Facts: docs/real-arm.md.
"""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, get_args

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from pydantic import BaseModel, Field

logger = logging.getLogger("real_arm")

PORT = os.getenv("ROBOT_PORT", "/dev/tty.usbmodem5A680112971")
ROBOT_ID = os.getenv(
    "ROBOT_ID", "my_follower"
)  # must match the --robot.id used for calibration
POSES_PATH = Path(__file__).resolve().parent.parent / "poses.json"

ArmJoint = Literal[
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
JOINTS: tuple[str, ...] = get_args(ArmJoint)

# Measured on the arm, viewed from behind it (docs/real-arm.md)
DIRECTIONS = {
    "shoulder_pan": "positive turns the whole arm to the right",
    "shoulder_lift": (
        "despite the name, positive LOWERS the upper arm forward toward the table; "
        "negative leans it back. About -12 is vertical, about +80 points it "
        "horizontally forward, -100 is folded fully back (rest)"
    ),
    "elbow_flex": (
        "positive folds the elbow shut (+97 is fully folded, as in rest), negative "
        "opens it; about -67 is straight, the forearm in line with the upper arm"
    ),
    "wrist_flex": (
        "positive tilts the gripper down in the reference pose; roughly 0 to +15 "
        "keeps the gripper in line with the forearm"
    ),
    "wrist_roll": "positive twists the gripper counterclockwise",
    "gripper": "0 is closed, 100 is fully open",
}

# Where the vertical/straight angles come from: the SO-100 URDF
# (assets/so100/so100.urdf) has the same joint ranges as the calibration, so
# calibrated = URDF + offset. Computed with forward kinematics, accurate to
# roughly +-10 deg. Details in docs/real-arm.md.

MAX_SPEED = 40.0  # degrees per second at the fastest point of a move
MIN_MOVE_SECONDS = 0.4
TICK_SECONDS = 0.02
MOTOR_ACCELERATION = 50  # LeRobot sets 254 (no ramp) on connect
LIMIT_MARGIN = 3.0  # stay this many degrees inside the calibrated end stops
# Off: stalls were often the arm's own weight, not an obstacle, and "Stopp" covers
# real collisions. True re-enables blocking detection and obstacle learning.
DETECT_BLOCKING = False
# Blocked = behind AND not moving. A joint lifting against gravity can lag more
# than LAG_LIMIT while still moving; only a joint that also stalls is stuck.
LAG_LIMIT = 8.0  # degrees behind the commanded target
STALL_DEGREES = 0.5  # moved less than this since the previous check = stalled
LAG_CHECKS = 3  # consecutive lagging-and-stalled checks before stopping (~0.2 s)
CHECK_EVERY_TICKS = 3
GRACE_SECONDS = 0.25  # motors need a moment to start following
MAX_KEYFRAMES = 30
MAX_REPEAT = 5
MAX_MOTION_SECONDS = 30.0
# Learned obstacles: a configuration counts as "the same place" when every other
# arm joint is within OBSTACLE_NEIGHBORHOOD of where the block happened.
OBSTACLE_NEIGHBORHOOD = 10.0
OBSTACLE_MARGIN = 3.0  # stop this many degrees short of the blocked angle
PATH_SAMPLE_DEGREES = 2.0  # check planned paths at this resolution
ARM_JOINTS = JOINTS[:-1]  # obstacles ignore the gripper
# follow(): live targets from a planner (Dream-1). Slower than glides: the planner's
# targets are not checked for this arm beyond the limits, so every jump becomes a ramp.
STREAM_TICK = 0.05  # 20 Hz, the rate Dream-1's rows are played at
STREAM_SPEED = 25.0  # deg/s per joint (Dream Labs: ramp in at most 0.5 rad/s = 28.6)
STREAM_GRIP_SPEED = 100.0  # gripper units/s (Dream Labs: at most 0.05 of the range a row)
MAX_FOLLOW_SECONDS = 60.0


class PlanDone(Exception):  # noqa: N818 - an outcome, not an error
    """A TargetSource has finished; the message says how."""


class TargetSource(Protocol):
    async def next(self, now: dict[str, float]) -> dict[str, float] | None: ...


class JointTarget(BaseModel):
    joint: ArmJoint
    degrees: float = Field(description="Target angle in degrees; gripper 0-100.")


class Keyframe(BaseModel):
    targets: list[JointTarget] = Field(description="Joints to move together.")
    seconds: float | None = Field(
        None,
        description="Duration of this keyframe. Omit for the safe default speed.",
    )


class MoveRejectedError(ValueError):
    pass


@dataclass
class Obstacle:
    """Something the arm ran into this session.

    Forbidden region: `joint` at or beyond `at[joint]` (minus OBSTACLE_MARGIN) in
    the `direction` it was travelling, while every other arm joint is within
    OBSTACLE_NEIGHBORHOOD of `at`. Further in the same direction is assumed
    blocked too; with the other joints elsewhere, the way may be free.
    """

    at: dict[str, float]
    joint: str
    direction: int  # +1 or -1

    def contains(self, config: dict[str, float]) -> bool:
        for j in ARM_JOINTS:
            if j != self.joint and abs(config[j] - self.at[j]) > OBSTACLE_NEIGHBORHOOD:
                return False
        return (
            self.direction * (config[self.joint] - self.at[self.joint])
            >= -OBSTACLE_MARGIN
        )

    def describe(self) -> str:
        past = "above" if self.direction > 0 else "below"
        near = ", ".join(f"{j} {self.at[j]:.0f}" for j in ARM_JOINTS if j != self.joint)
        return f"{self.joint} {past} {self.at[self.joint]:.0f} while near {near}"


def load_poses() -> dict[str, dict[str, float]]:
    return json.loads(POSES_PATH.read_text()) if POSES_PATH.exists() else {}


class RealArm:
    def __init__(self) -> None:
        self.robot = SO100Follower(
            SO100FollowerConfig(
                port=PORT,
                id=ROBOT_ID,
                # LeRobot's own safety net: no single command jumps further than this
                max_relative_target=10.0,
                # keep holding when the agent exits instead of dropping the arm
                disable_torque_on_disconnect=False,
            )
        )
        self.limits = self._limits()
        self._stop = asyncio.Event()
        self._motion = asyncio.Lock()
        self._bus = threading.Lock()
        self.obstacles: list[Obstacle] = []  # learned this session, never saved

    # -- connection ---------------------------------------------------------

    def connect(self) -> None:
        self.robot.connect(calibrate=False)
        for joint in JOINTS:
            self.robot.bus.write("Acceleration", joint, MOTOR_ACCELERATION)
        logger.info("connected to SO-100 on %s", PORT)

    def disconnect(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()

    # -- state --------------------------------------------------------------

    def _limits(self) -> dict[str, tuple[float, float]]:
        """Safe range per joint from the calibration file, LIMIT_MARGIN inside
        the end stops. LeRobot's 0 deg is the midpoint of the calibrated range."""
        result = {}
        for name, cal in self.robot.calibration.items():
            if name == "gripper":
                result[name] = (0.0, 100.0)
            else:
                half = (cal.range_max - cal.range_min) / 2 * 360 / 4095
                result[name] = (-half + LIMIT_MARGIN, half - LIMIT_MARGIN)
        return result

    def positions(self) -> dict[str, float]:
        with self._bus:
            obs = self.robot.get_observation()
        return {j: obs[f"{j}.pos"] for j in JOINTS}

    def status(self) -> dict:
        now = self.positions()
        joints = {
            j: {"now": round(now[j], 1), "min": round(lo, 1), "max": round(hi, 1)}
            for j, (lo, hi) in self.limits.items()
        }
        return {
            "joints": joints,
            "learned_obstacles": [o.describe() for o in self.obstacles],
        }

    # -- motion -------------------------------------------------------------

    def stop(self) -> None:
        """Safe to call from any callback: the running move notices within a tick."""
        self._stop.set()

    def clamp(self, targets: dict[str, float]) -> dict[str, float]:
        """Pull targets inside the safe range. Only for poses recorded by hand at
        the end stops; LLM-chosen targets go through validate() instead."""
        return {
            j: min(max(v, self.limits[j][0]), self.limits[j][1])
            for j, v in targets.items()
            if j in self.limits
        }

    def validate(self, targets: dict[str, float]) -> None:
        problems = []
        for joint, deg in targets.items():
            if joint not in self.limits:
                problems.append(f"unknown joint {joint}")
                continue
            low, high = self.limits[joint]
            if not low <= deg <= high:
                problems.append(f"{joint} {deg:g} is outside {low:.0f} to {high:.0f}")
        if problems:
            raise MoveRejectedError("; ".join(problems))

    def check_path(self, start: dict[str, float], targets: dict[str, float]) -> dict:
        """Reject a move whose straight joint-space path enters a learned obstacle.
        Returns the configuration at the end of the move."""
        end = start | targets
        steps = max(abs(end[j] - start[j]) for j in ARM_JOINTS) / PATH_SAMPLE_DEGREES
        n = max(int(steps), 1)
        # Right after a block the arm sits at the obstacle's edge. From there only
        # going deeper is forbidden; backing off or moving other joints is fine.
        inside = [o for o in self.obstacles if o.contains(start)]
        for k in range(1, n + 1):
            config = {j: start[j] + (end[j] - start[j]) * k / n for j in ARM_JOINTS}
            for obstacle in self.obstacles:
                deeper = obstacle.direction * (
                    config[obstacle.joint] - start[obstacle.joint]
                )
                if obstacle.contains(config) and (obstacle not in inside or deeper > 0):
                    raise MoveRejectedError(
                        f"the path runs into an obstacle found earlier this session "
                        f"({obstacle.describe()}). Go around it, for example change "
                        "another joint first"
                    )
        return end

    async def move(
        self, targets: dict[str, float], seconds: float | None = None
    ) -> str:
        self.validate(targets)
        async with self._motion:
            self.check_path(await asyncio.to_thread(self.positions), targets)
            self._stop.clear()
            return await self._glide(targets, seconds)

    async def run_motion(self, keyframes: list[Keyframe], repeat: int = 1) -> str:
        if not keyframes:
            raise MoveRejectedError("motion has no keyframes")
        if len(keyframes) * repeat > MAX_KEYFRAMES:
            raise MoveRejectedError(
                f"at most {MAX_KEYFRAMES} keyframes including repeats"
            )
        if not 1 <= repeat <= MAX_REPEAT:
            raise MoveRejectedError(f"repeat must be 1 to {MAX_REPEAT}")
        planned = 0.0
        for i, frame in enumerate(keyframes, 1):
            try:
                self.validate({t.joint: t.degrees for t in frame.targets})
            except MoveRejectedError as e:
                raise MoveRejectedError(f"keyframe {i}: {e}") from e
            planned += frame.seconds or MIN_MOVE_SECONDS
        if planned * repeat > MAX_MOTION_SECONDS:
            raise MoveRejectedError(f"motion longer than {MAX_MOTION_SECONDS:.0f} s")

        async with self._motion:
            config = await asyncio.to_thread(self.positions)
            for _ in range(repeat):
                for i, frame in enumerate(keyframes, 1):
                    try:
                        config = self.check_path(
                            config, {t.joint: t.degrees for t in frame.targets}
                        )
                    except MoveRejectedError as e:
                        raise MoveRejectedError(f"keyframe {i}: {e}") from e
            self._stop.clear()
            for _ in range(repeat):
                for i, frame in enumerate(keyframes, 1):
                    targets = {t.joint: t.degrees for t in frame.targets}
                    result = await self._glide(targets, frame.seconds)
                    if not result.startswith("reached"):
                        return f"keyframe {i}: {result}"
            return (
                f"motion done ({len(keyframes)} keyframes x {repeat}). {self._where()}"
            )

    async def follow(self, source: "TargetSource", seconds: float) -> str:
        """Track targets that a planner (Dream-1) produces live, for at most `seconds`.

        Every tick the source sees where the arm is and answers a target, None to hold, or
        raises PlanDone when it is done. Each target is validated; the commanded
        position moves toward it at most STREAM_SPEED deg/s (gripper STREAM_GRIP_SPEED/s),
        so a jump in the plan becomes a slow ramp. Ends on stop(), a rejected target, or
        a planner error, holding the arm in every case.
        """
        if not 0 < seconds <= MAX_FOLLOW_SECONDS:
            raise MoveRejectedError(f"seconds must be 0 to {MAX_FOLLOW_SECONDS:.0f}")
        async with self._motion:
            self._stop.clear()
            commanded = await asyncio.to_thread(self.positions)
            t0 = next_tick = time.monotonic()
            try:
                while time.monotonic() - t0 < seconds:
                    if self._stop.is_set():
                        return f"stopped by user. {await self._hold_and_where()}"
                    now = await asyncio.to_thread(self.positions)
                    try:
                        target = await source.next(now)
                    except PlanDone as done:
                        return f"{done}. {await self._hold_and_where()}"
                    if target is not None:
                        try:
                            self.validate(target)
                        except MoveRejectedError as e:
                            return f"rejected a planned position ({e}), stopped. {await self._hold_and_where()}"
                        for j, goal in target.items():
                            cap = (STREAM_GRIP_SPEED if j == "gripper" else STREAM_SPEED) * STREAM_TICK
                            commanded[j] += max(-cap, min(cap, goal - commanded[j]))
                        await asyncio.to_thread(self._send, {j: commanded[j] for j in target})
                    next_tick += STREAM_TICK
                    await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
            except Exception as e:  # any planner failure must leave the arm held
                logger.exception("follow failed")
                return f"planner failed ({e}), stopped. {await self._hold_and_where()}"
            return f"time up after {seconds:.0f} s. {await self._hold_and_where()}"

    async def _hold_and_where(self) -> str:
        await asyncio.to_thread(self._hold)
        return await asyncio.to_thread(self._where)

    async def _glide(self, targets: dict[str, float], seconds: float | None) -> str:
        """Move joints together along a smoothstep curve while watching for
        blocking and stop requests. Holds position if it ends early."""
        start = await asyncio.to_thread(self.positions)
        start = {j: start[j] for j in targets}
        longest = max(abs(targets[j] - start[j]) for j in targets)
        # never faster than MAX_SPEED, even if the LLM asks for a shorter time
        duration = max(longest * 1.5 / MAX_SPEED, MIN_MOVE_SECONDS, seconds or 0.0)

        over = dict.fromkeys(targets, 0)
        last = dict(start)
        tick = 0
        t0 = time.monotonic()
        while (elapsed := time.monotonic() - t0) < duration:
            if self._stop.is_set():
                await asyncio.to_thread(self._hold)
                return f"stopped by user. {self._where()}"
            s = elapsed / duration
            s = s * s * (3 - 2 * s)  # smoothstep: peak speed is 1.5x the average
            commanded = {j: start[j] + (targets[j] - start[j]) * s for j in targets}
            await asyncio.to_thread(self._send, commanded)

            tick += 1
            if (
                DETECT_BLOCKING
                and elapsed > GRACE_SECONDS
                and tick % CHECK_EVERY_TICKS == 0
            ):
                now = await asyncio.to_thread(self.positions)
                for j in targets:
                    lagging = abs(commanded[j] - now[j]) > LAG_LIMIT
                    stalled = abs(now[j] - last[j]) < STALL_DEGREES
                    over[j] = over[j] + 1 if lagging and stalled else 0
                    last[j] = now[j]
                blocked = [
                    j for j in targets if j != "gripper" and over[j] >= LAG_CHECKS
                ]
                if blocked:
                    await asyncio.to_thread(self._hold)
                    j = blocked[0]
                    obstacle = Obstacle(
                        at={a: now[a] for a in ARM_JOINTS},
                        joint=j,
                        direction=1 if targets[j] > start[j] else -1,
                    )
                    self.obstacles.append(obstacle)
                    logger.warning("%s blocked, learned: %s", j, obstacle.describe())
                    return (
                        f"blocked: {j} stuck at {now[j]:.0f} on the way to {targets[j]:.0f}, "
                        "stopped and holding. Something is in the way, probably the table. "
                        f"Remembered for this session: avoid {obstacle.describe()}."
                    )
            await asyncio.sleep(TICK_SECONDS)

        await asyncio.to_thread(self._send, targets)
        await asyncio.sleep(0.3)  # let the motors settle
        now = await asyncio.to_thread(self.positions)
        notes = []
        if (
            "gripper" in targets
            and abs(now["gripper"] - targets["gripper"]) > LAG_LIMIT
        ):
            notes.append(
                f"gripper stopped at {now['gripper']:.0f} instead of "
                f"{targets['gripper']:.0f}, probably holding something"
            )
        reached = ", ".join(f"{j} {now[j]:.0f}" for j in targets)
        return "reached: " + reached + ("; " + "; ".join(notes) if notes else "")

    def _send(self, targets: dict[str, float]) -> None:
        with self._bus:
            self.robot.send_action({f"{j}.pos": v for j, v in targets.items()})

    def _hold(self) -> None:
        """Freeze every joint at its present position."""
        now = self.positions()
        with self._bus:
            self.robot.bus.sync_write("Goal_Position", now)

    def _where(self) -> str:
        now = self.positions()
        return "Now at " + ", ".join(f"{j} {now[j]:.0f}" for j in JOINTS) + "."
