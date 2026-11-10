import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Literal, Protocol, get_args

from pydantic import BaseModel, Field

logger = logging.getLogger("robot")

URDF_PATH = Path(__file__).resolve().parent.parent / "assets" / "so100" / "so100.urdf"

JointName = Literal[
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINTS: tuple[str, ...] = get_args(JointName)

PoseName = Literal["home", "rest", "reach_forward", "look_up"]
GripperState = Literal["open", "closed"]

MAX_STEPS = 40
MAX_WAIT_SECONDS = 5.0

# Joint angles in degrees, in JOINTS order, measured in the URDF's zero frame
# (all zeros = arm folded). Placeholders until the real arm is calibrated:
# LeRobot's calibrated zero differs, so re-record these on the physical robot.
POSES: dict[str, tuple[float, ...]] = {
    "home": (0.0, 60.0, -120.0, 0.0, 0.0),
    "rest": (0.0, 0.0, 0.0, 0.0, 0.0),
    "reach_forward": (0.0, 130.0, -80.0, 50.0, 0.0),
    "look_up": (0.0, 60.0, -120.0, -90.0, 0.0),
}

# Safe range per joint in degrees, taken from the URDF limits. Every command is
# checked against this before it reaches the hardware.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-114.0, 114.0),
    "shoulder_lift": (0.0, 200.0),
    "elbow_flex": (-180.0, 0.0),
    "wrist_flex": (-143.0, 68.0),
    "wrist_roll": (-180.0, 180.0),
}

GRIPPER_ANGLES = {"open": 80.0, "closed": 0.0}


def validate(angles: tuple[float, ...]) -> None:
    if len(angles) != len(JOINTS):
        raise ValueError(f"expected {len(JOINTS)} joint angles, got {len(angles)}")
    for name, angle in zip(JOINTS, angles, strict=True):
        low, high = JOINT_LIMITS[name]
        if not low <= angle <= high:
            raise ValueError(f"{name}={angle} outside safe range [{low}, {high}]")


class Robot(Protocol):
    async def move_joints(self, angles: tuple[float, ...]) -> None: ...
    async def set_gripper(self, state: GripperState) -> None: ...
    def status(self) -> dict: ...


class Step(BaseModel):
    """One step of a motion sequence. Fill only the fields for its action."""

    action: Literal["pose", "joint", "gripper", "wait"]
    pose: PoseName | None = Field(None, description="For action=pose.")
    joint: JointName | None = Field(None, description="For action=joint.")
    degrees: float | None = Field(
        None, description="For action=joint: absolute target angle in degrees."
    )
    gripper: GripperState | None = Field(None, description="For action=gripper.")
    seconds: float | None = Field(
        None, description=f"For action=wait: pause, at most {MAX_WAIT_SECONDS}."
    )


def plan_steps(start: tuple[float, ...], steps: list[Step]) -> list[Step]:
    """Check every step against the joint limits before anything moves.

    Raises ValueError naming the first bad step, so a sequence either runs
    completely or not at all.
    """
    if not steps:
        raise ValueError("sequence is empty")
    if len(steps) > MAX_STEPS:
        raise ValueError(f"sequence has {len(steps)} steps, at most {MAX_STEPS}")
    angles = list(start)
    for i, step in enumerate(steps, 1):
        try:
            match step.action:
                case "pose":
                    if step.pose is None:
                        raise ValueError("missing pose")
                    angles = list(POSES[step.pose])
                case "joint":
                    if step.joint is None or step.degrees is None:
                        raise ValueError("missing joint or degrees")
                    angles[JOINTS.index(step.joint)] = step.degrees
                case "gripper":
                    if step.gripper is None:
                        raise ValueError("missing gripper state")
                case "wait":
                    if step.seconds is None or not 0 < step.seconds <= MAX_WAIT_SECONDS:
                        raise ValueError(f"wait must be 0 to {MAX_WAIT_SECONDS} s")
            validate(tuple(angles))
        except ValueError as e:
            raise ValueError(f"step {i} ({step.action}): {e}") from e
    return steps


async def run_steps(robot: Robot, steps: list[Step]) -> None:
    start = tuple(robot.status()["joints"].values())
    for step in plan_steps(start, steps):
        match step.action:
            case "pose":
                await robot.move_joints(POSES[step.pose])
            case "joint":
                angles = list(robot.status()["joints"].values())
                angles[JOINTS.index(step.joint)] = step.degrees
                await robot.move_joints(tuple(angles))
            case "gripper":
                await robot.set_gripper(step.gripper)
            case "wait":
                await asyncio.sleep(step.seconds)


class RerunView:
    """Draws the SO-100 model in the Rerun viewer and animates its joints."""

    def __init__(self) -> None:
        import rerun as rr

        self._rr = rr
        # Live view, not a recording: joints are logged as static (no timeline), so
        # the viewer always shows the latest state immediately. The fixed
        # recording_id makes every run (demo, manual, agent) update the same view.
        # Flush every 20 ms instead of Rerun's default 200 ms so the animation is smooth.
        os.environ.setdefault("RERUN_FLUSH_TICK_SECS", "0.02")
        rr.init("so100-voice", recording_id="so100-voice", spawn=True)
        self._tree = rr.urdf.UrdfTree.from_file_path(URDF_PATH)
        self._tree.log_urdf_to_recording()

    def show(self, angles: tuple[float, ...], gripper_deg: float) -> None:
        values = {**dict(zip(JOINTS, angles, strict=True)), "gripper": gripper_deg}
        for name, deg in values.items():
            joint = self._tree.get_joint_by_name(name)
            self._rr.log(
                f"joint_transforms/{name}",
                joint.compute_transform(math.radians(deg)),
                static=True,
            )


class FakeRobot:
    """Stands in for the SO-100 until the hardware is connected.

    Logs every command and, with visualize=True, animates the arm in Rerun.
    """

    MOVE_SECONDS = 1.0
    STEPS = 30

    def __init__(self, visualize: bool = False) -> None:
        self._angles = POSES["rest"]
        self._gripper: GripperState = "closed"
        self._view = RerunView() if visualize else None
        self._render(self._angles, GRIPPER_ANGLES[self._gripper])

    async def move_joints(self, angles: tuple[float, ...]) -> None:
        validate(angles)
        logger.info(
            "[FAKE ROBOT] moving joints to %s", dict(zip(JOINTS, angles, strict=True))
        )
        gripper = GRIPPER_ANGLES[self._gripper]
        await self._animate(self._angles, angles, gripper, gripper)
        self._angles = angles

    async def set_gripper(self, state: GripperState) -> None:
        logger.info("[FAKE ROBOT] gripper -> %s", state)
        start, end = GRIPPER_ANGLES[self._gripper], GRIPPER_ANGLES[state]
        await self._animate(self._angles, self._angles, start, end)
        self._gripper = state

    def status(self) -> dict:
        return {
            "joints": dict(zip(JOINTS, self._angles, strict=True)),
            "gripper": self._gripper,
        }

    async def _animate(
        self,
        start: tuple[float, ...],
        end: tuple[float, ...],
        grip_start: float,
        grip_end: float,
    ) -> None:
        if self._view is None:
            return
        for step in range(1, self.STEPS + 1):
            t = step / self.STEPS
            t = t * t * (3 - 2 * t)  # ease in/out
            angles = tuple(a + (b - a) * t for a, b in zip(start, end, strict=True))
            self._render(angles, grip_start + (grip_end - grip_start) * t)
            await asyncio.sleep(self.MOVE_SECONDS / self.STEPS)

    def _render(self, angles: tuple[float, ...], gripper_deg: float) -> None:
        if self._view is not None:
            self._view.show(angles, gripper_deg)
