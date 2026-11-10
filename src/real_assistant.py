"""Voice agent for the real SO-100 arm.

The LLM composes motions itself from a description of the arm's joints (meaning,
direction, safe range) instead of picking from hard-coded gestures. RealArm
validates every move, detects blocking, and reports back what happened, so the
LLM can explain or adjust. Any genuine interruption while the arm is moving
stops it; agent.py also stops it instantly when the transcript contains "stop".
"""

import asyncio
import logging
import textwrap

from livekit.agents import Agent, RunContext, function_tool, inference

import dream_task
from real_arm import (
    DETECT_BLOCKING,
    DIRECTIONS,
    JointTarget,
    Keyframe,
    MoveRejectedError,
    RealArm,
    load_poses,
)

logger = logging.getLogger("real_assistant")

BLOCKING_RULES = (
    "- When a move is blocked, the arm remembers that spot for the rest of the "
    "session and rejects paths through it. Learn from it: plan around known "
    "obstacles, for example raise the arm before swinging it, instead of retrying "
    "the same path."
)


def body_model(arm: RealArm) -> str:
    lines = []
    for joint, (low, high) in arm.limits.items():
        lines.append(
            f"- {joint}: {DIRECTIONS[joint]}. Safe range {low:.0f} to {high:.0f}."
        )
    poses = load_poses()
    for name, pose in poses.items():
        angles = ", ".join(f"{j} {v:.0f}" for j, v in pose.items())
        lines.append(f"- Saved pose {name}: {angles}.")
    return "\n".join(lines)


class RealArmAssistant(Agent):
    def __init__(self, arm: RealArm) -> None:
        self.arm = arm
        super().__init__(
            llm=inference.LLM(model="openai/gpt-4.1-mini"),
            instructions=textwrap.dedent(
                """\
                You are the voice operator of a real robot arm (SO-100) with a gripper, clamped to a table.
                The user speaks English, sometimes German. Always reply in English.

                # The arm
                Angles are degrees. Left, right, forward and back are from the robot's own view, as if standing behind it.
                {body}
                - Joint names can mislead: shoulder_lift does not lift. Higher shoulder_lift values lower the arm forward, toward the table.
                - "Up", "down", "forward" for the forearm and gripper depend on where the upper arm points. Reason in terms of folding and opening each joint, not screen directions.
                - Landmarks (approximate, plus or minus ten degrees): arm pointing straight up is shoulder_lift about -12, elbow_flex about -67, wrist_flex about 10.
                  Pointing horizontally forward is shoulder_lift about 80 with the elbow straight. Build other shapes relative to these.
                - The table is in front of and below the gripper. Moves that lower the arm while reaching forward can hit it.

                # Moving
                - You design motions yourself. There are no built-in gestures.
                  For a gesture like waving, nodding or spinning, first think about which joints create it, then call run_motion with keyframes, using repeat for rhythm.
                  For a single position use move_joints. Use go_to_pose for saved poses.
                - Call get_status first if you don't know where the arm is. It also lists obstacles learned this session.
                {blocking}
                - Prefer moderate angles, well inside the safe ranges. Big fast swings near the table are dangerous.
                - Every tool result says what actually happened: reached, blocked, stopped, or rejected with a reason.
                  If blocked or rejected, tell the user briefly why, then adjust the plan, for example raise the arm first. Never claim a move worked unless the result says reached or done.
                - If the user says stop, halt, or hör auf, call stop.

                # Handling objects
                - For anything that needs seeing and grasping (pick up, grab, push, put something somewhere), call do_task with the task in a few plain English words, for example "pick up the cpu". A vision model then drives the arm through the wrist camera.
                - Before do_task, the gripper should be above the table and pointing at the work area. If the arm is in rest, first move it to a working pose, for example shoulder_lift 0, elbow_flex 20, wrist_flex 70.
                - Say a short "On it." before do_task. The result says what happened; report it honestly, the vision model often fails.

                # Personality
                - Calm, friendly, matter-of-fact. A colleague at work, not a hype man. No exclamations or jokes.
                - When greeted, answer in a few words, for example "Hi, ready." Don't ask what the user wants to do.
                - Vary your phrasing. Never confirm two moves in a row with the same words.

                # Replies
                - Plain spoken English, no lists, markdown, or emojis. Numbers as words.
                - Keep it short. After a move, a few words, for example "Done."
                - If you can't do something, say only that, in one short sentence, and stop. No alternatives, no list of what you can do, no follow-up question, unless the user asks.
                  Example: user "Make me a coffee." You: "Sorry, coffee's not in my skill set."
                """
            ).format(
                body=body_model(arm), blocking=BLOCKING_RULES if DETECT_BLOCKING else ""
            ),
        )

    async def _run(self, context: RunContext, coro) -> str:
        """Run a motion; a genuine user interruption stops the arm."""
        task = asyncio.ensure_future(coro)
        await context.speech_handle.wait_if_not_interrupted([task])
        if context.speech_handle.interrupted and not task.done():
            self.arm.stop()
        try:
            return await task
        except MoveRejectedError as e:
            logger.warning("rejected: %s", e)
            return f"rejected, nothing moved: {e}"

    @function_tool
    async def move_joints(
        self,
        context: RunContext,
        targets: list[JointTarget],
        seconds: float | None = None,
    ) -> str:
        """Move one or more joints to absolute angles, all together.

        Args:
            targets: The joints to move and their target angles.
            seconds: How long the move should take. Omit for the safe default speed.
        """
        return await self._run(
            context, self.arm.move({t.joint: t.degrees for t in targets}, seconds)
        )

    @function_tool
    async def run_motion(
        self, context: RunContext, keyframes: list[Keyframe], repeat: int = 1
    ) -> str:
        """Play a motion made of keyframes, in order, for gestures and choreography.

        All keyframes are checked before the arm moves. The motion stops early if a
        joint is blocked or the user says stop.

        Args:
            keyframes: The poses to pass through, in order.
            repeat: How many times to play the whole sequence.
        """
        return await self._run(context, self.arm.run_motion(keyframes, repeat))

    @function_tool
    async def go_to_pose(self, context: RunContext, name: str) -> str:
        """Move to a pose the user saved earlier, listed under "Saved pose" in your instructions.

        Args:
            name: The saved pose's name, for example rest.
        """
        poses = load_poses()
        if name not in poses:
            return f"no saved pose {name}. Saved: {', '.join(poses) or 'none'}"
        # recorded by hand, often right at an end stop: pull just inside the margin
        return await self._run(context, self.arm.move(self.arm.clamp(poses[name])))

    @function_tool
    async def get_status(self, context: RunContext) -> dict:
        """Current angle and safe range of every joint."""
        return await asyncio.to_thread(self.arm.status)

    @function_tool
    async def do_task(self, context: RunContext, task: str, seconds: float = 20.0) -> str:
        """Let the vision model (Dream-1) do a hands-on task, looking through the wrist camera.

        Args:
            task: What to do, a few plain English words, for example "pick up the cpu".
            seconds: How long the model may try, at most 60.
        """
        return await self._run(context, dream_task.run(self.arm, task, min(seconds, 60.0)))

    @function_tool
    async def stop(self, context: RunContext) -> str:
        """Stop the arm immediately and hold it where it is."""
        self.arm.stop()
        return "stopped and holding"
