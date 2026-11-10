"""RealArm's motion logic against a simulated bus, no hardware needed."""

import asyncio
import json
import threading

import pytest
from livekit.agents import AgentSession, inference

import real_arm
from real_arm import JOINTS, JointTarget, Keyframe, MoveRejectedError, RealArm
from real_assistant import RealArmAssistant


class SimArm(RealArm):
    """RealArm with the serial bus replaced: joints follow commands instantly,
    except `stuck` joints, which can't pass their given angle."""

    def __init__(
        self,
        stuck: dict[str, float] | None = None,
        slow: dict[str, float] | None = None,
    ) -> None:
        self.limits = dict.fromkeys(JOINTS, (-90.0, 90.0)) | {"gripper": (0.0, 100.0)}
        self._stop = asyncio.Event()
        self._motion = asyncio.Lock()
        self._bus = threading.Lock()
        self.pos = dict.fromkeys(JOINTS, 0.0)
        self.stuck = stuck or {}
        self.slow = slow or {}  # joint -> max degrees per command (lags but moves)
        self.sent = 0
        self.held = False
        self.obstacles = []

    def positions(self) -> dict[str, float]:
        return dict(self.pos)

    def _send(self, targets: dict[str, float]) -> None:
        self.sent += 1
        for j, v in targets.items():
            if j in self.stuck:
                v = min(v, self.stuck[j])
            if j in self.slow:
                step = self.slow[j]
                v = self.pos[j] + max(-step, min(step, v - self.pos[j]))
            self.pos[j] = v

    def _hold(self) -> None:
        self.held = True


@pytest.fixture(autouse=True)
def fast_motion(monkeypatch):
    monkeypatch.setattr(real_arm, "MAX_SPEED", 400.0)
    monkeypatch.setattr(real_arm, "DETECT_BLOCKING", True)  # off in production
    monkeypatch.setattr(real_arm, "MIN_MOVE_SECONDS", 0.2)
    monkeypatch.setattr(real_arm, "GRACE_SECONDS", 0.02)


async def test_move_reaches_target() -> None:
    arm = SimArm()
    result = await arm.move({"shoulder_pan": 30, "elbow_flex": -20})
    assert result.startswith("reached")
    assert arm.pos["shoulder_pan"] == 30 and arm.pos["elbow_flex"] == -20


async def test_blocked_joint_stops_and_holds() -> None:
    arm = SimArm(stuck={"elbow_flex": 10})
    result = await arm.move({"elbow_flex": 80}, seconds=1.0)
    assert result.startswith("blocked: elbow_flex")
    assert arm.held


async def test_slow_but_moving_joint_is_not_blocked() -> None:
    # falls well over LAG_LIMIT behind, but keeps moving: load, not an obstacle
    arm = SimArm(slow={"elbow_flex": 1.0})
    result = await arm.move({"elbow_flex": -60}, seconds=0.6)
    assert result.startswith("reached"), result


async def test_stop_ends_move_early() -> None:
    arm = SimArm()
    move = asyncio.create_task(arm.move({"shoulder_pan": 80}, seconds=2.0))
    await asyncio.sleep(0.2)
    arm.stop()
    result = await move
    assert result.startswith("stopped by user")
    assert arm.held and arm.pos["shoulder_pan"] < 80


async def test_out_of_range_is_rejected_before_moving() -> None:
    arm = SimArm()
    with pytest.raises(MoveRejectedError, match="elbow_flex 120"):
        await arm.move({"elbow_flex": 120})
    assert arm.sent == 0


async def test_motion_validates_every_keyframe_first() -> None:
    arm = SimArm()
    frames = [
        Keyframe(targets=[JointTarget(joint="shoulder_pan", degrees=20)]),
        Keyframe(targets=[JointTarget(joint="wrist_flex", degrees=200)]),
    ]
    with pytest.raises(MoveRejectedError, match="keyframe 2"):
        await arm.run_motion(frames)
    assert arm.sent == 0


async def test_wave_is_composed_as_motion() -> None:
    arm = SimArm()
    async with (
        inference.LLM(model="openai/gpt-4.1-mini") as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(RealArmAssistant(arm))
        result = await session.run(user_input="Wink mal mit dem Arm.")

        call = result.expect.contains_function_call(name="run_motion")
        args = json.loads(call.event().item.arguments)
        frames = len(args["keyframes"]) * (args.get("repeat") or 1)
        assert frames >= 3, "a wave needs back-and-forth keyframes"


async def test_reach_up_raises_the_arm() -> None:
    """shoulder_lift's name misleads: positive lowers the arm forward. Reaching up
    must use the vertical landmark (about -12) and an open elbow (about -67)."""
    arm = SimArm()
    arm.limits = arm.limits | {
        "shoulder_lift": (-98.0, 98.0),
        "elbow_flex": (-94.0, 94.0),
    }
    async with (
        inference.LLM(model="openai/gpt-4.1-mini") as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(RealArmAssistant(arm))
        await session.run(user_input="Reach for the stars!")
        assert -40 < arm.pos["shoulder_lift"] < 15, arm.pos
        assert arm.pos["elbow_flex"] < -40, arm.pos


async def test_blocked_spot_is_avoided_for_the_session() -> None:
    arm = SimArm(stuck={"elbow_flex": 10})
    first = await arm.move({"elbow_flex": 80}, seconds=1.0)
    assert "Remembered" in first and len(arm.obstacles) == 1

    # same region again: rejected before anything moves
    sent = arm.sent
    with pytest.raises(MoveRejectedError, match="obstacle found earlier"):
        await arm.move({"elbow_flex": 50})
    assert arm.sent == sent

    # moving away from the obstacle is fine
    assert (await arm.move({"elbow_flex": -20})).startswith("reached")

    # a path that passes through it is rejected too, even if the target is beyond
    with pytest.raises(MoveRejectedError):
        await arm.move({"elbow_flex": 85})

    # with another joint far enough elsewhere, the same angle is allowed again
    arm.stuck = {}
    await arm.move({"shoulder_lift": -40})
    assert (await arm.move({"elbow_flex": 50})).startswith("reached")


async def test_motion_keyframes_check_learned_obstacles() -> None:
    arm = SimArm()
    arm.obstacles.append(
        real_arm.Obstacle(
            at=dict.fromkeys(real_arm.ARM_JOINTS, 0.0), joint="wrist_flex", direction=1
        )
    )
    frames = [
        Keyframe(targets=[JointTarget(joint="shoulder_pan", degrees=5)]),
        Keyframe(targets=[JointTarget(joint="wrist_flex", degrees=30)]),
    ]
    with pytest.raises(MoveRejectedError, match="keyframe 2"):
        await arm.run_motion(frames)
    assert arm.sent == 0


class ListSource:
    """A TargetSource playing fixed targets, one per tick, then done."""

    def __init__(self, targets: list) -> None:
        self.targets = list(targets)

    async def next(self, now: dict[str, float]) -> dict[str, float] | None:
        if not self.targets:
            raise real_arm.PlanDone("plan finished")
        target = self.targets.pop(0)
        if isinstance(target, Exception):
            raise target
        return target


async def test_follow_caps_speed_and_finishes() -> None:
    arm = SimArm()
    result = await arm.follow(ListSource([{"shoulder_pan": 50.0}] * 4), seconds=5)
    assert result.startswith("plan finished"), result
    step = real_arm.STREAM_SPEED * real_arm.STREAM_TICK
    assert arm.pos["shoulder_pan"] == pytest.approx(4 * step)  # ramped, not jumped to 50
    assert arm.held


async def test_follow_none_holds_the_last_command() -> None:
    arm = SimArm()
    await arm.follow(ListSource([{"shoulder_pan": 1.0}, None, None]), seconds=5)
    assert arm.pos["shoulder_pan"] == pytest.approx(1.0)
    assert arm.sent == 1


async def test_follow_rejects_a_target_outside_the_limits() -> None:
    arm = SimArm()
    result = await arm.follow(ListSource([{"shoulder_pan": 5.0}, {"elbow_flex": 120.0}]), seconds=5)
    assert result.startswith("rejected a planned position"), result
    assert arm.pos["elbow_flex"] == 0.0 and arm.held


async def test_follow_planner_error_holds() -> None:
    arm = SimArm()
    result = await arm.follow(ListSource([{"shoulder_pan": 1.0}, RuntimeError("no credit")]), seconds=5)
    assert result.startswith("planner failed (no credit)"), result
    assert arm.held


async def test_follow_stops_on_request() -> None:
    arm = SimArm()
    task = asyncio.create_task(arm.follow(ListSource([{"shoulder_pan": 80.0}] * 100), seconds=10))
    await asyncio.sleep(0.2)
    arm.stop()
    result = await task
    assert result.startswith("stopped by user"), result
    assert arm.pos["shoulder_pan"] < 20
