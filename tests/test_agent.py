import json

import pytest
from livekit.agents import AgentSession, inference

from agent import Assistant
from robot import POSES, FakeRobot, Step, plan_steps, run_steps

LLM_MODEL = "openai/gpt-4.1-mini"


async def test_moves_to_named_pose() -> None:
    robot = FakeRobot()
    async with (
        inference.LLM(model=LLM_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant(robot=robot))
        result = await session.run(user_input="Fahr bitte in die home Pose.")

        result.expect.next_event().is_function_call(
            name="move_to_pose", arguments={"pose": "home"}
        )
        result.expect.next_event().is_function_call_output()
        assert robot.status()["joints"]["shoulder_lift"] == POSES["home"][1]


async def test_opens_gripper() -> None:
    robot = FakeRobot()
    async with (
        inference.LLM(model=LLM_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant(robot=robot))
        result = await session.run(user_input="Mach den Greifer auf.")

        result.expect.next_event().is_function_call(
            name="set_gripper", arguments={"state": "open"}
        )
        assert robot.status()["gripper"] == "open"


async def test_sweep_with_gripper_builds_sequence() -> None:
    robot = FakeRobot()
    async with (
        inference.LLM(model=LLM_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant(robot=robot))
        result = await session.run(
            user_input="Mach einen Kreis mit der Basis und öffne und schließe dabei den Greifer."
        )

        call = result.expect.next_event().is_function_call(name="run_sequence")
        args = json.loads(call.event().item.arguments)
        steps = [Step.model_validate(s) for s in args["steps"]]
        pans = [s.degrees for s in steps if s.joint == "shoulder_pan"]
        grips = [s.gripper for s in steps if s.action == "gripper"]
        assert min(pans) < 0 < max(pans), "base should sweep both directions"
        assert "open" in grips and "closed" in grips


async def test_cannot_do_reply_is_short() -> None:
    async with (
        inference.LLM(model=LLM_MODEL) as llm,
        inference.LLM(model="openai/gpt-4.1-mini") as judge,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant(robot=FakeRobot()))
        result = await session.run(user_input="Mach mir einen Kaffee.")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge,
                intent="Says in one short sentence that it can't do this. "
                "Does NOT list the poses, joints, or other things it can do instead.",
            )
        )


def test_invalid_step_rejects_whole_sequence() -> None:
    steps = [
        Step(action="joint", joint="shoulder_pan", degrees=30),
        Step(action="joint", joint="elbow_flex", degrees=50),  # limit is -180..0
    ]
    with pytest.raises(ValueError, match="step 2"):
        plan_steps(POSES["rest"], steps)


async def test_sequence_runs_in_order() -> None:
    robot = FakeRobot()
    await run_steps(
        robot,
        [
            Step(action="joint", joint="shoulder_pan", degrees=-60),
            Step(action="gripper", gripper="open"),
            Step(action="joint", joint="shoulder_pan", degrees=60),
        ],
    )
    assert robot.status()["joints"]["shoulder_pan"] == 60
    assert robot.status()["gripper"] == "open"
