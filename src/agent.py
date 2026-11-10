import asyncio
import logging
import os
import re
import textwrap

from dotenv import load_dotenv
from kugelaudio import livekit as kugelaudio
from kugelaudio import load_audio
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    STTContextOptions,
    TurnHandlingOptions,
    UserInputTranscribedEvent,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics

from robot import POSES, FakeRobot, GripperState, PoseName, Robot, Step, run_steps

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self, robot: Robot | None = None) -> None:
        self.robot = robot or FakeRobot()
        super().__init__(
            # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
            # See all available models at https://docs.livekit.io/agents/models/llm/
            llm=inference.LLM(model="gemma-4-31b"),
            # To use a realtime model instead of a voice pipeline, replace the LLM
            # with a realtime model and remove the STT/TTS from the AgentSession
            # (Note: This is for OpenAI GPT-Live, the recommended speech-to-speech
            # model. For other providers, see https://docs.livekit.io/agents/models/realtime/)
            # Replace the llm argument with the model's LiveKit Inference ID:
            #    llm="openai/gpt-live-1",
            instructions=textwrap.dedent(
                """\
                You are the voice operator of a small robot arm with a gripper. You move it only by calling your tools.
                The user speaks English, sometimes German. Always reply in English. At the beginning of the conversation
                you don't ask the user what they want to do.

                # The arm
                - Poses: home (upright), rest (folded), reach_forward (gripper near the table in front), look_up (wrist tilted back).
                - Joints in degrees: shoulder_pan rotates the base left and right (minus 114 to 114, 0 is straight ahead).
                  shoulder_lift 0 to 200, elbow_flex minus 180 to 0, wrist_flex minus 143 to 68, wrist_roll minus 180 to 180.
                - Gripper: open or closed.

                # Moving
                - For a single pose or gripper command use move_to_pose or set_gripper.
                - For anything with several steps, single joints, or repetition use run_sequence and build the whole motion as one list of steps.
                  Example: a circle or sweep is shoulder_pan to minus 60, then 60, then back to 0. Opening and closing the gripper during it means gripper steps in between.
                - Steps run one after another, not at the same time.
                - Never claim the arm moved unless the tool call succeeded.

                # Replies
                - Plain spoken English, no lists, markdown, or emojis. Numbers as words.
                - After a move, confirm in a few words.
                - If you can't do something or a move is rejected, say only that, in one short sentence, and stop. No alternatives, no list of what you can do, no follow-up question, unless the user asks.
                  Example: user "Make me a coffee." You: "Sorry, coffee's not in my skill set."
                """
            ),
        )

    @function_tool
    async def move_to_pose(self, context: RunContext, pose: PoseName) -> str:
        """Move the arm to one of the predefined poses.

        Args:
            pose: The pose to move to.
        """
        try:
            await self.robot.move_joints(POSES[pose])
        except ValueError as e:
            logger.warning("rejected move to %s: %s", pose, e)
            return f"Move rejected: {e}"
        return f"Arm is now at {pose}."

    @function_tool
    async def set_gripper(self, context: RunContext, state: GripperState) -> str:
        """Open or close the gripper.

        Args:
            state: Whether the gripper should be open or closed.
        """
        await self.robot.set_gripper(state)
        return f"Gripper is {state}."

    @function_tool
    async def run_sequence(self, context: RunContext, steps: list[Step]) -> str:
        """Run a motion made of several steps, in order.

        Every step is checked before the arm moves; if one is invalid, nothing runs.

        Args:
            steps: The steps to run, in order.
        """
        try:
            await run_steps(self.robot, steps)
        except ValueError as e:
            logger.warning("rejected sequence: %s", e)
            return f"Sequence rejected, nothing moved: {e}"
        return f"Sequence of {len(steps)} steps done."

    @function_tool
    async def get_status(self, context: RunContext) -> dict:
        """Report the arm's current joint angles and gripper state."""
        return self.robot.status()


def noise_filter():
    """Input audio filter, applied before VAD, STT and turn detection.

    NOISE_FILTER=clarity (default): KugelAudio Clarity, runs on KugelAudio's servers.
      With CLARITY_SPEAKER_WAV set to a clean 2-8 s sample of your voice, it keeps
      only your voice (drops other people and the agent's own voice from the speakers).
    NOISE_FILTER=ai_coustics: ai-coustics voice isolation (always in Chinese mode).
    See https://docs.kugelaudio.com/integrations/livekit
    """
    if chinese() or os.getenv("NOISE_FILTER", "clarity") == "ai_coustics":
        return ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_VF_L)
    speaker_wav = os.getenv("CLARITY_SPEAKER_WAV")
    return kugelaudio.enhancement(
        speaker=load_audio(speaker_wav) if speaker_wav else None,
        on_error=lambda e: logger.error("Clarity disabled, using raw audio: %s", e),
    )


# Keyterms bias the STT toward distinctive words it would otherwise misspell.
KEYTERMS = [
    "Greifer",
    "gripper",
    "home",
    "rest",
    "reach forward",
    "look up",
    "shoulder pan",
    "shoulder lift",
    "elbow flex",
    "wrist flex",
    "wrist roll",
    "Stopp",
]


def chinese() -> bool:
    """LANGUAGE=zh: a Chinese conversation on LiveKit Inference models only
    (AssemblyAI STT, LiveKit turn detector, Fish Audio TTS)."""
    return os.getenv("LANGUAGE") == "zh"


def use_kugel_ear() -> bool:
    """STT=kugel (default): KugelAudio Kugel Ear, which also ends the turns.
    STT=kugel_livekit_turns: Kugel Ear for the text, the LiveKit turn detector ends the turns.
    STT=assemblyai: AssemblyAI via LiveKit Inference with the LiveKit turn detector."""
    return not chinese() and os.getenv("STT", "kugel") != "assemblyai"


def use_kugel_turns() -> bool:
    return use_kugel_ear() and os.getenv("STT", "kugel") == "kugel"


def speech_input() -> dict:
    """The AgentSession's speech-to-text arguments."""
    if use_kugel_ear():
        # Kugel Ear, reads KUGELAUDIO_API_KEY. No language set: detects German and English.
        # See https://docs.kugelaudio.com/api-reference/endpoints/speech-to-text
        if use_kugel_turns():
            # KugelTurn ends the turns; thinking pauses should not end the turn
            turns = {"turn_detection_level": "conservative"}
        else:
            # The LiveKit turn detector decides; Kugel Ear finalizes its text
            # after a fixed 1 s pause instead of running KugelTurn.
            turns = {"turn_detection": "vad"}
        return {
            "stt": kugelaudio.STT(
                **turns,
                # Boosted phrases come out in this spelling instead of a misheard one
                boosted_phrases=KEYTERMS,
            )
        }
    if chinese():
        return {
            "stt": inference.STT(model="assemblyai/universal-3-6-pro", language="zh"),
            # gripper, robot arm, wrist, elbow, shoulder, stop ("夹爪" was heard as "家抓")
            "stt_context_options": STTContextOptions(
                keyterms=["夹爪", "机械臂", "手腕", "肘部", "肩膀", "停止"]
            ),
        }
    # No language set: Universal-3.6 Pro detects German and English and code-switches.
    # Keyterm detection also extracts terms from the live conversation.
    # See https://docs.livekit.io/agents/models/stt/keyterms/
    return {
        "stt": inference.STT(model="assemblyai/universal-3-6-pro"),
        "stt_context_options": STTContextOptions(
            keyterms=KEYTERMS, keyterm_detection={"enabled": True}
        ),
    }


def turn_detection():
    """With KugelTurn, Kugel Ear ends each turn itself on the server, so its transcripts
    drive the agent. Otherwise the LiveKit turn detector, an end-of-turn model on the
    user's audio; AgentSession supplies the required VAD.
    See https://docs.livekit.io/agents/build/turns"""
    return "stt" if use_kugel_turns() else inference.TurnDetector()


# Not "halt": in German it is also a filler word ("das ist halt so"). 停 covers 停止 and 停下.
STOP_WORDS = re.compile(r"\bstop+\w*\b|停", re.IGNORECASE)


def text_to_speech():
    """KugelAudio, reads KUGELAUDIO_API_KEY from .env.local.
    In Chinese mode Fish Audio via LiveKit Inference; FISH_VOICE picks another voice
    from https://fish.audio/app/discovery/ (default: Adrian, speaking Chinese)."""
    if chinese():
        # See https://docs.livekit.io/agents/models/tts/fishaudio/
        return inference.TTS(
            model="fishaudio/s2.1-pro",
            voice=os.getenv("FISH_VOICE", "bf322df2096a46f18c579d0baa36f41d"),
            language="zh",
        )
    # See https://docs.kugelaudio.com/integrations/livekit
    return kugelaudio.TTS(
        model="kugel-3.5",
        voice_id=287,  # "Chucker": middle-aged, conversational
        language="en",
        speed=1.2,  # 0.8-1.2, a touch quicker sounds more energetic
        temperature=0.6,  # 0.0-1.0, more variation in delivery
    )


def in_chinese(instructions: str) -> str:
    """The prompts are written in English; only their language rules change."""
    for english, chinese_rule in [
        (
            "The user speaks English, sometimes German. Always reply in English.",
            "The user speaks Mandarin Chinese. Always reply in Mandarin Chinese (simplified characters).",
        ),
        ("Plain spoken English", "Plain spoken Mandarin Chinese"),
    ]:
        assert english in instructions, (
            f"prompt changed, update in_chinese(): {english!r}"
        )
        instructions = instructions.replace(english, chinese_rule)
    return instructions


async def robot_agent(ctx: JobContext, session: AgentSession) -> Agent:
    agent = await _robot_agent(ctx, session)
    if chinese():
        await agent.update_instructions(in_chinese(agent.instructions))
    return agent


async def _robot_agent(ctx: JobContext, session: AgentSession) -> Agent:
    """ROBOT=real drives the physical SO-100; anything else keeps the fake robot
    and its Rerun view. The real-arm code (and torch) only loads when needed."""
    if os.getenv("ROBOT", "fake") != "real":
        return Assistant(robot=FakeRobot(visualize=True))

    from real_arm import RealArm
    from real_assistant import RealArmAssistant

    arm = RealArm()
    await asyncio.to_thread(arm.connect)
    ctx.add_shutdown_callback(lambda: asyncio.to_thread(arm.disconnect))

    # Stop instantly on "stop" in the live transcript, before the turn ends and
    # without waiting for the LLM.
    @session.on("user_input_transcribed")
    def _stop_on_keyword(ev: UserInputTranscribedEvent) -> None:
        if STOP_WORDS.search(ev.transcript):
            logger.info("stop word heard: %r", ev.transcript)
            arm.stop()

    return RealArmAssistant(arm)


server = AgentServer()


@server.rtc_session(agent_name="so100-voice")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline: speech-to-text per STT/LANGUAGE, text-to-speech per LANGUAGE
    session = AgentSession(
        # Speech-to-text and turn detection: see speech_input()
        **speech_input(),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        tts=text_to_speech(),
        turn_handling=TurnHandlingOptions(
            turn_detection=turn_detection(),
            # Commands often contain thinking pauses ("fahr nach links ... und dann ..."),
            # so wait longer after silence before committing the turn.
            # See https://docs.livekit.io/agents/logic/turns/tuning/
            endpointing={"mode": "fixed", "min_delay": 0.8, "max_delay": 3.5},
            # VAD interruptions: any speech of 0.3 s cuts the agent off. "adaptive" tried to
            # tell interruptions from backchannels ("mhm") and kept talking over the user.
            # See https://docs.livekit.io/agents/logic/turns/tuning/
            interruption={"mode": "vad", "min_duration": 0.3},
            # allow the LLM to generate a response while waiting for the end of turn
            # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
            preemptive_generation={"enabled": True},
        ),
    )

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=await robot_agent(ctx, session),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_filter(),
            ),
        ),
    )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = anam.AvatarSession(
    #     persona_config=anam.PersonaConfig(
    #         name="...",
    #         avatarId="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/anam
    #     ),
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
