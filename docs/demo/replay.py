"""Scripted re-enactment of a voice session, for the README animation (docs/demo/demo.tape).

Not a recording: it prints what `lk agent console` shows, with timing, so the GIF is
reproducible. Run with: python docs/demo/replay.py
"""

import sys
import time

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
CYAN, GREEN, YELLOW, RED, MAGENTA = "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[35m"


def out(text: str = "", pause: float = 0.0) -> None:
    print(text, flush=True)
    time.sleep(pause)


def typed(prefix: str, text: str, speed: float = 0.035) -> None:
    """Streams text word by word, like a live transcript."""
    sys.stdout.write(prefix)
    for word in text.split(" "):
        sys.stdout.write(word + " ")
        sys.stdout.flush()
        time.sleep(speed * max(len(word), 2))
    print()


def you(text: str) -> None:
    typed(f"{CYAN}{BOLD}  you  ›{RESET} ", text)
    time.sleep(0.5)


def agent(text: str) -> None:
    typed(f"{GREEN}{BOLD}agent  ›{RESET} ", text, speed=0.025)
    time.sleep(0.8)


def tool(call: str, result: str, color: str = YELLOW, wait: float = 1.2) -> None:
    out(f"{DIM}         ⚙ {call}{RESET}", wait)
    out(f"{color}         ↳ {result}{RESET}", 0.6)


def main() -> None:
    out(f"{DIM}connecting to LiveKit Cloud…{RESET}", 0.6)
    out(
        f"{BOLD}so100-voice{RESET} {DIM}·{RESET} STT kugel-ear {DIM}·{RESET} LLM gpt-4.1-mini "
        f"{DIM}·{RESET} TTS kugel-3 {DIM}·{RESET} {MAGENTA}ROBOT=real{RESET}",
        0.3,
    )
    out(f"{DIM}SO-100 on /dev/tty.usbmodem… · 6 servos · limits loaded · 🎙 listening{RESET}", 1.0)
    out()

    you("Hi there")
    agent("Hi, ready.")

    you("Wave at me")
    tool(
        "run_motion(keyframes=4, repeat=3)",
        "motion done (4 keyframes x 3). Now at shoulder_lift -12, elbow_flex -60, wrist_roll 0…",
        GREEN,
        wait=1.8,
    )
    agent("Done.")

    you("Bend the elbow to two hundred degrees")
    tool("move_joints(elbow_flex=200)", "rejected, nothing moved: elbow_flex 200 is outside -94 to 94", RED)
    agent("That's past the elbow's limit, so I didn't move.")

    you("Go to a working pose and pick up the CPU")
    tool("move_joints(shoulder_lift=0, elbow_flex=20, wrist_flex=70)", "reached: shoulder_lift 0, elbow_flex 20, wrist_flex 70", GREEN)
    agent("On it.")
    out(f"{DIM}         ⚙ do_task(\"pick up the cpu\", seconds=20){RESET}", 0.5)
    for i, (rtt, note) in enumerate([(2114, "approach"), (2380, "descend"), (1975, "align jaws")], 1):
        out(
            f"{DIM}           wrist cam → {MAGENTA}Dream-1{RESET}{DIM} · plan {i} · 48 TCP poses · "
            f"round trip {rtt} ms · IK ok · ≤25°/s · {note}{RESET}",
            1.1,
        )
    you("Stop!")
    out(f"{RED}{BOLD}         ⛔ stop heard in the live transcript → arm holding{RESET}", 0.4)
    out(f"{YELLOW}         ↳ stopped by user. Now at shoulder_pan 2, shoulder_lift 9, elbow_flex 24…  Dream-1 sent 3 plans.{RESET}", 0.6)
    agent("Stopped. Holding where it is.")
    out("", 2.5)


if __name__ == "__main__":
    main()
