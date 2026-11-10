"""Ask Dream-1 (Dream Labs) for one motion plan from the wrist camera. Nothing is sent to the arm.

Run with: uv run python src/dream_try.py "pick up the red cube"

Needs the API key in ~/.dreamlabs/credentials. Each run costs a few credits.
The camera is OpenCV index 0, used as the right wrist camera; overhead and left
wrist get black frames. The hand pose sent is a placeholder (no IK yet), so the
printed plan is relative to that pose, not the real arm.
"""

import base64
import sys
import time

import cv2
import dream
import numpy as np

CHECKPOINT = "dreamlabs/dream-1-beta"
CAMERA_INDEX = 0
FRAME_SIZE = (320, 176)  # width x height the model sees

# Placeholder hand pose: 25 cm forward, 15 cm up, pointing forward. Six zero joints, gripper open.
PLACEHOLDER_STATE = {
    f"state.{side}_{part}": value
    for side in ("right", "left")
    for part, value in (
        ("arm", [0.0] * 6),
        ("gripper", [1.0]),
        ("grip", [1.0]),
        ("eef_xyz", [0.25, 0.0, 0.15]),
        ("eef_rot6", [0.0, 0.0, -1.0, 0.0, 1.0, 0.0]),
    )
}


def jpg(image: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".jpg", image)[1].tobytes()).decode("ascii")


def get_action(client: dream.DreamClient) -> dict:
    """get_action(), waiting while the model starts up or is busy."""
    failed = 0
    while True:
        try:
            return client.get_action()
        except dream.DreamServerError as error:
            failed = failed + 1 if error.reason == "error" else 0
            if not error.retryable or failed >= 3:
                raise
            wait = error.retry_after_s or 1.0
            print(f"waiting ({error.reason}), asking again in {wait:.0f} s")
            time.sleep(wait)


def main() -> None:
    task = " ".join(sys.argv[1:]) or "pick up the object in front of the gripper"
    observation = {"prompt": task, **PLACEHOLDER_STATE}
    black = jpg(np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8))

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    with dream.DreamClient() as client:
        client.policy({"checkpoint": CHECKPOINT, "robot": "other_embodiment"})
        for tick in range(9):  # the model reads the last 9 frames, 0.2 s apart
            for _ in range(6):  # 6 frames at 30 fps = 0.2 s
                ok, frame = cap.read()
            if not ok:
                sys.exit(f"camera {CAMERA_INDEX} gave no frame")
            frames = {
                "video.overhead": black,
                "video.wrist_l": black,
                "video.wrist_r": jpg(cv2.resize(frame, FRAME_SIZE)),
            }
            client.set_observation(frames, observation, timestamp=tick * 0.2)
        cap.release()

        action = get_action(client)["per_key"]
        xyz = np.asarray(action["action.right_eef_xyz"])
        grip = np.asarray(action["action.right_grip"])
        steps = [0, 16, 32, 47]
        print(f"task: {task}")
        print("right hand xyz (m) at steps 0/16/32/47:\n", np.round(xyz[steps], 3))
        print("right grip (0 closed, 1 open):", np.round(grip[steps], 2))
        print(f"round trip {client.last_timing['rtt_ms']:.0f} ms, model {client.last_timing['serve_infer_ms']} ms")


if __name__ == "__main__":
    main()
