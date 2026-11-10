"""Dream-1 (Dream Labs) as a planner for the real arm: wrist camera + task in, arm motion out.

Dream-1 answers where the fingertip point (TCP) should go over the next 48 rows (20 rows/s).
It runs as a model trained on another arm ("other_embodiment", the docs' route B): it reads the
camera and the task, ignores the arm's joints, and anchors its answer on the TCP pose we send.
We send that pose from our own FK, solve each answered pose back to joints with our own IK
(so100_kinematics), and RealArm.follow() checks the limits and caps the speed before anything
moves. Docs: https://platform.dreamlabs.ai/api-documentation/control-loop/#on-another-arm

Needs the API key in ~/.dreamlabs/credentials. Costs one paid call per chunk, about every 1.5 s.
"""

import asyncio
import base64
import logging
import os
import threading
import time

import cv2
import dream
import numpy as np

import so100_kinematics as kin
from real_arm import PlanDone, RealArm

logger = logging.getLogger("dream_task")

CHECKPOINT = os.getenv("DREAM_CHECKPOINT", "dreamlabs/dream-1-beta")
CAMERA_INDEX = int(os.getenv("WRIST_CAMERA_INDEX", "0"))
# The wrist camera may be mounted turned: 0, 90, 180 or 270 degrees clockwise to make it upright.
CAMERA_ROTATE = int(os.getenv("WRIST_CAMERA_ROTATE", "0"))
FRAME_SIZE = (320, 176)  # width x height the model sees
ROW_S = 0.05  # 20 rows per second, Dream-1's default action_hz
STALE_FRAME_S = (
    0.2  # hold when the newest camera frame is older (Dream Labs safety rule)
)
MAX_IK_ERROR_M = 0.02  # a planned point the arm can't get within 2 cm of is skipped
MAX_FAILED_CALLS = 3
ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# The model needs a left arm too: a fixed parked pose, gripper open, black camera.
_PARK_R = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)  # fingers forward, jaw left
LEFT_PARKED = {
    "state.left_arm": [0.0] * 6,
    "state.left_gripper": [1.0],
    "state.left_grip": [1.0],
    "state.left_eef_xyz": [0.2, 0.3, 0.1],
    "state.left_eef_rot6": kin.rot6(_PARK_R),
}


def _jpg(image: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".jpg", image)[1].tobytes()).decode("ascii")


BLACK = _jpg(np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8))


class WristCamera:
    """Reads the camera on its own thread; latest() returns the newest frame as JPEG and its age."""

    def __init__(self) -> None:
        self._cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            raise RuntimeError(f"wrist camera {CAMERA_INDEX} did not open")
        self._frame: str | None = None
        self._at = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if CAMERA_ROTATE in ROTATIONS:
                frame = cv2.rotate(frame, ROTATIONS[CAMERA_ROTATE])
            self._frame, self._at = (
                _jpg(cv2.resize(frame, FRAME_SIZE)),
                time.monotonic(),
            )

    def latest(self) -> tuple[str | None, float]:
        return self._frame, time.monotonic() - self._at

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


def _ik_limits(arm: RealArm) -> tuple[np.ndarray, np.ndarray]:
    """RealArm's safe ranges as URDF radians, so IK never answers a joint RealArm would reject."""
    low, high = [], []
    for j in kin.ARM_JOINTS:
        a, b = (kin.SIGN[j] * v + kin.OFFSET[j] for v in arm.limits[j])
        low.append(np.radians(min(a, b)))
        high.append(np.radians(max(a, b)))
    return np.maximum(low, kin.LOWER), np.minimum(high, kin.UPPER)


class DreamPlanner:
    """A RealArm TargetSource: hands Dream-1 the camera and the TCP pose, plays its chunks back as joints."""

    def __init__(
        self, client: dream.DreamClient, camera: WristCamera, task: str, arm: RealArm
    ) -> None:
        self.client = client
        self.camera = camera
        self.prompt = f"{task} with the right arm"  # one arm: run it as the right one (Dream Labs docs)
        self.lower, self.upper = _ik_limits(arm)
        # (t_obs, rows, start row) of the chunk being played
        self.plan: tuple[float, list[dict[str, float] | None], int] | None = None
        self.call: asyncio.Task | None = None
        self.call_while_moving = False
        self.retry_at = 0.0
        self.failed = 0
        self.chunks = 0

    async def next(self, now: dict[str, float]) -> dict[str, float] | None:
        frame, age = self.camera.latest()
        if frame is None or age > STALE_FRAME_S:
            return None  # no fresh view: hold

        q = kin.calibrated_to_urdf(now)
        xyz, rotation = kin.fk(q)
        grip = [now["gripper"] / 100.0]
        observation = {
            "prompt": self.prompt,
            "state.right_arm": [0.0]
            * 6,  # the model's joint slots; ignored on another arm
            "state.right_gripper": grip,
            "state.right_grip": grip,
            "state.right_eef_xyz": list(xyz),
            "state.right_eef_rot6": kin.rot6(rotation),
            **LEFT_PARKED,
        }
        frames = {
            "video.overhead": BLACK,
            "video.wrist_l": BLACK,
            "video.wrist_r": frame,
        }
        self.client.set_observation(frames, observation, timestamp=time.monotonic())

        t = time.monotonic()
        if self.call is not None and self.call.done():
            self._take(self.call, q, t)
            self.call = None

        row, executed, remaining = None, 0, 0
        if self.plan is not None:
            t_obs, rows, start = self.plan
            k = int(
                (t - t_obs) / ROW_S
            )  # the row due now; rows already past are skipped
            if k < len(rows):
                row, executed, remaining = rows[k], k - start, len(rows) - k
        if (
            self.call is None
            and t >= self.retry_at
            and self.client.refill_due(executed, remaining)
        ):
            self.call_while_moving = remaining > 0
            self.call = asyncio.create_task(
                asyncio.to_thread(self._get_action, q, time.monotonic())
            )
        return row

    def _get_action(
        self, q: np.ndarray, t_obs: float
    ) -> tuple[float, list[dict[str, float] | None]]:
        """One Dream-1 call and the IK for its 48 rows, off the control loop."""
        action = self.client.get_action()["per_key"]
        rows: list[dict[str, float] | None] = []
        seed = q
        missed = 0
        for xyz, r6, grip in zip(
            action["action.right_eef_xyz"],
            action["action.right_eef_rot6"],
            action["action.right_grip"],
            strict=True,
        ):
            q_row, error = kin.ik(
                np.asarray(xyz),
                kin.from_rot6(r6),
                seed,
                lower=self.lower,
                upper=self.upper,
            )
            if error > MAX_IK_ERROR_M:
                rows.append(None)  # out of reach: hold for this row
                missed += 1
                continue
            seed = q_row
            rows.append(
                kin.urdf_to_calibrated(q_row)
                | {"gripper": float(np.clip(grip, 0, 1)) * 100}
            )
        if missed:
            logger.warning("%d of %d planned points out of reach", missed, len(rows))
        if missed == len(rows):
            raise PlanDone("every point Dream-1 planned is out of the arm's reach")
        return t_obs, rows

    def _take(self, call: asyncio.Task, q: np.ndarray, t: float) -> None:
        error = call.exception()
        if error is None:
            t_obs, rows = call.result()
            if not self.call_while_moving:
                # The arm held still during the call, so row 0 is still where it is now: play from
                # row 0. (A round trip can be longer than a whole chunk; skipping the rows past
                # would skip all of them.)
                t_obs = t
            self.plan = (t_obs, rows, int((t - t_obs) / ROW_S))
            self.chunks += 1
            self.failed = 0
            return
        if isinstance(error, dream.DreamSessionExpired):
            self.plan = None  # new session: ask again
        elif isinstance(
            error,
            (dream.DreamServerError, dream.DreamTimeout, dream.DreamProtocolError),
        ) and getattr(error, "retryable", True):
            self.failed = (
                self.failed + 1 if getattr(error, "reason", None) == "error" else 0
            )
            if self.failed > MAX_FAILED_CALLS:
                raise error
            wait = getattr(error, "retry_after_s", None) or 1.0
            logger.info(
                "Dream-1 not ready (%s), asking again in %.0f s",
                getattr(error, "reason", error),
                wait,
            )
            self.retry_at = t + wait
        else:
            raise error  # bad key, bad observation, out of credit, out of reach


async def run(arm: RealArm, task: str, seconds: float) -> str:
    """Let Dream-1 do `task` for at most `seconds`, through RealArm.follow()."""
    camera = await asyncio.to_thread(WristCamera)
    client = dream.DreamClient()
    try:
        client.policy(
            {
                "checkpoint": CHECKPOINT,
                "robot": "other_embodiment",
                "action_hz": 1 / ROW_S,
            }
        )
        planner = DreamPlanner(client, camera, task, arm)
        result = await arm.follow(planner, seconds)
        if planner.call is not None:
            await asyncio.wait([planner.call])  # get_action may not overlap close()
        return f"{result} Dream-1 sent {planner.chunks} plans."
    finally:
        await asyncio.to_thread(client.close)
        await asyncio.to_thread(camera.close)
