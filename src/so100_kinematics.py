# ruff: noqa: N803, N806  (R, T, J: rotation, transform, Jacobian, as in the maths)
"""SO-100 kinematics from the URDF: where the fingertips are (FK) and which joints put them somewhere (IK).

Three frames meet here:
- calibrated degrees: what RealArm reads and sends (LeRobot, 0 = middle of the calibrated range)
- URDF radians: the 3D model in assets/so100/so100.urdf (0 = arm folded)
- Dream frame: metres, x forward, y to the arm's left, z up, origin at the URDF base; what Dream-1 reads and answers

The tool point (TCP) is the point between the closed fingertips. TCP_FROM_WRIST_FLEX is measured
on the real arm by the user; see docs/real-arm.md before changing it.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF_PATH = Path(__file__).resolve().parent.parent / "assets" / "so100" / "so100.urdf"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Measured by the user (2026-10-28): wrist_flex axle to the point between the closed fingertips,
# gripper closed and in line with the forearm. Sideways offset not measured yet, assumed 0.
TCP_FROM_WRIST_FLEX = 0.16
WRIST_FLEX_TO_ROLL = 0.0601  # wrist_roll joint origin in the URDF
# In the gripper link's frame (the link after wrist_roll): fingers point along -y, roll turns about y.
TCP_IN_GRIPPER = np.array([0.0, -(TCP_FROM_WRIST_FLEX - WRIST_FLEX_TO_ROLL), 0.0])
APPROACH_IN_GRIPPER = np.array([0.0, -1.0, 0.0])  # the way the fingers point
JAW_IN_GRIPPER = np.array(
    [1.0, 0.0, 0.0]
)  # the line the fingers close along (sign fixed below)

# calibrated degrees -> URDF degrees: urdf = SIGN * calibrated + OFFSET.
# Offsets from the calibrated ranges meeting the URDF limits (docs/real-arm.md); signs checked with
# src/check_mapping.py on the real arm.
# pan and roll are mirrored: calibrated + turns right / counterclockwise from behind, URDF + the other way.
SIGN = {
    "shoulder_pan": -1.0,
    "shoulder_lift": 1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_roll": -1.0,
}
OFFSET = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 101.1,
    "elbow_flex": -97.4,
    "wrist_flex": -35.0,
    "wrist_roll": 0.0,
}


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _load_chain() -> list[
    tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]
]:
    """Per arm joint: fixed origin transform (4x4), axis, and limits in radians."""
    joints = {j.get("name"): j for j in ET.parse(URDF_PATH).getroot().findall("joint")}
    chain = []
    for name in ARM_JOINTS:
        j = joints[name]
        origin = j.find("origin")
        T = np.eye(4)
        T[:3, :3] = _rpy(*map(float, origin.get("rpy").split()))
        T[:3, 3] = list(map(float, origin.get("xyz").split()))
        axis = np.array(list(map(float, j.find("axis").get("xyz").split())))
        limit = j.find("limit")
        chain.append((T, axis, (float(limit.get("lower")), float(limit.get("upper")))))
    return chain


CHAIN = _load_chain()
LOWER = np.array([lim[0] for _, _, lim in CHAIN])
UPPER = np.array([lim[1] for _, _, lim in CHAIN])


def _gripper_frame(q: np.ndarray) -> np.ndarray:
    """4x4 pose of the gripper link in the URDF base frame, for URDF joint angles q (radians)."""
    T = np.eye(4)
    for (origin, axis, _), angle in zip(CHAIN, q, strict=True):
        R = np.eye(4)
        R[:3, :3] = _axis_angle(axis, angle)
        T = T @ origin @ R
    return T


def _urdf_to_dream_base() -> np.ndarray:
    """Rotation from the URDF base frame to the Dream frame (x forward, y left, z up).

    Up is the shoulder_pan axis. Forward is where the TCP points with the arm stretched out
    horizontally at pan 0 (lift past vertical, elbow straight, wrist in line).
    """
    pan_origin, pan_axis, _ = CHAIN[0]
    up = pan_origin[:3, :3] @ pan_axis
    stretched = np.radians([0.0, 180.0, -164.0, -21.5, 0.0])
    tcp = _gripper_frame(stretched) @ np.append(TCP_IN_GRIPPER, 1.0)
    forward = tcp[:3] - up * (tcp[:3] @ up)
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    return np.stack([forward, left, up])  # rows: Dream axes in URDF coordinates


BASE = _urdf_to_dream_base()


def _tcp_rotation(T: np.ndarray) -> np.ndarray:
    """TCP orientation in the Dream frame, columns: x = jaw line (left when pointing forward), y = up, z = fingers."""
    z = BASE @ T[:3, :3] @ APPROACH_IN_GRIPPER
    x = BASE @ T[:3, :3] @ JAW_IN_GRIPPER
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _jaw_sign() -> float:
    """+1 if JAW_IN_GRIPPER points to the arm's left with the hand pointing forward at roll 0, else -1."""
    stretched = np.radians([0.0, 180.0, -164.0, -21.5, 0.0])
    x = BASE @ _gripper_frame(stretched)[:3, :3] @ JAW_IN_GRIPPER
    return 1.0 if x[1] >= 0 else -1.0


JAW_IN_GRIPPER = JAW_IN_GRIPPER * _jaw_sign()


def fk(q_urdf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TCP position (metres) and rotation (3x3) in the Dream frame."""
    T = _gripper_frame(q_urdf)
    xyz = BASE @ (T @ np.append(TCP_IN_GRIPPER, 1.0))[:3]
    return xyz, _tcp_rotation(T)


def rot6(R: np.ndarray) -> list[float]:
    """The first two columns of R, the rotation format Dream-1 uses."""
    return [*R[:, 0], *R[:, 1]]


def from_rot6(r6) -> np.ndarray:
    """Rotation matrix from Dream-1's 6 numbers (Gram-Schmidt on the two columns)."""
    a, b = np.asarray(r6[:3], float), np.asarray(r6[3:], float)
    x = a / np.linalg.norm(a)
    y = b - x * (x @ b)
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def ik(
    xyz: np.ndarray,
    R: np.ndarray | None,
    seed: np.ndarray,
    iterations: int = 60,
    orientation_weight: float = 0.05,
    lower: np.ndarray = LOWER,
    upper: np.ndarray = UPPER,
) -> tuple[np.ndarray, float]:
    """URDF joint angles that put the TCP at xyz, and as close to orientation R as 5 joints allow.

    Damped least squares from seed, kept inside lower/upper (default: the URDF limits). Position matters most; with
    5 joints not every orientation exists, so orientation is weighted down (metres vs radians).
    Returns (q, position error in metres).
    """
    q = np.clip(np.array(seed, float), lower, upper)
    damping = 1e-3
    for _ in range(iterations):
        p, Rq = fk(q)
        err = [xyz - p]
        if R is not None:
            # rotation error as a vector: half the sum of column cross products
            w = 0.5 * sum(np.cross(Rq[:, i], R[:, i]) for i in range(3))
            err.append(orientation_weight * w)
        e = np.concatenate(err)
        if np.linalg.norm(e[:3]) < 5e-4 and (R is None or np.linalg.norm(e[3:]) < 1e-3):
            break
        J = np.zeros((len(e), len(q)))
        h = 1e-5
        for k in range(len(q)):
            dq = q.copy()
            dq[k] += h
            p2, R2 = fk(dq)
            col = [(p2 - p) / h]
            if R is not None:
                col.append(
                    orientation_weight
                    * 0.5
                    * sum(np.cross(Rq[:, i], R2[:, i]) for i in range(3))
                    / h
                )
            J[:, k] = np.concatenate(col)
        step = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(len(e)), e)
        q = np.clip(q + step, lower, upper)
    return q, float(np.linalg.norm(xyz - fk(q)[0]))


def calibrated_to_urdf(cal: dict[str, float]) -> np.ndarray:
    """RealArm's calibrated degrees -> URDF radians, in ARM_JOINTS order."""
    return np.radians([SIGN[j] * cal[j] + OFFSET[j] for j in ARM_JOINTS])


def urdf_to_calibrated(q: np.ndarray) -> dict[str, float]:
    """URDF radians -> RealArm's calibrated degrees."""
    return {
        j: (math.degrees(a) - OFFSET[j]) / SIGN[j]
        for j, a in zip(ARM_JOINTS, q, strict=True)
    }
