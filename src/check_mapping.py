"""Check the calibrated <-> URDF mapping on the real arm: the 3D model mirrors the arm live.

Run with: uv run python src/check_mapping.py

Read-only: torque is never switched on, so move the arm by hand. Each joint of the model in
Rerun should move the same way as the real one. Also prints where the fingertip point (TCP)
is in Dream-1's frame (metres; x forward, y left, z up). Ctrl+C to quit.
"""

import json
import math
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

import so100_kinematics as kin
from robot import RerunView

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")
PORT = os.environ["ROBOT_PORT"]
CALIBRATION = (
    Path.home()
    / ".cache/huggingface/lerobot/calibration/robots/so_follower"
    / f"{os.getenv('ROBOT_ID', 'my_follower')}.json"
)


def main() -> None:
    calibration = {
        name: MotorCalibration(**c)
        for name, c in json.loads(CALIBRATION.read_text()).items()
    }
    motors = {
        name: Motor(
            c.id,
            "sts3215",
            MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES,
        )
        for name, c in calibration.items()
    }
    bus = FeetechMotorsBus(PORT, motors, calibration)
    bus.connect()  # only opens the port and pings; nothing is written
    view = RerunView()
    try:
        while True:
            cal = bus.sync_read("Present_Position")
            q = kin.calibrated_to_urdf(cal)
            xyz, _ = kin.fk(q)
            view.show(tuple(math.degrees(a) for a in q), cal["gripper"] * 0.8)
            joints = "  ".join(f"{j[:5]} {cal[j]:6.1f}" for j in kin.ARM_JOINTS)
            print(
                f"\r{joints}  | TCP x {xyz[0]:+.3f} y {xyz[1]:+.3f} z {xyz[2]:+.3f}",
                end="",
                flush=True,
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        bus.port_handler.closePort()  # not bus.disconnect(): that may switch torque off/on


if __name__ == "__main__":
    main()
