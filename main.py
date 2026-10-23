import json
import os
import time

import cv2
import numpy as np
from ctu_bosch_sr450 import RobotBosch
from robotics_toolbox.core import SE2
from robotics_toolbox.utils.perception import detect_hoop_center

# Střed obruče v souřadnicích koncového bodu (TODO: změřit, výkres říká ~135 mm)
T_EE_HOOP = SE2(translation=[0.135, 0.0], rotation=0.0)

SAFE_Q = np.array([-0.7, -0.4, 0.0, 0.0])   # bezpečná poloha pro natáčení q4

# Kloubové konfigurace [q1, q2, q3, q4]; q3 = konstantní výška (výška podložky)
CALIB_Q = [
    np.array([-0.4, -0.2, -0.23, 3 * np.pi / 2]),
    np.array([-0.3, -0.25, -0.23, 3 * np.pi / 2]),
    np.array([0.4, 0.2, -0.23, 5 * np.pi / 2]),
    np.array([0.3, 0.25, -0.23, 5 * np.pi / 2]),
]


def hoop_center(robot, q):
    """Střed obruče (cx, cy) v souřadnicích základny pro konfiguraci q."""
    x, y, _, phi = robot.fk(q)
    T_base_hoop = SE2([x, y], float(phi)) * T_EE_HOOP
    return T_base_hoop.translation


def wind_q4(robot, start=True):
    """Natočí q4 do prostřední části rozsahu (start=True) nebo zpět (False)."""
    q = SAFE_Q.copy()
    q[3] = 0.0 if start else 2 * np.pi
    robot.move_to_q(q)
    robot.wait_for_motion_stop()

    q = robot.get_q()
    q[3] = 2 * np.pi if start else 0.0
    robot.move_to_q(q)
    robot.wait_for_motion_stop()


def calibrate(robot, out_dir="calib"):
    """Navštíví kalibrační pozice, uloží snímky a pozice středu obruče."""
    os.makedirs(out_dir, exist_ok=True)
    data = []
    for i, q in enumerate(CALIB_Q):
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        time.sleep(0.5)                       # počkat, až odezní chvění
        q_real = robot.get_q()
        cv2.imwrite(f"{out_dir}/img{i:02d}.png", robot.grab_image())
        data.append({"q": q_real.tolist(),
                     "hoop_xy": hoop_center(robot, q_real).tolist()})
    with open(f"{out_dir}/data.json", "w") as f:
        json.dump(data, f, indent=2)


def compute_homography(out_dir="calib"):
    """Offline: ze snímků a uložených poloh spočítá H (pixel -> střed obruče v základně)."""
    with open(f"{out_dir}/data.json") as f:
        data = json.load(f)
    pix, plane = [], []
    for i, d in enumerate(data):
        c = detect_hoop_center(cv2.imread(f"{out_dir}/img{i:02d}.png"))
        if c is None:
            print(f"snímek {i}: kruh nenalezen, přeskočeno")
            continue
        pix.append(c)
        plane.append(d["hoop_xy"])
    pix = np.array(pix, dtype=np.float32)
    plane = np.array(plane, dtype=np.float32)
    H, _ = cv2.findHomography(pix, plane)

    proj = (H @ np.c_[pix, np.ones(len(pix))].T).T
    err = np.hypot(*(proj[:, :2] / proj[:, 2:] - plane).T)
    print(f"reprojekční chyba [mm]: střední {1000 * err.mean():.1f}, max {1000 * err.max():.1f}")
    np.save(f"{out_dir}/H.npy", H)
    return H


if __name__ == "__main__":
    robot = RobotBosch()
    try:
        robot.initialize()
        robot.set_speed_relative(0.3)
        wind_q4(robot, start=True)
        calibrate(robot)
        wind_q4(robot, start=False)
        robot.soft_home()
    finally:
        robot.close()
    compute_homography()
