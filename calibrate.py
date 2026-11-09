import json
import os
import time

import cv2
import numpy as np
from ctu_bosch_sr450 import RobotBosch

from image_processing import image_proc

OUT_DIR = "calib"
CAPTURE = False                   # True = znovu nafotit kalibrační polohy
HOOP_R = 135                      # vzdálenost středu obruče od osy hřídele [mm]
HOOP_ANGLE = np.deg2rad(187)      # směr obruče vůči q4

CALIB_Q = [
    [0.671, -0.137, -0.23, 8.378],
    [0.718, -0.785, -0.23, 8.203],
    [0.872, -1.096, -0.23, 7.679],
    [1.309, -1.083, -0.23, 7.505],
    [1.147, -0.848, -0.23, 7.854],
    [1.181, -1.411, -0.23, 8.552],
    [1.405, -1.599, -0.23, 7.505],
    [-0.821, 0.626, -0.23, 4.014],
    [-1.018, 0.970, -0.23, 4.538],
    [-1.272, 0.960, -0.23, 4.014],
    [-1.392, 1.032, -0.23, 4.887],
    [-1.245, 1.456, -0.23, 4.887],
]


def calibrate(robot):
    """Projede CALIB_Q, uloží snímky a skutečné q do OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    data = []
    for i, q in enumerate(CALIB_Q):
        robot.move_to_q(np.array(q))
        robot.wait_for_motion_stop()
        time.sleep(0.5)
        q_real = robot.get_q()
        cv2.imwrite(f"{OUT_DIR}/img{i:02d}.png", robot.grab_image())
        x, y, _, phi = robot.fk(q_real)
        data.append({"q": q_real.tolist(), "hoop_xy": [float(x + 0.135 * np.cos(phi)),
                                                       float(y + 0.135 * np.sin(phi))]})
        print(f"[{i + 1}/{len(CALIB_Q)}] hotovo")
    with open(f"{OUT_DIR}/data.json", "w") as f:
        json.dump(data, f, indent=2)


def fix_negative_uhel(robot, start=True):
    """Natočí q4 na 2π (start=True, pak odjede stranou) nebo ho vrátí na 0 (start=False)."""
    robot.move_to_q([0.7, 0.4, 0.0, 0.0 if start else 2 * np.pi])
    robot.wait_for_motion_stop()
    q = robot.get_q()
    q[3] = 2 * np.pi if start else 0.0
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    if start:
        robot.move_to_q([1.45, 0, 0, 2 * np.pi])
        robot.wait_for_motion_stop()


def compute_homography():
    """Homografie z uložených snímků a polohy středu obruče z kinematiky [mm]."""
    with open(f"{OUT_DIR}/data.json") as f:
        q = np.array([frame["q"] for frame in json.load(f)])
    images = np.array([cv2.imread(f"{OUT_DIR}/img{i:02d}.png") for i in range(len(q))])
    a1, a12, a4 = q[:, 0], q[:, 0] + q[:, 1], q[:, 3] + HOOP_ANGLE
    x = 250 * np.cos(a1) + 200 * np.cos(a12) + HOOP_R * np.cos(a4)
    y = 250 * np.sin(a1) + 200 * np.sin(a12) + HOOP_R * np.sin(a4)
    hoop_positions = [{"translation_vector": [cx, cy, 0]} for cx, cy in zip(x, y)]
    return image_proc().find_homography(images, hoop_positions)


if __name__ == "__main__":
    robot = RobotBosch()
    robot.initialize()
    robot.set_speed_relative(0.5)
    fix_negative_uhel(robot, True)
    if CAPTURE:
        calibrate(robot)
    fix_negative_uhel(robot, False)
    robot.soft_home()

    H = compute_homography()
    np.save("H.npy", H)
    print(H)
    robot.close()
