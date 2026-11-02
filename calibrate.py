import json
import os
import time

import cv2
import numpy as np
from ctu_bosch_sr450 import RobotBosch
from image_processing import image_proc

# --- nastavení ---------------------------------------------------------------
D = np.array([0.135, 0.0])      # střed obruče v souřadnicích koncového bodu [m] (TODO změřit!)
OUT_DIR = "calib"

# rozprostřeno po celém snímku, rameno mimo záběr (vybráno automaticky přes H_angle_candidate_mm.npy)
CALIB_Q = [
    [0.671, -0.137, -0.23, 8.378],          # obruč v obraze ~ (240, 240) px
    [0.718, -0.785, -0.23, 8.203],          # obruč v obraze ~ (960, 240) px
    [0.872, -1.096, -0.23, 7.679],          # obruč v obraze ~ (1080, 840) px
    [1.309, -1.083, -0.23, 7.505],          # obruč v obraze ~ (240, 1560) px
    [1.147, -0.848, -0.23, 7.854],          # obruč v obraze ~ (240, 1080) px
    [1.181, -1.411, -0.23, 8.552],          # obruč v obraze ~ (600, 600) px
    [1.405, -1.599, -0.23, 7.505],          # obruč v obraze ~ (720, 1680) px
    [-0.821, 0.626, -0.23, 4.014],          # obruč v obraze ~ (1800, 240) px
    [-1.018, 0.970, -0.23, 4.538],          # obruč v obraze ~ (1680, 840) px
    [-1.272, 0.960, -0.23, 4.014],          # obruč v obraze ~ (2280, 840) px
    [-1.392, 1.032, -0.23, 4.887],          # obruč v obraze ~ (2280, 1680) px
    [-1.245, 1.456, -0.23, 4.887],          # obruč v obraze ~ (1560, 1440) px
]

def se2(x, y, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, x],
                     [s, c, y],
                     [0, 0, 1]])


T_EE_HOOP = se2(D[0], D[1], 0.0)


def hoop_center(robot, q):
    """Střed obruče (cx, cy) v souřadnicích základny pro kloubovou konfiguraci q."""
    x, y, _, phi = robot.fk(q)
    T = se2(x, y, phi) @ T_EE_HOOP
    return [float(T[0, 2]), float(T[1, 2])]


def calibrate(robot):
    os.makedirs(OUT_DIR, exist_ok=True)
    data = []
    for i, q in enumerate(CALIB_Q):
        robot.move_to_q(np.array(q))
        robot.wait_for_motion_stop()
        time.sleep(0.5)
        q_real = robot.get_q()
        img = robot.grab_image()
        cv2.imwrite(f"{OUT_DIR}/img{i:02d}.png", img)
        data.append({"q": q_real.tolist(), "hoop_xy": hoop_center(robot, q_real)})
        print(f"[{i + 1}/{len(CALIB_Q)}] hotovo")
    with open(f"{OUT_DIR}/data.json", "w") as f:
        json.dump(data, f, indent=2)


def fix_negative_uhel(robot, start=True):
    """Natočí q4 do prostřední části rozsahu (start=True) nebo ho vrátí zpět (False)."""
    q = np.array([-0.7, -0.4, 0.0, 0.0])
    if not start:
        q[3] = 2 * np.pi
    robot.move_to_q(q)
    robot.wait_for_motion_stop()

    q = robot.get_q()
    q[3] = 2 * np.pi if start else 0.0
    robot.move_to_q(q)
    robot.wait_for_motion_stop()


def detect_hoop_center(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.medianBlur(gray, 5)
    R_PX = 110    # změřený vnější poloměr obruče v pixelech
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=gray.shape[0],
        param1=100, param2=30,
        minRadius=int(2 * R_PX), maxRadius=int(4 * R_PX),
    )
    return None if circles is None else circles[0, 0, :2]



if __name__ == "__main__":
    robot = RobotBosch()
    
    robot.initialize()          
    robot.set_speed_relative(0.5)            
    fix_negative_uhel(robot, True)

    calibrate(robot)

    fix_negative_uhel(robot, False)
    robot.soft_home()

    proc = image_proc()
    with open("calib/data.json") as f:
        calibration = json.load(f)
    calibration_images = np.array([cv2.imread(f"calib/img{i:02d}.png") for i in range(len(calibration))])
    q = np.array([frame["q"] for frame in calibration])
    # FK + posun stredu obruce 135 mm ve smeru 185 stupnu od osy x nastroje.
    angle = q[:, 3] + np.deg2rad(185)
    x = 250 * np.cos(q[:, 0]) + 200 * np.cos(q[:, 0] + q[:, 1]) + 135 * np.cos(angle)
    y = 250 * np.sin(q[:, 0]) + 200 * np.sin(q[:, 0] + q[:, 1]) + 135 * np.sin(angle)
    hoop_positions = [{"translation_vector": [cx, cy, 0]} for cx, cy in zip(x, y)]
    H = proc.find_homography(calibration_images, hoop_positions)
    np.save("H.npy", H)  # pixely -> mm
    print(H)
    robot.close()
