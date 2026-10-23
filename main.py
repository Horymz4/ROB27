import json
import os
import time

import cv2
import numpy as np
from ctu_bosch_sr450 import RobotBosch

# --- nastavení ---------------------------------------------------------------
D = np.array([-0.135, 0.0])      # střed obruče v souřadnicích koncového bodu [m] (TODO změřit!)
OUT_DIR = "calib"

CALIB_Q = [
    [-0.5, -0.1, -0.23, 3 * np.pi / 2],    # (0.384, -0.368)
    [-0.4, -0.2, -0.23, 3 * np.pi / 2],    # (0.395, -0.345)
    [-0.3, -0.25, -0.23, 3 * np.pi / 2],   # (0.409, -0.313)
    [-0.2, -0.3, -0.23, 3 * np.pi / 2],    # (0.421, -0.281)    
    [0.2, 0.3, -0.23, 5 * np.pi / 2],      # (0.421,  0.281)
    [0.3, 0.25, -0.23, 5 * np.pi / 2],     # (0.409,  0.313)
    [0.4, 0.2, -0.23, 5 * np.pi / 2],      # (0.395,  0.345)
    [0.5, 0.1, -0.23, 5 * np.pi / 2],      # (0.384,  0.368)
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
        time.sleep(0.5)                          # počkat, až odezní chvění
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
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=gray.shape[0], param1=100, param2=30)
    return None if circles is None else circles[0, 0, :2]


def compute_homography():
    """Ze uložených snímků a poloh spočítá H (pixel -> střed obruče v základně)."""
    with open(f"{OUT_DIR}/data.json") as f:
        data = json.load(f)
    pix, plane = [], []
    for i, d in enumerate(data):
        c = detect_hoop_center(cv2.imread(f"{OUT_DIR}/img{i:02d}.png"))
        if c is None:
            print(f"snímek {i}: kruh nenalezen, přeskočeno")
            continue
        pix.append(c)
        plane.append(d["hoop_xy"])
    if len(pix) < 4:
        print("Nalezeno méně než 4 kruhy, homografii nelze spočítat.")
        return None
    pix = np.array(pix, dtype=np.float32)
    plane = np.array(plane, dtype=np.float32)
    H, _ = cv2.findHomography(pix, plane)

    proj = (H @ np.c_[pix, np.ones(len(pix))].T).T
    err = np.hypot(*(proj[:, :2] / proj[:, 2:] - plane).T)
    print(f"reprojekční chyba [mm]: střední {1000 * err.mean():.1f}, max {1000 * err.max():.1f}")
    np.save(f"{OUT_DIR}/H.npy", H)
    return H


if __name__ == "__main__":
    robot = RobotBosch()
    try:
        robot.initialize()                       # vyžádá stisk ARM POWER
        robot.set_speed_relative(0.3)            # pomalu pro první spuštění
        fix_negative_uhel(robot, True)

        calibrate(robot)

        fix_negative_uhel(robot, False)
        robot.soft_home()
    finally:
        robot.close()

    compute_homography()
