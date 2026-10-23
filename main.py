"""
Kalibrace kamera -> rovina robota (Bosch SR450) pomocí obruče (hoop) z výkresu.

Postup:
  1) robot s obručí najede do 8 pozic (středy obruče na mřížce 2 x 4),
  2) v každé pozici se uloží snímek a poloha STŘEDU obruče v základně (z fk),
  3) ze snímků se detekuje střed obruče a spočítá homografie pixel -> střed obruče.

Spuštění:
  python calibrate_hoop.py --dry      # jen vypíše cílové konfigurace (bez robota)
  python calibrate_hoop.py            # sběr dat (robot!)
  python calibrate_hoop.py --compute  # jen spočítá H z uložených dat (bez robota)
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from ctu_bosch_sr450 import RobotBosch

# --- geometrie obruče (z výkresu) --------------------------------------------
HOOP_R_OUT = 0.0348          # m, vnější poloměr (Ø69.6)
HOOP_R_IN = 0.030            # m, vnitřní poloměr (Ø60)
D = np.array([0.135, 0.0])   # m, střed obruče v souřadnicích koncového bodu (TODO ZMĚŘIT!)

# --- kalibrace ---------------------------------------------------------------
Z_EE = -0.23 + 0.5           # výška koncového bodu (q3 = -0.23, z = 0.5 + q3) = výška podložky
CENTER_X = (0.30, 0.38)      # středy obruče v základně [m]: 2 sloupce x 4 řady = 8 bodů
CENTER_Y = (-0.33, -0.11, 0.11, 0.33)
OUT_DIR = "calib"
SAFE_Q = np.array([-0.7, -0.4, 0.0, 0.0])   # bezpečná poloha pro natáčení q4
SPEED = 0.3                                  # relativní rychlost (0-1)


def se2(x, y, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])


T_EE_HOOP = se2(D[0], D[1], 0.0)


def hoop_center(robot, q):
    """Střed obruče (cx, cy) v základně pro kloubovou konfiguraci q."""
    x, y, _, phi = robot.fk(q)
    T = se2(x, y, phi) @ T_EE_HOOP
    return [float(T[0, 2]), float(T[1, 2])]


def build_targets(robot):
    """Pro 8 středů obruče vrátí kloubové konfigurace.

    Natočení koncového bodu: phi = -pi/2 (q4 = 3pi/2) pro cy < 0 a +pi/2 (q4 = 5pi/2)
    pro cy > 0 (stejná konvence jako dřív). Ze dvou řešení IK (dvě konfigurace lokte)
    se vybere to s menším |q1|.
    """
    targets = []
    for cy in CENTER_Y:
        phi = -np.pi / 2 if cy < 0 else np.pi / 2
        q4 = 3 * np.pi / 2 if cy < 0 else 5 * np.pi / 2
        for cx in CENTER_X:
            ex = cx - D[0] * np.cos(phi) + D[1] * np.sin(phi)
            ey = cy - D[0] * np.sin(phi) - D[1] * np.cos(phi)
            sols = [q for q in robot.ik(ex, ey, Z_EE, phi) if abs(q[3] - q4) < 1e-6]
            if not sols:
                raise RuntimeError(f"Střed obruče ({cx}, {cy}) je nedosažitelný.")
            targets.append(min(sols, key=lambda q: abs(q[0])))
    return targets


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


# --- detekce obruče ----------------------------------------------------------
def detect_hoop(img, r_px=None, scale=0.5):
    """Najde obruč a vrátí (u, v, r) v pixelech původního snímku, nebo None.

    r_px: očekávaný vnější poloměr obruče v pixelech. Pokud není zadán, odhadne se
    širokým hledáním na zmenšeném snímku. Poloměr se pak omezí na +-25 %.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    if r_px is None:
        r_lo, r_hi = 8, int(0.25 * min(small.shape))
    else:
        r_lo, r_hi = int(0.75 * r_px * scale), int(1.25 * r_px * scale) + 1
    circles = cv2.HoughCircles(
        small, cv2.HOUGH_GRADIENT, dp=1, minDist=small.shape[0],
        param1=100, param2=25, minRadius=r_lo, maxRadius=r_hi,
    )
    if circles is None:
        return None
    u, v, r = circles[0, 0] / scale
    return float(u), float(v), float(r)


# --- sběr dat a výpočet H ----------------------------------------------------
def collect(robot):
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = build_targets(robot)
    data = []
    for i, q in enumerate(targets):
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        time.sleep(0.5)                          # odeznění chvění
        q_real = robot.get_q()
        cv2.imwrite(f"{OUT_DIR}/img{i:02d}.png", robot.grab_image())
        data.append({"q": q_real.tolist(), "hoop_xy": hoop_center(robot, q_real)})
        print(f"[{i + 1}/{len(targets)}] hotovo")
    with open(f"{OUT_DIR}/data.json", "w") as f:
        json.dump(data, f, indent=2)


def compute_homography():
    with open(f"{OUT_DIR}/data.json") as f:
        data = json.load(f)
    dets = []
    for i, d in enumerate(data):
        det = detect_hoop(cv2.imread(f"{OUT_DIR}/img{i:02d}.png"))
        dets.append(det)
    radii = [d[2] for d in dets if d is not None]
    if len(radii) < 4:
        print("Nalezeno méně než 4 obruče.")
        return None
    r_ref = float(np.median(radii))
    # druhý průchod s omezením poloměru + vyřazení odlehlých poloměrů
    pix, plane = [], []
    for i, d in enumerate(data):
        det = detect_hoop(cv2.imread(f"{OUT_DIR}/img{i:02d}.png"), r_px=r_ref)
        if det is None or abs(det[2] - r_ref) > 0.2 * r_ref:
            print(f"snímek {i}: detekce nespolehlivá, přeskočeno")
            continue
        pix.append(det[:2])
        plane.append(d["hoop_xy"])
    if len(pix) < 4:
        print("Po filtraci zbylo méně než 4 body.")
        return None
    pix = np.array(pix, dtype=np.float32)
    plane = np.array(plane, dtype=np.float32)
    H, _ = cv2.findHomography(pix, plane)

    proj = (H @ np.c_[pix, np.ones(len(pix))].T).T
    err = np.hypot(*(proj[:, :2] / proj[:, 2:] - plane).T)
    print(f"poloměr obruče v obraze: {r_ref:.1f} px -> {r_ref / HOOP_R_OUT / 1000:.2f} px/mm")
    print(f"bodů: {len(pix)}, reprojekční chyba [mm]: střední {1000 * err.mean():.1f}, "
          f"max {1000 * err.max():.1f}")
    np.save(f"{OUT_DIR}/H.npy", H)
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="jen vypsat cílové konfigurace")
    ap.add_argument("--compute", action="store_true", help="jen spočítat H z uložených dat")
    args = ap.parse_args()

    if args.compute:
        compute_homography()
    elif args.dry:
        robot = RobotBosch(tty_dev=None)
        for i, q in enumerate(build_targets(robot)):
            print(i, np.round(q, 3), "střed obruče:", np.round(hoop_center(robot, q), 3))
    else:
        robot = RobotBosch()
        try:
            robot.initialize()
            robot.set_speed_relative(SPEED)
            wind_q4(robot, True)
            collect(robot)
            wind_q4(robot, False)
            robot.soft_home()
        finally:
            robot.close()
        compute_homography()
