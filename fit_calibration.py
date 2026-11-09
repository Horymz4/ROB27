"""
Z kalibračních snímků (calib/data.json + calib/imgXX.png) společně odhadne:
  - D        polohu středu obruče v souřadnicích koncového bodu [mm] (délka i úhel),
  - Q2_OFF   posun nuly kloubu q2 [rad] (skutečné q2 = hlášené q2 + Q2_OFF),
  - H        homografii pixel -> XY [mm] v základně.
Pro každé (D, Q2_OFF) se H spočítá nejmenšími čtverci a minimalizuje se reprojekční chyba.
Výsledek: hoop_model.json (D_mm, q2_offset) a H_fit_mm.npy. Načítá je new_main1.py.

  python fit_calibration.py
"""
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

CALIB_DIR = "calib"
L1, L2 = 250.0, 200.0                       # délky ramen [mm]
D_START = 135 * np.array([np.cos(np.radians(185)), np.sin(np.radians(185))])
R_HOOP_PX = (150, 240)                      # rozsah poloměru obruče v obraze [px]


def detect_hoop(img):
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=gray.shape[0], param1=100, param2=30,
                         minRadius=R_HOOP_PX[0], maxRadius=R_HOOP_PX[1])
    return None if c is None else c[0, 0, :2]


def hoop_xy(q, D, q2_off):
    q1, q2, a = q[:, 0], q[:, 1] + q2_off, q[:, 3]
    ee = np.c_[L1 * np.cos(q1) + L2 * np.cos(q1 + q2), L1 * np.sin(q1) + L2 * np.sin(q1 + q2)]
    return ee + np.c_[np.cos(a) * D[0] - np.sin(a) * D[1], np.sin(a) * D[0] + np.cos(a) * D[1]]


def fit_H(pix, xy):
    H, _ = cv2.findHomography(np.float32(pix), np.float32(xy), 0)
    return H


def project(H, pix):
    p = (H @ np.c_[pix, np.ones(len(pix))].T).T
    return p[:, :2] / p[:, 2:]


def residuals(params, q, pix, use_off=True):
    D, off = params[:2], (params[2] if use_off else 0.0)
    xy = hoop_xy(q, D, off)
    return (project(fit_H(pix, xy), pix) - xy).ravel()


def fit(q, pix, use_off=True):
    x0 = np.r_[D_START, 0.0] if use_off else D_START
    return least_squares(residuals, x0, args=(q, pix, use_off)).x


def leave_one_out(q, pix, use_off=True):
    errs = []
    for k in range(len(q)):
        m = np.arange(len(q)) != k
        p = fit(q[m], pix[m], use_off)
        D, off = p[:2], (p[2] if use_off else 0.0)
        H = fit_H(pix[m], hoop_xy(q[m], D, off))
        errs.append(np.linalg.norm(project(H, pix[k:k + 1])[0] - hoop_xy(q[k:k + 1], D, off)[0]))
    return np.array(errs)


if __name__ == "__main__":
    with open(f"{CALIB_DIR}/data.json") as f:
        data = json.load(f)
    q_all, pix = [], []
    for i, d in enumerate(data):
        c = detect_hoop(cv2.imread(f"{CALIB_DIR}/img{i:02d}.png"))
        if c is None:
            print(f"snímek {i}: obruč nenalezena, přeskočeno")
            continue
        q_all.append(d["q"]); pix.append(c)
    q, pix = np.array(q_all), np.array(pix, float)
    print(f"použito snímků: {len(q)}")

    r0 = residuals(D_START, q, pix, use_off=False).reshape(-1, 2)
    print(f"původní model (D 135 mm / 185°, bez posunu q2): chyba střední {np.linalg.norm(r0, axis=1).mean():.1f} mm, "
          f"max {np.linalg.norm(r0, axis=1).max():.1f} mm")
    for use_off, name in [(False, "jen D"), (True, "D + posun q2")]:
        loo = leave_one_out(q, pix, use_off)
        print(f"{name:13s}: leave-one-out chyba střední {loo.mean():.2f} mm, max {loo.max():.2f} mm")

    p = fit(q, pix, use_off=True)
    D, off = p[:2], p[2]
    H = fit_H(pix, hoop_xy(q, D, off))
    r = np.linalg.norm(residuals(p, q, pix).reshape(-1, 2), axis=1)
    print(f"D = ({D[0]:.1f}, {D[1]:.1f}) mm = {np.linalg.norm(D):.1f} mm pod {np.degrees(np.arctan2(D[1], D[0])):.1f}°")
    print(f"posun q2 = {off:+.4f} rad ({np.degrees(off):+.2f}°)")
    print(f"reprojekční chyba: střední {r.mean():.2f} mm, max {r.max():.2f} mm")
    with open("hoop_model.json", "w") as f:
        json.dump({"D_mm": D.tolist(), "q2_offset": float(off)}, f, indent=2)
    np.save("H_fit_mm.npy", H)
    print("uloženo: hoop_model.json, H_fit_mm.npy")
