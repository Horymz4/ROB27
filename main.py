import json
import os
import time

import cv2
import numpy as np
from ctu_bosch_sr450 import RobotBosch
from calibrate import fix_negative_uhel

#x = res[0]/res[2]
#y = res[1]/res[2]
D = np.array([0.135,0])
Z_PODLOZKA = -0.1         # výška koncového bodu (z = 0.5 + q3)

def pixel_to_center(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]               # (cx, cy) v základně

def center_to_ee(cx, cy, phi):
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi),  np.cos(phi)]])
    return np.array([cx, cy]) - R @ D             # koncový bod

def target_q(robot, cx, cy, z, q_prev):
    phi = -np.pi / 2 if cy < 0 else np.pi / 2
    q4 = 3 * np.pi / 2 if cy < 0 else 5 * np.pi / 2   # tvoje konvence q4
    ex, ey = center_to_ee(cx, cy, phi)
    sols = []
    for q in robot.ik(ex,ey,z,q4):
        sols.append(q)
    return min(sols, key=lambda q: np.linalg.norm(q - q_prev))


if __name__ == "__main__":
    robot = RobotBosch()
    
    robot.initialize()          
    fix_negative_uhel(robot, True)

    H = np.load("calib/H.npy")
    stred_aruko = np.array([268, 1170,1])
    res = H @ stred_aruko.T
    cx, cy = pixel_to_center(H, 268, 1170)
    q = target_q(robot, cx, cy, Z_PODLOZKA + 0.05, robot.get_q())   # 5 cm nad podložkou
    if q is None:
        print("Poloha je nedosažitelná, nepohybuji.")
    else:
        robot.move_to_q(q)
        robot.wait_for_motion_stop()

  
    fix_negative_uhel(robot, False)
    robot.soft_home()
    robot.close()