import json
import os
import time

import cv2
import numpy as np
from ctu_bosch_sr450 import RobotBosch
from calibrate import fix_negative_uhel
from image_processing import image_proc

#x = res[0]/res[2]
#y = res[1]/res[2]
D = np.array([0.135,0])
Z_PODLOZKA = -0.23  + 0.5      # výška koncového bodu (z = 0.5 + q3)

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
    print(cx,cy)
    ex, ey = center_to_ee(cx, cy, phi)
    sols = [q for q in robot.ik(ex,ey,z,phi) if abs(q[3] - q4 < 0.00001)]
    if not sols:return None
    return min(sols, key=lambda q: np.linalg.norm(q - q_prev))

def observe(robot, n, period):
    robot.grab_image()                            # nastartuje kameru
    images, times = [], []
    t_start = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        images.append(robot.grab_image())
        times.append(0.5 * (t0 + time.perf_counter()))   # absolutní čas (perf_counter)
        if i < n - 1:
            time.sleep(max(0.0, t_start + (i + 1) * period - time.perf_counter()))
    return images, times



if __name__ == "__main__":
    robot = RobotBosch()
    robot.initialize()      
    proc = image_proc()
    H = np.load("H_angle_candidate_mm.npy")

    fix_negative_uhel(robot, True)
    proc.homography(H)

    images, times = observe(robot, 9, 2)
    

    centers, valid_times = proc.find_aruco_centers(images, times, (1, 2))
    print("Stredy znacek:", centers)
    print("Casy:", valid_times)
    
    rotation_center, radius = proc.find_rotation_center(centers)
    print("Stred rotace:", rotation_center)
    print("Polomer drahy stredu ctverce:", radius)

    square_angle = proc.find_square_angle(centers, rotation_center)
    print("Uhel ctverce vuci polomeru [deg]:", np.degrees(square_angle))
  
    q = [0.001*rotation_center[0],0.001*rotation_center[1],0,2*np.pi]
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    time.sleep(10)

    fix_negative_uhel(robot, False)
    robot.soft_home()
    robot.close()