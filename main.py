import json
import os
import time

import cv2
import numpy as np
from ctu_bosch_sr450 import RobotBosch

H = np.array([[ 4.67122001e-04 -4.91955551e-04  3.81225504e-01]
 [ 3.15772649e-05 -4.30167625e-04  6.01848288e-01]
 [ 1.11145333e-03 -1.24376802e-03  1.00000000e+00]])

stred_aruko = np.array([268, 1170,1])
res = H @ stred_aruko.T
x = res[0]/res[2]
y = res[1]/res[2]
D = np.array([0.135,0])
z = -0.1
if __name__ == "__main__":
    robot = RobotBosch()
    
    robot.initialize()          
    fix_negative_uhel(robot, True)

  

    fix_negative_uhel(robot, False)
    robot.soft_home()
    robot.close()