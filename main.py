from ctu_bosch_sr450 import RobotBosch
import time
import cv2
import numpy as np

def calibrate():
    calibration_coords = [[-0.4, -0.2,-0.23, 3*np.pi/2],[-0.3, -0.25,-0.23, 3*np.pi/2], [0.4, 0.2,-0.23, 5*np.pi/2], [0.3, 0.25,-0.23, 5*np.pi/2]]
    for i in range(len(calibration_coords)):
        robot.move_to_q(calibration_coords[i])  # move robot
        robot.wait_for_motion_stop()
        img = robot.grab_image()
        filename = f'img{i}.png'
        cv2.imwrite(filename, img)
    time.sleep(0)
    #robot.soft_home()

def fix_negative_uhel(start=True):
    q = [-0.7, -0.4, 0, 0]
    if not start: q[3] = 2*np.pi
    robot.move_to_q(q)
    robot.wait_for_motion_stop()

    q = robot.get_q()
    if start: q[3] = 2*np.pi
    else: q[3] = 0
    robot.move_to_q(q)
    robot.wait_for_motion_stop()



if __name__ == "__main__":
    robot = RobotBosch()
    robot.initialize()
    fix_negative_uhel(True)

    calibrate()

    fix_negative_uhel(False)
    robot.soft_home()
    robot.close()
 
