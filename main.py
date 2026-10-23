from ctu_bosch_sr450 import RobotBosch
import time
import cv2
import numpy as np

robot = RobotBosch()  # set argument tty_dev=None if you are not connected to robot, it will allow you to compute FK and IK offline
robot.initialize()  # initialize connection to the robot
robot.move_to_q([-0.4, -0.2,-0.23, np.pi/2])  # move robot
robot.wait_for_motion_stop()
img = robot.grab_image()
cv2.imwrite("img3.png", img)
time.sleep(0)
robot.soft_home()
robot.close()  # close the connection
