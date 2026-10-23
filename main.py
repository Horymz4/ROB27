from ctu_bosch_sr450 import RobotBosch
import time 
import numpy as np

robot = RobotBosch()  # set argument tty_dev=None if you are not connected to robot, it will allow you to compute FK and IK offline
robot.initialize()  # initialize connection to the robot
robot.move_to_q([0.0, 0.0,-0.23, np.pi/2])  # move robot
robot.wait_for_motion_stop()
time.sleep(5)
robot.soft_home()
robot.close()  # close the connection
