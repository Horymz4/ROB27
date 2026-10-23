<<<<<<< HEAD
from ctu_bosch_sr450 import RobotBosch 

=======
from ctu_bosch_sr450 import RobotBosch
import time 
>>>>>>> refs/remotes/origin/main

robot = RobotBosch()  # set argument tty_dev=None if you are not connected to robot, it will allow you to compute FK and IK offline
robot.initialize()  # initialize connection to the robot
robot.move_to_q([0.5, 0.0, 0.0, 0.0])  # move robot
robot.wait_for_motion_stop()
time.sleep(10)
robot.close()  # close the connection
