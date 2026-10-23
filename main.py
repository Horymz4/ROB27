from ctu_bosch_sr450 import RobotBosch 


robot = RobotBosch()  # set argument tty_dev=None if you are not connected to robot, it will allow you to compute FK and IK offline
robot.initialize()  # initialize connection to the robot
robot.move_to_q([0.1, 0.0, 0.0, 0.0])  # move robot
robot.wait_for_motion_stop()
robot.close()  # close the connection
