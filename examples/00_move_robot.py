#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-10-31
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from ctu_bosch_sr450 import RobotBosch

robot = RobotBosch(tty_dev=None)
robot.initialize()

print(f"Joint configuration: {robot.get_q()}")
robot.move_to_q([0.1, 0.0, 0.0, 0.0])
robot.wait_for_motion_stop()

print(f"Joint configuration: {robot.get_q()}")

robot.soft_home()
print(f"Joint configuration: {robot.get_q()}")

robot.close()
