##!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-10-31
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import numpy as np

from ctu_bosch_sr450 import RobotBosch

robot = RobotBosch(tty_dev=None)
robot.initialize()

ik_solutions = robot.ik(x=0.25, y=0.25, z=-0.1, phi=np.deg2rad(45))

for solution in ik_solutions:
    print(f"Joint configuration: {solution}")
    robot.move_to_q(solution)
    robot.wait_for_motion_stop()

robot.close()!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-10-31
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import numpy as np

from ctu_bosch_sr450 import RobotBosch

robot = RobotBosch(tty_dev=None)
robot.initialize()

ik_solutions = robot.ik(x=0.25, y=0.25, z=-0.1, phi=np.deg2rad(45))

for solution in ik_solutions:
    print(f"Joint configuration: {solution}")
    robot.move_to_q(solution)
    robot.wait_for_motion_stop()

robot.close()
