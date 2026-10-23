#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-11-3
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import matplotlib.pyplot as plt
import numpy as np

from ctu_bosch_sr450 import RobotBosch

robot = RobotBosch(tty_dev=None)

xy = []
for q0 in np.linspace(robot.q_min[0], robot.q_max[0]):
    for q1 in np.linspace(robot.q_min[1], robot.q_max[1]):
        x, y, _, _ = robot.fk([q0, q1, 0, 0])
        xy.append([x, y])

fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
ax.plot(*np.array(xy).T, "ko", ms=5)
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
plt.show()
