import json
import os
import time

import cv2
import numpy as np
from ctu_bosch_sr450 import RobotBosch
from calibrate import fix_negative_uhel
from image_processing import image_proc

robot = RobotBosch()
robot.initialize()      
robot.reset_motors()
robot.soft_home()
robot.close()
