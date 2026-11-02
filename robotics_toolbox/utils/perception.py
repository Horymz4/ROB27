#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin CÃfka <martin.cifka@cvut.cz>
#
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  # noqa


def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    """
    Find homography based on images containing the hoop and the hoop positions loaded from
    the hoop_positions.json file in the following format:

    [{
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093259019899434, -0.17564068853313258, 0.04918733225140541]
    },
    {
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093569397977782, -0.08814069881074972, 0.04918733225140541]
    },
    ...
    ]
    """

    images = np.asarray(images)

    # todo HW03: Detect circle in each image
    image_points = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Suppress small scratches and image noise before looking for circular edges.
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        height, width = gray.shape
        shorter_side = min(height, width)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=shorter_side / 4,
            param1=100,
            param2=60,
            minRadius=int(0.08 * shorter_side),
            maxRadius=int(0.18 * shorter_side),
        )

        if circles is None:
            raise ValueError("Hoop was not detected.")

        center_x, center_y, _ = circles[0, 0]
        image_points.append([center_x, center_y])

    # todo HW03: Find homography using cv2.findHomography. Use the hoop positions and circle centers.
    image_points = np.asarray(image_points, dtype=np.float32)
    plane_points = np.asarray(
        [pose["translation_vector"][:2] for pose in hoop_positions],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(image_points, plane_points)
    if H is None:
        raise ValueError("Homography could not be computed from the detected hoops.")

    return H
