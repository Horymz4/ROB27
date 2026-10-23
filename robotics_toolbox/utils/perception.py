#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  # noqa



def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)
    centers = []
    for img in images:
        circles = None
        img = img.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=gray.shape[0],
            param1=100, param2=30,
        )
        if circles is None: 
            print("None")
            continue
        centers.append(circles[0, 0, :2])

    plane_points = np.array([p["translation_vector"][:2] for p in hoop_positions])

    H, _ = cv2.findHomography(np.array(centers, dtype=np.float32), plane_points)
    return H
