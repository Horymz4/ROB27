#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 2D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO2:
    """This class represents an SO2 rotations internally represented by rotation
    matrix."""

    def __init__(self, angle: float = 0.0) -> None:
        """Creates a rotation transformation that rotates vector by a given angle, that
        is expressed in radians. Rotation matrix .rot is used internally, no other
        variables can be stored inside the class."""
        super().__init__()
        self.rot: np.ndarray = np.zeros((2, 2))
        c = np.cos(angle)
        s = np.sin(angle)
        self.rot = np.array([[c, -s],[s, c]])

    def __mul__(self, other: SO2) -> SO2:
        """Compose two rotations, i.e., self * other"""
        return  SO2(self.angle + other.angle)
        

    @property
    def angle(self) -> float:
        """Return angle [rad] from the internal rotation matrix representation."""
        return  np.arctan2(self.rot[1, 0], self.rot[0, 0])


    def inverse(self) -> SO2:
        """Return inverse of the transformation. Do not change internal property of the
        object."""
        return SO2(-self.angle)

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (2,)
        return self.rot @ v

    def __eq__(self, other: SO2) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    def __hash__(self):
        return id(self)
