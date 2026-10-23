#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        assert v.shape == (3,)
        t = SO3()
        theta = np.linalg.norm(v)

        skew_matrix = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1],v[0], 0]])
        if theta == 0: t.rot = np.eye(3)
        else: t.rot = (np.eye(3) + np.sin(theta) /theta * skew_matrix + (1-np.cos(theta))/(theta**2) * (skew_matrix @skew_matrix))
        return t

    def log(self) -> np.ndarray:
        R = self.rot

        theta = np.arccos((np.trace(R) - 1) / 2)

        if np.isclose(theta, 0):
            return np.zeros(3)

        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))

        return theta * axis

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""
        result = SO3()
        result.rot = self.rot @ other.rot
        return result

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        result = SO3()
        result.rot = self.rot.T
        return result

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""
        c = np.cos(angle)
        s = np.sin(angle)

        return SO3(np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s,  c]
        ]))

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""
        c = np.cos(angle)
        s = np.sin(angle)

        return SO3(np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ]))

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        c = np.cos(angle)   
        s = np.sin(angle)

        return SO3(np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ]))

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        q = np.asarray(q)
        q = q/np.linalg.norm(q)
        q_xyz = q[:3]
        qw = q[3]
        theta = 2 * np.arccos(qw)
        w = q_xyz/ np.linalg.norm(q_xyz)
        return SO3.exp(theta*w)

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        qw = 1/2 * np.sqrt(1 + np.trace(self.rot))
        qx = 1/(4 * qw) * (self.rot[2][1] - self.rot[1][2])
        qy = 1/(4 * qw) * (self.rot[0][2] - self.rot[2][0])
        qz = 1/(4 * qw) * (self.rot[1][0] - self.rot[0][1])
        return np.array([qx,qy,qz,qw])

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        return SO3.exp(angle*axis)

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        R = self.rot

        theta = np.arccos((np.trace(R) - 1) / 2)

        if np.isclose(theta, 0):
            return np.zeros(3)

        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
        return (theta, axis)

    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """
        result = SO3()

        for i in range(3):
            axis = seq[i]
            angle = angles[i]
            if axis == 'x':
                result = result * SO3.rx(angle)
            elif axis == 'y':
                result = result * SO3.ry(angle)
            elif axis == 'z':
                result = result * SO3.rz(angle)
        return result
    
    def __hash__(self):
        return id(self)
