import cv2 as cv
import numpy as np
from robotics_toolbox.utils.perception import find_hoop_homography


class image_proc:
    def __init__(self):
        self.H = None
        # tyč v souřadnicích desky [mm]: dolů a zpět nahoru
        self.points = np.array([[50, 50, 200], [50, 50, 150], [50, 0, 100], [0, 0, 50], [0, 0, 20],
                                [0, 0, 20], [0, 0, 50], [50, 0, 100], [50, 50, 150], [50, 50, 200]],
                               dtype=np.float32)

    def find_homography(self, images, hoop_positions):
        """Homografie pixely -> XY [mm] z kalibračních snímků obruče."""
        self.H = find_hoop_homography(images, hoop_positions)
        return self.H

    def homography(self, H):
        self.H = H

    def find_aruco_centers(self, images, times):
        """Středy značek 1 a 2 (N, 2, 2) [mm] a časy snímků, na kterých jsou vidět obě."""
        detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50))
        centers, valid_times = [], []
        for image, t in zip(images, times):
            corners, ids, _ = detector.detectMarkers(image)
            if ids is None:
                continue
            ids = ids.ravel()
            if 1 not in ids or 2 not in ids:
                continue
            centers.append([cv.perspectiveTransform(corners[np.flatnonzero(ids == i)[0]], self.H)[0].mean(axis=0)
                            for i in (1, 2)])
            valid_times.append(t)
        return np.asarray(centers, dtype=np.float32).reshape(-1, 2, 2), np.array(valid_times)

    @staticmethod
    def find_rotation_center(centers):
        """Kružnice proložená středy desky: střed otáčení (2,) a poloměr."""
        x, y = centers.mean(axis=1).astype(float).T
        A = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
        sol, _, rank, _ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
        if rank < 3:
            raise ValueError("Pro kružnici jsou potřeba alespoň tři nekolineární polohy.")
        center = sol[:2]
        return center, np.sqrt(sol[2] + center @ center)

    @staticmethod
    def find_square_angle(centers, rotation_center):
        """Úhel osy x desky vůči spojnici střed otáčení -> střed desky [rad]."""
        radial = centers.mean(axis=1) - rotation_center
        diagonal = centers[:, 0] - centers[:, 1]                   # od značky 2 ke značce 1
        angles = (np.arctan2(diagonal[:, 1], diagonal[:, 0]) + np.pi / 4
                  - np.arctan2(radial[:, 1], radial[:, 0]))
        return np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())

    def fit_rotation(self, centers, times):
        """Geometrie desky a rovnoměrné otáčení (úhel proložený přímkou v čase)."""
        self.rotation_center, self.radius = self.find_rotation_center(centers)
        self.square_angle = self.find_square_angle(centers, self.rotation_center)
        radial = centers.mean(axis=1) - self.rotation_center
        angles = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
        self.reference_time = times[-1]
        self.angular_velocity, self.reference_angle = np.polyfit(times - self.reference_time, angles, 1)

    def predict_points(self, t):
        """Body self.points ve světě v čase t (N, 3) [mm]; výška se nemění."""
        theta = self.reference_angle + self.angular_velocity * (t - self.reference_time)
        a = theta + self.square_angle
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        center = self.rotation_center + self.radius * np.array([np.cos(theta), np.sin(theta)])
        points = self.points.astype(float).copy()
        points[:, :2] = self.points[:, :2] @ R.T + center
        return points
