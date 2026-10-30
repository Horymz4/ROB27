import cv2 as cv
import numpy as np
import json


class image_proc:
    def __init__(self):
            self.H = None
            self.points = np.array([[0, 0, 0], [0, 0, 50], [50, 0, 100], [50, 50, 150], [50, 50, 200]], dtype=np.float32)

    def find_homography(self, images: np.ndarray, hoop_positions):
            """pixely -> XY kalibracni roviny"""
            from robotics_toolbox.utils.perception import find_hoop_homography
    
            self.H = find_hoop_homography(images, hoop_positions)
            return self.H

    def homography(self, homography_matrix: np.ndarray):
            """pixely -> XY kalibracni roviny"""
            self.H = homography_matrix
            return None

    def images_to_numpy(self, data_dir):
            with open(f"{data_dir}/meta.json") as f:
                frames = json.load(f)["frames"]
            return np.array([cv.imread(f"{data_dir}/{frame['file']}") for frame in frames])

    def meta_to_time(self, meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            return np.array([frame["t"] for frame in meta["frames"]])

    def find_aruco_centers(self, images, times, marker_ids):
            """Vrati stredy (N, 2, 2) v rovine stolku a casy (N,)."""
            dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
            detector = cv.aruco.ArucoDetector(dictionary)
            centers, valid_times = [], []
            for image, time in zip(images, times):
                corners, ids, _ = detector.detectMarkers(image)
                if ids is None or any(marker_id not in ids for marker_id in marker_ids):
                    continue
                points = []
                for marker_id in marker_ids:
                    index = np.flatnonzero(ids.ravel() == marker_id)[0]
                    plane_corners = cv.perspectiveTransform(corners[index], self.H)
                    points.append(plane_corners[0].mean(axis=0))
                centers.append(points)
                valid_times.append(time)
            return np.asarray(centers, dtype=np.float32).reshape(-1, 2, 2), np.array(valid_times)

    def find_rotation_center(self, centers):
            """Vrati stred rotace (2,) a polomer drahy stredu ctverce."""
            square_centers = centers.mean(axis=1).astype(float)
            x, y = square_centers.T
            A = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
            solution, _, rank, _ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
            if rank < 3:
                raise ValueError("Pro kruznici jsou potreba alespon tri nekolinearni polohy.")
            rotation_center = solution[:2]
            radius = np.sqrt(solution[2] + rotation_center @ rotation_center)
            return rotation_center, radius

    def find_square_angle(self, centers, rotation_center):
            """Uhel osy x ctverce vuci polomeru [rad], poradi znacek (1, 2)."""
            radial = centers.mean(axis=1) - rotation_center
            diagonal = centers[:, 0] - centers[:, 1]  # od znacky 2 ke znacce 1
            angles = (np.arctan2(diagonal[:, 1], diagonal[:, 0]) + np.pi / 4
                      - np.arctan2(radial[:, 1], radial[:, 0]))
            return np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())

    def fit_rotation(self, centers, times):
            """Zapamatuje geometrii a rovnomerny pohyb; casy v sekundach."""
            self.rotation_center, self.radius = self.find_rotation_center(centers)
            self.square_angle = self.find_square_angle(centers, self.rotation_center)
            radial = centers.mean(axis=1) - self.rotation_center
            angles = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
            self.reference_time = times[0]
            self.angular_velocity, self.reference_angle = np.polyfit(
                times - self.reference_time, angles, 1)

    def predict_points(self, time):
            """Predikuje XYZ bodu (pocet_bodu, 3) v mm; vysku zachova."""
            theta = self.reference_angle + self.angular_velocity * (time - self.reference_time)
            angle = theta + self.square_angle
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            square_center = self.rotation_center + self.radius * np.array([np.cos(theta), np.sin(theta)])
            points = self.points.astype(float).copy()
            points[:, :2] = self.points[:, :2] @ R.T + square_center
            return points
    
    
    



    

if __name__ == "__main__":
    proc = image_proc()
    H = np.load("image_testing/H_angle_candidate_mm.npy")
    proc.homography(H)
    data_dir = "data_rotace1"
    images = proc.images_to_numpy(data_dir)
    times = proc.meta_to_time(f"{data_dir}/meta.json")
    centers, times = proc.find_aruco_centers(images, times, (1, 2))
    print("Stredy znacek:", centers)
    print("Casy:", times)
    rotation_center, radius = proc.find_rotation_center(centers)
    print("Stred rotace:", rotation_center)
    print("Polomer drahy stredu ctverce:", radius)
    square_angle = proc.find_square_angle(centers, rotation_center)
    print("Uhel ctverce vuci polomeru [deg]:", np.degrees(square_angle))
