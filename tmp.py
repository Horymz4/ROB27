
def observe(robot, H, n=N_IMAGES, period=PERIOD):
    robot.grab_image()                            # nastartuje kameru
    images, times = [], []
    t_start = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        images.append(robot.grab_image())
        times.append(0.5 * (t0 + time.perf_counter()))   # absolutní čas (perf_counter)
        if i < n - 1:
            time.sleep(max(0.0, t_start + (i + 1) * period - time.perf_counter()))
    t, centers = detectmarkers(images, np.array(times), H, IDS)
    if len(t) < 4:
        raise RuntimeError(f"Značky nalezeny jen na {len(t)} snímcích")
    model,  = fit_rotation(t, centers)
    return model                                  # predict_board(model, time.perf_counter()) platí v tomto běhu


if name == "main":
    H = np.load("calib/H.npy")
    robot = RobotBosch()
    try:
        robot.initialize()
        robot.set_speed_relative(0.3)
        fix_negative_uhel(robot, True)
        robot.move_to_q(OBS_Q)
        robot.wait_for_motion_stop()

        model = observe(robot, H)
        print(f"střed {np.round(model['rotation_center'], 4)}, R {model['radius']:.4f}, "
              f"omega {model['omega_deg_s']:.2f} °/s, pokryto {model['angle_span_deg']:.0f}°")
        with open("table_model.json", "w") as f:
            json.dump(model, f, indent=2)

        fix_negative_uhel(robot, False)
        robot.soft_home()
    finally:
        robot.close()

import time

import numpy as np
from ctu_bosch_sr450 import RobotBosch

from calibrate import fix_negative_uhel
from image_processing import image_proc

H_FILE = "H_angle_candidate_mm.npy"                       # H je v mm, robot pracuje v m
IDS = (1, 2)
N_IMAGES, PERIOD = 9, 2.0
D = 0.135 * np.array([np.cos(np.radians(185)), np.sin(np.radians(185))])   # střed obruče v s.s. koncového bodu [m]
Z_PODLOZKA = 0.5 - 0.23                                   # z = 0.5 + q3, q3 = -0.23 (výška podložky)
PATH_ORDER = [4, 3, 2, 1, 0]                              # pořadí bodů proc.points: od horního konce dolů (OVĚŘIT)
T_PREP, DT, LEAD = 6.0, 3.0, 3.0                          # s: příprava, odstup bodů, odhad doby jednoho pohybu


def center_to_ee(cx, cy, phi):
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    return np.array([cx, cy]) - R @ D


def target_q(robot, cx, cy, z, q_prev):
    """Řešení IK nejbližší q_prev (včetně q4, takže nedochází ke skokům o 180°). None = nedosažitelné."""
    sols = []
    for phi in (-np.pi / 2, np.pi / 2, np.arctan2(cy, cx), 0.0, np.pi):
        ex, ey = center_to_ee(cx, cy, phi)
        sols += robot.ik(ex, ey, z, phi)
    return min(sols, key=lambda q: np.linalg.norm(q - q_prev)) if sols else None


def observe(robot, n, period):
    robot.grab_image()                                    # nastartuje kameru
    images, times = [], []
    t_start = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        images.append(robot.grab_image())
        times.append(0.5 * (t0 + time.perf_counter()))    # absolutní čas (perf_counter)
        if i < n - 1:
            time.sleep(max(0.0, t_start + (i + 1) * period - time.perf_counter()))
    return images, np.array(times)


def make_plan(proc, t_first):
    """Seznam (čas, cx, cy, z) v m: body proc.points v pořadí PATH_ORDER, k-tý v čase t_first + k*DT."""
    plan = []
    for k, idx in enumerate(PATH_ORDER):
        t = t_first + k * DT
        p = proc.predict_points(t)[idx]                   # mm, z = výška nad deskou
        plan.append((t, p[0] / 1000, p[1] / 1000, Z_PODLOZKA + p[2] / 1000))
    return plan


def solve_plan(robot, plan, q_start):
    """Spočítá q pro celý plán (bez pohybu). Vrací seznam q, nebo None při nedosažitelném bodu."""
    qs, q_prev = [], q_start
    for t, cx, cy, z in plan:
        q = target_q(robot, cx, cy, z, q_prev)
        if q is None or np.max(np.abs(q - q_prev)) > 2.5:
            print(f"nedosažitelné / skok IK v bodě ({cx:.3f}, {cy:.3f}, {z:.3f})")
            return None
        qs.append(q)
        q_prev = q
    return qs


if __name__ == "__main__":
    robot = RobotBosch()
    try:
        robot.initialize()
        robot.set_speed_relative(0.3)
        proc = image_proc()
        proc.homography(np.load(H_FILE))

        fix_negative_uhel(robot, True)
        images, times = observe(robot, N_IMAGES, PERIOD)

        centers, valid_times = proc.find_aruco_centers(images, times, IDS)
        if len(valid_times) < 4:
            raise RuntimeError(f"Značky nalezeny jen na {len(valid_times)} snímcích")
        proc.fit_rotation(centers, valid_times)
        print(f"střed {np.round(proc.rotation_center, 1)} mm, R {proc.radius:.1f} mm, "
              f"omega {np.degrees(proc.angular_velocity):.2f} °/s")

        plan = make_plan(proc, time.perf_counter() + T_PREP)
        qs = solve_plan(robot, plan, robot.get_q())
        if qs is None:
            print("Plán je nedosažitelný, nepohybuji.")
        else:
            for (t, *_), q in zip(plan, qs):
                time.sleep(max(0.0, t - LEAD - time.perf_counter()))   # pohyb začne LEAD před cílovým časem
                robot.move_to_q(q)
                robot.wait_for_motion_stop()
                print(f"bod hotov, zpoždění {time.perf_counter() - t:+.2f} s")

        fix_negative_uhel(robot, False)
        robot.soft_home()
    finally:
        robot.close()

