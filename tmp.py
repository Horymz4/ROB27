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
APPROACH_MM = 80.0                                        # výška nad prvním bodem dráhy, ve které se najíždí [mm]
STEP_MM, DT_STEP = 15.0, 1.0                              # max. délka kroku po dráze [mm], čas na jeden krok [s]
LEAD, LEAD_BIG, T_PREP = 0.7, 4.0, 6.0                    # s: předstih malého kroku, předstih velkého přesunu, příprava

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


def make_plan(proc, robot, t_above):
    """Fáze: (1) svisle na výšku najíždění, (2) přesun nad první bod, (3) sjezd a jízda po dráze po malých krocích.
    Vrací seznam (čas cíle, q, předstih); čas None = provést hned."""
    pts = proc.points[PATH_ORDER].astype(float)                  # mm, souřadnice desky
    poly = np.vstack([pts[0] + [0, 0, APPROACH_MM], pts])        # nad prvním bodem -> dráha
    dense = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / STEP_MM)))
        dense += [a + (b - a) * k / n for k in range(1, n + 1)]
    proc.points = np.array(dense)                                # predict_points pak transformuje celou dráhu
    times = [t_above + i * DT_STEP for i in range(len(dense))]

    q_now = robot.get_q()
    z_above = Z_PODLOZKA + (poly[0][2]) / 1000
    q_up = q_now.copy()
    q_up[2] = z_above - 0.5                                      # čistě svislý pohyb (mění se jen q3)
    if not robot.in_limits(q_up):
        print(f"výška najíždění z = {z_above:.3f} m je mimo rozsah (0.17 až 0.505), sniž APPROACH_MM")
        return None
    plan, q_prev = [(None, q_up, 0.0)], q_up
    for i, t in enumerate(times):
        p = proc.predict_points(t)[i]                            # mm, z = výška nad deskou
        q = target_q(robot, p[0] / 1000, p[1] / 1000, Z_PODLOZKA + p[2] / 1000, q_prev)
        limit = 3.0 if i == 0 else 0.6                           # 1. přesun je velký, další kroky malé
        if q is None or np.max(np.abs(q - q_prev)) > limit:
            print(f"nedosažitelné / skok IK v kroku {i} ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm")
            return None
        plan.append((t, q, LEAD_BIG if i == 0 else LEAD))
        q_prev = q
    return plan


def run_plan(robot, plan):
    for t, q, lead in plan:
        if t is not None:
            time.sleep(max(0.0, t - lead - time.perf_counter()))
        robot.move_to_q(q)
        robot.wait_for_motion_stop()


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

        plan = make_plan(proc, robot, time.perf_counter() + T_PREP)
        if plan is None:
            print("Plán je nedosažitelný, nepohybuji.")
        else:
            run_plan(robot, plan)

        fix_negative_uhel(robot, False)
        robot.soft_home()
    finally:
        robot.close()