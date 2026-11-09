"""
Zjednodušená verze: obruč jede po tyči a před každým krokem se spočítá, kde bude tyč
v okamžiku, kdy tam robot dojede. Žádný předem počítaný časový plán, žádná kontrola kolizí těla robota.

Postup:
  1. nafotit otáčející se desku -> střed otáčení, rychlost, natočení (image_processing)
  2. dráhu (body tyče) rozsekat na kroky po STEP_MM
  3. pro každý krok: odhad doby pohybu -> poloha tyče v čase příjezdu -> IK -> jeď
"""
import json
import time
import numpy as np
from ctu_bosch_sr450 import RobotBosch

from calibrate import fix_negative_uhel
from image_processing import image_proc

SPEED = 0.5                      # musí sedět s duration_model.json
N_IMAGES, PERIOD = 15, 0.7       # focení desky
D = 0.135 * np.array([np.cos(np.radians(187)), np.sin(np.radians(187))])  # střed obruče vůči koncovému bodu [m]
Z_PODLOZKA = 0.27                # výška desky v souřadnicích robota [m]
APPROACH_MM = 30.0               # o kolik nad horním koncem tyče se najíždí [mm]
STEP_MM = 8.0                    # délka kroku po tyči [mm]
LEAD_T = 0.23                    # o kolik za okamžik příjezdu se míří [s] (zpoždění kamery; ladit po 0.05)
PHI_STEP = np.radians(5)         # jemnost hledání natočení obruče
MAX_DPHI = np.radians(30)        # po dráze se q4 za krok otočí nejvýš o tolik (žádné otočky dokola)


def duration(a, v, dq):
    """Odhad doby pohybu o dq podle změřeného modelu (nejpomalejší kloub)."""
    dq = np.abs(dq)
    return float(np.max(np.where(dq > 1e-9, a + dq / v, 0.0)))


def observe(robot, n, period):
    """n snímků v pravidelném rytmu; čas snímku = začátek volání grab_image (tehdy snímek vzniká)."""
    robot.grab_image()                                   # rozjezd kamery
    images, times = [], []
    t_start = time.perf_counter()
    for i in range(n):
        times.append(time.perf_counter())
        images.append(robot.grab_image())
        if i < n - 1:
            time.sleep(max(0.0, t_start + (i + 1) * period - time.perf_counter()))
    return images, np.array(times)


def make_path(points_mm):
    """Bod nad horním koncem tyče + body tyče, rozsekané na kroky po STEP_MM [mm, souřadnice desky]."""
    pts = np.asarray(points_mm, float)
    poly = np.vstack([pts[0] + [0, 0, APPROACH_MM], pts])
    path = [poly[0]]
    for p0, p1 in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / STEP_MM)))
        path += [p0 + (p1 - p0) * k / n for k in range(1, n + 1)]
    return np.array(path)


def ik_nearest(robot, p_mm, q_prev, first):
    """Konfigurace, při které je střed obruče v bodě p_mm (svět, z nad deskou), nejbližší q_prev.
    Po dráze (first=False) jen natočení do MAX_DPHI od aktuálního a bez přeskoku lokte."""
    x, y, z = p_mm[0] / 1000, p_mm[1] / 1000, Z_PODLOZKA + p_mm[2] / 1000
    phis = np.arange(-np.pi, np.pi, PHI_STEP) if first else q_prev[3] + np.arange(-MAX_DPHI, MAX_DPHI + 1e-9, PHI_STEP)
    sols = []
    for phi in phis:
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        ex, ey = np.array([x, y]) - R @ D                # kde musí být koncový bod
        sols += [q for q in robot.ik(ex, ey, z, phi) if robot.in_limits(q)]
    if not first:
        sols = [q for q in sols if abs(q[3] - q_prev[3]) <= MAX_DPHI + 1e-9 and np.sign(q[1]) == np.sign(q_prev[1])]
    if not sols:
        return None
    return min(sols, key=lambda q: np.linalg.norm(q - q_prev))


def follow(robot, proc, path, model):
    """Projede dráhu. Vrací True, když dojel na konec."""
    a, v = np.array(model["a"]), np.array(model["v"])
    proc.points = path                                   # predict_points pak posouvá celou dráhu
    q = robot.get_q()
    late = []                                            # o kolik byl robot pomalejší než model [s]
    for i in range(len(path)):
        bias = float(np.median(late[-5:])) if late else 0.0
        q_new, t_arr = q, time.perf_counter()
        for _ in range(3):                               # cíl závisí na čase příjezdu a ten na cíli
            t_arr = time.perf_counter() + duration(a, v, q_new - q) + bias
            target = ik_nearest(robot, proc.predict_points(t_arr + LEAD_T)[i], q, first=(i == 0))
            if target is None:
                print(f"bod {i}: IK nemá řešení, zastavuji")
                return False
            q_new = target
        robot.move_to_q(q_new)
        robot.wait_for_motion_stop()
        late.append(time.perf_counter() - t_arr + bias)
        q = q_new
    print(f"robot byl za modelem v průměru o {np.mean(late[1:]):+.3f} s")
    return True


if __name__ == "__main__":
    with open("duration_model.json") as f:
        model = json.load(f)
    assert abs(model["speed_relative"] - SPEED) < 1e-9, "model doby pohybu je změřen při jiné rychlosti"

    robot = RobotBosch()
    robot.initialize()
    try:
        fix_negative_uhel(robot, True)
        robot.set_speed_relative(SPEED)
        proc = image_proc()
        proc.homography(np.load("H_angle_candidate_mm.npy"))

        images, times = observe(robot, N_IMAGES, PERIOD)
        centers, valid_times = proc.find_aruco_centers(images, times)
        if len(valid_times) < 5:
            raise RuntimeError(f"Značky nalezeny jen na {len(valid_times)} snímcích")
        proc.fit_rotation(centers, valid_times)
        print(f"střed otáčení {proc.rotation_center}, poloměr {proc.radius:.1f} mm, "
              f"omega {np.degrees(proc.angular_velocity):.1f} °/s")

        if follow(robot, proc, make_path(proc.points), model):
            q = robot.get_q(); q[2] = 0.0                # dráha končí nahoře -> svisle nahoru
            robot.move_to_q(q); robot.wait_for_motion_stop()
            fix_negative_uhel(robot, False)
            robot.soft_home()
        else:
            print("POZOR: robot stojí, obruč může být na tyči. Odjeď ručně.")
    finally:
        robot.close()
