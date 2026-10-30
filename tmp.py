import json
import time

import numpy as np
from ctu_bosch_sr450 import RobotBosch

from calibrate import fix_negative_uhel
from image_processing import image_proc

H_FILE = "H_angle_candidate_mm.npy"                       # H je v mm, robot pracuje v m
MODEL_FILE = "duration_model.json"                        # výstup measure_robot.py
SPEED = 0.3                                               # musí odpovídat rychlosti, při které se měřilo
IDS = (1, 2)
N_IMAGES, PERIOD = 9, 2.0
D = 0.135 * np.array([np.cos(np.radians(185)), np.sin(np.radians(185))])   # střed obruče v s.s. koncového bodu [m]
Z_PODLOZKA = 0.5 - 0.28                                # z = 0.5 + q3, q3 = -0.23 (výška podložky)
PATH_ORDER = [4, 3, 2, 1, 0]                              # pořadí bodů proc.points: od horního konce dolů (OVĚŘIT)
APPROACH_MM = 80.0                                        # výška nad prvním bodem dráhy, ve které se najíždí [mm]
STEP_MM = 10.0                                            # max. délka kroku po dráze [mm]
MAX_STEP_S, MAX_BIG_S = 1.5, 8.0                          # max. doba kroku / prvního přesunu [s] (nahrazuje limit skoku)
SAFETY, DT_MIN, MARGIN, N_ITER = 1.3, 0.3, 0.5, 3         # rezerva na dobu kroku, min. odstup kroků [s], rezerva startu [s], iterace plánu
PHI_GRID = np.radians(np.arange(-180, 180, 15))           # povolená natočení koncového bodu (obruč je symetrická)


def center_to_ee(cx, cy, phi):
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    return np.array([cx, cy]) - R @ D


def duration(a, v, dq):
    """Odhad doby pohybu o dq podle změřeného modelu: max přes klouby z a_j + |dq_j| / v_j."""
    dq = np.abs(np.asarray(dq, dtype=float))
    mv = dq > 1e-9
    return float(np.max(a[mv] + dq[mv] / v[mv])) if mv.any() else 0.0


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


def ik_all(robot, cx, cy, z):
    """Všechna řešení IK pro střed obruče (cx, cy, z) při různých natočeních koncového bodu."""
    sols = []
    for phi in np.append(PHI_GRID, np.arctan2(cy, cx)):
        ex, ey = center_to_ee(cx, cy, phi)
        sols += robot.ik(ex, ey, z, phi)
    return sols


def solve_path(sols_per_step, q_start, limits_s, a, v):
    """Z řešení IK každého kroku vybere jednu větev (dynamické programování): každý krok musí trvat nejvýš
    limit [s] podle modelu, minimalizuje se celková doba. Vrací (seznam q, None), nebo (None, důvod selhání)."""
    layers, costs, backs = [np.array([q_start])], [np.zeros(1)], []
    for i, (sols, lim) in enumerate(zip(sols_per_step, limits_s)):
        if not sols:
            return None, f"krok {i}: žádné řešení IK (nedosažitelné nebo mimo limity kloubů)"
        cur, prev = np.array(sols), layers[-1]
        diff = np.abs(cur[:, None, :] - prev[None, :, :])        # (nové, staré, kloub)
        dur = np.where(diff > 1e-9, a + diff / v, 0.0).max(axis=2)
        trans = costs[-1][None, :] + dur
        trans[dur > lim] = np.inf
        back, cost = trans.argmin(axis=1), trans.min(axis=1)
        ok = np.isfinite(cost)
        if not ok.any():
            n, m = np.unravel_index(dur.argmin(), dur.shape)
            return None, (f"krok {i}: nejkratší možný krok trvá {dur[n, m]:.2f} s "
                          f"(kloub q{(a + diff[n, m] / v).argmax() + 1}) > limit {lim} s")
        layers.append(cur[ok]); costs.append(cost[ok]); backs.append(back[ok])
    k = int(np.argmin(costs[-1]))
    path = []
    for i in range(len(backs), 0, -1):
        path.append(layers[i][k]); k = backs[i - 1][k]
    return path[::-1], None


def make_plan(proc, robot, model):
    """Fáze: (1) svisle na výšku najíždění, (2) přesun nad první bod, (3) sjezd a jízda po dráze po malých krocích.
    Časy se odvozují z modelu doby pohybu (iterativně, protože cíl závisí na čase a doba na cíli).
    Vrací seznam (čas cíle, q, předstih); čas None = provést hned."""
    a, v = np.array(model["a"]), np.array(model["v"])
    t_start = time.perf_counter()
    pts = proc.points[PATH_ORDER].astype(float)                  # mm, souřadnice desky
    poly = np.vstack([pts[0] + [0, 0, APPROACH_MM], pts])        # nad prvním bodem -> dráha
    dense = [poly[0]]
    for p0, p1 in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / STEP_MM)))
        dense += [p0 + (p1 - p0) * k / n for k in range(1, n + 1)]
    proc.points = np.array(dense)                                # predict_points pak transformuje celou dráhu
    n = len(dense)

    q_now = robot.get_q()
    q_up = q_now.copy()
    q_up[2] = Z_PODLOZKA + poly[0][2] / 1000 - 0.5               # čistě svislý pohyb (mění se jen q3)
    if not robot.in_limits(q_up):
        print("výška najíždění je mimo rozsah kloubu q3, sniž APPROACH_MM")
        return None
    d_up = duration(a, v, q_up - q_now)

    gaps = np.full(n, 1.0)                                       # odstupy kroků [s]; první odhad
    t_first, tau = t_start + 10.0, None
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        times = t_first + np.concatenate([[0.0], np.cumsum(gaps[1:])])
        sols = []
        for i, t in enumerate(times):
            p = proc.predict_points(t)[i]                        # mm, z = výška nad deskou
            sols.append(ik_all(robot, p[0] / 1000, p[1] / 1000, Z_PODLOZKA + p[2] / 1000))
        qs, err = solve_path(sols, q_up, [MAX_BIG_S] + [MAX_STEP_S] * (n - 1), a, v)
        if qs is None:
            print("Plán selhal:", err)
            return None
        durs = [duration(a, v, q - qp) for q, qp in zip(qs, [q_up] + qs[:-1])]
        gaps = np.array([0.0] + [max(DT_MIN, SAFETY * d) for d in durs[1:]])
        tau = tau or (time.perf_counter() - t0)                  # doba jednoho výpočtu plánu
        t_first = t_start + (N_ITER + 1) * tau + d_up + durs[0] + MARGIN
    if time.perf_counter() > t_first - d_up - durs[0]:
        print("upozornění: výpočet plánu trval déle, než se čekalo; start bude pozdě")
    return [(None, q_up, 0.0)] + [(t, q, d + 0.1) for t, q, d in zip(times, qs, durs)]


def run_plan(robot, plan):
    late = 0.0
    for t, q, lead in plan:
        if t is not None:
            time.sleep(max(0.0, t - lead - time.perf_counter()))
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        if t is not None:
            late = max(late, time.perf_counter() - t)
    print(f"největší zpoždění za plánem: {late:.2f} s")


if __name__ == "__main__":
    with open(MODEL_FILE) as f:
        model = json.load(f)
    assert abs(model["speed_relative"] - SPEED) < 1e-9, "model byl změřen při jiné rychlosti, změř znovu"

    robot = RobotBosch()
    robot.initialize()
    fix_negative_uhel(robot, True)
    robot.set_speed_relative(SPEED)
    proc = image_proc()

    proc.homography(np.load(H_FILE))
    images, times = observe(robot, N_IMAGES, PERIOD)
    centers, valid_times = proc.find_aruco_centers(images, times, IDS)
    if len(valid_times) < 4:
        raise RuntimeError(f"Značky nalezeny jen na {len(valid_times)} snímcích")

    proc.fit_rotation(centers, valid_times)
    print("Stredy znacek:", centers)
    print("Casy:", valid_times)
    rotation_center, radius = proc.find_rotation_center(centers)
    print("Stred rotace:", rotation_center)
    print("Polomer drahy stredu ctverce:", radius)
    square_angle = proc.find_square_angle(centers, rotation_center)
    print("Uhel ctverce vuci polomeru [deg]:", np.degrees(square_angle))

    plan = make_plan(proc, robot, model)
    if plan is None:
        print("Plán je nedosažitelný, nepohybuji.")
    else:
       run_plan(robot, plan)

    fix_negative_uhel(robot, False)
    robot.soft_home()
    robot.close()