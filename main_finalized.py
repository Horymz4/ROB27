"""
Robot si nejdřív spočítá celý plán (kdy a v jaké konfiguraci má být) a pak ho projede.

  1. nafotit otáčející se desku -> střed otáčení, rychlost, natočení
  2. dráhu po tyči rozsekat na kroky po STEP_MM
  3. pro každý krok najít čas příjezdu a konfiguraci se středem obruče na tyči
  4. projet plán tak, aby robot do každého bodu dojel v plánovaný čas
"""
import json
import time
import numpy as np
from ctu_bosch_sr450 import RobotBosch

from calibrate import fix_negative_uhel
from image_processing import image_proc

SPEED = 0.5                       # musí sedět s duration_model.json
N_IMAGES, PERIOD = 22, 0.67
MIN_IMAGES = 8                    # minimum snímků s oběma značkami
PHOTO_Q = [-0.7, -0.4, 0, 2 * np.pi]

D = 0.135 * np.array([np.cos(np.radians(189)), np.sin(np.radians(189))])  # střed obruče vůči koncovému bodu [m]
Q2_OFF = 0.06                     # skutečné q2 = povelové q2 + Q2_OFF [rad]
H_FILE = "H_angle_candidate_mm.npy"
Z_PODLOZKA = 0.27                 # výška desky [m]

APPROACH_MM = 30.0                # nájezd nad horní konec tyče [mm]
STEP_MM = 6.7
CAMERA_DELAY = 0.03               # od začátku grab_image do pořízení snímku [s]
SAFETY, GAP = 1.1, 0.05           # rozestup kroků = SAFETY * doba pohybu + GAP [s]
MARGIN = 0.35                     # rezerva před startem [s]
PHI_STEP = np.radians(5)
MAX_DPHI = np.radians(30)         # max. pootočení q4 za krok

# díl na rameni 2, který naráží do tyče (jezdí nahoru/dolů s obručí)
HOUSING_OFFSET = 0.085            # od osy hřídele [m]
HOUSING_ANGLE = np.radians(157)   # od směru ramene 2 (loket -> hřídel), + = vlevo
HOUSING_R = 0.035                 # poloměr dílu + tyče + rezerva [m]
HOUSING_DZ = 0.03                 # spodek dílu nad rovinou obruče [m]


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def subdivide(poly, step):
    """Lomená čára rozdělená na úseky nejvýš step dlouhé."""
    out = [poly[0]]
    for p0, p1 in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / step)))
        out += [p0 + (p1 - p0) * k / n for k in range(1, n + 1)]
    return np.array(out)


def make_path(points_mm):
    pts = np.asarray(points_mm, float)
    return subdivide(np.vstack([pts[0] + [0, 0, APPROACH_MM], pts]), STEP_MM)


def make_rod(points_mm):
    """Tyč po 5 mm od horního konce k nejnižšímu bodu a svisle až na desku."""
    pts = np.asarray(points_mm, float)
    pts = pts[:int(np.argmin(pts[:, 2])) + 1]
    return subdivide(np.vstack([pts, [pts[-1, 0], pts[-1, 1], 0.0]]), 5.0)


def duration(a, v, dq):
    dq = np.abs(dq)
    return float(np.max(np.where(dq > 1e-9, a + dq / v, 0.0)))


def observe(robot, proc):
    """N_IMAGES snímků po PERIOD, vrací středy značek a časy platných snímků."""
    robot.grab_image()                                   # rozjezd kamery
    images, times = [], []
    t0 = time.perf_counter()
    for i in range(N_IMAGES):
        times.append(time.perf_counter() + CAMERA_DELAY)
        images.append(robot.grab_image())
        if i < N_IMAGES - 1:
            time.sleep(max(0.0, t0 + (i + 1) * PERIOD - time.perf_counter()))
    return proc.find_aruco_centers(images, np.array(times))


def go_photo(robot):
    robot.move_to_q(PHOTO_Q)
    robot.wait_for_motion_stop()


def rod_world(proc, rod_mm, t):
    """Body tyče (deska, mm) -> svět v čase t (m)."""
    theta = proc.reference_angle + proc.angular_velocity * (t - proc.reference_time)
    c = proc.rotation_center + proc.radius * np.array([np.cos(theta), np.sin(theta)])
    xy = rod_mm[:, :2] @ rot(theta + proc.square_angle).T + c
    return np.c_[xy / 1000, Z_PODLOZKA + rod_mm[:, 2] / 1000]


def housing_gap(robot, q, rod_w):
    """Vodorovná vzdálenost dílu na rameni 2 od tyče nad jeho spodkem [m]."""
    q = np.array(q, float)
    q[1] += Q2_OFF
    x, y, z, _ = robot.fk(q)
    high = rod_w[rod_w[:, 2] > z + HOUSING_DZ][:, :2]
    if not len(high):
        return np.inf
    a = q[0] + q[1] + HOUSING_ANGLE
    c = np.array([x, y]) + HOUSING_OFFSET * np.array([np.cos(a), np.sin(a)])
    return np.min(np.linalg.norm(high - c, axis=1))


def ik_nearest(robot, p_mm, q_prev, first, rod_w):
    """Konfigurace se středem obruče v p_mm nejbližší q_prev, bez kolize s tyčí (rod_w=None: nekontroluje se)."""
    xy = p_mm[:2] / 1000
    z = Z_PODLOZKA + p_mm[2] / 1000
    if first:
        phis = np.arange(-np.pi, np.pi, PHI_STEP)
    else:
        phis = q_prev[3] + np.arange(-MAX_DPHI, MAX_DPHI + 1e-9, PHI_STEP)

    sols = []
    for phi in phis:
        ex, ey = xy - rot(phi) @ D
        for q in robot.ik(ex, ey, z, phi):
            q = q.copy()
            q[1] -= Q2_OFF
            if not robot.in_limits(q):
                continue
            if not first and (abs(q[3] - q_prev[3]) > MAX_DPHI + 1e-9 or np.sign(q[1]) != np.sign(q_prev[1])):
                continue
            gap = np.inf if rod_w is None else housing_gap(robot, q, rod_w)
            if gap >= HOUSING_R:
                sols.append((q, gap))
    if not sols:
        return None

    dist = lambda s: np.linalg.norm(s[0] - q_prev)
    if first:                                            # díl co nejdál od tyče
        return max(sols, key=lambda s: min(s[1], 1.0) - 0.01 * dist(s))[0]
    return min(sols, key=dist)[0]


def make_plan(robot, proc, points_mm, model, lead, n_iter=3):
    """Seznam (čas příjezdu, q, doba pohybu) pro každý bod dráhy, nebo None."""
    a, v = np.array(model["a"]), np.array(model["v"])
    rod_mm = make_rod(points_mm)
    path = make_path(points_mm)
    proc.points = path                                   # predict_points posouvá celou dráhu
    i_low = int(np.argmin(path[:, 2]))                   # kolize se hlídají jen cestou dolů
    q_start = robot.get_q()
    gaps = np.full(len(path), 0.5)
    t_first = time.perf_counter() + 5.0

    for _ in range(n_iter):
        t_calc = time.perf_counter()
        times = t_first + np.r_[0.0, np.cumsum(gaps[1:])]
        qs, durs, q = [], [], q_start
        for i, t in enumerate(times):
            rod_w = rod_world(proc, rod_mm, t + lead) if i <= i_low else None
            q_new = ik_nearest(robot, proc.predict_points(t + lead)[i], q, i == 0, rod_w)
            if q_new is None:
                print(f"Plán selhal: bod {i} nemá řešení IK bez kolize s tyčí")
                return None
            durs.append(duration(a, v, q_new - q))
            qs.append(q_new)
            q = q_new
        gaps = SAFETY * np.array(durs) + GAP
        t_first = time.perf_counter() + (time.perf_counter() - t_calc) + durs[0] + MARGIN

    if time.perf_counter() > times[0] - durs[0]:
        print("upozornění: plán se počítal moc dlouho, start bude pozdě")
    return list(zip(times, qs, durs))


def run_plan(robot, plan):
    late = []
    for t, q, d in plan:
        time.sleep(max(0.0, t - d - time.perf_counter()))
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        late.append(time.perf_counter() - t)
    print(f"zpoždění proti plánu: střední {np.mean(late[1:]):+.2f} s, max {np.max(late[1:]):+.2f} s")


def lead_time(proc):
    """Empirický časový předstih podle směru otáčení a polohy stolku [s]."""
    lead = 0.05 if proc.angular_velocity < 0 else -0.03
    if proc.rotation_center[1] > 0:
        lead -= 0.05
    return lead


if __name__ == "__main__":
    with open("duration_model.json") as f:
        model = json.load(f)
    assert model["speed_relative"] == SPEED, "duration_model.json je změřený pro jinou rychlost"

    robot = RobotBosch()
    robot.initialize()
    try:
        fix_negative_uhel(robot, True)
        robot.set_speed_relative(SPEED)
        proc = image_proc()
        points = proc.points.copy()
        proc.homography(np.load(H_FILE))

        centers, valid_times = observe(robot, proc)
        if len(valid_times) < MIN_IMAGES:
            print(f"Značky nalezeny jen na {len(valid_times)} snímcích")
            go_photo(robot)
            centers, valid_times = observe(robot, proc)
        if len(valid_times) < MIN_IMAGES:
            print(f"Značky nalezeny jen na {len(valid_times)} snímcích")
            raise RuntimeError("Aruco značky nejsou vidět")
        print(f"počet snímků: {len(valid_times)}")
        proc.fit_rotation(centers, valid_times)
        print(f"střed otáčení {proc.rotation_center}, poloměr {proc.radius:.1f} mm, "
              f"omega {np.degrees(proc.angular_velocity):.1f} °/s")

        lead = lead_time(proc)
        print(f"LEAD_T: {lead}")

        plan = make_plan(robot, proc, points, model, lead)
        if plan is None:                                 # druhý pokus při jiném natočení stolku
            go_photo(robot)
            proc.fit_rotation(*observe(robot, proc))
            plan = make_plan(robot, proc, points, model, lead)

        if plan is None:
            print("Bludiště je v této poloze nedosažitelné, nepohybuji se.")
        else:
            run_plan(robot, plan)

        q = robot.get_q()
        q[2] = 0.0
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
    finally:
        robot.set_speed_relative(1)
        fix_negative_uhel(robot, False)
        robot.soft_home()
        robot.close()
