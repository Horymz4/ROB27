"""
Zjednodušená verze new_main3: robot si nejdřív spočítá celý plán (kde a kdy má být) a pak ho projede.
Kolize s bludištěm se hlídají jen zjednodušeně: místo na rameni 2, které do tyče naráží (8.5 cm od osy
hřídele, 23° vlevo od směru ramene 2), je bráno jako svislý válec; žádný kus tyče, který sahá výš než jeho spodek, do něj nesmí.

  1. nafotit otáčející se desku -> střed otáčení, rychlost, natočení (image_processing)
  2. dráhu (body tyče) rozsekat na kroky po STEP_MM
  3. plán: pro každý krok čas příjezdu a konfigurace q, aby byl střed obruče na tyči
  4. projet plán: každý pohyb odeslat tak, aby robot dojel v plánovaný čas
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
CAMERA_DELAY = 0.03              # čas od začátku volání grab_image do pořízení snímku [s]
LEAD_T = 0.2                     # o kolik za okamžik příjezdu se míří [s]; předbíhá -> sniž, zaostává -> zvyš
SAFETY, GAP = 1.1, 0.05          # rozestup kroků = SAFETY * doba pohybu + GAP [s]
MARGIN = 0.5                     # rezerva před startem [s]
PHI_STEP = np.radians(5)         # jemnost hledání natočení obruče
MAX_DPHI = np.radians(30)        # po dráze se q4 za krok otočí nejvýš o tolik (žádné otočky dokola)
# místo na rameni 2, které naráží do tyče
HOUSING_OFFSET = 0.085           # vzdálenost od osy hřídele [m] (změřeno)
HOUSING_ANGLE = np.radians(157)   # směr od osy hřídele, měřeno od směru ramene 2 (loket -> hřídel);
                                 # + = vlevo při pohledu od základny, 0 = dál od lokte, 180° = zpět k lokti
HOUSING_R = 0.015                 # poloměr toho místa + poloměr tyče + rezerva [m] (ZMĚŘIT)
HOUSING_DZ = 0.03                # o kolik je spodek toho dílu výš než rovina obruče [m] (ZMĚŘIT);
                                 # díl je přichycený k hřídeli, takže jezdí nahoru/dolů spolu s obručí


def duration(a, v, dq):
    """Odhad doby pohybu o dq podle změřeného modelu (nejpomalejší kloub)."""
    dq = np.abs(dq)
    return float(np.max(np.where(dq > 1e-9, a + dq / v, 0.0)))


def observe(robot, n, period):
    """n snímků v pravidelném rytmu; čas snímku = začátek volání grab_image + CAMERA_DELAY."""
    robot.grab_image()                                   # rozjezd kamery
    images, times = [], []
    t_start = time.perf_counter()
    for i in range(n):
        times.append(time.perf_counter() + CAMERA_DELAY)
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


def rod_world(proc, rod_mm, t):
    """Body tyče [mm, souřadnice desky] -> svět v čase t [m, z od země] (stejně jako predict_points)."""
    theta = proc.reference_angle + proc.angular_velocity * (t - proc.reference_time)
    ang = theta + proc.square_angle
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    c = proc.rotation_center + proc.radius * np.array([np.cos(theta), np.sin(theta)])
    return np.c_[(rod_mm[:, :2] @ R.T + c) / 1000, Z_PODLOZKA + rod_mm[:, 2] / 1000]


def housing_gap(robot, q, rod_w):
    """Vodorovná vzdálenost [m] místa na rameni 2 (HOUSING_OFFSET od osy hřídele ve směru HOUSING_ANGLE) od nejbližšího
    kusu tyče, který sahá výš než jeho spodek (díl jezdí s hřídelí, takže je vždy HOUSING_DZ nad obručí).
    Kolize = méně než HOUSING_R."""
    x, y, z, _ = robot.fk(q)                                     # z = výška obruče
    high = rod_w[rod_w[:, 2] > z + HOUSING_DZ][:, :2]
    if not len(high):
        return np.inf
    ang = q[0] + q[1] + HOUSING_ANGLE                            # směr ramene 2 + HOUSING_ANGLE
    c = np.array([x, y]) + HOUSING_OFFSET * np.array([np.cos(ang), np.sin(ang)])   # místo, které naráží
    return np.min(np.linalg.norm(high - c, axis=1))


def make_rod(points_mm, step=5.0):
    """Tyč jako hustá řada bodů po step mm, včetně svislého kusu od spodního bodu až na desku."""
    pts = np.vstack([points_mm, [points_mm[-1, 0], points_mm[-1, 1], 0.0]])
    rod = [pts[0]]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / step)))
        rod += [p0 + (p1 - p0) * k / n for k in range(1, n + 1)]
    return np.array(rod)


def ik_nearest(robot, p_mm, q_prev, first, rod_w):
    """Konfigurace, při které je střed obruče v bodě p_mm (svět, z nad deskou), nejbližší q_prev
    a skříní ramene 2 nenarazí do tyče (rod_w). Po dráze (first=False) jen natočení do MAX_DPHI
    od předchozího a bez přeskoku lokte."""
    x, y, z = p_mm[0] / 1000, p_mm[1] / 1000, Z_PODLOZKA + p_mm[2] / 1000
    if first:
        phis = np.arange(-np.pi, np.pi, PHI_STEP)
    else:
        phis = q_prev[3] + np.arange(-MAX_DPHI, MAX_DPHI + 1e-9, PHI_STEP)
    sols = []
    for phi in phis:
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        ex, ey = np.array([x, y]) - R @ D                # kde musí být koncový bod
        sols += [q for q in robot.ik(ex, ey, z, phi) if robot.in_limits(q)]
    if not first:
        sols = [q for q in sols
                if abs(q[3] - q_prev[3]) <= MAX_DPHI + 1e-9 and np.sign(q[1]) == np.sign(q_prev[1])]
    gaps = [housing_gap(robot, q, rod_w) for q in sols]
    sols = [(q, g) for q, g in zip(sols, gaps) if g >= HOUSING_R]   # bez kolize skříně ramene 2 s tyčí
    if not sols:
        return None
    if first:                                            # na začátku natočit obruč co nejdál od skříně
        return max(sols, key=lambda qg: min(qg[1], 1.0) - 0.01 * np.linalg.norm(qg[0] - q_prev))[0]
    return min(sols, key=lambda qg: np.linalg.norm(qg[0] - q_prev))[0]


def make_plan(robot, proc, path, model, n_iter=3):
    """Pro každý bod dráhy čas příjezdu t a konfiguraci q. Kde má robot být, záleží na čase,
    a čas na tom, jak daleko jede -> počítá se opakovaně. Vrací seznam (t, q, doba pohybu) nebo None."""
    a, v = np.array(model["a"]), np.array(model["v"])
    pts = np.asarray(proc.points, float)                 # tyč = body dolů po nejnižší bod (cesta zpět je stejná tyč)
    rod_mm = make_rod(pts[:int(np.argmin(pts[:, 2])) + 1])
    proc.points = path                                   # predict_points pak posouvá celou dráhu
    q_start = robot.get_q()
    gaps = np.full(len(path), 0.5)                       # první odhad rozestupů [s]
    t_first = time.perf_counter() + 5.0
    for _ in range(n_iter):
        t_calc = time.perf_counter()
        times = t_first + np.r_[0.0, np.cumsum(gaps[1:])]
        qs, durs, q = [], [], q_start
        for i, t in enumerate(times):
            q_new = ik_nearest(robot, proc.predict_points(t + LEAD_T)[i], q, first=(i == 0),
                               rod_w=rod_world(proc, rod_mm, t + LEAD_T))
            if q_new is None:
                print(f"Plán selhal: bod {i} nemá řešení IK bez kolize skříně ramene 2 s tyčí")
                return None
            durs.append(duration(a, v, q_new - q))
            qs.append(q_new)
            q = q_new
        gaps = SAFETY * np.array(durs) + GAP
        # start = až skončí další výpočet plánu a robot dojede na první bod
        t_first = time.perf_counter() + (time.perf_counter() - t_calc) + durs[0] + MARGIN
    if time.perf_counter() > times[0] - durs[0]:
        print("upozornění: plán se počítal moc dlouho, start bude pozdě")
    return list(zip(times, qs, durs))


def run_plan(robot, plan):
    """Každý pohyb odešle tak, aby robot dojel v plánovaný čas, a na konci vypíše, jak plán dodržel."""
    late = []
    for t, q, d in plan:
        time.sleep(max(0.0, t - d - time.perf_counter()))
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        late.append(time.perf_counter() - t)
    print(f"zpoždění proti plánu: střední {np.mean(late[1:]):+.2f} s, max {np.max(late[1:]):+.2f} s")


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

        plan = make_plan(robot, proc, make_path(proc.points), model)
        if plan is not None:
            run_plan(robot, plan)
            q = robot.get_q(); q[2] = 0.0                # dráha končí nahoře -> svisle nahoru
            robot.move_to_q(q); robot.wait_for_motion_stop()
            fix_negative_uhel(robot, False)
            robot.soft_home()
    finally:
        robot.close()
