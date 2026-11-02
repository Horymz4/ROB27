import json
import time
import numpy as np
from ctu_bosch_sr450 import RobotBosch

from calibrate import fix_negative_uhel
from image_processing import image_proc

SPEED = 0.5
IDS = (1, 2) #id aruco značek
N_IMAGES, PERIOD = 9, 1.5
D = 0.135 * np.array([np.cos(np.radians(187)), np.sin(np.radians(187))]) # střed obruče v s.s. koncového bodu [m]
Z_PODLOZKA = 0.27 # vyska podlozky od zeme (v q[2] == -0.22)
APPROACH_MM = 30.0                                        # výška nad prvním bodem dráhy, ve které se najíždí [mm]
STEP_MM = 8 # nemenit                                           # max. délka kroku po dráze [mm]
MAX_STEP_S, MAX_BIG_S = 4.0, 8.0                          # max. doba kroku / prvního přesunu [s] (nahrazuje limit skoku)
SAFETY, DT_MIN, MARGIN, N_ITER = 0.9, 0.05, 0.5, 3         # rezerva na dobu kroku, min. odstup kroků [s], rezerva startu [s], iterace plánu
PHI_GRID = np.radians(np.arange(-180, 180, 15))           # povolená natočení koncového bodu (obruč je symetrická)

# --- kolize těla robota s tyčí (všechno v m, z = výška od země jako robot.fk) ---------------------
ROD_TO_BOARD = True        # tyč pokračuje od posledního bodu dráhy svisle dolů až na desku
ROD_SAMPLE_MM = 5.0        # hustota bodů tyče pro kontrolu kolizí
R_ROD = 0.004              # poloměr tyče
R_QUILL = 0.04             # poloměr hřídele + příruby kolem osy koncového bodu (sahá od obruče nahoru)
QUILL_BELOW = 0.01         # o kolik příruba sahá pod rovinu obruče
R_ELBOW, Z_ELBOW = 0.09, 0.50       # poloměr pouzdra lokte a výška jeho spodku
R_FOREARM, R_UPPER, Z_ARM = 0.06, 0.08, 0.55   # poloviční šířky ramen a výška jejich spodku
CLEAR_MIN = 0.015          # min. vůle těla od tyče, jinak kolize
LEAD_T = 0.55              # o kolik dopředu se míří za okamžik příjezdu [s] (= dřívější +0.4 v plánu + příjezd o 0.15 s dřív)
TRANSIT_S = (0.5, 1.0)     # kde na přejezdu mezi kroky se kontrolují kolize (0.5 = půlka, 1 = cíl)
T_CALC = 0.08              # čas na přepočet cíle mezi kroky [s] (započte se do rozestupu kroků v plánu)


def center_to_ee(cx, cy, phi):
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    return np.array([cx, cy]) - R @ D


def duration(a, v, dq):
    """Odhad doby pohybu o dq podle změřeného modelu: max přes klouby z a_j + |dq_j| / v_j."""
    dq = np.abs(np.asarray(dq, dtype=float))
    mv = dq > 1e-9
    return float(np.max(a[mv] + dq[mv] / v[mv])) if mv.any() else 0.0


def observe(robot, n, period):
    robot.grab_image() #meaby pointless, fotime predtim aby se kamera startla
    images, times = [], []
    t_start = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        images.append(robot.grab_image())
        times.append(0.5 * (t0 + time.perf_counter()))  
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


def rod_samples(rod_mm):
    """Hustě navzorkovaná tyč (polyline) v souřadnicích desky [mm]."""
    rod = np.asarray(rod_mm, float)
    if ROD_TO_BOARD:
        rod = np.vstack([rod, [rod[-1, 0], rod[-1, 1], 0.0]])
    out = [rod[0]]
    for p0, p1 in zip(rod[:-1], rod[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / ROD_SAMPLE_MM)))
        out += [p0 + (p1 - p0) * k / n for k in range(1, n + 1)]
    return np.array(out)


def rod_world(proc, rod_mm, t):
    """Tyč v čase t ve světě [m, z od země] (stejná transformace jako proc.predict_points)."""
    theta = proc.reference_angle + proc.angular_velocity * (t - proc.reference_time)
    ang = theta + proc.square_angle
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    c = proc.rotation_center + proc.radius * np.array([np.cos(theta), np.sin(theta)])
    out = np.empty_like(rod_mm, dtype=float)
    out[:, :2] = (rod_mm[:, :2] @ R.T + c) / 1000
    out[:, 2] = Z_PODLOZKA + rod_mm[:, 2] / 1000
    return out


def _seg_dist(p, a, b):
    """Vzdálenost bodů p (M,2) od úseček a-b (K,2) -> (K, M)."""
    ab = (b - a)[:, None, :]
    ap = p[None, :, :] - a[:, None, :]
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(ap - t[..., None] * ab, axis=-1)


def clearance(robot, qs, rod_w):
    """Nejmenší vůle [m] mezi tělem robota (hřídel, loket, ramena) a tyčí pro každé řešení qs (K,4)."""
    qs = np.atleast_2d(qs)
    l1, l2 = robot.link_lengths
    elbow = l1 * np.c_[np.cos(qs[:, 0]), np.sin(qs[:, 0])]
    ee = elbow + l2 * np.c_[np.cos(qs[:, 0] + qs[:, 1]), np.sin(qs[:, 0] + qs[:, 1])]
    z_ee = robot._z_offset + qs[:, 2]
    p, rz = rod_w[:, :2], rod_w[:, 2][None, :]
    shape = (len(qs), len(p))
    bodies = [  # (vzdálenost K×M, poloměr, které body tyče jsou ve výšce dílu)
        (np.linalg.norm(p[None] - ee[:, None], axis=-1), R_QUILL, rz > (z_ee - QUILL_BELOW)[:, None]),
        (np.linalg.norm(p[None] - elbow[:, None], axis=-1), R_ELBOW, np.broadcast_to(rz > Z_ELBOW, shape)),
        (_seg_dist(p, elbow, ee), R_FOREARM, np.broadcast_to(rz > Z_ARM, shape)),
        (_seg_dist(p, np.zeros_like(ee), elbow), R_UPPER, np.broadcast_to(rz > Z_ARM, shape)),
    ]
    clr = np.full(len(qs), np.inf)
    for dist, r, mask in bodies:
        clr = np.minimum(clr, np.where(mask, dist - r - R_ROD, np.inf).min(axis=1))
    return clr


def select_ik(robot, sols, q_prev, rod_ws):
    """Z řešení IK vybere to, které je v limitech kloubů, nekoliduje s tyčí a je nejblíž q_prev
    (nejmenší změna kloubů ||q - q_prev||, ne podle času). Kolize se kontrolují v cíli i v polovině přejezdu
    z q_prev a proti tyči ve všech časech z rod_ws (tyč se během kroku otáčí). Vrací q nebo None."""
    sols = np.array([q for q in sols if robot.in_limits(q)]).reshape(-1, 4)
    if not len(sols):
        return None
    ok = np.ones(len(sols), bool)
    for s in TRANSIT_S:
        q_s = q_prev + s * (sols - q_prev)                       # robot jede lineárně v kloubech
        for rod_w in rod_ws:
            ok &= clearance(robot, q_s, rod_w) >= CLEAR_MIN
    sols = sols[ok]
    if not len(sols):
        return None
    return sols[np.argmin(np.linalg.norm(sols - q_prev, axis=1))]


def rod_window(proc, rod_mm, t0, t1, n=3):
    """Tyč v n časech mezi t0 a t1 (pro kontrolu kolizí po celou dobu kroku)."""
    return [rod_world(proc, rod_mm, t) for t in np.linspace(t0, t1, n)]


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
    pts = proc.points.astype(float)    
    rod_mm = rod_samples(pts[:int(np.argmin(pts[:, 2])) + 1])    # tyč = cesta dolů po nejnižší bod (bez cesty zpět)
    proc.rod_mm = rod_mm

    #bod nad bludistěm
    poly = np.vstack([pts[0] + [0, 0, APPROACH_MM], pts])
    dense = [poly[0]]

    for p0, p1 in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / STEP_MM)))
        dense += [p0 + (p1 - p0) * k / n for k in range(1, n + 1)]
    proc.points = np.array(dense) # predict_points pak transformuje celou dráhu
    n = len(dense)

    q_now = robot.get_q()
    q_up = q_now.copy()
    q_up[2] = Z_PODLOZKA + poly[0][2] / 1000 - 0.5               # čistě svislý pohyb (mění se jen q3)

    if not robot.in_limits(q_up):
        print("výška najíždění je mimo rozsah, sniž APPROACH_MM")
        return None

    d_up = duration(a, v, q_up - q_now)

    gaps = np.full(n, 1.0)                                       # odstupy kroků [s]; první odhad
    t_first, tau = t_start + 10.0, None
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        times = t_first + np.concatenate([[0.0], np.cumsum(gaps[1:])])
        sols = []

        #vypocet ik: v každém kroku bezkolizní řešení nejbližší předchozímu
        q_prev = q_up
        for i, t in enumerate(times):
            p = proc.predict_points(t+0.5)[i]   # mm, z = výška nad deskou
            q = select_ik(robot, ik_all(robot, p[0] / 1000, p[1] / 1000, Z_PODLOZKA + p[2] / 1000),
                          q_prev, rod_window(proc, rod_mm, t + 0.4 - LEAD_T - gaps[i], t + 0.4 + 0.3 * gaps[i]))
            if q is None:
                print(f"Plán selhal: krok {i}: žádné bezkolizní řešení IK")
                return None
            sols.append([q])                                     # solve_path už jen kontroluje limity doby kroků
            q_prev = q
        qs, err = solve_path(sols, q_up, [MAX_BIG_S] + [MAX_STEP_S] * (n - 1), a, v)
        if qs is None:
            print("Plán selhal:", err)
            return None
        durs = [duration(a, v, q - qp) for q, qp in zip(qs, [q_up] + qs[:-1])]
        gaps = np.array([0.0] + [max(DT_MIN, SAFETY * d + T_CALC) for d in durs[1:]])
        tau = tau or (time.perf_counter() - t0)                  # doba jednoho výpočtu plánu
        t_first = t_start + (N_ITER + 1) * tau + d_up + durs[0] + MARGIN
    if time.perf_counter() > t_first - d_up - durs[0]:
        print("upozornění: výpočet plánu trval déle, než se čekalo; start bude pozdě")
    return [(None, q_up, 0.0, None)] + [(t, q, d + 0.15, i) for i, (t, q, d) in enumerate(zip(times, qs, durs))]


def retarget(robot, proc, i, q_cur, q_plan, a, v, bias):
    """Cíl kroku i spočítaný pro SKUTEČNÝ čas příjezdu (teď + doba pohybu + bias): bezkolizní řešení IK
    nejbližší aktuální poloze. Vrací (q, odhad času příjezdu)."""
    t_send = time.perf_counter()
    t_arr = t_send + duration(a, v, q_plan - q_cur) + bias
    q_best = q_plan
    for _ in range(3):                                           # cíl závisí na čase a čas na cíli
        p = proc.predict_points(t_arr + LEAD_T)[i]
        q = select_ik(robot, ik_all(robot, p[0] / 1000, p[1] / 1000, Z_PODLOZKA + p[2] / 1000),
                      q_cur, rod_window(proc, proc.rod_mm, t_send, t_arr + LEAD_T + 0.1))
        if q is None:
            print(f"krok {i}: žádné bezkolizní řešení, ZASTAVUJI")
            return None, None
        q_best = q
        t_new = time.perf_counter() + duration(a, v, q - q_cur) + bias
        if abs(t_new - t_arr) < 0.01:
            break
        t_arr = t_new
    return q_best, t_arr


def run_plan(robot, plan, proc, model):
    """Jede podle plánu, ale cíl každého kroku přepočítá na skutečný čas příjezdu (když robot nestíhá,
    míří tam, kde tyč bude, ne kde měla být). Vypíše, kde robot zaostával."""
    a, v = np.array(model["a"]), np.array(model["v"])
    errs, rows = [], []
    q_cur = None
    for t, q, lead, i in plan:
        if t is None:
            robot.move_to_q(q); robot.wait_for_motion_stop()
            q_cur = np.array(q, float)
            continue
        time.sleep(max(0.0, t - lead - time.perf_counter()))
        bias = float(np.median(errs[-5:])) if errs else 0.0     # o kolik je robot pomalejší než model
        t_send = time.perf_counter()
        q, t_arr = retarget(robot, proc, i, q_cur, q, a, v, bias)
        if q is None:
            break
        clr = min(clearance(robot, q, rw)[0] for rw in rod_window(proc, proc.rod_mm, t_arr, t_arr + LEAD_T))
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        t_done = time.perf_counter()
        errs.append(t_done - t_arr + bias)
        d = proc.points[i] - proc.points[i - 1] if i > 0 else np.zeros(3)
        kind = "sikmy" if abs(d[2]) > 1e-6 and np.hypot(d[0], d[1]) > 1e-6 else ("svisly" if abs(d[2]) > 1e-6 else "jiny")
        rows.append((kind, t_done - (t - 0.15), t_done - t_arr + bias, t_done - t_send, clr, i))
        q_cur = q
    for kind in ("svisly", "sikmy"):
        r = np.array([x[1:5] for x in rows if x[0] == kind])
        if len(r):
            print(f"{kind:7s}: zpoždění za plánem střední {r[:, 0].mean():+.2f} s (max {r[:, 0].max():+.2f}), "
                  f"krok trvá o {r[:, 1].mean():+.3f} s déle než model, skutečná doba kroku {r[:, 2].mean():.2f} s")
    if rows:
        k = int(np.argmin([x[4] for x in rows]))
        print(f"nejmenší vůle těla od tyče (podle modelu): {1000 * rows[k][4]:.0f} mm v kroku {rows[k][5]} "
              f"-> pokud robot narazil a tady je víc než ~0, jsou rozměry R_*/Z_* menší než skutečnost")
       


if __name__ == "__main__":
    with open("duration_model.json") as f:
        model = json.load(f)
    assert abs(model["speed_relative"] - SPEED) == 0, "změnils rychlost blbecku"

    robot = RobotBosch()

    robot.initialize()
    fix_negative_uhel(robot, True)
    robot.set_speed_relative(SPEED)
    proc = image_proc()

    proc.homography(np.load("H_angle_candidate_mm.npy"))
    images, times = observe(robot, N_IMAGES, PERIOD)
    centers, valid_times = proc.find_aruco_centers(images, times, IDS)
    if len(valid_times) <= 4:
        print(f"Značky nalezeny jen na {len(valid_times)} snímcích")
        robot.move_to_q([0.7, 0.4, 0, 2*np.pi])
        images, times = observe(robot, N_IMAGES, PERIOD)
        centers, valid_times = proc.find_aruco_centers(images, times, IDS)
    if len(valid_times) > 4:
        proc.fit_rotation(centers, valid_times)
        print("Stredy znacek:", centers)
        print("Casy:", valid_times)
        rotation_center, radius = proc.find_rotation_center(centers)
        print("Stred rotace:", rotation_center)
        print("Polomer drahy stredu ctverce:", radius)
        square_angle = proc.find_square_angle(centers, rotation_center)
        print("Uhel ctverce vuci polomeru [deg]:", np.degrees(square_angle))
        plan = make_plan(proc, robot, model)
        if plan is not None: run_plan(robot, plan, proc, model)
        else: print("Plán je nedosažitelný, nepohybuji.")
        q = robot.get_q()
        q[2] = 0
        robot.move_to_q(q)
    else: 
        print(f"Značky nalezeny jen na {len(valid_times)} snímcích")

    
    #time.sleep(10)
    fix_negative_uhel(robot, False)
    robot.soft_home()
    robot.close()

