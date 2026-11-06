import json
import time
import numpy as np
from ctu_bosch_sr450 import RobotBosch

from calibrate import fix_negative_uhel
from image_processing import image_proc

SPEED = 0.5
N_IMAGES, PERIOD = 15, 0.7
ang = np.radians(187)
D = 0.135 * np.array([np.cos(ang), np.sin(ang)])    # střed obruče [m]
Z_PODLOZKA = 0.27                                   # vyska podlozky od zeme (odpovídá q[2] == -0.23)
APPROACH_MM = 30.0                                  # výška nad prvním bodem dráhy, ve které se najíždí [mm]
STEP_MM = 8                                         #  nemenit max. délka kroku po dráze [mm]
MAX_STEP_S, MAX_BIG_S = 4.0, 8.0                    # max. doba kroku / prvního přesunu [s] (nahrazuje limit skoku)
SAFETY, DT_MIN, MARGIN, N_ITER = 1.0, 0.05, 0.5, 3  # rezerva na dobu kroku, min. odstup kroků [s], rezerva startu [s], iterace plánu
PHI_GRID = np.radians(np.arange(-180, 180, 5))     # povolená natočení koncového bodu

ROD_TO_BOARD = True                                                      # tyč pokračuje od posledního bodu dráhy svisle dolů až na desku
ROD_SAMPLE_MM = 2        # hustota bodů tyče pro kontrolu kolizí
R_ROD = 0.0045              # poloměr tyče
R_QUILL = 0.04             # poloměr hřídele + příruby kolem osy koncového bodu (sahá od obruče nahoru)
QUILL_BELOW = 0.03        # o kolik příruba sahá pod rovinu obruče
R_ELBOW, Z_ELBOW = 0.09, 0.51       # poloměr pouzdra lokte a výška jeho spodku
R_FOREARM, R_UPPER, Z_ARM = 0.06, 0.08, 0.51   # poloviční šířky ramen a výška jejich spodku
CLEAR_MIN = 0.015          # min. vůle těla od tyče, jinak kolize

LEAD_T = 0.2               # o kolik za okamžik příjezdu [s]; jen dorovnání zpoždění kamery


CAMERA_DELAY = 0.03        # čas od začátku volání grab_image do pořízení snímku [s]
ARRIVE_EARLY = 0.15        # plán posílá pohyb tak, aby robot dojel o tolik dřív než plánovaný čas [s]
MAX_DPHI = np.radians(45)  # max. pootočení q4 za jeden krok po dráze (velké otočky by držákem/hřídelí smetly tyč)
TRANSIT_S = (0.33, 0.67, 1.0)  # kde na přejezdu mezi kroky se kontrolují kolize (1 = cíl)
T_CALC = 0.06                 # čas na přepočet cíle mezi kroky [s] (započte se do rozestupu kroků v plánu)

L_LIMB = 0.07        # délka ramene od koncového bodu směrem ke středu obruče [m]
R_LIMB = 0.04        # poloviční šířka tohoto ramene [m] 


def center_to_ee(cx, cy, phi):
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    return np.array([cx, cy]) - R @ D

def ee_phi(qs):
    return qs[:, 3] #+ qs[:, 1] 

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
        times.append(t0 + CAMERA_DELAY)                   # snímek vzniká na začátku volání, ne uprostřed
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

    phi = ee_phi(qs)
    d0 = D / np.linalg.norm(D)
    dir_ring = np.c_[np.cos(phi) * d0[0] - np.sin(phi) * d0[1],
                     np.sin(phi) * d0[0] + np.cos(phi) * d0[1]]
    limb_end = ee + L_LIMB * dir_ring

    p, rz = rod_w[:, :2], rod_w[:, 2][None, :]
    shape = (len(qs), len(p))
    bodies = [
        (np.linalg.norm(p[None] - ee[:, None], axis=-1), R_QUILL, rz > (z_ee - QUILL_BELOW)[:, None]),
        (_seg_dist(p, ee, limb_end), R_LIMB, rz > (z_ee - QUILL_BELOW)[:, None]),   # druha tyc blizko k kruhu
        (np.linalg.norm(p[None] - elbow[:, None], axis=-1), R_ELBOW, np.broadcast_to(rz > Z_ELBOW, shape)),
        (_seg_dist(p, elbow, ee), R_FOREARM, np.broadcast_to(rz > Z_ARM, shape)),
        (_seg_dist(p, np.zeros_like(ee), elbow), R_UPPER, np.broadcast_to(rz > Z_ARM, shape)),
    ]
    clr = np.full(len(qs), np.inf)
    for dist, r, mask in bodies:
        clr = np.minimum(clr, np.where(mask, dist - r - R_ROD, np.inf).min(axis=1))
    return clr

def select_ik(robot, sols, q_prev, rod_ws, q_ref=None, small_step=False):
    """Z řešení IK vybere to, které je v limitech kloubů, nekoliduje s tyčí a je nejblíž q_prev
    (nejmenší změna kloubů ||q - q_prev||, ne podle času); s q_ref vybírá nejbližší q_ref (plánované poloze).
    Kolize se kontrolují v bodech přejezdu TRANSIT_S z q_prev; rod_ws[k] je tyč v okamžiku, kdy je robot
    v bodě TRANSIT_S[k] (tyč se během kroku otáčí). Vrací q nebo None."""
    sols = np.array([q for q in sols if robot.in_limits(q)]).reshape(-1, 4)
    if small_step and len(sols):
        # po dráze jen malé kroky: q4 se nesmí otočit o víc než MAX_DPHI (jinak by zvolil stejnou polohu
        # o 360° jinak a otočil se dokola) a loket nesmí přeskočit na druhou stranu
        sols = sols[(np.abs(sols[:, 3] - q_prev[3]) <= MAX_DPHI)
                    & (np.sign(sols[:, 1]) == np.sign(q_prev[1]))]
    if not len(sols):
        return None
    ok = np.ones(len(sols), bool)
    for s, rod_w in zip(TRANSIT_S, rod_ws):
        q_s = q_prev + s * (sols - q_prev)                       # robot jede lineárně v kloubech
        ok &= clearance(robot, q_s, rod_w) >= CLEAR_MIN
    sols = sols[ok]
    if not len(sols):
        return None
    ref = q_prev if q_ref is None else q_ref
    return sols[np.argmin(np.linalg.norm(sols - ref, axis=1))]


def free_ik(robot, sols, rod_ws):
    """Všechna řešení IK v limitech kloubů, která v cíli nekolidují s tyčí v žádném z časů rod_ws."""
    sols = np.array([q for q in sols if robot.in_limits(q)]).reshape(-1, 4)
    ok = np.ones(len(sols), bool)
    for rod_w in rod_ws:
        ok &= clearance(robot, sols, rod_w) >= CLEAR_MIN
    return sols[ok]


def solve_path(sols_per_step, q_start, limits_s, a, v):
    """Z řešení IK každého kroku vybere jednu větev (dynamické programování): každý krok musí trvat nejvýš
    limit [s] podle modelu, minimalizuje se součet změn kloubů ||q - q_prev|| (nejbližší kroky, ne čas).
    Na rozdíl od výběru krok po kroku vidí dopředu, takže nezvolí polohu, ze které dál nevede bezkolizní cesta.
    Vrací (seznam q, None), nebo (None, důvod selhání)."""
    layers, costs, backs = [np.array([q_start])], [np.zeros(1)], []
    for i, (sols, lim) in enumerate(zip(sols_per_step, limits_s)):
        if not len(sols):
            return None, f"krok {i}: žádné bezkolizní řešení IK"
        cur, prev = np.array(sols), layers[-1]
        diff = np.abs(cur[:, None, :] - prev[None, :, :])        # (nové, staré, kloub)
        dur = np.where(diff > 1e-9, a + diff / v, 0.0).max(axis=2)
        trans = costs[-1][None, :] + np.linalg.norm(cur[:, None, :] - prev[None, :, :], axis=2)
        trans[dur > lim] = np.inf
        if i >= 1:                                               # po dráze: žádné velké otočky q4 ani přeskok lokte
            trans[diff[:, :, 3] > MAX_DPHI] = np.inf
            trans[np.sign(cur[:, None, 1]) != np.sign(prev[None, :, 1])] = np.inf
        back, cost = trans.argmin(axis=1), trans.min(axis=1)
        ok = np.isfinite(cost)
        if not ok.any():
            if (dur <= lim).any():
                return None, f"krok {i}: bezkolizní poloha by vyžadovala pootočit q4 o víc než {np.degrees(MAX_DPHI):.0f}°"
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
    q_up[2] = Z_PODLOZKA + poly[0][2] / 1000 - 0.5 

    if not robot.in_limits(q_up):   
        print("výška najíždění je mimo rozsah, sniž APPROACH_MM")
        return None

    d_up = duration(a, v, q_up - q_now)

    gaps = np.full(n, 1.0)  # odstupy kroků [s]; první odhad
    t_first, tau = t_start + 10.0, None
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        times = t_first + np.concatenate([[0.0], np.cumsum(gaps[1:])])
        sols = []

        #vypocet ik: v každém kroku všechna bezkolizní řešení, výběr nejbližších kroků udělá solve_path
        for i, t in enumerate(times):
            t_arr = t - ARRIVE_EARLY                              # kdy robot podle plánu dojede
            p = proc.predict_points(t_arr + LEAD_T)[i]   # mm, z = výška nad deskou
            sols.append(free_ik(robot, ik_all(robot, p[0] / 1000, p[1] / 1000, Z_PODLOZKA + p[2] / 1000),
                                [rod_world(proc, rod_mm, t_arr + LEAD_T)]))
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
    return [(None, q_up, 0.0, None)] + [(t, q, d + ARRIVE_EARLY, i) for i, (t, q, d) in enumerate(zip(times, qs, durs))]


def retarget(robot, proc, i, q_cur, q_plan, a, v, bias):
    """Cíl kroku i spočítaný pro SKUTEČNÝ čas příjezdu (teď + doba pohybu + bias): bezkolizní řešení IK
    (i na přejezdu z q_cur) nejbližší plánované poloze q_plan, takže drží větev, kterou plán vybral s výhledem dopředu.
    Vrací (q, odhad času příjezdu)."""
    t_send = time.perf_counter()
    t_arr = t_send + duration(a, v, q_plan - q_cur) + bias
    q_best = q_plan
    for _ in range(3):                                           # cíl závisí na čase a čas na cíli
        p = proc.predict_points(t_arr + LEAD_T)[i]
        q = select_ik(robot, ik_all(robot, p[0] / 1000, p[1] / 1000, Z_PODLOZKA + p[2] / 1000),
                      q_cur, [rod_world(proc, proc.rod_mm, t_send + s * (t_arr - t_send) + LEAD_T) for s in TRANSIT_S],
                      q_ref=q_plan, small_step=i > 0)
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
    """Jede podle plánu, cíl každého kroku přepočítá na skutečný čas příjezdu.
    Vrací (dojel celou dráhu?, seznam dosažených indexů bodů, poslední q)."""
    a, v = np.array(model["a"]), np.array(model["v"])
    errs, reached, calc = [], [], []
    q_cur = None
    for t, q, lead, i in plan:
        if t is None:
            robot.move_to_q(q); robot.wait_for_motion_stop()
            q_cur = np.array(q, float)
            continue
        time.sleep(max(0.0, t - lead - time.perf_counter()))
        bias = float(np.median(errs[-5:])) if errs else 0.0     # o kolik je robot pomalejší než model
        t0 = time.perf_counter()
        q, t_arr = retarget(robot, proc, i, q_cur, q, a, v, bias)
        calc.append(time.perf_counter() - t0)
        if q is None:
            return False, reached, q_cur
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        errs.append(time.perf_counter() - t_arr + bias)
        reached.append(i)
        q_cur = q
    print(f"přepočet mezi kroky trvá střední {np.mean(calc):.3f} s (T_CALC = {T_CALC}); "
          f"krok trvá o {np.mean(errs):+.3f} s déle než model")
    return True, reached, q_cur


def go_back(robot, proc, model, reached, q_cur):
    """Nouzový návrat: projede dosažené body pozpátku (po tyči zpět nahoru) se stejnou kontrolou kolizí.
    Vrací True, když se dostal až nad tyč."""
    a, v = np.array(model["a"]), np.array(model["v"])
    for i in reversed(reached[:-1]):
        q, _ = retarget(robot, proc, i, q_cur, q_cur, a, v, 0.0)
        if q is None:
            return False
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        q_cur = q
    return True


if __name__ == "__main__":
    with open("duration_model.json") as f:
        model = json.load(f)
    assert abs(model["speed_relative"] - SPEED) < 1e-9, "změnils rychlost blbecku"

    robot = RobotBosch()
    robot.initialize()
    safe_to_leave = True                                  # False = obruč zůstala na tyči, nehýbat
    try:
        fix_negative_uhel(robot, True)
        robot.set_speed_relative(SPEED)
        proc = image_proc()

        proc.homography(np.load("H_angle_candidate_mm.npy"))
        images, times = observe(robot, N_IMAGES, PERIOD)
        centers, valid_times = proc.find_aruco_centers(images, times)

        if len(valid_times) <= 4:
            print(f"Značky nalezeny jen na {len(valid_times)} snímcích")
            robot.move_to_q([0.7, 0.4, 0, 2*np.pi])
            robot.wait_for_motion_stop()                  # fotit až po dojetí
            images, times = observe(robot, N_IMAGES, PERIOD)
            centers, valid_times = proc.find_aruco_centers(images, times)

        if len(valid_times) > 4:
            proc.fit_rotation(centers, valid_times)
            print("Stredy znacek:", centers)
            print("Casy:", valid_times)
            print("Stred rotace:", proc.rotation_center)
            print("Polomer drahy stredu ctverce:", proc.radius)
            print("Uhel ctverce vuci polomeru [deg]:", np.degrees(proc.square_angle))
            plan = make_plan(proc, robot, model)
            if plan is not None:
                safe_to_leave = False
                ok, reached, q_cur = run_plan(robot, plan, proc, model)
                if ok:
                    safe_to_leave = True                  # dráha končí nahoře nad tyčí
                else:
                    print("Jízda zastavena, vracím se po tyči zpět nahoru.")
                    safe_to_leave = go_back(robot, proc, model, reached, q_cur)
            else:
                print("Plán je nedosažitelný, nepohybuji.")
        else:
            print(f"Značky nalezeny jen na {len(valid_times)} snímcích")

        if safe_to_leave:
            q = robot.get_q()
            q[2] = 0
            robot.move_to_q(q)
            robot.wait_for_motion_stop()                  # nejdřív dojet nahoru, pak teprve dál
            robot.set_speed_relative(1)
            fix_negative_uhel(robot, False)
            robot.soft_home()
        else:
            print("POZOR: obruč zůstala na tyči, robot stojí. Odjeď ručně.")
    except Exception:
        print("Chyba za běhu, robot zastaven na místě (neodjíždím, obruč může být na tyči).")
        raise
    finally:
        robot.close()
