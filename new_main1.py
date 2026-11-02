"""
new_main1: new_main + (1) kontrola kolizí ramene s tyčí, (2) cíl každého kroku se přepočítá na skutečný čas příjezdu.

Proč tyč nebyla uprostřed obruče (new_main):
  - run_plan posílal pohyb o (d + 0.1) s dřív -> obruč dojela o 0.1 s dřív a pak stála, deska se mezitím točila dál,
  - když krok trval déle než model, zpoždění se sčítalo a robot jel na polohu, kde tyč BYLA,
  - DP mohlo mezi kroky libovolně měnit natočení phi i větev IK (loket nahoře/dole); robot se pohybuje lineárně
    v kloubech, takže při takové změně se obruč i rameno uprostřed kroku vychýlí -> obruč mimo střed, rameno do tyče.
Proč rameno naráželo do tyče:
  - IK řešení se vybíralo jen podle času; nic nehlídalo, kde je vůči tyči hřídel/příruba, loket a ramena,
  - na konci se robot z dolního konce tyče odjížděl bokem (fix_negative_uhel) a táhl obruč přes tyč.

Co dělá new_main1:
  - každé řešení IK se zkontroluje proti tyči natočené do polohy v daném čase (hřídel + příruba, loket,
    ramena; obruč a držák se nekontrolují); kolizní se zahodí, blízká se penalizují (robot volí natočení, kde je tělo dál od tyče),
  - mezi kroky po dráze se nesmí změnit větev IK a phi se smí změnit max o MAX_DPHI,
  - za jízdy: před každým krokem se cíl přepočítá pro čas skutečného příjezdu (now + doba pohybu) a znovu
    zkontroluje kolize; robot jede hned po dojetí předchozího kroku, nečeká na plánovaný čas,
  - na konci se obruč vrátí po tyči nahoru (stejnou cestou zpět) a teprve pak odjede.

Rozměry těla robota (R_*, Z_*) jsou odhady -> ZMĚŘIT a upravit.
"""
import json
import time
import numpy as np
from ctu_bosch_sr450 import RobotBosch

from calibrate import fix_negative_uhel
from image_processing import image_proc

SPEED = 0.5
IDS = (1, 2) #id aruco značek
N_IMAGES, PERIOD = 15, 0.7
D = 0.135 * np.array([np.cos(np.radians(185)), np.sin(np.radians(185))]) # střed obruče v s.s. koncového bodu [m]
Z_PODLOZKA = 0.28 # vyska podlozky od zeme (v q[2] == -0.22)
PATH_ORDER = [4, 3, 2, 1, 0]                              # pořadí bodů proc.points: od horního konce dolů (OVĚŘIT)
APPROACH_MM = 10.0                                        # výška nad prvním bodem dráhy, ve které se najíždí [mm]
STEP_MM = 6.7                                            # max. délka kroku po dráze [mm]
MAX_STEP_S, MAX_BIG_S = 0.7, 4.0                          # max. doba kroku / prvního přesunu [s]
MARGIN, N_ITER = 0.5, 3                                   # rezerva startu [s], iterace plánu
PHI_GRID = np.radians(np.arange(-180, 180, 15))           # povolená natočení koncového bodu (obruč je symetrická)
CAMERA_DELAY = 0.0                                        # o kolik je snímek starší než střed volání grab_image [s] (změřit)

# --- tyč a tělo robota (všechno v m, z = výška od země jako robot.fk) --------------------------------
ROD_TO_BOARD = True        # tyč pokračuje od posledního bodu dráhy svisle dolů až na desku
ROD_SAMPLE_MM = 5.0        # hustota bodů tyče pro kontrolu kolizí
R_ROD = 0.004              # poloměr tyče
R_QUILL = 0.04             # poloměr hřídele + příruby kolem osy koncového bodu (sahá od obruče nahoru)
QUILL_BELOW = 0.01         # o kolik příruba sahá pod rovinu obruče
R_ELBOW, Z_ELBOW = 0.09, 0.50       # poloměr pouzdra lokte a výška jeho spodku (ZMĚŘIT)
R_FOREARM, R_UPPER, Z_ARM = 0.06, 0.08, 0.55   # poloviční šířky ramen a výška jejich spodku (ZMĚŘIT)
CLEAR_MIN = 0.015          # min. vůle těla od tyče, jinak kolize
CLEAR_SOFT, W_CLEAR = 0.05, 20.0    # pod touto vůlí se řešení penalizuje (W_CLEAR [s/m])
MAX_DPHI = np.radians(20)  # max. změna natočení q4 za jeden krok po dráze
RETARGET_DPHI = np.radians([0, -10, 10])   # natočení zkoušená při přepočtu za jízdy (kolem plánovaného)


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
        times.append(0.5 * (t0 + time.perf_counter()) - CAMERA_DELAY)
        if i < n - 1:
            time.sleep(max(0.0, t_start + (i + 1) * period - time.perf_counter()))
    return images, np.array(times)


def ik_all(robot, cx, cy, z, phis=None):
    """Všechna řešení IK (v limitech) pro střed obruče (cx, cy, z) při různých natočeních koncového bodu."""
    if phis is None:
        phis = np.append(PHI_GRID, np.arctan2(cy, cx))
    sols = []
    for phi in phis:
        ex, ey = center_to_ee(cx, cy, phi)
        sols += [q for q in robot.ik(ex, ey, z, phi) if robot.in_limits(q)]
    return np.array(sols).reshape(-1, 4)


# --- tyč v čase ----------------------------------------------------------------
def to_world(proc, pts_mm, t):
    """Body v souřadnicích desky [mm, z nad deskou] -> svět v čase t [m, z od země] (stejně jako predict_points)."""
    theta = proc.reference_angle + proc.angular_velocity * (t - proc.reference_time)
    ang = theta + proc.square_angle
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    c = proc.rotation_center + proc.radius * np.array([np.cos(theta), np.sin(theta)])
    pts = np.atleast_2d(np.asarray(pts_mm, float))
    out = np.empty_like(pts)
    out[:, :2] = (pts[:, :2] @ R.T + c) / 1000
    out[:, 2] = Z_PODLOZKA + pts[:, 2] / 1000
    return out


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


# --- kolize těla robota s tyčí -----------------------------------------------------
def _seg_dist(p, a, b):
    """Vzdálenost bodů p (M,2) od úseček a-b (K,2) -> (K, M)."""
    ab = (b - a)[:, None, :]
    ap = p[None, :, :] - a[:, None, :]
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(ap - t[..., None] * ab, axis=-1)


def clearance(robot, qs, rod_w):
    """Nejmenší vůle [m] mezi tělem robota a tyčí pro každé řešení qs (K,4); rod_w (M,3) ve světě."""
    qs = np.atleast_2d(qs)
    l1, l2 = robot.link_lengths
    elbow = l1 * np.c_[np.cos(qs[:, 0]), np.sin(qs[:, 0])]
    ee = elbow + l2 * np.c_[np.cos(qs[:, 0] + qs[:, 1]), np.sin(qs[:, 0] + qs[:, 1])]
    z_ee = robot._z_offset + qs[:, 2]
    base = np.zeros_like(ee)
    p, rz = rod_w[:, :2], rod_w[:, 2][None, :]
    inf = np.inf
    bodies = [  # (vzdálenost K×M, poloměr, maska výšky K×M)
        (np.linalg.norm(p[None] - ee[:, None], axis=-1), R_QUILL, rz > (z_ee - QUILL_BELOW)[:, None]),
        (np.linalg.norm(p[None] - elbow[:, None], axis=-1), R_ELBOW, np.broadcast_to(rz > Z_ELBOW, (len(qs), len(p)))),
        (_seg_dist(p, elbow, ee), R_FOREARM, np.broadcast_to(rz > Z_ARM, (len(qs), len(p)))),
        (_seg_dist(p, base, elbow), R_UPPER, np.broadcast_to(rz > Z_ARM, (len(qs), len(p)))),
    ]
    clr = np.full(len(qs), inf)
    for dist, r, mask in bodies:
        clr = np.minimum(clr, np.where(mask, dist - r - R_ROD, inf).min(axis=1))
    return clr


def solve_path(sols_per_step, clear_per_step, q_start, limits_s, a, v):
    """Z řešení IK každého kroku vybere jednu větev (dynamické programování). Krok musí trvat nejvýš limit [s],
    po dráze (krok >= 1) se nesmí měnit větev IK a q4 se smí změnit max o MAX_DPHI.
    Minimalizuje se celková doba + penalizace malé vůle od tyče. Vrací (seznam q, None), nebo (None, důvod)."""
    layers, costs, backs = [np.array([q_start])], [np.zeros(1)], []
    for i, (cur, clr, lim) in enumerate(zip(sols_per_step, clear_per_step, limits_s)):
        if not len(cur):
            return None, f"krok {i}: žádné bezkolizní řešení IK (nedosažitelné, mimo limity nebo rameno v tyči)"
        prev = layers[-1]
        diff = np.abs(cur[:, None, :] - prev[None, :, :])        # (nové, staré, kloub)
        dur = np.where(diff > 1e-9, a + diff / v, 0.0).max(axis=2)
        pen = W_CLEAR * np.maximum(0.0, CLEAR_SOFT - clr)
        trans = costs[-1][None, :] + dur + pen[:, None]
        bad = dur > lim
        if i >= 1:
            bad |= np.sign(cur[:, None, 1]) != np.sign(prev[None, :, 1])   # stejná větev (loket)
            bad |= diff[:, :, 3] > MAX_DPHI
        trans[bad] = np.inf
        back, cost = trans.argmin(axis=1), trans.min(axis=1)
        ok = np.isfinite(cost)
        if not ok.any():
            return None, (f"krok {i}: žádný přechod nesplní limit {lim} s / stejnou větev / změnu phi <= "
                          f"{np.degrees(MAX_DPHI):.0f}° (nejkratší krok {dur.min():.2f} s)")
        layers.append(cur[ok]); costs.append(cost[ok]); backs.append(back[ok])
    k = int(np.argmin(costs[-1]))
    path = []
    for i in range(len(backs), 0, -1):
        path.append(layers[i][k]); k = backs[i - 1][k]
    return path[::-1], None


def make_plan(proc, robot, model, dense, rod_mm):
    """Plán: (1) svisle na výšku najíždění, (2) přesun nad první bod, (3) sjezd a jízda po dráze.
    Časy jsou jen odhad pro výběr bezkolizních konfigurací; za jízdy se cíle přepočítávají (run_chase).
    Vrací (q_up, seznam q pro body dense) nebo None."""
    a, v = np.array(model["a"]), np.array(model["v"])
    t_start = time.perf_counter()
    n = len(dense)

    q_now = robot.get_q()
    q_up = q_now.copy()
    q_up[2] = Z_PODLOZKA + dense[0][2] / 1000 - 0.5               # čistě svislý pohyb (mění se jen q3)
    if not robot.in_limits(q_up):
        print("výška najíždění je mimo rozsah, sniž APPROACH_MM")
        return None
    d_up = duration(a, v, q_up - q_now)

    gaps = np.full(n, 1.0)
    t_first, tau = t_start + 10.0, None
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        times = t_first + np.concatenate([[0.0], np.cumsum(gaps[1:])])
        sols, clrs = [], []
        for i, t in enumerate(times):
            p = to_world(proc, dense[i], t)[0]
            s = ik_all(robot, p[0], p[1], p[2])
            c = clearance(robot, s, to_world(proc, rod_mm, t)) if len(s) else np.zeros(0)
            sols.append(s[c >= CLEAR_MIN]); clrs.append(c[c >= CLEAR_MIN])
        qs, err = solve_path(sols, clrs, q_up, [MAX_BIG_S] + [MAX_STEP_S] * (n - 1), a, v)
        if qs is None:
            print("Plán selhal:", err)
            return None
        durs = [duration(a, v, q - qp) for q, qp in zip(qs, [q_up] + qs[:-1])]
        gaps = np.array([0.0] + durs[1:])
        tau = tau or (time.perf_counter() - t0)
        t_first = t_start + (N_ITER + 1) * tau + d_up + durs[0] + MARGIN
    print(f"plán: {n} kroků, nejmenší vůle těla od tyče "
          f"{1000 * min(clearance(robot, q, to_world(proc, rod_mm, t))[0] for q, t in zip(qs, times)):.0f} mm")
    return q_up, qs


def retarget(robot, proc, a, v, q_cur, q_ref, p_mm, rod_mm, bias=0.0):
    """Cíl pro bod p_mm tak, aby tam obruč byla v okamžiku skutečného příjezdu. Hledá u plánovaného natočení
    a ve stejné větvi jako q_ref, bezkolizně. bias = průměrné zpoždění robota za modelem [s]. Vrací (q, odhad času příjezdu) nebo (None, None)."""
    t_arr = time.perf_counter() + duration(a, v, q_ref - q_cur) + bias
    best = None
    for _ in range(2):                                     # čas příjezdu závisí na cíli -> 2 iterace
        p = to_world(proc, p_mm, t_arr)[0]
        s = ik_all(robot, p[0], p[1], p[2], q_ref[3] + RETARGET_DPHI)
        if not len(s):
            return None, None
        s = s[(np.sign(s[:, 1]) == np.sign(q_ref[1])) & (np.abs(s[:, 3] - q_ref[3]) <= MAX_DPHI)]
        if not len(s):
            return None, None
        c = clearance(robot, s, to_world(proc, rod_mm, t_arr))
        s = s[c >= CLEAR_MIN]
        if not len(s):
            return None, None
        best = min(s, key=lambda q: duration(a, v, q - q_cur))
        t_arr = time.perf_counter() + duration(a, v, best - q_cur) + bias
    return best, t_arr


def run_chase(robot, proc, model, dense, qs_ref, order, rod_mm):
    """Projede body dense v pořadí order; každý cíl se přepočítá na skutečný čas příjezdu.
    Vrací (dojel celou cestu?, index posledního dosaženého bodu nebo None)."""
    a, v = np.array(model["a"]), np.array(model["v"])
    errs, reached = [], None
    q_cur = robot.get_q()
    for i in order:
        bias = float(np.median(errs[-5:])) if errs else 0.0     # učí se, o kolik robot zaostává za modelem
        q, t_arr = retarget(robot, proc, a, v, q_cur, qs_ref[i], dense[i], rod_mm, bias)
        if q is None:
            print(f"bod {i}: žádný bezkolizní cíl, zastavuji")
            return False, reached
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        errs.append(time.perf_counter() - t_arr + bias)
        q_cur, reached = q, i
    errs = np.array(errs)
    print(f"skutečný příjezd vs. odhad [s]: střední {errs.mean():+.3f}, max {np.abs(errs).max():.3f}")
    return True, reached


if __name__ == "__main__":
    with open("duration_model.json") as f:
        model = json.load(f)
    assert abs(model["speed_relative"] - SPEED) == 0, "změnils rychlost blbecku"

    robot = RobotBosch()
    robot.initialize()
    fix_negative_uhel(robot, True)
    robot.set_speed_relative(SPEED)
    proc = image_proc()

    proc.homography(np.load("H.npy"))
    images, times = observe(robot, N_IMAGES, PERIOD)
    centers, valid_times = proc.find_aruco_centers(images, times, IDS)
    if len(valid_times) < 4:
        print(f"Značky nalezeny jen na {len(valid_times)} snímcích")
    else:
        proc.fit_rotation(centers, valid_times)
        print("Stred rotace:", proc.rotation_center, " polomer:", proc.radius,
              " omega [deg/s]:", np.degrees(proc.angular_velocity))

        pts = proc.points[PATH_ORDER].astype(float)            # tyč od horního konce dolů [mm, souř. desky]
        rod_mm = rod_samples(pts)
        poly = np.vstack([pts[0] + [0, 0, APPROACH_MM], pts])
        dense = [poly[0]]
        for p0, p1 in zip(poly[:-1], poly[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / STEP_MM)))
            dense += [p0 + (p1 - p0) * k / n for k in range(1, n + 1)]
        dense = np.array(dense)

        plan = make_plan(proc, robot, model, dense, rod_mm)
        if plan is None:
            print("Plán je nedosažitelný, nepohybuji.")
        else:
            q_up, qs = plan
            robot.move_to_q(q_up); robot.wait_for_motion_stop()
            ok, reached = run_chase(robot, proc, model, dense, qs, range(len(dense)), rod_mm)
            # zpět po tyči nahoru nad horní konec (z bodu, kam dojel), ať obruč netáhne přes tyč
            back_ok = reached is None or run_chase(robot, proc, model, dense, qs, range(reached, -1, -1), rod_mm)[0]
            if not back_ok:
                print("POZOR: návrat po tyči se nepovedl, robot stojí - odjeď ručně")
                robot.close()
                raise SystemExit(1)

    q = robot.get_q(); q[2] = 0.0                          # svisle nahoru nad vše
    robot.move_to_q(q); robot.wait_for_motion_stop()
    fix_negative_uhel(robot, False)
    robot.soft_home()
    robot.close()
