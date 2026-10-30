"""
Měření doby pohybu robota Bosch SR450: pro každý kloub režie a_j [s] a rychlost v_j [rad/s, u q3 m/s].
Model: doba koordinovaného pohybu o dq = max_j (a_j + |dq_j| / v_j)   (domněnka, skript ji ověří na kombinovaných pohybech)

Měří se při STEJNÉ relativní rychlosti, jakou pak používá main (SPEED). Robot se pohybuje kolem aktuální polohy,
proto ho před spuštěním dej do volného místa (nad pracovní plochou, daleko od zdí).

  python measure_robot.py            # změří a uloží duration_model.json
"""
import json
import time

import numpy as np

SPEED = 0.3                                                   # musí odpovídat set_speed_relative v main
DISTANCES = [(0.10, 0.20, 0.40),                              # q1 [rad]
             (0.10, 0.20, 0.40),                              # q2 [rad]
             (0.02, 0.05, 0.10),                              # q3 [m]
             (0.20, 0.50, 1.00)]                              # q4 [rad]
REPEATS = 3


def timed_move(robot, q):
    t0 = time.perf_counter()
    robot.move_to_q(q)
    t1 = time.perf_counter()
    robot.wait_for_motion_stop()
    t2 = time.perf_counter()
    return t1 - t0, t2 - t0                                   # (doba volání move_to_q, celková doba pohybu)


def measure_joint(robot, q0, j, distances):
    """Pohyb kloubu j o několik délek tam a zpět. Vrací (a_j, v_j, rezidua, doba volání move_to_q)."""
    sign = 1.0 if q0[j] + max(distances) <= robot.q_max[j] else -1.0
    ds, ts, calls = [], [], []
    for d in distances:
        q = q0.copy(); q[j] += sign * d
        for _ in range(REPEATS):
            call, total = timed_move(robot, q)
            ds.append(d); ts.append(total); calls.append(call)
            timed_move(robot, q0)                              # zpět (neměří se)
    ds, ts = np.array(ds), np.array(ts)
    slope, a = np.polyfit(ds, ts, 1)                           # t = a + d / v
    v = 1.0 / slope
    resid = ts - (a + ds / v)
    return a, v, resid, float(np.median(calls))


def predict_duration(a, v, dq):
    dq = np.abs(np.asarray(dq, dtype=float))
    moving = dq > 1e-9
    return float(np.max(a[moving] + dq[moving] / v[moving])) if moving.any() else 0.0


def validate(robot, q0, a, v, n=6, seed=0):
    """Kombinované pohyby (víc kloubů najednou): predikce vs. měření."""
    rng = np.random.default_rng(seed)
    scale = np.array([0.3, 0.3, 0.04, 0.6])
    errs = []
    for _ in range(n):
        dq = rng.uniform(-1, 1, 4) * scale
        q = np.clip(q0 + dq, robot.q_min, robot.q_max)
        _, total = timed_move(robot, q)
        pred = predict_duration(a, v, q - q0)
        errs.append((pred, total))
        timed_move(robot, q0)
    return np.array(errs)


def main(robot, path="duration_model.json"):
    robot.set_speed_relative(SPEED)
    q0 = robot.get_q()
    a = np.zeros(4); v = np.zeros(4)
    for j in range(4):
        a[j], v[j], resid, call = measure_joint(robot, q0, j, DISTANCES[j])
        print(f"q{j + 1}: a = {a[j]:.2f} s, v = {v[j]:.3f} /s, rezidua do {np.abs(resid).max():.2f} s, "
              f"move_to_q se vrací za {call:.3f} s")
    errs = validate(robot, q0, a, v)
    rel = np.abs(errs[:, 0] - errs[:, 1]) / errs[:, 1]
    print(f"kombinované pohyby: predikce vs. měření, střední chyba {100 * rel.mean():.0f} %, max {100 * rel.max():.0f} %")
    model = {"speed_relative": SPEED, "a": a.tolist(), "v": v.tolist(),
             "validation_rel_err_max": float(rel.max())}
    with open(path, "w") as f:
        json.dump(model, f, indent=2)
    return model


if __name__ == "__main__":
    from ctu_bosch_sr450 import RobotBosch

    robot = RobotBosch()
    try:
        robot.initialize()
        main(robot)
        robot.soft_home()
    finally:
        robot.close()