import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ctu_bosch_sr450 import RobotBosch

OUT_DIR = "dataset_stolek"
N_IMAGES = 20        # počet snímků
PERIOD = 1.0         # s, odstup mezi snímky (pevný rytmus)


def build_dataset(robot, out_dir=OUT_DIR, n=N_IMAGES, period=PERIOD):
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    t_start = time.perf_counter()
    try:
        for i in range(n):
            t_before = time.perf_counter()
            img = robot.grab_image()
            t_after = time.perf_counter()

            name = f"img{i:03d}.png"
            cv2.imwrite(f"{out_dir}/{name}", img)

            meta.append({
                "index": i,
                "file": name,
                "t": 0.5 * (t_before + t_after) - t_start,   # odhad okamžiku expozice [s od 1. snímku]
                "t_before": t_before - t_start,
                "t_after": t_after - t_start,
                "grab_duration": t_after - t_before,
                "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            })
            print(f"[{i + 1}/{n}] t = {meta[-1]['t']:.3f} s, grab = {meta[-1]['grab_duration']:.3f} s")

            if i < n - 1:                                   # pevný rytmus nezávislý na zápisu
                wait = t_start + (i + 1) * period - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    print(f"  upozornění: snímek {i + 1} se zpozdil o {-wait:.3f} s")
    finally:
        with open(f"{out_dir}/meta.json", "w") as f:
            json.dump({"period_requested": period, "frames": meta}, f, indent=2)

    dts = np.diff([m["t"] for m in meta])
    if len(dts):
        print(f"skutečný odstup: střední {dts.mean():.3f} s, min {dts.min():.3f}, max {dts.max():.3f}")


if __name__ == "__main__":
    robot = RobotBosch(tty_dev=None)     # bez připojení k robotu, jen kamera
    build_dataset(robot)