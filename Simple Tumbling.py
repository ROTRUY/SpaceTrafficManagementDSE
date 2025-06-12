#!/usr/bin/env python3
"""
Monte‑Carlo that estimates how long a tumbling spacecraft needs to accumulate
all 30 words (18 s) of the first three GPS L1 C/A sub‑frames with two
antennas:

Initial LOS is guaranteed but it the initial contact window is randomized.

* Ant A on the +Z face – boresight = +Z.
* Ant B on the +Y face – boresight = +Y (adjacent panel).

Both patches have identical 110 ° full‑angle (55 ° half‑cone) FOVs.

No CLI flags are required; just press *Run* and the script sweeps the default
rpm grid `[0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`, 1 000 trials each, for tumble axes +X +Y +Z.

"""
from __future__ import annotations
import argparse, math, csv, pathlib
from typing import Sequence, List
import numpy as np
from numpy.random import default_rng

rng = default_rng()

# NAV signal / receiver parameters
WORD_SEC      = 0.6          # NAV word duration (s)
NEEDED_WORDS  = 30           # first 3 sub‑frames
FOV_HALF_DEG  = 55.0
COS_FOV       = math.cos(math.radians(FOV_HALF_DEG))

# Defaults
DEFAULT_RPMS    = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
DEFAULT_TRIALS  = 1000
DEFAULT_LAYOUT  = "Z+Y"

# Body axes for spin
AXES = {
    'X': np.array([1.0,0.0,0.0]),
    'Y': np.array([0.0,1.0,0.0]),
    'Z': np.array([0.0,0.0,1.0])
}

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_step(q: np.ndarray, axis: np.ndarray, omega: float) -> np.ndarray:
    half = 0.5*omega*WORD_SEC
    dq = np.concatenate(([math.cos(half)], math.sin(half)*axis))
    qn = quat_mul(q, dq)
    return qn/np.linalg.norm(qn)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])


def rand_unit() -> np.ndarray:
    v = rng.normal(size=3)
    return v/np.linalg.norm(v)

def initial_quat(axis: np.ndarray) -> np.ndarray:
    phi = rng.uniform(0, 2*math.pi)
    half = 0.5*phi
    return np.array([math.cos(half), *(math.sin(half)*axis)])


def boresights_from_layout(layout: str) -> List[np.ndarray]:
    if layout == "Z_ONLY":
        return [np.array([0,0,1], dtype=float)]
    if layout == "Z+Y":
        return [np.array([0,0,1], dtype=float), np.array([0,1,0], dtype=float)]
    if layout.startswith("CUSTOM:"):
        path = pathlib.Path(layout.split(":",1)[1]).expanduser()
        vecs = []
        with open(path, newline='') as f:
            for row in csv.reader(f):
                if len(row)==3:
                    v = np.array([float(x) for x in row])
                    vecs.append(v/np.linalg.norm(v))
        if not vecs:
            raise ValueError("Custom layout CSV empty or malformed")
        return vecs
    raise ValueError("Unknown --ant-layout value")

def single_trial(rpm: float, axis_key: str, bores: List[np.ndarray], guarantee_los: bool, t_max: float) -> float:
    axis  = AXES[axis_key]
    omega = rpm*2*math.pi/60.0
    q     = initial_quat(axis)

    # Pick LOS; ensure at least one antenna sees it if guarantee_los
    while True:
        s = rand_unit()
        if not guarantee_los:
            break
        R0 = quat_to_matrix(q)
        if any(R0@b @ s >= COS_FOV for b in bores):
            break

    captured = np.zeros(NEEDED_WORDS, dtype=bool)
    ticks = int(t_max/WORD_SEC)
    R = quat_to_matrix(q)
    for k in range(ticks+1):
        if any(R@b @ s >= COS_FOV for b in bores):
            idx = k % NEEDED_WORDS
            captured[idx] = True
            if captured.all():
                return (k+1)*WORD_SEC
        q = quat_step(q, axis, omega)
        R = quat_to_matrix(q)
    return math.nan

# Monte‑Carlo per axis

def mc_axis(rpms: Sequence[float], axis_key: str, bores, trials: int, guarantee_los: bool, t_max: float):
    print(f"\nTumbling about +{axis_key}-axis  |  antennas = {len(bores)}  (layout preset)")
    print("-"*70)
    for rpm in rpms:
        tarr = np.empty(trials)
        for i in range(trials):
            tarr[i] = single_trial(rpm, axis_key, bores, guarantee_los, t_max)
        med,p95 = np.nanpercentile(tarr,[50,95])
        succ = np.count_nonzero(~np.isnan(tarr))/trials
        print(f"rpm {rpm:6.2f} | med {med:7.2f} s | p95 {p95:7.2f} s | succ {succ:5.3f}")

def parse_args():
    p = argparse.ArgumentParser(description="GNSS ephemeris Monte‑Carlo with two antennas (word‑tick)")
    p.add_argument('--rpm', nargs='+', type=float, default=None, help='rpm values (default auto grid)')
    p.add_argument('--trials', type=int, default=DEFAULT_TRIALS, help='trials per rpm')
    p.add_argument('--axes', nargs='+', default=['ALL'], choices=['X','Y','Z','ALL'], help='which tumble axes to simulate')
    p.add_argument('--no-initial-los', action='store_true', help='allow start out of view')
    p.add_argument('--t-max', type=float, default=3600.0, help='timeout seconds')
    p.add_argument('--ant-layout', type=str, default=DEFAULT_LAYOUT, help='Z_ONLY | Z+Y | CUSTOM:/path/to/csv')
    return p.parse_args()


def main():
    args = parse_args()
    rpms = args.rpm if args.rpm is not None else DEFAULT_RPMS
    axes = ['X','Y','Z'] if 'ALL' in args.axes else args.axes
    guarantee = not args.no_initial_los

    bores = boresights_from_layout(args.ant_layout)

    for ax in axes:
        mc_axis(rpms, ax, bores, args.trials, guarantee, args.t_max)

if __name__ == '__main__':
    main()
