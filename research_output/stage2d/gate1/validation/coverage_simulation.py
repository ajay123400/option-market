"""Synthetic dependence study for the Gate 1 resampling procedures (known truth).

Data-generating process (calibrated to the observed sample): 260 clusters ('expiries') x 25 rows; row = cluster effect + noise, noise sd 1,
cluster-effect sd 1.175 (within-cluster correlation 0.58, the observed ICC of the calendar spread), cluster effects AR(1) across consecutive clusters with
coefficient phi (observed lag-1 autocorrelation of the per-expiry series: 0.13-0.23). The estimand is the all-observation median; its population value is 0
by symmetry. Reported: empirical coverage and mean width of 95% percentile intervals (R replications, B bootstrap draws), for
row-level iid resampling (ignores clustering), whole-cluster iid, circular moving blocks of clusters (b = 2, 5, 8) and stationary blocks (mean 5).
Usage: python research_output/stage2d/gate1/validation/coverage_simulation.py [R] [B]
"""
import math
import os
import sys
import time

sys.path.insert(0, os.getcwd())          # run from the repository root

import numpy as np

from optionsengine.research.stage2d import clusters as C

R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
B = int(sys.argv[2]) if len(sys.argv) > 2 else 300
N_CL, M, SIG_MU = 260, 25, 1.175


class RowMedian:
    def __init__(self, cs):
        self.x = cs.cols["x"]

    def evaluate(self, rows, seq):
        return np.array([np.median(self.x[rows])])


def dataset(rng, phi):
    z = rng.normal(size=N_CL + 100)
    a = np.zeros(N_CL + 100)
    for t in range(1, N_CL + 100):
        a[t] = phi * a[t - 1] + z[t] * math.sqrt(1 - phi ** 2)
    mu = SIG_MU * a[100:]
    x = np.repeat(mu, M) + rng.normal(0, 1.0, N_CL * M)
    return np.repeat([f"c{i:04d}" for i in range(N_CL)], M), x


METHODS = [("row_iid (ignores clustering)", "row", "iid", None), ("cluster_iid", "cl", "iid", None), ("cluster_moving_block_b2", "cl", "moving_block", 2),
           ("cluster_moving_block_b5", "cl", "moving_block", 5), ("cluster_moving_block_b8", "cl", "moving_block", 8), ("cluster_stationary_mean5", "cl", "stationary", 5)]
print(f"R={R} replications, B={B} bootstrap draws, {N_CL} clusters x {M} rows; truth = 0")
print(f"{'phi':>5s}  {'method':32s} {'coverage':>9s} {'mean width':>11s} {'MC se(cov)':>11s}")
t0 = time.time()
for phi in (0.0, 0.25, 0.6):
    rng = np.random.default_rng(4242 + int(phi * 100))
    hit = {m[0]: 0 for m in METHODS}
    wid = {m[0]: 0.0 for m in METHODS}
    for r in range(R):
        labels, x = dataset(rng, phi)
        cs = C.ClusterSet(labels, {"x": x})
        rs = C.ClusterSet(np.char.zfill(np.arange(len(x)).astype(str), 7), {"x": x}, row_level=True)
        for name, kind, scheme, b in METHODS:
            c = cs if kind == "cl" else rs
            lo, hi = C.percentile_ci(C.bootstrap(c, RowMedian(c).evaluate, scheme, b, B, 9000 + r, k=1))
            hit[name] += bool(lo[0] <= 0.0 <= hi[0])
            wid[name] += hi[0] - lo[0]
    for name, *_ in METHODS:
        cov = hit[name] / R
        print(f"{phi:5.2f}  {name:32s} {cov:9.3f} {wid[name] / R:11.3f} {math.sqrt(cov * (1 - cov) / R):11.3f}")
    print(f"       ({time.time() - t0:.0f} s elapsed)", flush=True)
