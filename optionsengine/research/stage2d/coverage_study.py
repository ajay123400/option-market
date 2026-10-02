"""T11 Tier B: synthetic coverage / dependence / size-and-power study of the Gate 1 and Gate 2 inference procedures (Stage 2D, Gate 3b).

For every scenario of the known-truth process (`synth_process`) and R independent histories (260 expiries, ~7,700 rows, the shape of the real sample) the study applies the REAL Gate 1/2
code and records
  * coverage of the population value of each estimand by 95% percentile intervals from: row-iid resampling (ignores clustering), date-block (the Stage 2C method, b = 10), expiry-iid,
    expiry moving-block b = 2, 5, 8;
  * the dev-vs-holdout contrast (Gate 2 `split_contrast`): coverage of the true difference and the rejection / power rate under a TRUE NULL and a TRUE SHIFT, for b = 2, 5, 8;
  * the bootstrap Wald heterogeneity test: false-rejection rate on placebo strata (assigned per row and per date) and power for a planted stratum effect.
Truth = the estimands evaluated on a very long simulation of the same scenario (`synth_process.truth_estimates`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import clusters as C, contrasts as K, estimands as E, strata as ST, synth_process as SP

BLOCKS = (2, 5, 8)
EST_ALL = E.EST_NAMES
ROW_LEVEL_OK = [i for i, n in enumerate(EST_ALL) if i not in (1, 3, 5, 7)]
METHODS = ("row_iid", "date_block10", "expiry_iid", "expiry_block2", "expiry_block5", "expiry_block8")
HEADLINE = ("S1_session_all_obs_median", "S1_calendar_all_obs_median", "R1_geometric_mean_all_obs", "R2_median_ratio_all_obs", "R3_ratio_of_summed_total_variance")


@dataclass(frozen=True)
class Scenario:
    name: str
    dgp: SP.DGP
    description: str
    contrast: Tuple[str, ...] = ()                       # subset of ("null", "shift")
    wald: bool = False
    shift_factor: float = 1.15
    n_expiries: int = 260
    n_dev: int = 170


def scenarios() -> List[Scenario]:
    base = SP.DGP()
    return [
        Scenario("B0_calibrated", base, "calibrated process: regime shifts, jumps, heavy tails, persistent mispricing, overlapping weekly windows", ("null", "shift"), True),
        Scenario("B1_strong_dependence", replace(base, stay=0.97, eps=0.005, phi_m=0.93), "much more persistent vol regimes and mispricing (stress test of the dependence handling)", ("null",)),
        Scenario("B2_no_overlap_iid", replace(base, max_dte_days=5.5, eps=1.0, phi_m=0.0), "i.i.d. vol states, no mispricing persistence, windows <= 5.5 days: weekly expiries do not overlap", ()),
        Scenario("B4_overlap_only", replace(base, eps=1.0, phi_m=0.0), "i.i.d. vol states and no mispricing persistence with <= 14-day windows: cross-expiry dependence arises ONLY from the overlapping windows", ()),
        Scenario("B3_extreme_tails", replace(base, p_jump=0.06, jump_mean=12.0, sigma_z=0.55, sigma_zo=0.9, levels=(7.5, 9.5, 12.0, 16.0, 24.0, 40.0)), "frequent large jumps and fatter-tailed variance noise", ()),
    ]


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def _ci(est: E.Estimands, scheme: str, block: Optional[int], reps: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = C.bootstrap(est.cs, est.evaluate, scheme, block, reps, seed, k=len(EST_ALL))
    lo, hi = C.percentile_ci(d)
    return est.point(), lo, hi


def history_cis(a: pd.DataFrame, reps: int, seed: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    out["row_iid"] = _ci(E.build(a, row_level=True, min_clusters=30), "iid", None, reps, seed)
    out["date_block10"] = _ci(E.build(a, cluster_col="day"), "moving_block", 10, reps, seed + 1)
    est = E.build(a)
    out["expiry_iid"] = _ci(est, "iid", None, reps, seed + 2)
    for j, b in enumerate(BLOCKS):
        out[f"expiry_block{b}"] = _ci(est, "moving_block", b, reps, seed + 3 + j)
    return out


def run_history(args) -> dict:
    sc, h, reps, wald_reps, seed0 = args
    rows = SP.simulate(sc.dgp, sc.n_expiries, seed0 + h)
    res: dict = {"h": h}
    a = SP.to_aligned(rows, sc.dgp.c)
    res["cis"] = history_cis(a, reps, seed0 * 7 + h * 101)
    from . import dependence as D
    per = E.build(a).per_cluster_table()
    x = per.median_spread_calendar.to_numpy(float)
    x = x[np.isfinite(x)]
    res["icc"] = D.icc_oneway(a.spread_calendar.to_numpy(float), a.expiry.to_numpy())["icc"]
    res["acf1"] = float(D.acf(x, 1)[1])
    res["n_rows"] = len(a)
    split = SP.assign_split(rows, sc.n_dev)
    contr = {}
    for kind in sc.contrast:
        c_row = np.where(split == "dev", sc.dgp.c, sc.dgp.c * (sc.shift_factor if kind == "shift" else 1.0))
        al = SP.to_aligned(rows, c_row)
        dev, hold = al[split == "dev"].reset_index(drop=True), al[split == "holdout"].reset_index(drop=True)
        for b in BLOCKS:
            con = K.split_contrast(dev, hold, b, reps, seed0 * 11 + h * 13 + b)
            contr[(kind, b)] = con[["estimand", "difference_holdout_minus_dev", "ci_lo", "ci_hi", "p_two_sided_bootstrap"]].to_numpy()
    res["contrast"] = contr
    if sc.wald:
        w = {}
        rng = np.random.default_rng(seed0 * 17 + h)
        a2 = a.copy()
        a2["lv_row"] = rng.integers(0, 3, len(a2))
        day_level = {d: int(rng.integers(0, 3)) for d in a2.day.unique()}
        a2["lv_date"] = a2.day.map(day_level)
        cc = np.where(a2.dte_bucket == "1-3d", sc.dgp.c * 1.25, sc.dgp.c)
        a3 = SP.to_aligned(rows, cc)
        a3["lv_dte"] = a3.dte_bucket
        for name, frame_, col, lvls in (("placebo_by_row", a2, "lv_row", [0, 1, 2]), ("placebo_by_date", a2, "lv_date", [0, 1, 2]), ("planted_dte_effect", a3, "lv_dte", ["1-3d", "3-7d", "7-14d"])):
            al = ST.Aligned(frame_)
            masks = [(al.df[col] == l).to_numpy() for l in lvls]
            tab, draws, views = ST.strata_table(al, name, lvls, masks, 5, wald_reps, seed0 * 19 + h)
            het = ST.heterogeneity_table(name, lvls, draws, views, stat_names=("S1_calendar", "R1_geometric_mean"))
            w[name] = {r.statistic: r.p_value for r in het.itertuples()}
        res["wald"] = w
    return res


def run_scenario(sc: Scenario, n_hist: int, reps: int, wald_reps: int, seed0: int, workers: int = 4) -> List[dict]:
    jobs = [(sc, h, reps, wald_reps, seed0) for h in range(n_hist)]
    if workers <= 1:
        return [run_history(j) for j in jobs]
    with Pool(workers) as pool:
        return list(pool.imap(run_history, jobs, chunksize=2))


# ------------------------------------------------------------------------------ aggregation
def coverage_rows(sc: Scenario, results: List[dict], truth: Dict[str, float]) -> pd.DataFrame:
    rows = []
    n = len(results)
    for m in METHODS:
        est = np.array([r["cis"][m][0] for r in results])
        lo = np.array([r["cis"][m][1] for r in results])
        hi = np.array([r["cis"][m][2] for r in results])
        for i, name in enumerate(EST_ALL):
            t = truth[name]
            if m in ("row_iid", "date_block10") and i in (1, 3, 5, 7):
                continue
            ok = np.isfinite(lo[:, i]) & np.isfinite(hi[:, i])
            k = int(np.sum((lo[ok, i] <= t) & (t <= hi[ok, i])))
            nn = int(ok.sum())
            w = wilson(k, nn)
            rows.append(dict(scenario=sc.name, method=m, estimand=name, truth=t, coverage=k / nn, wilson_lo=w[0], wilson_hi=w[1], n_histories=nn, mean_width=float(np.mean(hi[ok, i] - lo[ok, i])),
                             mean_estimate=float(np.mean(est[ok, i])), bias=float(np.mean(est[ok, i]) - t), in_target_band_93_97=bool(0.93 <= k / nn <= 0.97),
                             consistent_with_band=bool(w[1] >= 0.93 and w[0] <= 0.97), undercovers=bool(w[1] < 0.93)))
    return pd.DataFrame(rows)


def contrast_rows(sc: Scenario, results: List[dict], truth_diff: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for kind in sc.contrast:
        for b in BLOCKS:
            arr = np.array([r["contrast"][(kind, b)] for r in results], dtype=object)
            n = len(results)
            for i, name in enumerate(EST_ALL):
                diff = np.array([float(arr[h][i][1]) for h in range(n)])
                lo = np.array([float(arr[h][i][2]) for h in range(n)])
                hi = np.array([float(arr[h][i][3]) for h in range(n)])
                p = np.array([float(arr[h][i][4]) for h in range(n)])
                t = truth_diff[name]
                ok = np.isfinite(lo) & np.isfinite(hi)
                kk = int(np.sum((lo[ok] <= t) & (t <= hi[ok])))
                w = wilson(kk, int(ok.sum()))
                rej = int(np.sum(p[ok] < 0.05))
                rw = wilson(rej, int(ok.sum()))
                rows.append(dict(scenario=sc.name, truth_kind={"null": "true_null", "shift": "true_shift"}[kind], block=b, estimand=name, true_difference=t, coverage_of_true_difference=kk / ok.sum(), coverage_wilson_lo=w[0], coverage_wilson_hi=w[1],
                                 coverage_in_93_97=bool(0.93 <= kk / ok.sum() <= 0.97), rejection_rate_p_lt_005=rej / ok.sum(), rejection_wilson_lo=rw[0], rejection_wilson_hi=rw[1],
                                 ci_excludes_zero_rate=float(np.mean((lo[ok] > 0) | (hi[ok] < 0))), mean_width=float(np.mean(hi[ok] - lo[ok])), mean_estimated_difference=float(np.mean(diff[ok])), n_histories=int(ok.sum())))
    return pd.DataFrame(rows)


def wald_rows(sc: Scenario, results: List[dict]) -> pd.DataFrame:
    rows = []
    if not sc.wald:
        return pd.DataFrame(rows)
    for name in ("placebo_by_row", "placebo_by_date", "planted_dte_effect"):
        for stat in ("S1_calendar", "R1_geometric_mean"):
            p = np.array([r["wald"][name][stat] for r in results], float)
            n = int(np.isfinite(p).sum())
            rej = int(np.sum(p[np.isfinite(p)] < 0.05))
            w = wilson(rej, n)
            rows.append(dict(scenario=sc.name, test=name, statistic=stat, rejection_rate_alpha_005=rej / n, wilson_lo=w[0], wilson_hi=w[1], n_histories=n,
                             truth="null (placebo)" if name.startswith("placebo") else "planted c x 1.25 in the 1-3 d stratum",
                             near_5pct=bool(w[0] <= 0.05 <= w[1]) if name.startswith("placebo") else np.nan))
    return pd.DataFrame(rows)


def truth_difference(dgp: SP.DGP, shift: float, n_expiries: int = 12000, seed: int = 987654) -> Dict[str, float]:
    """True holdout - dev difference of each estimand when implied variance is scaled by `shift` in the holdout (c x shift): population values from a long simulation."""
    rows = SP.simulate(dgp, n_expiries, seed)
    a0 = E.build(SP.to_aligned(rows, dgp.c), min_clusters=1).point()
    a1 = E.build(SP.to_aligned(rows, dgp.c * shift), min_clusters=1).point()
    return dict(zip(EST_ALL, a1 - a0))


# ------------------------------------------------------------------------------ raw per-history records (so every aggregate can be recomputed independently)
HIST_ESTIMANDS = ("S1_calendar_all_obs_median", "R1_geometric_mean_all_obs")


def history_interval_rows(sc: Scenario, results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for m in METHODS:
            est, lo, hi = r["cis"][m]
            for n in HIST_ESTIMANDS:
                i = EST_ALL.index(n)
                rows.append(dict(scenario=sc.name, history=r["h"], method=m, estimand=n, estimate=est[i], ci_lo=lo[i], ci_hi=hi[i]))
    return pd.DataFrame(rows)


def history_contrast_rows(sc: Scenario, results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for (kind, b), arr in r["contrast"].items():
            for n in HIST_ESTIMANDS:
                i = EST_ALL.index(n)
                rows.append(dict(scenario=sc.name, history=r["h"], truth_kind={"null": "true_null", "shift": "true_shift"}[kind], block=b, estimand=n, difference=float(arr[i][1]), ci_lo=float(arr[i][2]),
                                 ci_hi=float(arr[i][3]), p_value=float(arr[i][4])))
    return pd.DataFrame(rows)


def history_wald_rows(sc: Scenario, results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for name, d in r.get("wald", {}).items():
            for stat, p in d.items():
                rows.append(dict(scenario=sc.name, history=r["h"], test=name, statistic=stat, p_value=p))
    return pd.DataFrame(rows)
