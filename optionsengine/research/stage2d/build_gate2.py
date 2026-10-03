"""Stage 2D Gate 2 builder (T5-T8). Writes ONLY under a `stage2d` directory; the design is frozen in PREREGISTRATION.md (its SHA-256 is recorded).

python -m optionsengine.research.stage2d.build_gate2 --out research_output/stage2d/gate2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import alt_rv, build_gate1 as g1, clusters as C, contrasts as K, estimands as E, spec_curve as SC, strata as S

SEED = 20260101
FAMILY_A = ["S1_calendar_all_obs_median", "S2_calendar_median_of_expiry_medians", "R1_geometric_mean_all_obs", "R2_median_ratio_all_obs"]
STRATA_DIMS = [("dte_bucket", ["1-3d", "3-7d", "7-14d"]), ("time", ["10:00", "13:00", "15:00"]), ("iv_quartile", ["Q1 low", "Q2", "Q3", "Q4 high"]),
               ("recent_rv_regime", ["low-vol", "mid-vol", "high-vol", "unknown"]), ("expiry_year", None), ("weekend_in_window", [False, True])]
HET_STATS = ("S1_calendar", "R1_geometric_mean")

LEDGER = [
    ("Dev/holdout split date 2025-01-01", "Stage 2A", "fixed before results were seen (Stage 2A report)", "no", "T5 (holdout is 'previously viewed')"),
    ("Risk-free rate 6.5% (assumption)", "Phase 1 / 2A", "repo constant", "no", "T7 rate 5.5% / 7.5%"),
    ("Forward-parity dispersion gate 5 bps", "Stage 2A", "approved default before results", "no", "T8 forward gate 3/8/12 bps"),
    ("Stale-quote age <= 5 minutes", "Stage 2A", "approved before results", "no", "T8 stale age 2/15 min"),
    ("Snapshot times 10:00 / 13:00 / 15:00", "Stage 2A", "specified before results", "no", "T8 drop one time at a time"),
    ("Strict ATM IV (interpolated between reliable bracketing strikes)", "Stage 2A", "specified before results; loose population identical (0 loose-only rows)", "no", "n/a (no difference)"),
    ("Primary universe <= 14 calendar DTE", "Stage 2C", "specified before results (strike-coverage bias)", "no", "T8 DTE <= 7 and <= 21"),
    ("RV measures: hybrid / intraday-only / close-to-close, hybrid as the interval-complete one", "Stage 2B/2C", "all three specified before results; hybrid singled out for the aligned comparison AFTER the session-basis results", "yes", "T7 intraday-only"),
    ("Session-basis annualization (252) in the original comparison", "Stage 2B/2C", "specified before results", "no", "T7 250/256"),
    ("Calendar-basis RV and the total-variance ratio", "Stage 2C audit", "introduced AFTER the session-basis results showed the time-basis mismatch", "yes", "T7 365.25; Gate 1 intervals"),
    ("Uniform weighting of the partial snapshot session", "Stage 2C", "convention; sensitivity shown in Stage 2C", "partly", "T7 profile weighting"),
    ("1-minute return sampling", "Stage 2B", "specified before results", "no", "T7 5/15 minutes"),
    ("Regime cut-offs from the development sample only", "Stage 2C", "specified before results", "no", "used unchanged in T5/T6"),
    ("Population for Stage 2D: EXP / hybrid", "Stage 2D proposal", "chosen AFTER seeing Stage 2C", "yes", "stated"),
    ("Block-bootstrap length 5 (largest rounded Politis-White)", "Gate 1", "rule fixed before intervals were computed", "no", "Gate 1 grid; T5/T6 use 5, sensitivity 2 and 8"),
    ("Minimum 30 clusters per interval", "Gate 1", "guard fixed in advance", "no", "applies to every stratum"),
    ("Pre-registered Gate 2 design (families, contrasts, specifications)", "Gate 2", "written before any Gate 2 code or number", "no", "PREREGISTRATION.md (SHA-256 in run metadata)"),
]


def sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _expiry_year(df: pd.DataFrame) -> pd.Series:
    return df.expiry.str[:4]


def run_t5(e: pd.DataFrame, out: str, reps: int) -> Dict[str, pd.DataFrame]:
    dev, hold = e[e.split == "dev"].reset_index(drop=True), e[e.split == "holdout"].reset_index(drop=True)
    con = K.split_contrast(dev, hold, 5, reps, SEED)
    fam = con.estimand.isin(FAMILY_A)
    con["family_A"] = fam
    con.loc[fam, "p_holm_family_A"] = K.holm(con.loc[fam, "p_two_sided_bootstrap"].to_numpy())
    con.to_csv(os.path.join(out, "t5_split_contrasts.csv"), index=False)
    ps, comp = K.post_stratified_contrast(dev, hold, "iv_quartile", 5, reps, SEED)
    ps.to_csv(os.path.join(out, "t5_post_stratified_contrast.csv"), index=False)
    comp.to_csv(os.path.join(out, "t5_iv_quartile_composition.csv"), index=False)
    K.within_level_contrasts(dev, hold, "iv_quartile", 5, max(1000, reps // 2), SEED).to_csv(os.path.join(out, "t5_within_iv_quartile_contrasts.csv"), index=False)
    st = K.straddling_expiries(e)
    pd.DataFrame(dict(expiry=st)).to_csv(os.path.join(out, "t5_straddling_expiries.csv"), index=False)
    led = pd.DataFrame(LEDGER, columns=["analysis_choice", "stage", "when_decided", "results_seen_when_decided", "alternative_evaluated_in"])
    led.to_csv(os.path.join(out, "forking_paths_ledger.csv"), index=False)
    return dict(contrast=con, n_straddling=len(st))


def run_block_sensitivity(e: pd.DataFrame, out: str, reps: int) -> pd.DataFrame:
    """Pre-registered sensitivity of the headline intervals and of the dev-vs-holdout contrasts to the block length (b = 2 and 8; the headline is b = 5)."""
    dev, hold = e[e.split == "dev"].reset_index(drop=True), e[e.split == "holdout"].reset_index(drop=True)
    rows = []
    for b in (2, 8):
        con = K.split_contrast(dev, hold, b, reps, SEED + 700 + b)
        for r in con.itertuples():
            rows.append(dict(block=b, analysis="dev_vs_holdout", estimand=r.estimand, estimate=r.difference_holdout_minus_dev, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                             p_two_sided_bootstrap=r.p_two_sided_bootstrap))
        cur = SC.estimate_spec(SC.Spec(SC.BASELINE, "baseline", "", e), b, reps, SEED + 800 + b)
        for r in cur.itertuples():
            rows.append(dict(block=b, analysis="pooled_baseline", estimand=r.estimand, estimate=r.estimate, ci_lo=r.ci_lo, ci_hi=r.ci_hi, p_two_sided_bootstrap=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "block_length_sensitivity.csv"), index=False)
    return df


def run_t6(e: pd.DataFrame, aligned_all: pd.DataFrame, out: str, reps: int) -> Dict[str, pd.DataFrame]:
    e = e.assign(expiry_year=_expiry_year(e))
    al = S.Aligned(e, extra=dict(iv=e.iv_pct.to_numpy(float), rv=e.rv_calendar_pct.to_numpy(float)))
    tabs, hets = [], []
    for j, (dim, levels) in enumerate(STRATA_DIMS):
        col = al.df[dim]
        lv = sorted(col.unique(), key=str) if levels is None else levels
        masks = [(col == l).to_numpy() for l in lv]
        t, draws, views = S.strata_table(al, dim, lv, masks, 5, reps, SEED + 100 + j)
        tabs.append(t)
        hets.append(S.heterogeneity_table(dim, lv, draws, views, exclude_levels=("unknown",)))
    tab = pd.concat(tabs, ignore_index=True)
    het = pd.concat(hets, ignore_index=True)
    het["family_B"] = True
    het["p_holm_family_B"] = K.holm(het.p_value.fillna(1.0).to_numpy())
    tab.to_csv(os.path.join(out, "t6_strata_estimates.csv"), index=False)
    het.to_csv(os.path.join(out, "t6_heterogeneity_tests.csv"), index=False)
    mz = []
    for name, d in (("all", e), ("dev", e[e.split == "dev"]), ("holdout", e[e.split == "holdout"])):
        for log in (False, True):
            mz.append(dict(sample=name, **S.mincer_zarnowitz(d, "iv_pct", "rv_calendar_pct", log, 5, max(1000, reps // 2), SEED + 300)))
    pd.DataFrame(mz).to_csv(os.path.join(out, "t6_mincer_zarnowitz.csv"), index=False)
    # expiry-day stratum through the F1 / F5 hybrid targets (horizon mismatch; no ratio statistics)
    rows = []
    for h in ("F1", "F5"):
        f = aligned_all[(aligned_all.horizon == h) & (aligned_all.measure == "hybrid")].reset_index(drop=True)
        if len(f) < 100:
            continue
        a = S.Aligned(f)
        masks = [(~a.df.expiry_day.astype(bool)).to_numpy(), a.df.expiry_day.astype(bool).to_numpy()]
        t, draws, views = S.strata_table(a, f"expiry_day_{h}", ["not expiry day", "expiry day"], masks, 5, reps, SEED + 400)
        t = t[t.statistic.isin(["S1_calendar", "S1_session"])]
        w = S.heterogeneity_table(f"expiry_day_{h}", ["not expiry day", "expiry day"], draws, views, stat_names=("S1_calendar",))
        for r in t.itertuples():
            rows.append(dict(horizon=h, **r._asdict()))
        rows.append(dict(horizon=h, dimension=f"expiry_day_{h}", level="difference test", statistic="S1_calendar", estimate=w.wald.iloc[0], ci_note=f"Wald={w.wald.iloc[0]:.3f}, p={w.p_value.iloc[0]:.4f}"))
    pd.DataFrame(rows).drop(columns=["Index"], errors="ignore").to_csv(os.path.join(out, "t6_expiry_day_stratum.csv"), index=False)
    return dict(strata=tab, het=het)


def default_variants(out: str, spot: str, participant_dir: str, root: str, stage2a_root: str = "research_output/stage2a") -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    vdir = os.path.join(out, "variants")
    os.makedirs(vdir, exist_ok=True)
    with open(os.path.join(vdir, ".gitignore"), "w") as f:
        f.write("*\n!.gitignore\n")
    variants, rates = {}, {}
    for name, src in (("fwd_3bps", "sens_3bps"), ("fwd_8bps", "sens_8bps"), ("fwd_12bps", "sens_12bps"), ("age_2min", "sens_age2"), ("age_15min", "sens_age15")):
        variants[name] = SC.exp_hybrid(SC.aligned_from_stage2a(os.path.join(stage2a_root, src), os.path.join(vdir, name), spot, participant_dir))
    variants["dte_le21"] = SC.exp_hybrid(SC.aligned_from_stage2a(os.path.join(stage2a_root, "full"), os.path.join(vdir, "dte_le21"), spot, participant_dir, dte_max=21))
    for r in ("5.5", "7.5"):
        d = SC.stage2a_rate_variant(root, os.path.join(vdir, f"rate_{r}", "stage2a"), float(r) / 100.0)
        rates[r] = SC.exp_hybrid(SC.aligned_from_stage2a(d, os.path.join(vdir, f"rate_{r}"), spot, participant_dir))
    return variants, rates


def build(aligned: str, stage2c: str, out: str, spot: str = "data/hist1m/NIFTY50_1m.parquet", participant_dir: str = "data/participant_oi", root: str = "data/hist1m",
          reps: int = 5000, grid_reps: int = 5000, variants_provider: Optional[Callable] = None, prereg: Optional[str] = None, use_alt_rv: bool = True) -> dict:
    if "stage2d" not in os.path.normpath(out).split(os.sep):
        raise ValueError("Stage 2D outputs must live under a 'stage2d' directory (existing research artifacts are never overwritten)")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    e = g1.load_baseline(aligned, stage2c)
    aligned_all = pd.read_csv(aligned)
    t5 = run_t5(e, out, reps)
    run_t6(e, aligned_all, out, reps)
    run_block_sensitivity(e, out, reps)
    W = alt_rv.WindowVariance(alt_rv.load_index(spot, participant_dir)) if use_alt_rv else None
    variants, rates = (variants_provider or (lambda: default_variants(out, spot, participant_dir, root)))()
    specs = SC.build_specs(aligned_all, W, variants, rates)
    curve = pd.concat([SC.estimate_spec(s, 5, grid_reps, SEED + 500) for s in specs], ignore_index=True)
    curve.to_csv(os.path.join(out, "t7_t8_specification_curve.csv"), index=False)
    SC.robustness_reading(curve).to_csv(os.path.join(out, "t7_t8_robustness_reading.csv"), index=False)
    meta = dict(git_commit=g1._git(), preregistration_sha256=sha256(prereg or os.path.join(out, "PREREGISTRATION.md")), n_rows=len(e), n_expiries=int(e.expiry.nunique()),
                reps=reps, grid_reps=grid_reps, base_seed=SEED, block=5, n_specs=len(specs), n_straddling_expiries=t5["n_straddling"], seconds=round(time.time() - t0, 1),
                started=datetime.now().isoformat(timespec="seconds"), numpy=np.__version__, pandas=pd.__version__)
    with open(os.path.join(out, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", default="research_output/stage2c/annualization_audit/aligned_observations.csv")
    ap.add_argument("--stage2c", default="research_output/stage2c")
    ap.add_argument("--out", default="research_output/stage2d/gate2")
    ap.add_argument("--reps", type=int, default=5000)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.aligned, a.stage2c, a.out, reps=a.reps, grid_reps=a.reps), indent=1))


if __name__ == "__main__":
    main()
