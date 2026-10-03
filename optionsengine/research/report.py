"""Coverage / exclusion / reliability reporting for Stage 2A outputs.

    python -m optionsengine.research.report --dir research_output/stage2a/full \
        --sens research_output/stage2a/sens_3bps:3,research_output/stage2a/sens_8bps:8

All numbers are computed from the CSVs written by `build_surface`; nothing is
re-priced here. Development vs holdout is the `split` column (decided by date
before looking at any result). Regime cut-offs are computed on DEVELOPMENT
data only and then applied unchanged to the holdout.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

M_BINS = [-1, 0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 10]
M_LABELS = ["<0.5%", "0.5-1%", "1-2%", "2-4%", "4-7%", "7-12%", ">12%"]
T_BINS = [-1, 1.0, 3.0, 7.0, 14.0, 30.0, 1e9]
T_LABELS = ["<=1d", "1-3d", "3-7d", "7-14d", "14-30d", ">30d"]


def load(directory: str):
    s = pd.read_csv(os.path.join(directory, "smiles.csv"))
    p = pd.read_csv(os.path.join(directory, "points.csv"))
    return s, p


def add_buckets(df: pd.DataFrame, t_col="T_days", m_col: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    df["T_bucket"] = pd.cut(df[t_col], T_BINS, labels=T_LABELS)
    if m_col:
        df["m_bucket"] = pd.cut(df[m_col].abs(), M_BINS, labels=M_LABELS)
    return df


def snapshot_inventory(s: pd.DataFrame) -> pd.DataFrame:
    """Every attempted (snapshot, expiry): forward status or the reason no smile exists."""
    d = s.copy()
    d["outcome"] = d["forward_status"].fillna(d["group_reason"]).fillna("unknown")
    t = d.pivot_table(index="outcome", columns="split", values="smile_id", aggfunc="count", fill_value=0)
    t["all"] = t.sum(axis=1)
    return t


def forward_status_rates(s: pd.DataFrame) -> pd.DataFrame:
    d = s[s["forward_status"].notna()]
    out = d.groupby(["split", "forward_status"]).size().unstack(fill_value=0)
    out["total"] = out.sum(axis=1)
    for c in ("ok", "low_confidence", "unavailable"):
        if c in out:
            out[c + "_share"] = (out[c] / out["total"]).round(4)
    return out


def forward_by_tenor(s: pd.DataFrame) -> pd.DataFrame:
    """Forward-gate outcome by time to expiry (the gate limit is in bps of spot, independent of tenor)."""
    d = add_buckets(s[s["forward_status"].notna()])
    t = d.groupby(["T_bucket", "forward_status"], observed=True).size().unstack(fill_value=0)
    t["total"] = t.sum(axis=1)
    for c in ("ok", "low_confidence", "unavailable"):
        if c in t:
            t[c + "_share"] = (t[c] / t["total"]).round(3)
    return t


def snapshots_by_year(s: pd.DataFrame) -> pd.DataFrame:
    d = s.assign(year=s["day"].str[:4], primary=(s["forward_status"] == "ok"), has_atm=s["atm_iv"].notna())
    return d.groupby(["year", "split"]).agg(snapshots=("smile_id", "size"), primary=("primary", "sum"), with_atm=("has_atm", "sum"),
                                            expiries=("expiry", "nunique"), days=("day", "nunique"))


def point_exclusions(s: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Exclusion accounting. ITM-side contracts are counted from the smile table (they are not points)."""
    rows = []
    for split, g in p.groupby("split"):
        n = len(g)
        row = {"split": split, "otm_points": n, "used": int(g["used"].sum())}
        for k, v in g["exclusion"].dropna().value_counts().items():
            row["excl:" + ("iv_failed" if str(k).startswith("iv_failed:") else k)] = row.get(
                "excl:" + ("iv_failed" if str(k).startswith("iv_failed:") else k), 0) + int(v)
        sm = s[s["split"] == split]
        row["itm_side_not_used"] = int(sm["n_itm_side_excluded"].fillna(0).sum())
        row["strikes_without_otm_quote"] = int(sm["n_strikes_without_otm_quote"].fillna(0).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("split").fillna(0).astype(int)


def iv_convergence_and_reliability(p: pd.DataFrame, primary_only: bool = True) -> pd.DataFrame:
    """Among FRESH OTM points (an IV was attempted): convergence, reliability and resolution-limited rates."""
    d = p[p["iv_status"].notna()].copy()
    if primary_only:
        d = d[d["forward_status"] == "ok"]
    d["converged"] = d["iv_status"] == "converged"
    d["reliable"] = d["reliable"].astype(bool)
    out = d.groupby("split").agg(attempted=("converged", "size"), converged=("converged", "sum"),
                                 reliable=("reliable", "sum"), resolution_limited=("resolution_limited", "sum"),
                                 ill_conditioned=("ill_conditioned", "sum"))
    out["converged_rate"] = (out["converged"] / out["attempted"]).round(4)
    out["reliable_rate_of_attempted"] = (out["reliable"] / out["attempted"]).round(4)
    out["reslimited_rate_of_converged"] = (out["resolution_limited"] / out["converged"]).round(4)
    return out


def reliability_matrix(p: pd.DataFrame, split: Optional[str] = None, value: str = "reliable") -> pd.DataFrame:
    """Share of attempted fresh OTM points (OK forwards) that are `value`, by |ln K/F| x time-to-expiry."""
    d = p[(p["iv_status"].notna()) & (p["forward_status"] == "ok")]
    if split:
        d = d[d["split"] == split]
    d = add_buckets(d, m_col="log_moneyness").copy()
    d["flag"] = d[value].astype(float) if value != "converged" else (d["iv_status"] == "converged").astype(float)
    return d.pivot_table(index="m_bucket", columns="T_bucket", values="flag", aggfunc="mean", observed=False).round(3)


def smile_coverage(s: pd.DataFrame) -> pd.DataFrame:
    """Among PRIMARY smiles (forward OK): availability of ATM IV and of 25-delta RR/BF (T > 1 day)."""
    d = add_buckets(s[s["forward_status"] == "ok"])
    d["has_atm"] = d["atm_iv"].notna()
    d["has_atm_loose"] = d["atm_loose_iv"].notna()
    d["rr_eligible"] = d["T_days"] > 1.0
    d["has_rr"] = d["rr25"].notna()
    g = d.groupby(["split", "T_bucket"], observed=True)
    out = g.agg(smiles=("smile_id", "size"), atm=("has_atm", "sum"), atm_loose=("has_atm_loose", "sum"),
                rr_eligible=("rr_eligible", "sum"), rr_bf=("has_rr", "sum"))
    out["atm_share"] = (out["atm"] / out["smiles"]).round(3)
    out["atm_loose_share"] = (out["atm_loose"] / out["smiles"]).round(3)
    out["rr_bf_share_of_eligible"] = (out["rr_bf"] / out["rr_eligible"].replace(0, np.nan)).round(3)
    return out


def expiry_day_flags(p: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Resolution-limited share of CONVERGED OTM points, expiry day vs other days, by snapshot time."""
    d = p[(p["iv_status"] == "converged") & (p["forward_status"] == "ok")].merge(
        s[["smile_id", "expiry_day"]], on="smile_id", how="left")
    out = d.groupby(["expiry_day", "time"]).agg(converged=("iv", "size"), resolution_limited=("resolution_limited", "sum"),
                                                reliable=("reliable", "sum"))
    out["resolution_limited_share"] = (out["resolution_limited"] / out["converged"]).round(4)
    out["reliable_share"] = (out["reliable"] / out["converged"]).round(4)
    return out


def regime_cutoffs(s: pd.DataFrame, lo_q: float = 1 / 3, hi_q: float = 2 / 3) -> Dict[str, float]:
    """Tercile cut-offs of ATM IV measured on DEVELOPMENT smiles only (3-30 day expiries, OK forwards)."""
    dev = s[(s["split"] == "dev") & (s["forward_status"] == "ok") & s["atm_iv"].notna() & s["T_days"].between(3, 30)]
    if dev.empty:
        raise ValueError("no development ATM IVs to define regimes")
    return {"low_high": float(dev["atm_iv"].quantile(lo_q)), "mid_high": float(dev["atm_iv"].quantile(hi_q))}


def regime_table(s: pd.DataFrame, cuts: Dict[str, float]) -> pd.DataFrame:
    d = s[(s["forward_status"] == "ok") & s["T_days"].between(3, 30)].copy()
    d["regime"] = pd.cut(d["atm_iv"], [-np.inf, cuts["low_high"], cuts["mid_high"], np.inf],
                         labels=["low-vol", "mid-vol", "high-vol"])
    d["has_atm"] = d["atm_iv"].notna()
    # regimes can only be assigned where an ATM IV exists (smiles without one are reported in smile_coverage)
    reg = d[d["has_atm"]].groupby(["split", "regime"], observed=True).agg(
        smiles=("smile_id", "size"), median_atm_iv=("atm_iv", "median"), rr_bf_available=("rr25", lambda x: x.notna().mean()),
        median_rr25=("rr25", "median"), median_bf25=("bf25", "median"))
    return reg.round(4)


def sensitivity(default_dir: str, others: Sequence[tuple], default_label: float = 5.0,
                label: str = "max_dispersion_bps") -> pd.DataFrame:
    """Compare runs that differ in ONE parameter (forward-gate bps by default). `others` = [(directory, value)].
    Shows gate outcome rates and, for smiles that are primary in BOTH runs, how much the ATM IV moves."""
    base_s, _ = load(default_dir)
    base = base_s.set_index("smile_id")
    rows = []
    for d, bps in [(default_dir, default_label)] + list(others):
        s, _ = load(d)
        s = s.set_index("smile_id")
        f = s[s["forward_status"].notna()]
        both = s.join(base[["atm_iv", "forward_status"]], rsuffix="_default", how="inner")
        both = both[(both["forward_status"] == "ok") & (both["forward_status_default"] == "ok") &
                    both["atm_iv"].notna() & both["atm_iv_default"].notna()]
        diff = (both["atm_iv"] - both["atm_iv_default"]).abs() * 100
        ok_rows = s[s["forward_status"] == "ok"]
        rows.append(dict(**{label: bps}, groups=len(f), ok_share=round((f["forward_status"] == "ok").mean(), 4),
                         low_confidence_share=round((f["forward_status"] == "low_confidence").mean(), 4),
                         unavailable_share=round((f["forward_status"] == "unavailable").mean(), 4),
                         atm_available_share_of_ok=round(ok_rows["atm_iv"].notna().mean(), 4),
                         common_ok_atm=len(both), atm_iv_abs_change_median_volpts=round(float(diff.median()), 4) if len(diff) else None,
                         atm_iv_abs_change_p99_volpts=round(float(diff.quantile(0.99)), 4) if len(diff) else None))
    return pd.DataFrame(rows).set_index(label)


def build_report(directory: str, sens: Sequence[tuple] = ()) -> Dict[str, pd.DataFrame]:
    s, p = load(directory)
    cuts = regime_cutoffs(s)
    tables = {
        "snapshot_inventory": snapshot_inventory(s),
        "forward_status": forward_status_rates(s),
        "forward_by_tenor": forward_by_tenor(s),
        "snapshots_by_year": snapshots_by_year(s),
        "point_exclusions": point_exclusions(s, p),
        "iv_convergence_reliability": iv_convergence_and_reliability(p),
        "reliable_matrix_dev": reliability_matrix(p, "dev"),
        "reliable_matrix_holdout": reliability_matrix(p, "holdout"),
        "converged_matrix_all": reliability_matrix(p, None, "converged"),
        "smile_coverage": smile_coverage(s),
        "expiry_day_flags": expiry_day_flags(p, s),
        "regimes": regime_table(s, cuts),
    }
    if sens:
        tables["forward_threshold_sensitivity"] = sensitivity(directory, sens)
    out = os.path.join(directory, "report")
    os.makedirs(out, exist_ok=True)
    for k, v in tables.items():
        v.to_csv(os.path.join(out, k + ".csv"))
    with open(os.path.join(out, "regime_cutoffs.json"), "w") as f:
        json.dump(cuts, f, indent=1)
    return tables


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sens", default="", help="dir:bps,dir:bps")
    a = ap.parse_args(argv)
    sens = [(x.split(":")[0], float(x.split(":")[1])) for x in a.sens.split(",") if x]
    pd.set_option("display.width", 220)
    for k, v in build_report(a.dir, sens).items():
        print(f"\n## {k}\n{v.to_string()}")


if __name__ == "__main__":
    main()
