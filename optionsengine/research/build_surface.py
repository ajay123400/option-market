"""Stage 2A driver: historical expiry-wise IV surface -> CSV tables.

    python -m optionsengine.research.build_surface --root data/hist1m --out research_output/stage2a/full

Outputs (source data is only read):
  smiles.csv    one row per attempted (snapshot, expiry), INCLUDING groups with no smile (group_reason)
  points.csv    one row per OTM-side listed strike of every smile that could be built
  run_metadata.json   arguments, configuration, input inventory, row counts, git commit

Conventions are documented in optionsengine/research/README.md and surface.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time as _time
from dataclasses import asdict
from datetime import date, datetime, timedelta
from multiprocessing import Pool
from typing import Dict, List, Optional

import pandas as pd

from ..analytics import expiry_at_close
from ..forward import ForwardConfig
from ..surface import (IV_FAILED_PREFIX, SurfaceConfig, SmileResult, build_smile)
from . import loaders

DEFAULT_TIMES = ("10:00", "13:00", "15:00")
DEFAULT_SPLIT_DATE = "2025-01-01"      # development < split <= holdout
SPOT_START = date(2021, 9, 16)
EXCLUSION_COLUMNS = ("stale", "never_traded", "zero_price", "forward_low_confidence", "resolution_limited",
                     "ill_conditioned")

SMILE_COLUMNS = (
    "smile_id", "day", "time", "expiry", "split", "expiry_day", "spot", "T_days", "n_quotes", "group_reason",
    "forward_status", "primary", "forward_used", "fwd_n_pairs", "fwd_n_inliers", "fwd_dispersion_pts",
    "fwd_dispersion_limit_pts", "fwd_reason", "carry_yield", "n_points", "n_used", "n_iv_converged", "n_reliable",
    "n_itm_side_excluded", "n_strikes_without_otm_quote", "n_iv_failed", "atm_iv", "atm_k_low", "atm_k_high",
    "atm_reason", "atm_loose_iv", "atm_loose_unreliable_inputs", "rr25", "bf25", "iv25_call", "iv25_put",
    "rr_bf_reason", "n_stale", "n_never_traded", "n_zero_price", "n_forward_low_confidence",
    "n_resolution_limited", "n_ill_conditioned")
POINT_COLUMNS = (
    "smile_id", "day", "time", "expiry", "split", "forward_status", "T_days", "forward", "strike", "kind", "price",
    "age_min", "log_moneyness", "iv", "iv_status", "reliable", "resolution_limited", "ill_conditioned", "iv_low",
    "iv_high", "delta_fwd", "used", "exclusion")

_SPOT: Dict[int, float] = {}


def _init_worker(spot_path: str):
    global _SPOT
    _SPOT = loaders.load_spot_closes(spot_path)


def _git_commit() -> Optional[str]:
    """HEAD commit, suffixed '+dirty' if optionsengine/ has uncommitted changes (so outputs are traceable)."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "optionsengine"],
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        return None


def smile_row(day: date, hhmm: str, expiry: str, split: str, res: Optional[SmileResult], group_reason: Optional[str],
              spot: Optional[float], T_days: Optional[float], n_quotes: int) -> dict:
    row = dict(smile_id=f"{day.isoformat()}T{hhmm}_{expiry}", day=day.isoformat(), time=hhmm, expiry=expiry, split=split,
               expiry_day=(day.isoformat() == expiry), spot=spot, T_days=T_days, n_quotes=n_quotes, group_reason=group_reason)
    if res is None:
        return row
    fe = res.forward
    ec = res.exclusion_counts
    n_conv = sum(1 for p in res.points if p.iv is not None)
    row.update(
        T_days=res.T_days, forward_status=fe.status.value, primary=res.primary, forward_used=res.forward_used,
        fwd_n_pairs=fe.n_candidates, fwd_n_inliers=fe.n_inliers, fwd_dispersion_pts=fe.dispersion,
        fwd_dispersion_limit_pts=fe.dispersion_limit, fwd_reason=fe.reason, carry_yield=res.q,
        n_points=len(res.points), n_used=sum(p.used for p in res.points), n_iv_converged=n_conv,
        n_reliable=sum(p.reliable for p in res.points), n_itm_side_excluded=res.n_itm_side_excluded,
        n_strikes_without_otm_quote=res.n_strikes_without_otm_quote,
        n_iv_failed=sum(v for k, v in ec.items() if k.startswith(IV_FAILED_PREFIX)),
        atm_iv=res.atm.iv if res.atm else None, atm_k_low=res.atm.strike_low if res.atm else None,
        atm_k_high=res.atm.strike_high if res.atm else None, atm_reason=res.atm_reason,
        atm_loose_iv=res.atm_loose.iv if res.atm_loose else None,
        atm_loose_unreliable_inputs=res.atm_loose.n_unreliable_inputs if res.atm_loose else None,
        rr25=res.rr_bf.rr25 if res.rr_bf else None, bf25=res.rr_bf.bf25 if res.rr_bf else None,
        iv25_call=res.rr_bf.iv25_call if res.rr_bf else None, iv25_put=res.rr_bf.iv25_put if res.rr_bf else None,
        rr_bf_reason=res.rr_bf_reason)
    for c in EXCLUSION_COLUMNS:
        row["n_" + c] = ec.get(c, 0)
    return row


def process_expiry(job: dict):
    """All snapshots of one expiry file. Returns (smile_rows, point_rows)."""
    expiry, path = job["expiry"], job["path"]
    cfg: SurfaceConfig = job["config"]
    exp_date = date.fromisoformat(expiry)
    exp_dt = expiry_at_close(exp_date)
    split_date = date.fromisoformat(job["split_date"])
    df = loaders.load_expiry_frame(path)
    smiles, points = [], []
    for day in loaders.trading_days(df):
        if day > exp_date or (exp_date - day).days > job["max_dte"]:
            continue
        if job["start"] and day < job["start"] or job["end"] and day > job["end"]:
            continue
        split = "dev" if day < split_date else "holdout"
        for hhmm in job["times"]:
            t0 = loaders.snapshot_ts(day, hhmm)
            obs = datetime.fromtimestamp(t0 + loaders.BAR_SECONDS, tz=loaders.IST)
            T = max((exp_dt - obs).total_seconds(), 0.0) / (365 * 86400)
            if day < SPOT_START or t0 not in _SPOT:
                smiles.append(smile_row(day, hhmm, expiry, split, None, "no_spot_bar_at_snapshot", None, T * 365, 0))
                continue
            quotes = loaders.quotes_at(df, t0)
            if not quotes:
                smiles.append(smile_row(day, hhmm, expiry, split, None, "no_option_bars_at_snapshot", _SPOT[t0], T * 365, 0))
                continue
            res = build_smile(quotes, _SPOT[t0], T, job["r"], cfg)
            smiles.append(smile_row(day, hhmm, expiry, split, res, res.group_reason, _SPOT[t0], T * 365, len(quotes)))
            sid = smiles[-1]["smile_id"]
            for p in res.points:
                points.append(dict(smile_id=sid, day=day.isoformat(), time=hhmm, expiry=expiry, split=split,
                                   forward_status=res.forward.status.value, T_days=res.T_days, forward=res.forward_used,
                                   strike=p.strike, kind=p.kind.value, price=p.price, age_min=p.age_minutes,
                                   log_moneyness=p.log_moneyness, iv=p.iv, iv_status=p.iv_status, reliable=p.reliable,
                                   resolution_limited=p.resolution_limited, ill_conditioned=p.ill_conditioned,
                                   iv_low=p.iv_low, iv_high=p.iv_high, delta_fwd=p.delta_fwd, used=p.used,
                                   exclusion=p.exclusion))
    return smiles, points


def build(root: str, out: str, times=DEFAULT_TIMES, r: float = 0.065, max_dte: int = 45, workers: int = 4,
          expiries: Optional[List[str]] = None, sample: Optional[int] = None, seed: int = 7,
          start: Optional[date] = None, end: Optional[date] = None, forward_bps: Optional[float] = None,
          max_age_min: float = 5.0, split_date: str = DEFAULT_SPLIT_DATE) -> dict:
    os.makedirs(out, exist_ok=True)
    files = loaders.list_expiry_files(root)
    if expiries:
        files = [f for f in files if f[0] in set(expiries)]
    if sample and sample < len(files):
        random.Random(seed).shuffle(files)
        files = sorted(files[:sample])
    fcfg = ForwardConfig() if forward_bps is None else ForwardConfig(max_dispersion_bps=forward_bps)
    cfg = SurfaceConfig(max_age_minutes=max_age_min, forward=fcfg)
    jobs = [dict(expiry=e, path=p, config=cfg, r=r, times=tuple(times), max_dte=max_dte, start=start, end=end,
                 split_date=split_date) for e, p in files]
    spot_path = os.path.join(root, "NIFTY50_1m.parquet")
    t_start = _time.time()
    with Pool(workers, initializer=_init_worker, initargs=(spot_path,)) as pool:
        results = pool.map(process_expiry, jobs, chunksize=1)
    smiles = [row for res in results for row in res[0]]
    points = [row for res in results for row in res[1]]
    # fixed schemas: an empty result still writes a header (never a column-less file)
    pd.DataFrame(smiles).reindex(columns=list(SMILE_COLUMNS)).to_csv(os.path.join(out, "smiles.csv"), index=False)
    pd.DataFrame(points).reindex(columns=list(POINT_COLUMNS)).to_csv(os.path.join(out, "points.csv"), index=False)
    meta = dict(command=" ".join(sys.argv), git_commit=_git_commit(), started=datetime.fromtimestamp(t_start).isoformat(),
                seconds=round(_time.time() - t_start, 1), root=root, n_expiry_files=len(files),
                input_bytes=sum(os.path.getsize(p) for _, p in files), times=list(times), risk_free_rate=r,
                risk_free_rate_note="ASSUMPTION (repo constant 6.5%), not data", max_dte_days=max_dte,
                split_date=split_date, forward_max_dispersion_bps=fcfg.max_dispersion_bps,
                forward_default_bps=ForwardConfig().max_dispersion_bps, max_age_minutes=max_age_min,
                surface_config={k: v for k, v in asdict(cfg).items() if k not in ("forward", "solver")},
                pandas=pd.__version__, n_smile_rows=len(smiles), n_point_rows=len(points))
    with open(os.path.join(out, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/hist1m")
    ap.add_argument("--out", default="research_output/stage2a/full")
    ap.add_argument("--times", default=",".join(DEFAULT_TIMES))
    ap.add_argument("--rate", type=float, default=0.065, help="risk-free rate ASSUMPTION (default 6.5%%)")
    ap.add_argument("--max-dte", type=int, default=45)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--expiries", default=None, help="comma-separated expiry dates")
    ap.add_argument("--sample", type=int, default=None, help="random sample of N expiry files")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--forward-bps", type=float, default=None, help="SENSITIVITY ONLY: override the 5 bps forward gate")
    ap.add_argument("--max-age-min", type=float, default=5.0)
    ap.add_argument("--split-date", default=DEFAULT_SPLIT_DATE)
    a = ap.parse_args(argv)
    meta = build(a.root, a.out, tuple(a.times.split(",")), a.rate, a.max_dte, a.workers,
                 a.expiries.split(",") if a.expiries else None, a.sample, a.seed,
                 date.fromisoformat(a.start) if a.start else None, date.fromisoformat(a.end) if a.end else None,
                 a.forward_bps, a.max_age_min, a.split_date)
    print(json.dumps({k: meta[k] for k in ("n_expiry_files", "n_smile_rows", "n_point_rows", "seconds", "git_commit")}))


if __name__ == "__main__":
    main()
