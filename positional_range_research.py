"""positional_range_research.py -- how often does NIFTY expire inside the
09:20 volume-leader range? Research for a positional version of the method.

For every weekly expiry E (previous expiry P) and every trading day D in
(P, E] -- Day 1 = first day after P, ..., last day = expiry day:
  range at D 09:20  from D's first 5-min candle: CE/PE volume leaders
                    (100-pt strikes, cumulative volume 09:15-09:19),
                    UPPER = CE leader + PE leader's first-candle high,
                    LOWER = PE leader - CE leader's first-candle high
  expiry close      NIFTY's last 1-min close on E (~ the settlement price)
  inside?           LOWER <= close <= UPPER; else which side and how far
  touched?          did NIFTY trade above UPPER / below LOWER between D 09:20
                    and the expiry close
  hold to expiry    sell the 50-pt strikes nearest UPPER (CE) / LOWER (PE) at
                    D 09:20's next-bar open, no stop, no adjustment, settle at
                    intrinsic vs the expiry close (points per lot, 0.5 slippage)

Output: results/positional_range_research/rows.csv + report.html
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import paths
import simulator as S
from intraday_range_bt import Day, _nearest

OUT = os.path.join(paths.BASE_DIR, "results", "positional_range_research")
SLIP = 0.5


def run(verbose=True):
    man = S._manifest()
    exps = sorted(man)
    sp = S._load_spot()
    st, sc, sh, sl = (sp[c].to_numpy() for c in ("ts", "close", "high", "low"))
    rows = []
    for n, exp in enumerate(exps):
        if (man[exp] or {}).get("status") != "done":
            continue
        prev = exps[n - 1] if n else None
        lo_day = pd.Timestamp(prev).date() if prev else S._week_of(exp)[0]
        df = S._load_options(exp)
        mins = df["date"].dt.hour * 60 + df["date"].dt.minute
        df = df[mins <= 15 * 60 + 30]
        days = [d for d in sorted(df["day"].unique()) if lo_day < d <= pd.Timestamp(exp).date()]
        if not days:
            continue
        # expiry close = NIFTY's last print on expiry day
        e_end = int(pd.Timestamp(f"{exp} 23:59", tz=S.IST).timestamp())
        e_start = int(pd.Timestamp(f"{exp} 09:00", tz=S.IST).timestamp())
        j = np.searchsorted(st, e_end, side="right") - 1
        if j < 0 or st[j] < e_start:
            continue
        exp_close = float(sc[j])
        for k, day in enumerate(days, 1):
            D = Day(df[df["day"] == day], 100, 5, st, sc)
            if D.n < 30 or D.hhmm[0] != "09:15":
                continue
            ld = D.leaders(4)
            R = D.rng(ld) if ld else None
            if not R:
                continue
            up, lo = R["upper"], R["lower"]
            t920 = int(pd.Timestamp(f"{day} 09:20", tz=S.IST).timestamp())
            a, b = np.searchsorted(st, t920), j + 1
            hi_after, lo_after = float(sh[a:b].max()), float(sl[a:b].min())
            spot920 = float(D.spot[4]) if np.isfinite(D.spot[4]) else None
            ce_k, pe_k = _nearest(up, 50, "CE"), _nearest(lo, 50, "PE")
            ce_p, pe_p = D.price("CE", ce_k, 5), D.price("PE", pe_k, 5)
            hold = None
            if ce_p is not None and pe_p is not None:
                hold = (ce_p - SLIP + pe_p - SLIP) - max(0.0, exp_close - ce_k) - max(0.0, pe_k - exp_close)
            side = "inside" if lo <= exp_close <= up else ("above" if exp_close > up else "below")
            rows.append({
                "expiry": exp, "expiry_wd": pd.Timestamp(exp).strftime("%a"), "day": day.isoformat(),
                "day_no": k, "days_in_week": len(days), "dte": len(days) - k,
                "ce_leader": R["ce_leader"], "pe_leader": R["pe_leader"],
                "upper": up, "lower": lo, "width": round(up - lo, 2), "spot_920": spot920,
                "exp_close": exp_close, "side": side,
                "outside_by": round(exp_close - up if side == "above" else lo - exp_close if side == "below" else 0.0, 2),
                "touched_up": hi_after > up, "touched_down": lo_after < lo,
                "max_above": round(hi_after - up, 2), "max_below": round(lo - lo_after, 2),
                "ce_strike": ce_k, "pe_strike": pe_k, "ce_prem": ce_p, "pe_prem": pe_p,
                "hold_pts": None if hold is None else round(hold, 2),
            })
        if verbose:
            print(f"\r{exp}: {len(rows)} rows", end="", flush=True)
    if verbose:
        print()
    d = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "rows.csv"), index=False)
    return d


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    d = run()
    d1 = d[d.day_no == 1]
    print(len(d), "rows;", len(d1), "Day-1 ranges")
    print("Day-1 expiry side %:", (d1.side.value_counts(normalize=True) * 100).round(1).to_dict())
