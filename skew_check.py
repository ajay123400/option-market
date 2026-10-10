"""skew_check.py -- live IV-rule readings vs the evening history, side by side.

The plan's live 09:30 reading (sell_rules.status: the chain's last-traded
prices at the moment of the check) and the evening history (sell_rules.
compute_expiry: the 1-min closes of the 09:30 bar) are meant to measure the
same thing. On 9 Oct 2026 the 25-delta put skew came out 1.10 live vs 1.43 in
the history -- enough to flip "steep". This script lines them up for every
day in results/iv_rule/live_log.jsonl and recomputes the history method at
two moments, to separate a TIMING difference from a METHOD difference:

  hist_0930   the stored history value (bars up to and including 09:30)
  hist_live   the same method on the bars completed when the live reading
              was taken (the minute before its timestamp)

Output: printed table + results/iv_rule/live_vs_history.csv
Usage:  python skew_check.py            (add --seed to copy today's today.json reading into the log once)
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import paths
import sell_rules as SR
import simulator as S

OUT = os.path.join(SR.DIR, "live_vs_history.csv")


def seed_from_today():
    """Copy today.json's readings into the log (no strike detail) if that day is not logged yet."""
    tj = SR._load_today()
    if not tj.get("readings"):
        return 0
    have = set()
    if os.path.exists(SR.LIVE_LOG):
        for line in open(SR.LIVE_LOG, encoding="utf-8"):
            r = json.loads(line)
            have.add((r["day"], r["slot"]))
    n = 0
    with open(SR.LIVE_LOG, "a", encoding="utf-8") as f:
        for slot, r in sorted(tj["readings"].items()):
            if (tj["day"], slot) in have:
                continue
            f.write(json.dumps({"day": tj["day"], "slot": slot, "at": (r.get("at") or slot) + ":00", "expiry": tj.get("expiry"),
                                "atm_iv": r.get("atm_iv"), "skew25": r.get("skew25"), "detail": None, "source": "seeded from today.json"}) + "\n")
            n += 1
    return n


def _hist_at(exp, day, t_unix):
    """sell_rules' history method on the bars with ts <= t_unix (same filters as compute_expiry)."""
    df = S._load_options(exp)
    dd = df[(df["day"] == pd.Timestamp(day).date()) & (df.ts <= t_unix)]
    if dd.empty:
        return None
    last = dd.sort_values("ts").groupby(["strike", "type"]).close.last()
    sp = S._load_spot()
    j = np.searchsorted(sp["ts"].to_numpy(), t_unix, side="right") - 1
    spot = float(sp["close"].to_numpy()[j])
    q = {(int(k), ty): float(v) for (k, ty), v in last.items() if abs(k - spot) <= 1600}
    _, iv, skew, info = SR.measure(q, spot, (SR._exp_close_ts(exp) - t_unix) / (365 * 86400), detail=True)
    return {"atm_iv": iv, "skew25": skew, "detail": info}


def _leg(d, side):
    x = (d or {}).get(side) or {}
    return f"{x.get('k')}@{round(float(x['px']), 2)}" if x.get("px") is not None else None


def compare():
    if not os.path.exists(SR.LIVE_LOG):
        return pd.DataFrame()
    live = [json.loads(l) for l in open(SR.LIVE_LOG, encoding="utf-8")]
    h = SR.history()
    h["day"] = h["day"].astype(str)
    rows = []
    for r in live:
        hr = h[(h.day == r["day"]) & (h.slot == r["slot"])]
        rec = {"day": r["day"], "slot": r["slot"], "live_at": r["at"], "source": r.get("source"),
               "live_iv": r.get("atm_iv"), "live_skew": r.get("skew25"),
               "hist_iv": float(hr.atm_iv.iloc[0]) if len(hr) else None,
               "hist_skew": float(hr.skew25.iloc[0]) if len(hr) and pd.notna(hr.skew25.iloc[0]) else None}
        exp = r.get("expiry")
        p = os.path.join(paths.BASE_DIR, "data", "hist1m", "options", f"{exp}.parquet")
        if exp and os.path.exists(p):
            at = datetime.strptime(f"{r['day']} {r['at'][:5]}", "%Y-%m-%d %H:%M")
            t_live = int(pd.Timestamp(at, tz=S.IST).timestamp()) - 60            # the last bar completed before the live reading
            x = _hist_at(exp, r["day"], t_live)
            if x:
                rec.update(hist_live_iv=x["atm_iv"], hist_live_skew=x["skew25"])
                d = x["detail"] or {}
                rec.update(hist_pe=_leg(d, "pe"), hist_ce=_leg(d, "ce"))
            t_slot = int(pd.Timestamp(f"{r['day']} {r['slot']}", tz=S.IST).timestamp())
            y = _hist_at(exp, r["day"], t_slot)
            if y:
                d = y["detail"] or {}
                rec.update(hist0930_pe=_leg(d, "pe"), hist0930_ce=_leg(d, "ce"))
        ld = r.get("detail") or {}
        if ld:
            rec.update(live_pe=_leg(ld, "pe"), live_ce=_leg(ld, "ce"))
        for a, b, name in (("live_skew", "hist_skew", "skew_gap"), ("live_iv", "hist_iv", "iv_gap")):
            if rec.get(a) is not None and rec.get(b) is not None:
                rec[name] = round(rec[a] - rec[b], 2)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    return out


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    if "--seed" in sys.argv:
        print("seeded", seed_from_today(), "readings from today.json")
    d = compare()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(d.to_string(index=False) if len(d) else "no live readings logged yet")
    if len(d) and d.skew_gap.notna().any():
        g = d.dropna(subset=["skew_gap"])
        print(f"\nskew gap live - history: mean {g.skew_gap.mean():+.2f}, mean |gap| {g.skew_gap.abs().mean():.2f} over {len(g)} readings")
    print(OUT)
