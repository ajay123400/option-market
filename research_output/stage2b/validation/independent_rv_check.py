"""Three-way reconciliation of Stage 2B realized volatility on the REAL spot file.

Calculation A  pure-Python streaming loop (pyarrow only to read; math.log; no numpy/pandas; own chain logic)
Calculation B  pandas groupby + reindex-to-the-375-minute-grid + Series.diff style logic (own code, no module imports)
Module         optionsengine.research.* output files (research_output/stage2b/*.csv)

None of A or B imports optionsengine. They reuse only the written DEFINITIONS (not code). A treats a session as
regular if it has >= 150 bars in 09:15-15:29 on a weekday (its own rule, no calendar), B uses the same rule via pandas.
Usage: python research_output/stage2b/validation/independent_rv_check.py [out_dir]
"""
import math
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2b"
SPOT = "data/hist1m/NIFTY50_1m.parquet"
A_DAYS = 252.0

# ---------------------------------------------------------------- A: pure python
tbl = pq.read_table(SPOT, columns=["ts", "open", "high", "low", "close"])
ts, op, hi, lo, cl = (tbl.column(c).to_pylist() for c in ("ts", "open", "high", "low", "close"))
sessions = {}                                  # day_number -> {minute_slot: (o,h,l,c)}
for t, o, h, l, c in zip(ts, op, hi, lo, cl):
    ist = t + 19800
    dn, m = divmod(ist, 86400)
    slot = m // 60 - (9 * 60 + 15)
    if 0 <= slot < 375 and all(isinstance(v, float) and math.isfinite(v) and v > 0 for v in (o, h, l, c)) and h >= l and l <= o <= h and l <= c <= h:
        sessions.setdefault(dn, {})
        if slot in sessions[dn]:                # keep first, count nothing here (file has no duplicates; module reports them)
            continue
        sessions[dn][slot] = (o, h, l, c)
    else:
        sessions.setdefault(dn, {})
def weekday(dn):                                # 1970-01-01 was a Thursday
    return (dn + 3) % 7
A = {}
chain = []
for dn in sorted(sessions):
    bars = sessions[dn]
    if weekday(dn) < 5 and len(bars) >= 150:
        n = 0; ss = 0.0; sr = 0.0
        if 0 in bars:
            r = math.log(bars[0][3] / bars[0][0]); n += 1; ss += r * r; sr += r
        for i in range(1, 375):
            if i in bars and (i - 1) in bars:
                r = math.log(bars[i][3] / bars[i - 1][3]); n += 1; ss += r * r; sr += r
        chain.append(dn)
        A[dn] = dict(n=n, ss=ss, sr=sr, o=bars[0][0] if 0 in bars else None, c=bars[374][3] if 374 in bars else None)
import datetime as _dt
import os
_part = set()
for f in os.listdir("data/participant_oi"):                 # own reading of the calendar evidence (dates with NSE participant OI files)
    if f.endswith(".csv") and len(f) == 14:
        _part.add((_dt.date.fromisoformat(f[:-4]) - _dt.date(1970, 1, 1)).days)
def chain_unbroken(prev_dn, dn):
    """no participant-evidence trading date strictly between prev and dn that has ZERO spot bars (a missing session)"""
    return not any((prev_dn < d < dn) and d not in sessions for d in _part)
prev = None
for dn in chain:
    a = A[dn]
    if prev is not None and chain_unbroken(prev, dn):
        pc = A[prev]["c"]
        a["cc"] = math.log(a["c"] / pc) if (pc and a["c"]) else None
        a["on"] = math.log(a["o"] / pc) if (pc and a["o"]) else None
    else:
        a["cc"] = a["on"] = None
    prev = dn

# ---------------------------------------------------------------- B: pandas
df = pd.read_parquet(SPOT)
ist = df.ts + 19800
df["dn"] = ist // 86400
df["slot"] = (ist % 86400) // 60 - (9 * 60 + 15)
reg = df[(df.slot >= 0) & (df.slot < 375)]
Brows = {}
for dn, g in reg.groupby("dn"):
    if (dn + 3) % 7 >= 5 or len(g) < 150:
        continue
    g = g.drop_duplicates("slot").set_index("slot").reindex(range(375))
    prevc = g.close.shift(1)
    r = np.log(g.close / prevc)
    r.iloc[0] = np.log(g.close.iloc[0] / g.open.iloc[0])
    Brows[dn] = dict(n=int(r.notna().sum()), ss=float((r.dropna() ** 2).sum()), sr=float(r.dropna().sum()),
                     o=g.open.iloc[0], c=g.close.iloc[374])
Bs = pd.DataFrame(Brows).T
Bs["pc"] = Bs.c.shift(1); Bs["cc"] = np.log(Bs.c / Bs.pc); Bs["on"] = np.log(Bs.o / Bs.pc)

# ---------------------------------------------------------------- module
mod = pd.read_csv(f"{OUT}/daily_rv.csv")
mod = mod[mod.session_type == "regular"].copy()
mod["dn"] = pd.to_datetime(mod.session_date).map(lambda d: (d.date() - pd.Timestamp("1970-01-01").date()).days)
mod = mod.set_index("dn")

print("=== per-session reconciliation (regular sessions) ===")
print(f"sessions: module {len(mod)}  A {len(A)}  B {len(Bs)}")
common = sorted(set(mod.index) & set(A) & set(Bs.index))
print("common to all three:", len(common))
rows = []
for dn in common:
    m, a, b = mod.loc[dn], A[dn], Bs.loc[dn]
    rows.append(dict(dn=dn, n_mod=m.n_valid_returns, n_A=a["n"], n_B=b["n"], ss_mod_A=abs(m.intraday_sumsq - a["ss"]), ss_mod_B=abs(m.intraday_sumsq - b["ss"]),
                     cc_mod_A=abs(m.cc_return - a["cc"]) if (a["cc"] is not None and not pd.isna(m.cc_return)) else np.nan,
                     cc_mod_B=abs(m.cc_return - b["cc"]) if (not pd.isna(b["cc"]) and not pd.isna(m.cc_return)) else np.nan,
                     on_mod_A=abs(m.overnight_return - a["on"]) if (a["on"] is not None and not pd.isna(m.overnight_return)) else np.nan,
                     cc_valid_mismatch=(a["cc"] is None) != pd.isna(m.cc_return)))
r = pd.DataFrame(rows)
print("return-count mismatches  module vs A:", int((r.n_mod != r.n_A).sum()), " module vs B:", int((r.n_mod != r.n_B).sum()))
print("max |sum r^2| diff       module-A:", r.ss_mod_A.max(), " module-B:", r.ss_mod_B.max())
print("max |cc return| diff     module-A:", r.cc_mod_A.max(), " module-B:", r.cc_mod_B.max(), " | max |overnight| diff module-A:", r.on_mod_A.max())
print("sessions where cc validity differs (module vs A):", int(r.cc_valid_mismatch.sum()))
if r.cc_valid_mismatch.any():
    for dn in r[r.cc_valid_mismatch].dn:
        print("   ", (pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(dn))).date(), "module cc_reason:", mod.loc[dn].cc_reason, "| A cc:", A[dn]["cc"])

print("\n=== rolling windows (strict): module vs A-derived vs B-derived, annualised vol % ===")
roll = pd.read_csv(f"{OUT}/rolling_rv.csv")
roll = roll[roll.method == "strict"].copy()
roll["dn"] = pd.to_datetime(roll.session_date).map(lambda d: (d.date() - pd.Timestamp("1970-01-01").date()).days)
chain_days = sorted(A)
for measure in ("intraday", "close_to_close", "hybrid"):
    for N in (5, 10, 20):
        mm = roll[(roll.measure == measure) & (roll.window_sessions == N)].set_index("dn").rv_ann_pct
        diffs_A, diffs_B, valid_A_only, mod_only, both = [], [], 0, 0, 0
        for i, dn in enumerate(chain_days):
            if i - N + 1 < 0:
                continue
            w = chain_days[i - N + 1:i + 1]
            if measure == "intraday":
                valsA = [A[x]["ss"] if A[x]["n"] == 375 else None for x in w]
                valsB = [Bs.loc[x].ss if Bs.loc[x].n == 375 else None for x in w]
            elif measure == "close_to_close":
                valsA = [A[x]["cc"] ** 2 if A[x]["cc"] is not None else None for x in w]
                valsB = [Bs.loc[x].cc ** 2 if not pd.isna(Bs.loc[x].cc) else None for x in w]
            else:
                valsA = [(A[x]["on"] ** 2 + A[x]["ss"]) if (A[x]["on"] is not None and A[x]["n"] == 375) else None for x in w]
                valsB = [(Bs.loc[x].on ** 2 + Bs.loc[x].ss) if (not pd.isna(Bs.loc[x].on) and Bs.loc[x].n == 375) else None for x in w]
            va = 100 * math.sqrt(A_DAYS * sum(valsA) / N) if all(v is not None for v in valsA) else None
            vb = 100 * math.sqrt(A_DAYS * sum(valsB) / N) if all(v is not None for v in valsB) else None
            vm = mm.get(dn, np.nan)
            if (va is None) != pd.isna(vm) or (vb is None) != pd.isna(vm):
                mod_only += 1
            if va is not None and not pd.isna(vm):
                both += 1; diffs_A.append(abs(va - vm)); diffs_B.append(abs(vb - vm))
        print(f"{measure:15s} N={N:2d}: windows compared {both:5d}  max |module-A| {max(diffs_A):.2e}  max |module-B| {max(diffs_B):.2e}  availability disagreements {mod_only}")

print("\n=== representative hand-check: last regular session ===")
dn = chain_days[-1]; bars = sessions[dn]
first5 = [math.log(bars[0][3] / bars[0][0])] + [math.log(bars[i][3] / bars[i - 1][3]) for i in range(1, 5)]
print("date", (pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(dn))).date(), " first 5 minute returns:", [f"{x:.6e}" for x in first5])
print(f"A: n={A[dn]['n']}  sum r^2={A[dn]['ss']:.10e}  intraday ann vol = 100*sqrt(252*sum r^2) = {100*math.sqrt(A_DAYS*A[dn]['ss']):.6f} %   cc={A[dn]['cc']:.8f}  overnight={A[dn]['on']:.8f}")
m = mod.loc[dn]
print(f"module: n={int(m.n_valid_returns)}  sum r^2={m.intraday_sumsq:.10e}  intraday ann vol = {m.intraday_vol_ann_pct:.6f} %  cc={m.cc_return:.8f}  overnight={m.overnight_return:.8f}")
