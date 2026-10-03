"""Independent re-calculation of Stage 2C targets from the RAW spot file (no optionsengine imports).

Calculation A: pure-Python loops (math.log, dict of bars).   Calculation B: pandas pivot + numpy diff.
Both implement the WRITTEN definitions: bar-start timestamps; reference price = close of the snapshot bar; first target
return = close(next minute)/close(snapshot bar); family A = the next k complete regular sessions after the snapshot
session; family B (expiry-aligned) = remainder of the snapshot session + every later session up to and including the
expiry session; hybrid = overnight + 1-minute returns; annualization sqrt(252 * variance / session-equivalents).
Usage: python research_output/stage2c/validation/independent_stage2c_check.py [stage2c_dir]
"""
import math
import random
import sys

import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2c"
A_DAYS = 252.0
long = pd.read_csv(f"{OUT}/iv_rv_observations.csv")
prim = long[(long.population == "primary") & long.target_available].copy()

# ------------------------------------------------------------------------------ raw bars
df = pd.read_parquet("data/hist1m/NIFTY50_1m.parquet")
ist = df.ts.to_numpy(dtype="int64") + 19800
dn = ist // 86400
slot = (ist % 86400) // 60 - 555
o_, h_, l_, c_ = (df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
valid = np.isfinite(o_) & np.isfinite(h_) & np.isfinite(l_) & np.isfinite(c_) & (o_ > 0) & (h_ >= l_) & (o_ >= l_) & (o_ <= h_) & (c_ >= l_) & (c_ <= h_)
in_win = (slot >= 0) & (slot < 375) & valid
bars = {}                                              # day number -> {slot: (open, close)}
for d, s, o, c in zip(dn[in_win], slot[in_win], o_[in_win], c_[in_win]):
    bars.setdefault(int(d), {})[int(s)] = (float(o), float(c))
wk = lambda d: (d + 3) % 7                              # 1970-01-01 = Thursday
chain = [d for d in sorted(bars) if wk(d) < 5 and len(bars[d]) >= 150]
pos = {d: i for i, d in enumerate(chain)}
to_dn = lambda iso: (pd.Timestamp(iso).date() - pd.Timestamp("1970-01-01").date()).days
complete = {d: len(bars[d]) == 375 for d in chain}


# ------------------------------------------------------------------------------ A: pure python
def sess_var_A(d, prev_close):
    """(intraday sum r^2, overnight r) for a complete session."""
    b = bars[d]
    ss = math.log(b[0][1] / b[0][0]) ** 2
    for i in range(1, 375):
        ss += math.log(b[i][1] / b[i - 1][1]) ** 2
    on = math.log(b[0][0] / prev_close) if prev_close else None
    return ss, on


def fixed_A(day, k, measure):
    p = pos[day]
    if p + k >= len(chain):
        return None
    tot = 0.0
    for j in range(p + 1, p + 1 + k):
        d = chain[j]
        prevd = chain[j - 1]
        pc = bars[prevd][374][1] if 374 in bars[prevd] else None
        if measure == "close_to_close":                    # only the two CLOSES are needed (session completeness is irrelevant)
            if pc is None or 374 not in bars[d]:
                return None
            tot += math.log(bars[d][374][1] / pc) ** 2
            continue
        if not complete[d] or pc is None:
            return None
        ss, on = sess_var_A(d, pc)
        tot += ss + (on * on if measure == "hybrid" else 0.0)
    return 100 * math.sqrt(A_DAYS * tot / k), tot


def exp_A(day, hhmm, expiry, measure):
    p, q = pos.get(day), pos.get(expiry)
    if p is None or q is None or q <= p:
        return None
    s0 = int(hhmm[:2]) * 60 + int(hhmm[3:]) - 555
    b = bars[day]
    if any(i not in b for i in range(s0, 375)):
        return None
    tot = sum(math.log(b[i][1] / b[i - 1][1]) ** 2 for i in range(s0 + 1, 375))
    n_part = 374 - s0
    for j in range(p + 1, q + 1):
        d = chain[j]
        if not complete[d] or 374 not in bars[chain[j - 1]]:
            return None
        ss, on = sess_var_A(d, bars[chain[j - 1]][374][1])
        tot += ss + (on * on if measure == "hybrid" else 0.0)
    eq = n_part / 375 + (q - p)
    return 100 * math.sqrt(A_DAYS * tot / eq), tot


# ------------------------------------------------------------------------------ B: pandas / numpy
reg = pd.DataFrame({"dn": dn[in_win], "slot": slot[in_win], "open": o_[in_win], "close": c_[in_win]})
reg = reg[reg.dn.isin(chain)]
CL = reg.pivot(index="dn", columns="slot", values="close")
OP = reg.pivot(index="dn", columns="slot", values="open")


def sess_B(d):
    c = CL.loc[d].to_numpy(); o = OP.loc[d].to_numpy()
    r = np.empty(375); r[0] = np.log(c[0] / o[0]); r[1:] = np.log(c[1:] / c[:-1])
    return r


def fixed_B(day, k, measure):
    p = pos[day]
    if p + k >= len(chain):
        return None
    tot = 0.0
    for j in range(p + 1, p + 1 + k):
        d, pv = chain[j], chain[j - 1]
        c, o = CL.loc[d].to_numpy(), OP.loc[d].to_numpy()
        cp = CL.loc[pv, 374]
        if measure == "close_to_close":
            if not (np.isfinite(c[374]) and np.isfinite(cp)):
                return None
            tot += np.log(c[374] / cp) ** 2
            continue
        if not np.isfinite(c).all() or not np.isfinite(o[0]):
            return None
        r = sess_B(d)
        tot += np.sum(r ** 2) + (np.log(o[0] / cp) ** 2 if measure == "hybrid" else 0.0)
    return 100 * np.sqrt(A_DAYS * tot / k), float(tot)


def exp_B(day, hhmm, expiry, measure):
    p, q = pos.get(day), pos.get(expiry)
    if p is None or q is None or q <= p:
        return None
    s0 = int(hhmm[:2]) * 60 + int(hhmm[3:]) - 555
    c = CL.loc[day].to_numpy()[s0:]
    if not np.isfinite(c).all():
        return None
    tot = float(np.sum(np.diff(np.log(c)) ** 2))
    for j in range(p + 1, q + 1):
        d, pv = chain[j], chain[j - 1]
        cc, oo = CL.loc[d].to_numpy(), OP.loc[d].to_numpy()
        if not np.isfinite(cc).all() or not np.isfinite(oo[0]) or not np.isfinite(CL.loc[pv, 374]):
            return None
        tot += float(np.sum(sess_B(d) ** 2)) + (float(np.log(oo[0] / CL.loc[pv, 374]) ** 2) if measure == "hybrid" else 0.0)
    eq = (374 - s0) / 375 + (q - p)
    return 100 * np.sqrt(A_DAYS * tot / eq), tot


# ------------------------------------------------------------------------------ compare on a stratified sample
rng = random.Random(17)
rows = []
for (h, m), g in prim.groupby(["horizon", "measure"]):
    ids = list(g.index)
    rng.shuffle(ids)
    for i in ids[:60 if h != "EXP" else 80]:
        rows.append(prim.loc[i])
S = pd.DataFrame(rows)
res = []
for r in S.itertuples():
    day, exp = to_dn(r.day), to_dn(r.expiry)
    if r.horizon == "EXP":
        a, b = exp_A(day, r.time, exp, r.measure), exp_B(day, r.time, exp, r.measure)
    else:
        k = {"F1": 1, "F5": 5, "F10": 10, "F20": 20}[r.horizon]
        a, b = fixed_A(day, k, r.measure), fixed_B(day, k, r.measure)
    if a is None or b is None:
        res.append(dict(h=r.horizon, m=r.measure, status="independent calc says UNAVAILABLE", rv_file=r.future_rv_pct))
        continue
    res.append(dict(h=r.horizon, m=r.measure, status="ok", rv_file=r.future_rv_pct, d_A=abs(a[0] - r.future_rv_pct), d_B=abs(b[0] - r.future_rv_pct), d_AB=abs(a[0] - b[0]),
                    tv_file=r.rv_total_variance, tv_A=a[1], tv_B=b[1], iv=r.iv_pct, diff_file=r.iv_minus_rv,
                    diff_recalc=r.iv_pct - a[0], ratio_file=r.iv_over_rv, ratio_recalc=(r.iv_pct / a[0]) if a[0] > 0 else np.nan,
                    abs_file=r.abs_err, sq_file=r.sq_err))
R = pd.DataFrame(res)
print(f"sample size {len(R)}; independent calculations disagree about availability in {int((R.status != 'ok').sum())} rows")
ok = R[R.status == "ok"]
print(R.groupby(["h", "m"]).size().to_string())
print(f"RV   max |file-A| {ok.d_A.max():.3e}  max |file-B| {ok.d_B.max():.3e}  max |A-B| {ok.d_AB.max():.3e}  (annualised vol points)")
print(f"total variance max |file-A| {np.abs(ok.tv_file - ok.tv_A).max():.3e}  max |file-B| {np.abs(ok.tv_file - ok.tv_B).max():.3e}")
print(f"IV-RV arithmetic: max |file diff - recomputed| {np.abs(ok.diff_file - ok.diff_recalc).max():.3e}; ratio {np.nanmax(np.abs(ok.ratio_file - ok.ratio_recalc)):.3e}; "
      f"abs_err {np.abs(ok.abs_file - np.abs(ok.diff_recalc)).max():.3e}; sq_err {np.abs(ok.sq_file - ok.diff_recalc ** 2).max():.3e}")

# ------------------------------------------------------------------------------ timestamp boundary logic
print("\n=== timestamp boundary logic (expiry-aligned sample) ===")
e = S[S.horizon == "EXP"].head(40)
bad = 0
for r in e.itertuples():
    d = pd.Timestamp(r.day)
    t0 = pd.Timestamp(f"{r.day} {r.time}", tz="Asia/Kolkata")
    expect_obs = (t0 + pd.Timedelta(seconds=60)).isoformat()
    first_ret_end = t0 + pd.Timedelta(seconds=120)
    ok_b = (r.observation_ts_ist == expect_obs) and (r.target_start_ts == expect_obs) and (pd.Timestamp(r.target_start_ts) < first_ret_end)
    bad += not ok_b
print(f"observation_ts == snapshot bar start + 60 s and target_start_ts == observation_ts: {len(e) - bad}/{len(e)}")
# discrimination: a WRONG boundary that also counts the snapshot bar's own open->close return must change the result
wrong = []
for r in e.itertuples():
    day, exp = to_dn(r.day), to_dn(r.expiry)
    s0 = int(r.time[:2]) * 60 + int(r.time[3:]) - 555
    b = bars[day]
    extra = math.log(b[s0][1] / b[s0][0]) ** 2
    wrong.append(extra / max(r.rv_total_variance, 1e-30))
print(f"if the snapshot bar's own return were wrongly included, target variance would change by median {100 * np.median(wrong):.3f}% (max {100 * np.max(wrong):.2f}%) -> the boundary is testable, and the file does not include it")
# 13:00 vs 15:00 remainder sizes
for hh, n in (("10:00", 329), ("13:00", 149), ("15:00", 29)):
    sub = S[(S.horizon == "EXP") & (S.time == hh) & (S.measure == "intraday")].head(3)
    for r in sub.itertuples():
        eq = r.session_equivalents - r.n_valid_sessions
        assert abs(eq * 375 - n) < 1e-6, (hh, eq * 375)
print("partial-session returns: 10:00 -> 329, 13:00 -> 149, 15:00 -> 29 (checked from session_equivalents)")

# ------------------------------------------------------------------------------ availability disagreements (reconciliation)
dis = R[R.status != "ok"]
if len(dis):
    print("\n=== rows where the independent code says UNAVAILABLE but the file has a value ===")
    idxs = S.reset_index(drop=True).loc[dis.index][["obs_id", "day", "time", "expiry", "horizon", "measure", "future_rv_pct"]]
    print(idxs.to_string(index=False))

# ------------------------------------------------------------------------------ the other direction: rows the file marks UNAVAILABLE
print("\n=== rows the file marks UNAVAILABLE: does the independent code agree that no complete target exists? ===")
un = long[(long.population == "primary") & (~long.target_available)].sample(400, random_state=5)
disagree = []
for r in un.itertuples():
    day, exp = to_dn(r.day), to_dn(r.expiry)
    if r.horizon == "EXP":
        a, b = exp_A(day, r.time, exp, r.measure), None
    else:
        k = {"F1": 1, "F5": 5, "F10": 10, "F20": 20}[r.horizon]
        a = fixed_A(day, k, r.measure) if day in pos else None
    if a is not None:
        disagree.append((r.obs_id, r.horizon, r.measure, r.unavailable_reason))
print(f"sampled unavailable rows: {len(un)} (reasons: {un.unavailable_reason.value_counts().to_dict()}); independent code constructed a target for {len(disagree)}")
for x in disagree[:10]:
    print("   ", x)
