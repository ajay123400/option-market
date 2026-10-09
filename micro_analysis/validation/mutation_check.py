"""Mutation check for micro_analysis (usage, repo root: python micro_analysis/validation/mutation_check.py WORKDIR): each fault is applied to a COPY and tests/test_micro_analysis.py must fail."""
import os
import shutil
import subprocess
import sys

SRC = os.getcwd()
W = sys.argv[1]
if os.path.exists(W):
    shutil.rmtree(W)
os.makedirs(W)
for d in ("micro_analysis", "collector", "tests", "optionsengine"):
    shutil.copytree(os.path.join(SRC, d), os.path.join(W, d), ignore=shutil.ignore_patterns("__pycache__"))
if os.path.exists(os.path.join(SRC, "pytest.ini")):
    shutil.copy(os.path.join(SRC, "pytest.ini"), W)
A = "micro_analysis/analyze.py"
M = [
    ("ivcalc: skew sign", "micro_analysis/ivcalc.py", "(expiry_ts - (capture_ts - (skew_s or 0.0)))", "(expiry_ts - (capture_ts + (skew_s or 0.0)))"),
    ("ivcalc: vega per point -> per 10", "micro_analysis/ivcalc.py", "math.sqrt(T) / 100.0", "math.sqrt(T) / 10.0"),
    ("ivcalc: vega sqrt(T) dropped", "micro_analysis/ivcalc.py", "norm_pdf(d1) * math.sqrt(T) / 100.0", "norm_pdf(d1) / 100.0"),
    ("ivcalc: ATM weight uses the upper point", "micro_analysis/ivcalc.py", "w = (0.0 - lo[0]) / (hi[0] - lo[0])", "w = (0.0 - hi[0]) / (hi[0] - lo[0])"),
    ("ivcalc: bracket limit exclusive", "micro_analysis/ivcalc.py", "khi - klo > max_bracket_pts", "khi - klo >= max_bracket_pts"),
    ("ivcalc: forward strike counted below", "micro_analysis/ivcalc.py", "if v[0] < 0]", "if v[0] <= 0]"),
    ("ivcalc: unreliable IVs accepted", "micro_analysis/ivcalc.py", "return res.iv, res.reliable", "return res.iv, True"),
    ("ivcalc: ATM usable flags ignored", "micro_analysis/ivcalc.py", "not (lo[2] and hi[2] and lo[1] is not None and hi[1] is not None)", "not (lo[1] is not None and hi[1] is not None)"),
    ("analyze: crossed boundary", A, "if bid > ask:", "if bid >= ask:"),
    ("analyze: stale boundary", A, "(cmf - skew) > cfg.max_quote_age_s", "(cmf - skew) >= cfg.max_quote_age_s"),
    ("analyze: skew not removed from the feed age", A, "(cmf - skew) > cfg.max_quote_age_s", "cmf > cfg.max_quote_age_s"),
    ("analyze: websocket check dropped", A, 'if r["data_source"] not in ("fyers:ws-full", "arrow:ws-full"):', "if False:"),
    ("analyze: never-traded rows kept", A, 'if not r["volume"] or pd.isna(r["volume"]):', "if False:"),
    ("analyze: mid = bid", A, "mid = (bid + ask) / 2\n        ltp_age", "mid = bid\n        ltp_age"),
    ("analyze: spread pct base", A, "spread_pct=100.0 * (ask - bid) / mid", "spread_pct=100.0 * (ask - bid) / ask"),
    ("analyze: ltp - mid sign", A, "d = ltp - mid if not pd.isna(ltp) else NAN", "d = mid - ltp if not pd.isna(ltp) else NAN"),
    ("analyze: ltp_at swapped", A, '("ask" if ltp >= ask else "bid" if ltp <= bid else "inside")', '("bid" if ltp >= ask else "ask" if ltp <= bid else "inside")'),
    ("analyze: ltp age skew", A, "(ltp_age - skew) <= cfg.max_quote_age_s", "ltp_age <= cfg.max_quote_age_s"),
    ("analyze: OTM side flipped", A, "otm = (typ == \"CE\") if k >= F else (typ == \"PE\")", "otm = (typ == \"PE\") if k >= F else (typ == \"CE\")"),
    ("analyze: forward status ignored", A, "if fe.status is ForwardStatus.OK:", "if fe.status is not ForwardStatus.OK:"),
    ("analyze: sell scenario sign", A, "sell = m - f * (m - b)", "sell = m + f * (m - b)"),
    ("analyze: vol point scale", A, "atm_rec[f\"sell_minus_ltp_vol_pts_cross{f:g}\"] = 100.0 * (sell - l)", "atm_rec[f\"sell_minus_ltp_vol_pts_cross{f:g}\"] = (sell - l)"),
    ("analyze: ltp iv freshness ignored", A, '("ltp", ltp, ltp_ok)', '("ltp", ltp, True)'),
    ("analyze: half-spread in vol points", A, 'rec["half_spread_vol_pts_iv"] = rec["iv_spread_vol_pts"] / 2 if both else NAN', 'rec["half_spread_vol_pts_iv"] = rec["iv_spread_vol_pts"] if both else NAN'),
    ("analyze: primary instants", A, "primary=hhmm in cfg.primary_times", "primary=False"),
    ("config: DTE edge exclusive", "micro_analysis/config.py", "    if t_days <= e[0]:", "    if t_days < e[0]:"),
    ("config: moneyness ATM", "micro_analysis/config.py", "if a == e[0]:", "if a <= e[0] + 1:"),
    ("config: time bucket edge", "micro_analysis/config.py", "if mins < edge:", "if mins <= edge:"),
    ("summaries: share at ask uses bid", "micro_analysis/summaries.py", 'share_ltp_at_ask=float((g["ltp_at"] == "ask").mean())', 'share_ltp_at_ask=float((g["ltp_at"] == "bid").mean())'),
    ("summaries: stale last trades kept", "micro_analysis/summaries.py", 'g0 = rows[rows["ltp_fresh"] & rows["ltp_age_bucket"].notna()]', "g0 = rows"),
    ("summaries: cost scenarios without bid IV", "micro_analysis/summaries.py", 'a = atm[atm["atm_iv_mid"].notna() & atm["atm_iv_bid"].notna() & atm["atm_iv_ask"].notna()]', 'a = atm[atm["atm_iv_mid"].notna()]'),
    ("summaries: primary filter", "micro_analysis/summaries.py", 'atm[atm["primary"]][cols]', "atm[cols]"),
    ("summaries: bias day window", "micro_analysis/summaries.py", '(rows["offset_strikes"].abs() <= max_abs_offset)', '(rows["offset_strikes"].abs() <= max_abs_offset + 5)'),
    ("run: input hash", "micro_analysis/run.py", "sha256=sha256(p)", "sha256=sha256(dbs[0])"),
    ("loader: day from filename", "micro_analysis/loader.py", 'replace("micro_", "")', 'replace("micro", "")'),
]
res = []
for name, f, a, b in M:
    p = os.path.join(W, f)
    s = open(p).read()
    if a not in s:
        res.append((name, "PATTERN NOT FOUND"))
        continue
    open(p, "w").write(s.replace(a, b, 1))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_micro_analysis.py", "-x", "-q", "-p", "no:cacheprovider"], cwd=W, capture_output=True, text=True, timeout=300)
        res.append((name, "KILLED" if r.returncode != 0 else "SURVIVED"))
    except subprocess.TimeoutExpired:
        res.append((name, "KILLED (timeout)"))
    open(p, "w").write(s)
for n, r in res:
    print(f"{r:20s} {n}")
