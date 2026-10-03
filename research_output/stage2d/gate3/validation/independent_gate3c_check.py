"""Independent recomputation of the Gate 3c claim numbers (no optionsengine imports; csv/pandas/numpy only).

* Headline T11 coverage ranges are recomputed from the RAW per-history intervals (`t11b_history_intervals.csv`) and the population truths (`t11b_truth.csv`), not from the aggregated table.
* Key numbers of claims C01, C03, C04, C07, C10, C12, C13 are re-read from the Gate 1-3a CSVs by explicit row picking and located in `claim_ledger.csv`.
* Structural re-check by plain string tests: every ledger row with an R-estimand and an interval carries the under-coverage warning; no tier-1 row is NOT_ASSESSABLE; the final report contains the warning column.
Usage: python research_output/stage2d/gate3/validation/independent_gate3c_check.py [stage2d_dir]
"""
import re
import sys
import numpy as np
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2d"
bad = 0


def ok(name, cond, detail=""):
    global bad
    bad += not cond
    print(f"  {'ok ' if cond else 'BAD'} {name} {detail}")


led = pd.read_csv(f"{R}/gate3/claim_ledger.csv").set_index("claim_id")
rep = open(f"{R}/gate3/FINAL_REPORT.md", encoding="utf-8").read()

# ---- T11 headline ranges from the raw per-history records
hi = pd.read_csv(f"{R}/gate3/t11b_history_intervals.csv")
tr = pd.read_csv(f"{R}/gate3/t11b_truth.csv")
print("T11 coverage from raw per-history intervals (headline estimands S1_calendar and R1 are stored per history)")
cov = {}
for (sc, m, e), g in hi.groupby(["scenario", "method", "estimand"]):
    t = tr[(tr.scenario == sc) & (tr.estimand == e)].population_value.iloc[0]
    g = g[g.ci_lo.notna() & g.ci_hi.notna()]
    cov[(sc, m, e)] = float(((g.ci_lo <= t) & (t <= g.ci_hi)).mean())
agg = pd.read_csv(f"{R}/gate3/t11b_coverage.csv")
for (sc, m, e), v in cov.items():
    a = agg[(agg.scenario == sc) & (agg.method == m) & (agg.estimand == e)].coverage.iloc[0]
    ok(f"raw vs aggregated coverage {sc}/{m}/{e}", abs(v - a) < 1e-12, f"{v:.4f}")
b0 = agg[(agg.scenario == "B0_calibrated") & (agg.method == "expiry_block5")]
rat, spr = b0[b0.estimand.str.startswith("R")].coverage, b0[b0.estimand.str.startswith("S")].coverage
for label, lo, hi_ in (("ratio", rat.min(), rat.max()), ("spread", spr.min(), spr.max())):
    s = f"{lo:.1%}"
    ok(f"{label} range {lo:.1%}-{hi_:.1%} in FINAL_REPORT", s in rep and f"{hi_:.1%}" in rep)
ok("ratio under-coverage range is 88.7%-91.7%", round(rat.min(), 3) == 0.887 and round(rat.max(), 3) == 0.917)
ok("spread range is 93.7%-94.7%", round(spr.min(), 3) == 0.937 and round(spr.max(), 3) == 0.947)

# ---- claim numbers
print("Claim numbers re-read from the source CSVs")
ci = pd.read_csv(f"{R}/gate1/ci_methods_comparison.csv")
pick = lambda e: ci[(ci.estimand == e) & (ci.method == "expiry_moving_block_b5")].iloc[0]
s1 = pick("S1_calendar_all_obs_median"); r1 = pick("R1_geometric_mean_all_obs"); r3 = pick("R3_ratio_of_summed_total_variance")
ok("C01 estimate and interval", f"{s1.estimate:.3f}" in led.key_numbers["C01"] and f"{s1.ci_lo:.2f}-{s1.ci_hi:.2f}" in led.key_numbers["C01"], led.key_numbers["C01"][:70])
ok("C03 R1 estimate and interval", f"{r1.estimate:.3f} ({r1.ci_lo:.3f}-{r1.ci_hi:.3f})" in led.key_numbers["C03"])
ok("C04 R3 interval contains 1", r3.ci_lo < 1 < r3.ci_hi and f"{r3.ci_lo:.3f}-{r3.ci_hi:.3f}" in led.key_numbers["C04"] and led.status["C04"] == "NOT_SUPPORTED")
spec = pd.read_csv(f"{R}/gate2/t7_t8_specification_curve.csv")
r3s = spec[spec.estimand == "R3_ratio_of_summed_total_variance"]
n_in = int((r3s.ci_lo < 1).__and__(r3s.ci_hi > 1).sum())
ok("C04 interval contains 1 in N specs", f"contains 1 in {n_in} of {len(r3s)} specifications" in led.key_numbers["C04"], f"{n_in}/{len(r3s)}")
t5 = pd.read_csv(f"{R}/gate2/t5_split_contrasts.csv")
famA = t5[t5.family_A == True]
ok("C07 family A Holm p all >= 1.00", float(famA.p_holm_family_A.min()) >= 0.995 and led.status["C07"] == "INCONCLUSIVE")
man = pd.read_csv(f"{R}/gate3/t9_manski_bounds.csv")
m = man[man.estimand.str.contains("calendar") & man.missing_scope.str.startswith("all missing")].iloc[0]
ok("C10 Manski bounds", f"[{m.manski_lower:.3f}, {m.manski_upper:.3f}]" in led.key_numbers["C10"])
tip = pd.read_csv(f"{R}/gate3/t10_tipping_bias.csv").iloc[0]
ok("C12 tipping bias and relative IV reduction", f"Rs {tip.price_bias_rs_median:.1f}" in led.key_numbers["C12"] and f"{tip.iv_reduction_pct_for_unit_ratio:.1f}% of the IV level" in led.key_numbers["C12"])
cp = pd.read_csv(f"{R}/gate3/t10_call_put_disagreement.csv").iloc[0]
ok("C13 call-minus-put median", f"{cp.median_diff_vol_points:+.3f}" in led.key_numbers["C13"])
w = lambda mth: float(ci[(ci.estimand == "S1_calendar_all_obs_median") & (ci.method == mth)].width.iloc[0])
ok("C05 expiry-block / row-iid width ratio", f"{w('expiry_moving_block_b5') / w('row_iid_NAIVE_reference'):.1f}x" in led.key_numbers["C05"])

# ---- structural re-check by plain strings
print("Structural checks")
for cid, r in led.iterrows():
    ests = str(r.estimands).split(";") if isinstance(r.estimands, str) else []
    if r.interval_based and any(e in ("R1", "R1e", "R2", "R2e", "R3") for e in ests):
        ok(f"{cid} ratio interval claim has the T11 warning", "covered the true value of the ratio estimands only 88.7%-91.7%" in r.required_warnings and "not transferable" in r.required_warnings)
    if r.evidence_tier == "1":
        ok(f"{cid} tier 1 has all five conditions True", str(r.tier1_conditions).count("=True") == 5)
    if r.status in ("NOT_ASSESSABLE", "NOT_CLAIMED"):
        ok(f"{cid} is not tier 1", r.evidence_tier != "1")
ok("C19 never upgraded", led.status["C19"] == "NOT_ASSESSABLE" and led.status["C18"] == "NOT_CLAIMED" and led.status["C20"] == "NOT_ASSESSABLE")
ok("FINAL_REPORT carries the under-coverage column for ratio claims", rep.count("RATIO INTERVAL UNDER-COVERS") >= 5)
# independent mini-lint on the generated prose (ledger + report): forbidden words must sit in a sentence with a negation/limit
neg = re.compile(r"\b(not|no|never|cannot|without|nor)\b|NOT_|n/a", re.I)
tern = re.compile(r"\b(tradable|profitable|alpha|arbitrage|risk premium)\b", re.I)
hits = 0
for txt in [rep] + [str(x) for c in ("claim", "key_numbers", "required_warnings", "allowed_wording") for x in led[c]]:
    for sent in re.split(r"(?<=[.!?;])\s+|\n", txt):
        if tern.search(sent) and not neg.search(sent):
            hits += 1; print("   un-negated:", sent[:120])
ok("independent lint: no un-negated forbidden term in ledger prose or final report", hits == 0)
print("\nRESULT:", "ALL CHECKS PASSED" if bad == 0 else f"{bad} CHECKS FAILED")
sys.exit(1 if bad else 0)
