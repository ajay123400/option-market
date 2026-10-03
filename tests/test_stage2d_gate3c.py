"""Stage 2D Gate 3c (T12): claim ledger, language lint, final report and the prospective protocol."""
import copy
import hashlib
import json
import os
import shutil

import pandas as pd
import pytest

from optionsengine.research.stage2d import build_gate3c as B, claim_lint as L, claims as K

ROOT = "research_output/stage2d"


@pytest.fixture(scope="module")
def src():
    return K.Src(ROOT)


@pytest.fixture(scope="module")
def claims(src):
    return K.build_claims(src)


# ============================================================================================ lint
@pytest.mark.parametrize("text,rule", [
    ("This shows a tradable edge in the data.", "edge_profit"),
    ("The strategy is profitable.", "edge_profit"),
    ("A variance risk premium exists.", "risk_premium"),
    ("The holdout gives out-of-sample confirmation.", "out_of_sample"),
    ("The result is significant.", "significance"),
    ("This proves the effect.", "proof"),
    ("Options are overpriced.", "mispricing"),
    ("The interval has guaranteed 95% coverage.", "nominal_coverage"),
])
def test_lint_flags_forbidden_claims(text, rule):
    r = L.lint_text(text)
    assert any(x["rule"] == rule and x["status"] == "FLAG" for x in r), r


@pytest.mark.parametrize("text", [
    "This is not evidence of a tradable edge.", "No variance risk premium is claimed.", "The holdout was previously viewed, never out-of-sample.",
    "The difference is statistically significant (p = 0.01).", "Significant after Holm adjustment.", "This does not prove a profitable strategy.",
    "A same-direction bias cannot be ruled out without bid/ask; tradable status is not assessable.", "The DTE bucket edge is inclusive.",
])
def test_lint_allows_negated_or_qualified(text):
    assert all(x["status"] != "FLAG" for x in L.lint_text(text)), L.lint_text(text)


def test_lint_reports_line_and_skips_forbidden_wording_lines():
    t = "ok line\nThe edge is large.\n**Forbidden wording.** tradable edge, profitable, proves\n"
    r = L.lint_text(t, "f.md")
    assert [(x["line"], x["rule"], x["status"]) for x in r] == [(2, "edge_profit", "FLAG")]
    assert r[0]["source"] == "f.md"


def test_lint_word_boundaries():
    assert L.lint_text("The knowledge of the cheapest path and a hedge fund.") == []


def test_lint_files_roundtrip(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("This is profitable.\n")
    r = L.lint_files([str(p)])
    assert len(r) == 1 and r[0]["status"] == "FLAG" and r[0]["source"] == str(p)


# ============================================================================================ ledger structure
def test_ledger_has_twenty_claims_with_valid_vocabulary(claims):
    assert [c.claim_id for c in claims] == [f"C{i:02d}" for i in range(1, 21)]
    assert all(c.status in K.STATUSES and c.tier in K.TIERS for c in claims)
    assert all(c.allowed and c.forbidden and c.warnings and c.sources for c in claims)


def test_ledger_passes_its_own_rules(src, claims):
    assert L.check_ledger(claims, K.tier1_checks(src, True)) == []


def test_ratio_interval_claims_carry_the_t11_warning(claims):
    ratio = [c for c in claims if set(c.estimands) & set(K.RATIO_ESTIMANDS)]
    assert {c.claim_id for c in ratio} >= {"C03", "C04", "C07", "C08", "C09", "C10"}
    for c in ratio:
        assert c.interval_based and c.undercoverage_warning
        assert "covered the true value of the ratio estimands only" in c.warnings and "not transferable" in c.warnings


def test_missing_warning_is_detected(src, claims):
    c3 = copy.deepcopy(claims)
    c3[2].warnings = "Descriptive."
    errs = L.check_ledger(c3, K.tier1_checks(src, True))
    assert any("C03" in e and "under-coverage" in e for e in errs)
    c4 = copy.deepcopy(claims)
    c4[0].warnings = "Descriptive."
    assert any("C01" in e and "not transferable" in e for e in L.check_ledger(c4, K.tier1_checks(src, True)))


def test_tier1_requires_all_conditions_and_assessable_claims(src, claims):
    errs = L.check_ledger(claims, {"a": True, "b": False})
    assert any("Tier 1 without all pre-registered conditions" in e for e in errs)
    bad = copy.deepcopy(claims)
    bad[18].tier = "1"
    assert any("C19" in e and "cannot be Tier 1" in e for e in L.check_ledger(bad, {"a": True}))


def test_forbidden_language_in_a_claim_is_detected(src, claims):
    bad = copy.deepcopy(claims)
    bad[0].claim = "The sample shows a tradable edge."
    assert any("C01.claim" in e and "edge_profit" in e for e in L.check_ledger(bad, K.tier1_checks(src, True)))


def test_tier1_checks_values(src):
    for spread in (True, False):
        chk = K.tier1_checks(src, spread)
        assert set(chk) == {"interval_excludes_null", "dev_holdout_same_side", "every_expiry_year_same_side", "every_specification_excludes_null", "no_detectable_outcome_dependent_selection"}
        assert all(chk.values())


def test_claim_numbers_come_from_the_csvs(src, claims):
    """C01's numbers are exactly the committed Gate 1 values; the T11 warning quotes the committed coverage table."""
    row = src.df("gate1/ci_methods_comparison.csv")
    r = row[(row.estimand == "S1_calendar_all_obs_median") & (row.method == "expiry_moving_block_b5")].iloc[0]
    assert f"{r.estimate:.3f}" in claims[0].key_numbers and f"{r.ci_lo:.2f}-{r.ci_hi:.2f}" in claims[0].key_numbers
    t = K.t11_numbers(src)
    assert 0.93 < t["slo"] < t["shi"] < 0.96 and 0.88 < t["rlo"] < t["rhi"] < 0.92
    assert f"{t['rlo']:.1%}-{t['rhi']:.1%}" in claims[2].warnings
    assert claims[3].status == "NOT_SUPPORTED" and "contains 1" in claims[3].key_numbers


def test_t11_numbers_by_hand_on_a_fake_source(tmp_path):
    os.makedirs(tmp_path / "gate3")
    rows = []
    for sc in ("B0_calibrated", "B1_strong_dependence"):
        for m in ("expiry_block5", "row_iid", "expiry_iid"):
            for e, v in (("S1_x", 0.94), ("S2_x", 0.95), ("R1_x", 0.90), ("R3_x", 0.88)):
                rows.append(dict(scenario=sc, method=m, estimand=e, coverage=v if m != "row_iid" else 0.3 if sc == "B0_calibrated" else 0.1))
    pd.DataFrame(rows).to_csv(tmp_path / "gate3" / "t11b_coverage.csv", index=False)
    t = K.t11_numbers(K.Src(str(tmp_path)))
    assert (t["slo"], t["shi"], t["rlo"], t["rhi"]) == (0.94, 0.95, 0.88, 0.90)
    assert t["row_lo"] == t["row_hi"] == 0.3


def test_src_val_requires_exactly_one_row(src):
    with pytest.raises(KeyError):
        src.val("gate1/ci_methods_comparison.csv", "estimate", method="expiry_moving_block_b5")
    assert src.val("gate1/estimands_point.csv", "n_rows", estimand="S1_calendar_all_obs_median") == 6490


# ============================================================================================ builder and outputs
def test_build_refuses_output_outside_stage2d(tmp_path):
    with pytest.raises(ValueError):
        B.build(ROOT, str(tmp_path / "plain"))


def test_build_is_deterministic_and_clean(tmp_path):
    a = tmp_path / "x" / "stage2d" / "o1"
    b = tmp_path / "x" / "stage2d" / "o2"
    for d in (a, b):
        os.makedirs(d)
        shutil.copy(os.path.join(ROOT, "gate3", B.PROTOCOL), d / B.PROTOCOL)
    ma, mb = B.build(ROOT, str(a)), B.build(ROOT, str(b))
    for f in ("claim_ledger.csv", "claim_ledger.md", "FINAL_REPORT.md", "lint_report.csv"):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    assert ma["lint"]["new_flags"] == 0 and ma["n_claims"] == 20
    assert ma["protocol_sha256"] == hashlib.sha256(open(os.path.join(ROOT, "gate3", B.PROTOCOL), "rb").read()).hexdigest()
    rep = (a / "FINAL_REPORT.md").read_text()
    assert rep.count("RATIO INTERVAL UNDER-COVERS") >= 5
    assert "88.7%" in rep and "91.7%" in rep and "93.7%" in rep and "94.7%" in rep
    for phrase in ("previously viewed", "not extrapolated", "must not be extrapolated"):
        assert phrase in rep or phrase in (a / "claim_ledger.md").read_text()


def test_committed_outputs_exist_and_are_consistent():
    g = os.path.join(ROOT, "gate3")
    df = pd.read_csv(os.path.join(g, "claim_ledger.csv"))
    assert len(df) == 20 and set(df.status) <= set(K.STATUSES)
    lint = pd.read_csv(os.path.join(g, "lint_report.csv"))
    new = lint[lint.source.str.endswith(("FINAL_REPORT.md", "claim_ledger.md", B.PROTOCOL, "GATE3C_IMPLEMENTATION_PLAN.md"))]
    assert (new.status != "FLAG").all()
    meta = json.load(open(os.path.join(g, "run_metadata_3c.json")))
    assert meta["protocol_sha256"] == hashlib.sha256(open(os.path.join(g, B.PROTOCOL), "rb").read()).hexdigest()
    # frozen inputs: the source CSV hashes recorded at generation still match
    for p, h in meta["source_hashes"].items():
        if os.path.basename(os.path.dirname(p)) == "gate3" and os.path.basename(p) in ("claim_ledger.csv", "lint_report.csv"):
            continue
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h, p


# ============================================================================================ protocol
def test_protocol_freezes_the_required_items_without_results():
    t = open(os.path.join(ROOT, "gate3", B.PROTOCOL), encoding="utf-8").read()
    for must in ("2026-09-30", "52 prospective expiries", "b = 5 and b = 8", "no prospective expiry has been analysed", "88.7", "fewer than 30 expiries", "SHA-256"):
        assert must in t
    assert not any(w in t.lower() for w in ("stop loss", "take profit", "entry price", "lot size"))
    assert all(x["status"] != "FLAG" for x in L.lint_text(t))


def test_other_claim_numbers_match_the_csvs(src, claims):
    g1 = src.df("gate1/ci_methods_comparison.csv")
    w = lambda m: float(g1[(g1.estimand == "S1_calendar_all_obs_median") & (g1.method == m)].width.iloc[0])
    assert f"{w('expiry_moving_block_b5') / w('row_iid_NAIVE_reference'):.1f}x (S1 calendar)" in claims[4].key_numbers
    assert f"{w('expiry_moving_block_b5') / w('date_block_stage2c_method'):.2f}x as wide" in claims[5].key_numbers
    man = src.df("gate3/t9_manski_bounds.csv")
    m = man[(man.estimand.str.startswith("S1_calendar")) & (man.missing_scope == "all missing (L1+L2)")].iloc[0]
    assert f"[{m.manski_lower:.3f}, {m.manski_upper:.3f}]" in claims[9].key_numbers and m.manski_lower < m.manski_upper
    tip = src.df("gate3/t10_tipping_bias.csv").iloc[0]
    assert f"Rs {tip.price_bias_rs_median:.1f}" in claims[11].key_numbers and "relative reduction of 12.0% of the IV level" in claims[11].key_numbers
    cp = src.df("gate3/t10_call_put_disagreement.csv").iloc[0]
    assert f"{cp.median_diff_vol_points:+.3f}" in claims[12].key_numbers
    assert claims[6].status == "INCONCLUSIVE" and "all Holm p = 1.00" in claims[6].key_numbers
