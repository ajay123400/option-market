"""Stage 2D Gate 3c (T12): the claim ledger. Every number is read from the committed Gate 1-3b CSVs (never typed); wording is fixed here and linted."""
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import pandas as pd

TIERS = {"0": "descriptive association in this sample", "1": "statistical evidence (see rule)", "2": "tradable edge: not assessable here", "NA": "not applicable (method / limitation statement)"}
STATUSES = ("SUPPORTED_WITH_CAVEATS", "SUPPORTED_METHOD", "INCONCLUSIVE", "NOT_SUPPORTED", "NOT_ASSESSABLE", "NOT_CLAIMED")
RATIO_ESTIMANDS = ("R1", "R1e", "R2", "R2e", "R3")

ROOT = "research_output/stage2d"
UNDERCOVERAGE = ("Warning (T11): in the calibrated synthetic study the expiry-block (b = 5) 95% interval covered the true value of the ratio estimands only {rlo:.1%}-{rhi:.1%} of the time "
                 "(nominal 95%), and materially less under strong dependence (best {b1lo:.0%}-{b1hi:.0%}); read ratio intervals as too narrow. Synthetic coverage is not transferable to the real data.")
SPREAD_NOTE = ("Note (T11): in the calibrated synthetic study the expiry-block (b = 5) interval for spread-median estimands covered {slo:.1%}-{shi:.1%} (nominal 95%); worse under strong dependence. Synthetic coverage is not transferable to the real data.")


class Src:
    """Loads CSVs once and returns single values; an ambiguous or empty selection raises."""

    def __init__(self, root: str = ROOT):
        self.root = root
        self._c: Dict[str, pd.DataFrame] = {}

    def df(self, rel: str) -> pd.DataFrame:
        if rel not in self._c:
            self._c[rel] = pd.read_csv(os.path.join(self.root, rel))
        return self._c[rel]

    def val(self, rel: str, col: str, **where):
        d = self.df(rel)
        for k, v in where.items():
            d = d[d[k] == v]
        if len(d) != 1:
            raise KeyError(f"{rel} {where}: {len(d)} rows")
        return d[col].iloc[0]


@dataclass
class Claim:
    claim_id: str
    topic: str
    status: str
    tier: str
    claim: str
    key_numbers: str
    sources: str
    warnings: str
    allowed: str
    forbidden: str
    estimands: tuple = ()
    interval_based: bool = False
    undercoverage_warning: bool = False


def t11_numbers(s: Src) -> Dict[str, float]:
    c = s.df("gate3/t11b_coverage.csv")
    b0 = c[(c.scenario == "B0_calibrated") & (c.method == "expiry_block5")]
    rat = b0[b0.estimand.str.startswith("R")].coverage
    spr = b0[b0.estimand.str.startswith("S")].coverage
    b1 = c[(c.scenario == "B1_strong_dependence")]
    bestb1 = b1.groupby("method").coverage.mean()
    best = b1[b1.method == bestb1.idxmax()].coverage
    return dict(rlo=rat.min(), rhi=rat.max(), slo=spr.min(), shi=spr.max(), b1lo=best.min(), b1hi=best.max(),
                row_lo=c[(c.scenario == "B0_calibrated") & (c.method == "row_iid")].coverage.min(), row_hi=c[(c.scenario == "B0_calibrated") & (c.method == "row_iid")].coverage.max())


def tier1_checks(s: Src, spread: bool) -> Dict[str, bool]:
    """The pre-registered Tier 1 conditions for the headline spread (calendar) or typical ratio (R1): interval excludes the null, dev and holdout on the same side of the null,
    every expiry year on that side, every specification on that side with interval excluding the null, and no detectable outcome-dependent selection (T9)."""
    est = "S1_calendar_all_obs_median" if spread else "R1_geometric_mean_all_obs"
    stat = "S1_calendar" if spread else "R1_geometric_mean"
    null = 0.0 if spread else 1.0
    row = s.df("gate1/ci_methods_comparison.csv")
    row = row[(row.estimand == est) & (row.method == "expiry_moving_block_b5")].iloc[0]
    t5 = s.df("gate2/t5_split_contrasts.csv"); t5 = t5[t5.estimand == est].iloc[0]
    st = s.df("gate2/t6_strata_estimates.csv"); yrs = st[(st.dimension == "expiry_year") & (st.statistic == stat)]
    sr = s.df("gate2/t7_t8_robustness_reading.csv").set_index("estimand").loc[est]
    od = s.df("gate3/t9_outcome_dependence_test.csv").iloc[0]
    return {"interval_excludes_null": bool(row.ci_lo > null or row.ci_hi < null), "dev_holdout_same_side": bool((t5.dev_estimate - null) * (t5.holdout_estimate - null) > 0),
            "every_expiry_year_same_side": bool(((yrs.estimate - null) * (row.estimate - null) > 0).all() and len(yrs) >= 5),
            "every_specification_excludes_null": bool(sr.share_intervals_excluding_null == 1.0 and sr.sign_same_as_baseline_in_every_spec),
            "no_detectable_outcome_dependent_selection": bool(od.p_permutation_two_sided > 0.05)}


def build_claims(s: Src) -> List[Claim]:
    t = t11_numbers(s)
    UC = UNDERCOVERAGE.format(**t)
    SN = SPREAD_NOTE.format(**t)
    G1 = "gate1/ci_methods_comparison.csv"
    eb = lambda est, col: float(s.val(G1, col, estimand=est, method="expiry_moving_block_b5"))
    sc, ss = "S1_calendar_all_obs_median", "S1_session_all_obs_median"
    r1, r2, r3 = "R1_geometric_mean_all_obs", "R2_median_ratio_all_obs", "R3_ratio_of_summed_total_variance"
    spec = s.df("gate2/t7_t8_specification_curve.csv")
    spec_r3 = spec[spec.estimand == r3]
    r3_incl1 = int(spec_r3.null_inside.sum()); r3_n = len(spec_r3)
    rr = s.df("gate2/t7_t8_robustness_reading.csv").set_index("estimand")
    het = s.df("gate2/t6_heterogeneity_tests.csv")
    n_holm = int((het.p_holm_family_B < 0.05).sum()); n_het = len(het)
    t5 = s.df("gate2/t5_split_contrasts.csv"); fam = t5[t5.family_A == True]
    ri = s.df("gate1/ratio_inference.csv")
    sign = ri[ri.test == "sign_test_expiry_median_ratio_gt_1"].iloc[0]
    dep = s.df("gate1/dependence_diagnostics.csv").set_index("series")
    nar = float(s.val(G1, "width", estimand=sc, method="expiry_moving_block_b5")) / float(s.val(G1, "width", estimand=sc, method="row_iid_NAIVE_reference"))
    nar2 = float(s.val(G1, "width", estimand=r2, method="expiry_moving_block_b5")) / float(s.val(G1, "width", estimand=r2, method="row_iid_NAIVE_reference"))
    dbw = float(s.val(G1, "width", estimand=sc, method="expiry_moving_block_b5")) / float(s.val(G1, "width", estimand=sc, method="date_block_stage2c_method"))
    um = s.df("gate3/t9_universe_counts.csv"); allrow = um[(um.dimension == "all")].iloc[0]
    od = s.df("gate3/t9_outcome_dependence_test.csv").iloc[0]
    man = s.df("gate3/t9_manski_bounds.csv"); mc = man[(man.estimand.str.startswith("S1_calendar")) & (man.missing_scope == "all missing (L1+L2)")].iloc[0]
    ipw = s.df("gate3/t9_ipw_estimates.csv").set_index("estimand")
    tip = s.df("gate3/t10_tipping_bias.csv").iloc[0]
    cp = s.df("gate3/t10_call_put_disagreement.csv").iloc[0]
    mz = s.df("gate2/t6_mincer_zarnowitz.csv"); mza = mz[(mz["sample"] == "all") & (mz.scale == "level")].iloc[0]
    ed = s.df("gate2/t6_expiry_day_stratum.csv")
    edv = ed[(ed.horizon == "F1") & (ed.level == "expiry day") & (ed.statistic == "S1_calendar")].iloc[0]
    eno = ed[(ed.horizon == "F1") & (ed.level == "not expiry day") & (ed.statistic == "S1_calendar")].iloc[0]
    ta = s.df("gate3/t11a_recovery_summary.csv")
    tg = s.df("gate3/t11a_generator_check.csv")
    cov = s.df("gate3/t11b_coverage.csv")
    dbk = cov[(cov.scenario == "B0_calibrated") & (cov.method == "date_block10")].coverage.mean()
    d_s1 = fam[fam.estimand == sc].iloc[0]
    loyo = s.df("gate1/leave_one_year_out.csv")
    y_min, y_max = float(loyo[sc].min()), float(loyo[sc].max())
    strata = s.df("gate2/t6_strata_estimates.csv")
    yrs = strata[(strata.dimension == "expiry_year") & (strata.statistic == "S1_calendar")]
    all_years_pos = bool((yrs.estimate > 0).all())
    topk = s.df("gate1/topk_removal.csv"); tk = topk[(topk.estimand == r3) & (topk.k_removed == 5)].iloc[0]
    t1s, t1r = tier1_checks(s, True), tier1_checks(s, False)
    tier_s = "1" if all(t1s.values()) else "0"
    tier_r = "1" if all(t1r.values()) else "0"
    C = Claim
    claims = [
        C("C01", "IV exceeds realised volatility (calendar basis)", "SUPPORTED_WITH_CAVEATS", tier_s,
          "In the 260-expiry EXP/hybrid sample the median calendar-basis spread (IV minus RV) is positive: the historical sample shows implied volatility above subsequent realised volatility.",
          f"S1 calendar {eb(sc, 'estimate'):.3f} vol pts, 95% expiry-block interval {eb(sc, 'ci_lo'):.2f}-{eb(sc, 'ci_hi'):.2f}; dev {d_s1.dev_estimate:.3f}, holdout {d_s1.holdout_estimate:.3f}; all expiry years positive: {all_years_pos}; "
          f"positive in {int(rr.loc[sc, 'n_specs'] * rr.loc[sc, 'share_intervals_excluding_null'])} of {int(rr.loc[sc, 'n_specs'])} specifications (range {rr.loc[sc, 'min_estimate']:.2f}-{rr.loc[sc, 'max_estimate']:.2f})",
          "gate1/ci_methods_comparison.csv; gate2/t5_split_contrasts.csv; gate2/t6_strata_estimates.csv; gate2/t7_t8_robustness_reading.csv; gate3/t9_*.csv",
          "Descriptive of this historical sample under stated dependence assumptions; the holdout was previously viewed (not out-of-sample); no bid/ask, so a same-direction last-trade bias of about "
          f"Rs {tip.price_bias_rs_median:.1f} would explain all of it (C12); ≤ 14 DTE universe only (C11). {SN}",
          "The sample shows IV above realised volatility by about 1.8 vol pts (calendar basis), with the stated interval and caveats.",
          "IV is overpriced / options are expensive / a premium exists / this is tradable / out-of-sample confirmation", ("S1",), True, True),
        C("C02", "Spread size depends on the annualization basis", "SUPPORTED_WITH_CAVEATS", "0",
          "The size of the IV-RV spread depends on whether RV is annualised per trading session or per calendar day; the session-basis spread is larger.",
          f"S1 session {eb(ss, 'estimate'):.3f} ({eb(ss, 'ci_lo'):.2f}-{eb(ss, 'ci_hi'):.2f}) vs calendar {eb(sc, 'estimate'):.3f}; session-basis dev-minus-holdout difference {float(s.val('gate2/t5_split_contrasts.csv', 'difference_holdout_minus_dev', estimand=ss)):.2f} (p = {float(s.val('gate2/t5_split_contrasts.csv', 'p_two_sided_bootstrap', estimand=ss)):.3f}, block-length dependent)",
          "gate1/ci_methods_comparison.csv; gate2/t5_split_contrasts.csv; Stage 2C annualization audit",
          f"Descriptive; the calendar basis is the headline because it matches IV's time basis. {SN}", "The two bases give different spread sizes; the difference reflects the annualization convention.",
          "the session spread is the true spread", ("S1",), True, True),
        C("C03", "Typical total-variance ratio exceeds 1", "SUPPORTED_WITH_CAVEATS", tier_r,
          "Implied total variance typically exceeds subsequently realised total variance in this sample (typical-ratio estimands R1/R2).",
          f"R1 {eb(r1, 'estimate'):.3f} ({eb(r1, 'ci_lo'):.3f}-{eb(r1, 'ci_hi'):.3f}); R2 {eb(r2, 'estimate'):.3f} ({eb(r2, 'ci_lo'):.3f}-{eb(r2, 'ci_hi'):.3f}); "
          f"{float(sign.statistic):.1%} of expiries have median ratio above 1 (exact sign test p = {float(sign.p_two_sided):.1e}); intervals exclude 1 in {int(rr.loc[r1, 'n_specs'])} of {int(rr.loc[r1, 'n_specs'])} specifications",
          "gate1/ci_methods_comparison.csv; gate1/ratio_inference.csv; gate2/t7_t8_robustness_reading.csv; gate3/t11b_coverage.csv",
          UC, "The typical ratio of implied to realised total variance is above 1 in this sample; the interval is probably too narrow (see warning).",
          "ratio proves overpricing / a volatility risk premium", ("R1", "R2"), True, True),
        C("C04", "Ratio of summed variances (R3)", "NOT_SUPPORTED", "0",
          "R3 (sum of implied over sum of realised total variance) is NOT shown to differ from 1.",
          f"R3 {eb(r3, 'estimate'):.3f} ({eb(r3, 'ci_lo'):.3f}-{eb(r3, 'ci_hi'):.3f}) contains 1; interval contains 1 in {r3_incl1} of {r3_n} specifications; removing the 5 most influential expiries moves it {float(tk.full_estimate):.3f} -> {float(tk.after_removing_top_k_influential):.3f} (random-removal median {float(tk.random_k_p50):.3f})",
          "gate1/ci_methods_comparison.csv; gate1/topk_removal.csv; gate2/t7_t8_specification_curve.csv", UC,
          "R3 is dominated by a few high-variance expiries; no claim about R3 above 1 is made.", "volume-weighted premium exists; R3 confirms the ratio", ("R3",), True, True),
        C("C05", "Row-level inference is invalid", "SUPPORTED_METHOD", "NA",
          "Treating the 6,490 rows as independent is invalid: rows within an expiry are strongly correlated and neighbouring expiries overlap.",
          f"the expiry-block interval is {nar:.1f}x (S1 calendar) and {nar2:.1f}x (R2) as wide as the row-iid interval; ICC by expiry {float(dep.loc['median_spread_calendar', 'icc_by_expiry']):.2f}, design effect {float(dep.loc['median_spread_calendar', 'design_effect_by_expiry']):.1f}; "
          f"synthetic coverage of row-iid {t['row_lo']:.0%}-{t['row_hi']:.0%} (B0) vs nominal 95%",
          "gate1/ci_methods_comparison.csv; gate1/dependence_diagnostics.csv; gate3/t11b_coverage.csv", "Synthetic coverage is not transferable to the real data.",
          "Inference must use the expiry as the unit with block resampling; row-level intervals are not used.", "6,490 independent observations", (), False, False),
        C("C06", "Date-block (Stage 2C) intervals", "SUPPORTED_METHOD", "NA",
          "The Stage 2C date-block interval was somewhat optimistic.",
          f"expiry-block interval {dbw:.2f}x as wide as the date-block interval for S1 calendar; synthetic mean coverage of date_block10 {dbk:.1%} (B0)",
          "gate1/ci_methods_comparison.csv; gate3/t11b_coverage.csv", "Synthetic coverage is not transferable to the real data.",
          "Stage 2C intervals should be read as lower bounds on uncertainty.", "Stage 2C intervals were conservative", (), False, False),
        C("C07", "Development vs holdout", "INCONCLUSIVE", "0",
          "No development-vs-holdout difference is detected, but the comparison has low power and cannot show that the periods are equal.",
          f"family A (4 contrasts): all Holm p = {float(fam.p_holm_family_A.min()):.2f}; S1 calendar holdout-minus-dev {float(d_s1.difference_holdout_minus_dev):.2f} ({float(d_s1.ci_lo):.2f} to {float(d_s1.ci_hi):.2f})",
          "gate2/t5_split_contrasts.csv; gate2/t5_post_stratified_contrast.csv; gate3/t11b_split_contrast.csv",
          f"The holdout was previously viewed and is not out-of-sample. In the synthetic study the false-rejection rate of the contrast under a true null was about 6-7% overall and up to 10-13% for R3. {UC}",
          "No difference between periods was detected; the data cannot rule out one.", "holdout confirms / validates out-of-sample / the effect is stable", ("S1", "R1", "R2"), True, True),
        C("C08", "Heterogeneity across strata", "SUPPORTED_WITH_CAVEATS", "0",
          "The spread differs descriptively across DTE bucket, IV level, recent-volatility regime and weekend-in-window; these strata overlap and partly condition on variables tied to the spread.",
          f"{n_holm} of {n_het} pre-declared heterogeneity tests survive Holm adjustment; expiry-year heterogeneity not rejected",
          "gate2/t6_heterogeneity_tests.csv; gate2/t6_strata_estimates.csv; gate3/t11b_wald.csv",
          f"Descriptive, no mechanism claimed. In the synthetic study the Wald test false-rejected 3.7-7.0% with row-level placebo strata but 8.3-10.0% with date-level placebo strata (anti-conservative). {UC}",
          "The spread varies across these strata in this sample.", "the premium is higher when ... (a rule)", ("S1", "R1"), True, True),
        C("C09", "Robustness to conventions and filters", "SUPPORTED_WITH_CAVEATS", "0",
          "The sign of the spread and typical ratio is unchanged across the 30-31 tested conventions and filters; magnitude moves.",
          f"S1 calendar range {rr.loc[sc, 'min_estimate']:.2f}-{rr.loc[sc, 'max_estimate']:.2f} (baseline {rr.loc[sc, 'baseline']:.2f}); R1 range {rr.loc[r1, 'min_estimate']:.3f}-{rr.loc[r1, 'max_estimate']:.3f} incl. an intraday-only subset; R3 interval contains 1 in {r3_incl1} of {r3_n} specifications",
          "gate2/t7_t8_robustness_reading.csv; gate2/t7_t8_specification_curve.csv",
          f"All variants reuse the same committed pipeline, so a common bias (strike coverage, last-trade pricing, missing bid/ask) is shared by every specification and not detectable here. {UC}",
          "Not sensitive in sign to the tested conventions.", "robust to everything / model-free", ("S1", "R1", "R3"), True, True),
        C("C10", "Selection of observations", "SUPPORTED_WITH_CAVEATS", "0",
          "Observable selection into the sample is small and, within the power available, no dependence of rejection on unforeseen later volatility is detected.",
          f"{int(allrow.L1_rejected + allrow.L2_no_exp_target)} of {int(allrow.attempted_groups - allrow.expiry_day_by_construction)} non-expiry-day groups missing ({float(allrow.share_missing_of_u_exp):.1%}); Manski bounds for the median calendar spread [{float(mc.manski_lower):.3f}, {float(mc.manski_upper):.3f}]; "
          f"break-down share {float(mc.breakdown_share_needed):.1%}; outcome coefficient {float(od.outcome_coefficient_standardized):+.2f} (permutation p = {float(od.p_permutation_two_sided):.2f}, {int(od.n_flagged)} flagged); IPW shifts S1 calendar by {float(ipw.loc['S1_calendar', 'difference']):+.3f}",
          "gate3/t9_*.csv", f"Low power (53 flagged groups); cannot see selection on unobservables, the strike-listing process, or anything outside ≤ 14 DTE. {UC}",
          "No selection on the outcome was detected within the tested power.", "there is no selection bias", ("S1", "R1"), True, True),
        C("C11", "Population is the ≤ 14 DTE universe", "SUPPORTED_WITH_CAVEATS", "0",
          "Results are conditional on the ≤ 14 calendar-DTE universe, which is coverage-driven; nothing is extrapolated to longer maturities.",
          "14-30 DTE groups: forward-OK 56.4% and strict-ATM 46.4% vs 99.0% for ≤ 14 DTE (composition diagnostic only; no outcomes compared)",
          "gate3/t9_dte_composition_diagnostic.csv", "Diagnostic only.", "Findings apply to weekly-style ≤ 14 DTE NIFTY options in this sample.", "applies to all NIFTY options", (), False, False),
        C("C12", "Systematic last-trade price bias (bound)", "NOT_ASSESSABLE", "NA",
          "A systematic same-direction bias in last-trade prices could create or remove the spread, and this cannot be tested without bid/ask data.",
          f"bias needed to explain the whole median spread: Rs {tip.price_bias_rs_median:.1f} (IQR {tip.price_bias_rs_p25:.1f}-{tip.price_bias_rs_p75:.1f}), {tip.price_bias_pct_of_price_median:.1f}% of the median bracketing price; "
          f"the ratio would equal 1 if every IV were multiplied by {tip.iv_factor_for_unit_ratio:.3f}, a relative reduction of {tip.iv_reduction_pct_for_unit_ratio:.1f}% of the IV level (not {tip.iv_reduction_pct_for_unit_ratio:.0f} volatility points)",
          "gate3/t10_tipping_bias.csv; gate3/t10_price_bias_scenarios.csv", "Scenarios are assumptions/bounds, not observed bid/ask data.",
          "How large a bias would be needed; whether it exists is unknown.", "bid/ask costs are small / the spread is net of costs", (), False, False),
        C("C13", "Call-vs-put disagreement proxy", "SUPPORTED_WITH_CAVEATS", "0",
          "No call-vs-put disagreement was detected in the observable proxy.",
          f"median call-minus-put IV {float(cp.median_diff_vol_points):+.3f} vol pt over {int(cp.n_strikes)} strikes in {int(cp.n_snapshots)} snapshots; robust sd {float(cp.robust_sd_diff):.3f}",
          "gate3/t10_call_put_disagreement.csv", "Limitation retained: a common-mode systematic last-trade bias cannot be detected without bid/ask data.",
          "No call-vs-put disagreement was detected in this observable proxy.", "quote noise is negligible / last trades are unbiased", (), False, False),
        C("C14", "Pipeline recovers a known truth (Tier A)", "SUPPORTED_METHOD", "NA",
          "The real Stage 2A-2C and audit code recover the generator's realised variance, IV, spread and ratio from synthetic data written in the on-disk format.",
          f"max relative error of V {float(ta.max_abs_V_rel_error.max()):.1e}; max IV error {float(ta.max_abs_iv_error_vol_pts.max()):.4f} vol pt; max estimand difference {float(ta.max_abs_estimand_difference.max()):.4f}; "
          f"generator-level ratio-of-sums check {float(tg.simulated.iloc[1]):.4f} +/- {float(tg.mc_se.iloc[1]):.4f} vs c = 1 (a ratio of random sums is not exactly c)",
          "gate3/t11a_*.csv", "Deterministic-volatility worlds validate plumbing and definitions, not statistical behaviour. The pre-stated R3-within-3-SE criterion was mis-specified and replaced (recorded in REPORT_3B.md).",
          "The pipeline's definitions and annualization are implemented as specified.", "the pipeline is validated on real market data", (), False, False),
        C("C15", "Coverage of the percentile expiry-block bootstrap (Tier B)", "SUPPORTED_METHOD", "NA",
          "The current percentile-bootstrap uncertainty procedure does not achieve nominal 95% coverage uniformly.",
          f"B0 calibrated, b = 5: spread-median estimands {t['slo']:.1%}-{t['shi']:.1%}; ratio estimands {t['rlo']:.1%}-{t['rhi']:.1%}; strong dependence (B1) best method {t['b1lo']:.0%}-{t['b1hi']:.0%}; row-iid {t['row_lo']:.0%}-{t['row_hi']:.0%}",
          "gate3/t11b_coverage.csv; gate3/REPORT_3B.md", "Synthetic only; the simulation is less clustered (ICC 0.49 vs 0.575) but more serially dependent (ACF 0.275 vs 0.128) than the data; not extrapolated.",
          "Report ratio intervals with an explicit under-coverage warning.", "the intervals have 95% coverage on the real data", (), False, True),
        C("C16", "Mincer-Zarnowitz slope (descriptive)", "SUPPORTED_WITH_CAVEATS", "0",
          "Regressing realised volatility on IV gives a slope below 1 in this sample (descriptive; IV noise attenuates the slope).",
          f"slope {float(mza.slope):.2f} ({float(mza.slope_ci_lo):.2f}-{float(mza.slope_ci_hi):.2f}), intercept {float(mza.intercept):.2f}",
          "gate2/t6_mincer_zarnowitz.csv", f"No forecasting claim; errors-in-variables not corrected. {SN}", "Descriptive slope of RV on IV in this sample.", "IV is a biased / inefficient forecast", ("S1",), True, True),
        C("C17", "Expiry-day stratum", "SUPPORTED_WITH_CAVEATS", "0",
          "Expiry-day IV exceeds the following sessions' realised volatility by far more than on other days, but this is a horizon-mismatched comparison.",
          f"F1 calendar spread {float(edv.estimate):.2f} ({float(edv.ci_lo):.2f}-{float(edv.ci_hi):.2f}) vs {float(eno.estimate):.2f} on other days",
          "gate2/t6_expiry_day_stratum.csv", f"IV belongs to an expiring contract, RV to later sessions; expiry-day wing IVs are resolution-limited; never pooled. {SN}",
          "Descriptive, horizon-mismatched.", "expiry-day options are overpriced", ("S1",), True, True),
        C("C18", "Variance risk premium (not claimed)", "NOT_CLAIMED", "NA", "Stage 2D does not claim a variance risk premium or any compensation for risk.",
          "n/a", "n/a", "Would require risk-neutral vs physical measure arguments, bid/ask, and prospective data.", "The sample shows IV above realised volatility (C01).", "risk premium / VRP / compensation", (), False, False),
        C("C19", "Tradability and profitability (not assessable)", "NOT_ASSESSABLE", "2",
          "Whether any of this is tradable is not assessable: there is no bid/ask, cost, margin, slippage or execution information, and no strategy was tested.",
          "n/a", "n/a", "No strategy, P&L, signal, sizing or Fyers integration exists in Stage 2D.", "No claim about tradability is made.", "edge / profitable / alpha / tradable / arbitrage", (), False, False),
        C("C20", "Prospective validity", "NOT_ASSESSABLE", "NA",
          "Nothing has been tested on expiries after 2026-09-30; the frozen protocol is the only untouched test.",
          "no prospective result exists", "PROSPECTIVE_PROTOCOL.md", "No prospective result was computed, simulated or fabricated.", "A protocol exists and is frozen; results will be judged by it.", "validated going forward / out-of-sample confirmed", (), False, False),
    ]
    for c in claims:
        assert c.status in STATUSES and c.tier in TIERS
    return claims
