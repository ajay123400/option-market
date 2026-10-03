"""Stage 2D Gate 2 (T5-T8): contrasts, post-stratification, strata/heterogeneity, Mincer-Zarnowitz, alternative RV, specification curve, builder guards."""
import json
import math
import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from optionsengine.research import build_annualization_audit as baa, build_iv_rv as biv, iv_rv
from optionsengine.research.stage2d import alt_rv as A, build_gate2 as B2, clusters as C, contrasts as K, estimands as E, spec_curve as SC, strata as S
from tests.test_iv_rv import WD, alt as alt_returns, build as build_sessions, make_index
from tests.test_stage2d_gate1 import frame, many_clusters

IST = timezone(timedelta(hours=5, minutes=30))


# ------------------------------------------------------------------------------------------------ helpers
def write_stage2c(tmp_path, df):
    """Baseline tables (as the committed Stage 2C pipeline would write them) for the EXP/hybrid rows of `df`, and the aligned table itself."""
    e = df[(df.horizon == "EXP") & (df.measure == "hybrid")].reset_index(drop=True)
    est = E.build(e).point()
    d = tmp_path / "stage2c" / "annualization_audit"
    d.mkdir(parents=True)
    pd.DataFrame([dict(horizon="EXP", measure="hybrid", dimension="overall", level="all", n_obs=len(e), spread_session_median=est[0], spread_calendar_median=est[2])]).to_csv(d / "original_vs_aligned.csv", index=False)
    pd.DataFrame([dict(measure="hybrid", dimension="overall", level="all", ratio_geometric_mean=est[4], ratio_median=est[6])]).to_csv(d / "expiry_total_variance.csv", index=False)
    al = tmp_path / "aligned.csv"
    df.to_csv(al, index=False)
    return str(al), str(tmp_path / "stage2c")


def level_frame(n_exp=60, m=6, seed=0, shift=0.0, level_probs=(0.25, 0.25, 0.25, 0.25), level_effect=None, split="dev", start=0):
    """Rows with spread/ratio columns, an iv_quartile level and expiry labels; spread mean = level effect + shift."""
    rng = np.random.default_rng(seed)
    levels = np.array(["Q1 low", "Q2", "Q3", "Q4 high"])
    eff = level_effect or {l: 0.0 for l in levels}
    rows = []
    for c in range(n_exp):
        mu = rng.normal(0, 0.5)
        for i in range(m):
            lv = rng.choice(levels, p=level_probs)
            sc = eff[lv] + shift + mu + rng.normal(0, 1.0)
            v = 1e-4
            rows.append(dict(expiry=f"2022-{start + c:05d}", day=f"d{c}-{i}", time="10:00", split=split, iv_quartile=lv, spread_session=sc + 1, spread_calendar=sc,
                             implied_total_variance=v * math.exp(rng.normal(0.2, 0.3)), rv_total_variance=v, total_variance_ratio_valid=True))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ Holm / bootstrap p
def test_holm_known_values_monotone_and_capped():
    adj = K.holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
    np.testing.assert_allclose(K.holm([0.6, 0.7]), [1.0, 1.0])
    assert K.holm([0.2])[0] == 0.2
    p = np.random.default_rng(0).random(20)
    a = K.holm(p)
    assert (a >= p - 1e-15).all() and (a <= 1).all()
    order = np.argsort(p)
    assert (np.diff(a[order]) >= -1e-15).all()


def test_bootstrap_p_value_floor_and_symmetry():
    assert K.boot_p(np.ones(999)) == pytest.approx(2 / 1000)
    assert K.boot_p(np.concatenate([np.ones(500), -np.ones(499)])) == pytest.approx(1.0)
    assert K.boot_p(np.array([np.nan, 1.0, 1.0, 1.0])) == pytest.approx(2 / 4)


# ------------------------------------------------------------------------------------------------ split contrasts
def test_split_contrast_recovers_a_known_shift_and_is_null_without_one():
    dev = level_frame(80, 6, 1, 0.0)
    hold = level_frame(60, 6, 2, 1.0, start=1000, split="holdout")
    con = K.split_contrast(dev, hold, 5, 600, 7).set_index("estimand")
    r = con.loc["S1_calendar_all_obs_median"]
    assert r.difference_holdout_minus_dev == pytest.approx(1.0, abs=0.3)
    assert r.ci_lo > 0 and r.p_two_sided_bootstrap < 0.05
    assert r.dev_clusters == 80 and r.holdout_clusters == 60
    null = K.split_contrast(dev, level_frame(60, 6, 3, 0.0, start=1000, split="holdout"), 5, 600, 7).set_index("estimand").loc["S1_calendar_all_obs_median"]
    assert null.ci_lo < 0 < null.ci_hi and null.p_two_sided_bootstrap > 0.05


def test_split_contrast_is_deterministic_and_guards_small_splits():
    dev, hold = level_frame(40, 4, 1), level_frame(40, 4, 2, start=500, split="holdout")
    a, b = K.split_contrast(dev, hold, 5, 200, 3), K.split_contrast(dev, hold, 5, 200, 3)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(K.split_contrast(dev, hold, 5, 200, 4))
    with pytest.raises(C.ClusterError):
        K.split_contrast(dev, level_frame(20, 4, 2, start=500, split="holdout"), 5, 100, 3)


def test_straddling_expiries_are_listed():
    df = pd.DataFrame(dict(expiry=["e1", "e1", "e2", "e3", "e3"], split=["dev", "holdout", "dev", "holdout", "holdout"]))
    assert K.straddling_expiries(df) == ["e1"]


# ------------------------------------------------------------------------------------------------ post-stratification
def test_weighted_median_hand_values():
    assert K.weighted_median(np.array([3.0, 1.0, 2.0]), np.ones(3)) == 2.0
    assert K.weighted_median(np.array([1.0, 2.0, 3.0]), np.array([5.0, 1.0, 1.0])) == 1.0
    assert K.weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0])) == 3.0


def _ps_cs(df, levels):
    cols = E.cluster_columns(df)
    cols["q"] = df.iv_quartile.map({l: i for i, l in enumerate(levels)}).to_numpy()
    return C.ClusterSet(df.expiry.to_numpy(), cols)


def test_post_stratification_reweights_to_the_target_composition():
    levels = ["Q1 low", "Q2", "Q3", "Q4 high"]
    df = level_frame(60, 10, 5, 0.0, level_probs=(0.1, 0.0, 0.0, 0.9), level_effect={"Q1 low": 0.0, "Q2": 0.0, "Q3": 0.0, "Q4 high": 10.0})
    df.loc[df.iv_quartile == "Q1 low", "spread_calendar"] = 0.0
    df.loc[df.iv_quartile == "Q4 high", "spread_calendar"] = 10.0
    cs = _ps_cs(df, levels)
    raw = float(np.median(df.spread_calendar))
    ps = K.PostStratified(cs, 4, np.array([0.6, 0.0, 0.0, 0.4])).point()
    assert raw == 10.0 and ps[0] == 0.0                                                    # 60% weight on the low level moves the weighted median
    same = K.PostStratified(cs, 4, np.array([(df.iv_quartile == l).mean() for l in levels])).point()
    assert same[0] == pytest.approx(np.median(df.spread_calendar))                          # own composition => identity
    assert same[2] == pytest.approx(math.exp(np.log(df.implied_total_variance / df.rv_total_variance).mean()), rel=1e-12)


def test_post_stratified_contrast_removes_a_pure_level_shift():
    eff = {"Q1 low": 0.0, "Q2": 2.0, "Q3": 4.0, "Q4 high": 6.0}
    dev = level_frame(80, 8, 1, 0.0, level_probs=(0.4, 0.3, 0.2, 0.1), level_effect=eff)
    hold = level_frame(70, 8, 2, 0.0, level_probs=(0.1, 0.2, 0.3, 0.4), level_effect=eff, start=1000, split="holdout")
    raw = K.split_contrast(dev, hold, 5, 400, 1).set_index("estimand").loc["S1_calendar_all_obs_median"]
    ps, comp = K.post_stratified_contrast(dev, hold, "iv_quartile", 5, 400, 1)
    ps = ps.set_index("estimand").loc["S1_calendar_ps"]
    assert raw.ci_lo > 0 and abs(raw.difference_holdout_minus_dev) > 1.5                    # composition shift looks like a spread shift
    assert abs(ps.difference) < 0.6 and ps.ci_lo < 0 < ps.ci_hi                              # ... and disappears after re-weighting
    assert comp.dev_share.sum() == pytest.approx(1.0) and comp.holdout_share.sum() == pytest.approx(1.0)


def test_within_level_contrasts_flag_levels_with_too_few_expiries():
    dev = level_frame(60, 6, 1, level_probs=(0.97, 0.01, 0.01, 0.01))
    hold = level_frame(60, 6, 2, level_probs=(0.97, 0.01, 0.01, 0.01), start=900, split="holdout")
    out = K.within_level_contrasts(dev, hold, "iv_quartile", 5, 150, 1)
    assert out.loc[out.level == "Q1 low", "S1_calendar_diff_ci_lo"].notna().all()
    small = out[out.level == "Q4 high"].iloc[0]
    assert np.isnan(small.S1_calendar_diff_ci_lo) and "point estimates only" in str(small.get("note", ""))


# ------------------------------------------------------------------------------------------------ strata and heterogeneity
def strata_frame(n_exp=60, m=6, seed=0, effects=(0.0, 0.0, 0.0), expiry_effect=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_exp):
        mu = rng.normal(0, expiry_effect) if expiry_effect else 0.0
        for i in range(m):
            lv = int(rng.integers(0, len(effects)))
            sc = effects[lv] + mu + rng.normal(0, 1.0)
            rows.append(dict(expiry=f"2022-{c:05d}", lv=f"L{lv}", spread_session=sc, spread_calendar=sc, implied_total_variance=2e-4 * math.exp(effects[lv] / 10 + mu / 10 + rng.normal(0, 0.2)),
                             rv_total_variance=1e-4, total_variance_ratio_valid=True))
    return pd.DataFrame(rows)


def run_het(df, reps=200, seed=1):
    al = S.Aligned(df)
    lv = sorted(df.lv.unique())
    masks = [(al.df.lv == l).to_numpy() for l in lv]
    tab, draws, views = S.strata_table(al, "lv", lv, masks, 5, reps, seed)
    het = S.heterogeneity_table("lv", lv, draws, views, stat_names=("S1_calendar",))
    return tab, het, draws, views


def test_aligned_views_share_one_cluster_index_space_and_count_rows():
    df = strata_frame(40, 5, 1, (0, 1, 2))
    al = S.Aligned(df)
    masks = [(al.df.lv == f"L{i}").to_numpy() for i in range(3)]
    views = [al.view(m) for m in masks]
    assert sum(v.n_rows for v in views) == len(df)
    np.testing.assert_array_equal(sum(v.cnt for v in views), np.bincount(al.codes, minlength=al.n_clusters))
    seq = np.arange(al.n_clusters)
    assert views[0].stats(seq)[0] == pytest.approx(np.median(df[df.lv == "L0"].spread_calendar))


def test_strata_with_fewer_than_30_expiries_get_point_estimates_only():
    df = strata_frame(60, 6, 2, (0, 0))
    df.loc[df.expiry.isin(sorted(df.expiry.unique())[:5]), "lv"] = "L2"                   # a level confined to 5 expiries
    tab, het, _, _ = run_het(df, 100)
    small = tab[(tab.level == "L2") & (tab.statistic == "S1_calendar")].iloc[0]
    assert small.n_expiries == 5 and np.isnan(small.ci_lo) and "point only" in small.ci_note
    assert "L2" not in het.usable_levels.iloc[0]


def test_heterogeneity_test_is_calibrated_and_has_power():
    reject_null = sum(run_het(strata_frame(60, 6, s, (0, 0, 0)), 150, s)[1].p_value.iloc[0] < 0.05 for s in range(40))
    assert reject_null <= 6                                                                  # nominal 5% of 40 = 2
    reject_alt = sum(run_het(strata_frame(60, 6, 100 + s, (0, 0, 1.5)), 150, s)[1].p_value.iloc[0] < 0.05 for s in range(10))
    assert reject_alt >= 9


def test_joint_draws_respect_covariance_between_strata_that_share_expiries():
    df = strata_frame(80, 6, 3, (0, 0), expiry_effect=3.0)
    _, _, draws, _ = run_het(df, 400, 5)
    assert np.corrcoef(draws[:, 0, 0], draws[:, 1, 0])[0, 1] > 0.5


def test_wald_statistic_matches_a_manual_computation():
    rng = np.random.default_rng(0)
    draws = np.zeros((2000, 3, 4))
    draws[:, 0, 0] = rng.normal(0, 1, 2000)
    draws[:, 1, 0] = draws[:, 0, 0] + rng.normal(0.0, 0.5, 2000)
    draws[:, 2, 0] = rng.normal(0, 1, 2000)
    point = np.zeros((3, 4))
    point[1, 0], point[2, 0] = 1.0, -0.5
    r = S.wald_heterogeneity(draws, point, 0, [0, 1, 2])
    D = np.stack([draws[:, 1, 0] - draws[:, 0, 0], draws[:, 2, 0] - draws[:, 0, 0]], axis=1)
    d = np.array([1.0, -0.5])
    assert r["wald"] == pytest.approx(float(d @ np.linalg.pinv(np.cov(D, rowvar=False)) @ d), rel=1e-10)
    assert r["df"] == 2 and 0 < r["p_value"] <= 1
    assert np.isnan(S.wald_heterogeneity(draws, point, 0, [0])["wald"])


def test_mincer_zarnowitz_recovers_a_known_slope_and_ols_is_exact():
    a, b = S.ols(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 5.0, 7.0]))
    assert (a, b) == (pytest.approx(1.0), pytest.approx(2.0))
    rng = np.random.default_rng(0)
    rows = []
    for c in range(80):
        mu = rng.normal(0, 1)
        for i in range(6):
            x = rng.uniform(8, 20)
            rows.append(dict(expiry=f"e{c:04d}", iv_pct=x, rv_calendar_pct=2.0 + 0.8 * x + mu + rng.normal(0, 1)))
    out = S.mincer_zarnowitz(pd.DataFrame(rows), reps=300, seed=2)
    assert out["slope"] == pytest.approx(0.8, abs=0.1) and out["slope_ci_lo"] < 0.8 < out["slope_ci_hi"]
    lg = S.mincer_zarnowitz(pd.DataFrame(rows), log=True, reps=100, seed=2)
    assert lg["scale"] == "log" and np.isfinite(lg["slope"])
    with pytest.raises(C.ClusterError):
        S.mincer_zarnowitz(pd.DataFrame(rows[:60]), reps=50)


# ------------------------------------------------------------------------------------------------ alternative RV constructions
def random_specs(n_days, seed=0, scale=0.0007):
    rng = np.random.default_rng(seed)
    return [(WD[i], list(rng.normal(0, scale, 375)), float(rng.normal(0, 0.002)) if i else 0.0) for i in range(n_days)]


def window_variance(n_days=8, seed=0):
    rows, _ = build_sessions(random_specs(n_days, seed))
    idx, _ = make_index(rows, WD[:n_days])
    return idx, A.WindowVariance(idx, dev_end=WD[n_days - 1])


@pytest.mark.parametrize("hhmm", ["10:00", "13:00", "15:00"])
def test_one_minute_alt_rv_reproduces_the_stage2c_target_exactly(hhmm):
    idx, W = window_variance()
    exp = iv_rv.expiry_aligned_target(idx, WD[1], hhmm, WD[5], "hybrid")
    v = W.variance(WD[1], hhmm, WD[5], 1)
    assert v == pytest.approx(exp.total_variance, rel=1e-12)
    assert W.session_equivalents(WD[1], hhmm, WD[5]) == pytest.approx(exp.session_equivalents, abs=1e-15)


def test_sampled_variance_hand_values_for_a_known_path():
    rows, cl = build_sessions([(WD[0], [0.001] * 375, 0.0), (WD[1], [0.002] * 375, 0.0)])
    idx, _ = make_index(rows, WD[:2])
    W = A.WindowVariance(idx, dev_end=WD[1])
    closes = np.array(cl[WD[0]][1])
    s = 45
    # k = 5: closes at 45, 50, ..., 370 then 374 (the final step is shortened)
    pr = closes[list(range(45, 374, 5)) + [374]]
    part = float(np.sum(np.diff(np.log(pr)) ** 2))
    assert A.partial_var_sampled(closes, s, 5) == pytest.approx(part, rel=1e-12)
    start, cl1 = cl[WD[1]]
    P = np.concatenate([[start], cl1])
    grid = P[::5]
    sess = float(np.sum(np.diff(np.log(grid)) ** 2))
    assert sess == pytest.approx(A.sampled_intraday_var(P, 5), rel=1e-12)
    assert W.variance(WD[0], "10:00", WD[1], 5) == pytest.approx(part + sess + 0.0, rel=1e-9)
    with pytest.raises(ValueError):
        A.sampled_intraday_var(P, 7)


def test_alt_rv_ignores_everything_at_or_before_the_snapshot_and_sees_the_future():
    base_specs = random_specs(6, 3)
    rows, _ = build_sessions(base_specs)
    idx, W0 = make_index(rows, WD[:6])[0], None
    W0 = A.WindowVariance(idx, dev_end=WD[5])
    s = iv_rv.slot_of("13:00")
    past = list(base_specs[0][1])
    for i in range(0, s + 1):
        past[i] = 0.03 if i % 2 == 0 else -0.03                                         # violent past, including the snapshot bar
    rows2, _ = build_sessions([(WD[0], past, 0.0)] + base_specs[1:])
    idx2, _ = make_index(rows2, WD[:6])
    W2 = A.WindowVariance(idx2, dev_end=WD[5])
    fut = list(base_specs[0][1])
    fut[s + 7] = 0.04                                                                    # a move strictly after the snapshot
    rows3, _ = build_sessions([(WD[0], fut, 0.0)] + base_specs[1:])
    idx3, _ = make_index(rows3, WD[:6])
    W3 = A.WindowVariance(idx3, dev_end=WD[5])
    for k in (1, 5, 15):
        v0 = W0.variance(WD[0], "13:00", WD[4], k)
        assert W2.variance(WD[0], "13:00", WD[4], k) == pytest.approx(v0, rel=1e-9)
        assert W3.variance(WD[0], "13:00", WD[4], k) != pytest.approx(v0, rel=1e-6)


def test_flat_profile_reduces_to_uniform_and_dev_profile_is_a_distribution():
    flat = np.full(375, 1 / 375)
    for s in (45, 225, 345):
        assert A.remainder_weight(flat, s) == pytest.approx((374 - s) / 375, abs=1e-15)
    idx, W = window_variance(8, 5)
    assert W.profile.sum() == pytest.approx(1.0) and (W.profile >= 0).all()
    u = W.session_equivalents(WD[1], "10:00", WD[5], "uniform")
    p = W.session_equivalents(WD[1], "10:00", WD[5], "profile")
    assert u - p == pytest.approx(329 / 375 - A.remainder_weight(W.profile, 45), abs=1e-12)


def test_unavailable_windows_return_none_not_filled():
    idx, W = window_variance(5)
    assert W.variance(WD[3], "10:00", WD[3], 1) is None                                  # expiry day snapshot
    assert W.variance(WD[3], "10:00", WD[2], 1) is None                                  # expiry before snapshot
    assert W.variance(date(2030, 1, 1), "10:00", WD[2], 1) is None


def aligned_row(**kw):
    base = dict(iv_pct=20.0, rv_total_variance=3e-4, session_equivalents=3.0, span_days=4.0, implied_total_variance=0.04 * 4.0 / 365, rv_session_pct=0.0, rv_calendar_pct=0.0,
                spread_session=0.0, spread_calendar=0.0, total_variance_ratio_valid=True, total_variance_ratio=1.0, total_variance_diff=0.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_recompute_annualization_and_calendar_year_conversions():
    df = aligned_row()
    a = A.recompute(df).iloc[0]
    assert a.rv_session_pct == pytest.approx(100 * math.sqrt(252 * 3e-4 / 3.0))
    assert a.rv_calendar_pct == pytest.approx(100 * math.sqrt(365 * 3e-4 / 4.0))
    s250 = A.recompute(df, session_days=250.0).iloc[0]
    assert s250.rv_session_pct == pytest.approx(a.rv_session_pct * math.sqrt(250 / 252))
    assert s250.rv_calendar_pct == pytest.approx(a.rv_calendar_pct)                     # calendar basis unaffected by the session-day count
    c = A.recompute(df, calendar_days=365.25).iloc[0]
    f = math.sqrt(365.25 / 365)
    assert c.iv_pct == pytest.approx(20.0 * f) and c.rv_calendar_pct == pytest.approx(a.rv_calendar_pct * f)
    assert c.spread_calendar == pytest.approx(a.spread_calendar * f, rel=1e-12) if a.spread_calendar else True
    assert c.total_variance_ratio == pytest.approx(a.total_variance_ratio, rel=1e-14)    # IV^2 T invariant => ratio unchanged
    z = A.recompute(df, V=np.array([0.0])).iloc[0]
    assert z.rv_calendar_pct == 0.0 and not z.total_variance_ratio_valid and math.isnan(z.total_variance_ratio)


# ------------------------------------------------------------------------------------------------ specification curve
class StubWindow:
    """Deterministic stand-in for WindowVariance: V scales with k so sampling specs differ from the baseline."""

    def variance(self, day, hhmm, expiry, k):
        return 1e-4 * (1 + 0.1 * (k - 1) / 14) * (1 + (hash((day.day, expiry.day)) % 3) * 0.01 / 1000)

    def session_equivalents(self, day, hhmm, expiry, weighting="uniform"):
        return 3.0 if weighting == "uniform" else 2.5


def gate2_aligned(n_exp=130, seed=0, with_f=True):
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2022, 1, 6)
    for k in range(n_exp):
        exp = start + timedelta(days=7 * k)
        end = datetime(exp.year, exp.month, exp.day, 15, 30, tzinfo=IST)
        mu = rng.normal(2.0, 1.0)
        for off in (12, 10, 8, 6, 5, 4, 3, 2, 1, 0):
            day = exp - timedelta(days=off)
            if day.weekday() >= 5:
                continue
            for tm in ("10:00", "13:00", "15:00"):
                h, m = (int(x) for x in tm.split(":"))
                obs = datetime(day.year, day.month, day.day, h, m, tzinfo=IST) + timedelta(seconds=60)
                T = (end - obs).total_seconds() / (365 * 86400)
                iv = 12 + rng.normal(0, 2)
                span = T * 365
                for hz in ("EXP", "F5"):
                    if hz == "EXP" and off == 0:
                        continue
                    if hz == "F5" and not with_f:
                        continue
                    v = abs(rng.normal(2e-4, 5e-5)) * (T if hz == "EXP" else 5 / 252) * 20 + 1e-6
                    if hz == "EXP":
                        w = (iv / 100) ** 2 * T
                        se = max(span * 252 / 365, 0.2)
                        sp = span
                    else:
                        w = (iv / 100) ** 2 * 7 / 365
                        se, sp = 5.0, 7.0
                    rv_s = 100 * math.sqrt(252 * v / se)
                    rv_c = 100 * math.sqrt(365 * v / sp)
                    rows.append(dict(horizon=hz, measure="hybrid", expiry=exp.isoformat(), day=day.isoformat(), time=tm, split="dev" if day < date(2023, 9, 1) else "holdout",
                                     expiry_day=(off == 0), T_days=T * 365, iv_pct=iv, span_days=sp, session_equivalents=se, target_start_ts=obs.isoformat(), target_end_ts=end.isoformat(),
                                     rv_total_variance=v, implied_total_variance=w, rv_session_pct=rv_s, rv_calendar_pct=rv_c, spread_session=iv - rv_s + mu, spread_calendar=iv - rv_c + mu,
                                     total_variance_ratio_valid=True, total_variance_ratio=w / v, total_variance_diff=w - v,
                                     dte_bucket="1-3d" if T * 365 < 3 else ("3-7d" if T * 365 < 7 else "7-14d"), weekend_in_window=bool(day.weekday() + (exp - day).days >= 5),
                                     recent_rv_regime=str(rng.choice(["low-vol", "mid-vol", "high-vol", "unknown"])), iv_quartile=str(rng.choice(["Q1 low", "Q2", "Q3", "Q4 high"]))))
    return pd.DataFrame(rows)


def test_estimate_spec_baseline_matches_gate1_estimates_and_marks_the_null():
    df = gate2_aligned(60, 1, with_f=False)
    base = SC.exp_hybrid(df)
    cur = SC.estimate_spec(SC.Spec("baseline", "baseline", "x", base), 5, 100, 1).set_index("estimand")
    est = E.build(base).point()
    for i, n in enumerate(E.EST_NAMES):
        assert cur.loc[n, "estimate"] == pytest.approx(est[i])
        assert cur.loc[n, "null_value"] == (0.0 if n.startswith("S") else 1.0)
    assert cur.n_rows.iloc[0] == len(base)


def test_build_specs_lists_every_preregistered_specification_and_filters_correctly():
    df = gate2_aligned(90, 2, with_f=False)
    base = SC.exp_hybrid(df)
    variants = {n: base.copy() for n in ("fwd_3bps", "fwd_8bps", "fwd_12bps", "age_2min", "age_15min", "dte_le21")}
    rates = {"5.5": base.copy(), "7.5": base.copy()}
    specs = SC.build_specs(df, StubWindow(), variants, rates)
    names = [s.name for s in specs]
    for need in ("baseline", "session_days_250", "session_days_256", "calendar_days_365.25", "rv_sampling_5min", "rv_sampling_15min", "partial_session_profile", "rate_5.5pct", "rate_7.5pct",
                 "factorial_rate6.5_k1", "factorial_rate5.5_k15", "factorial_rate7.5_k5", "fwd_3bps", "fwd_12bps", "age_2min", "age_15min", "dte_le7", "dte_le21",
                 "drop_expiry_year_2021", "drop_expiry_year_2026", "drop_time_1000", "drop_time_1300", "drop_time_1500"):
        assert need in names, need
    assert len(names) == len(set(names))
    by = {s.name: s for s in specs}
    assert (by["dte_le7"].df.T_days <= 7).all() and len(by["dte_le7"].df) < len(base)
    edge = df.copy()
    idx = edge.index[(edge.horizon == "EXP")][:5]
    edge.loc[idx, "T_days"] = [6.9, 7.0, 7.1, 7.9, 8.1]
    cut = {sp.name: sp for sp in SC.build_specs(edge, StubWindow(), {}, {})}["dte_le7"].df
    assert len(cut[cut.T_days.isin([6.9, 7.0])]) == 2 and not cut.T_days.isin([7.1, 7.9, 8.1]).any()      # boundary: <= 7 keeps exactly 7.0
    assert not by["drop_time_1000"].df.time.eq("10:00").any() and len(by["drop_time_1000"].df) < len(base)
    assert not by["drop_expiry_year_2021"].df.expiry.str.startswith("2021").any()
    assert by["factorial_rate6.5_k1"].df.equals(base)
    s250 = by["session_days_250"].df
    np.testing.assert_allclose(s250.rv_session_pct, base.rv_session_pct * math.sqrt(250 / 252), rtol=1e-12)
    assert len(by["rv_sampling_5min"].df) == len(base)


def test_build_specs_without_alt_rv_skips_sampling_specs_only():
    df = gate2_aligned(60, 3, with_f=False)
    names = [s.name for s in SC.build_specs(df, None, {}, {})]
    assert "rv_sampling_5min" not in names and "partial_session_profile" not in names and "session_days_250" in names and "factorial_rate6.5_k1" in names


def test_robustness_reading_logic():
    def curve(vals):
        rows = []
        for spec, (a, b) in vals.items():
            for n, est in (("S1_calendar_all_obs_median", a), ("R1_geometric_mean_all_obs", b)):
                null = 0.0 if n.startswith("S") else 1.0
                rows.append(dict(spec=spec, estimand=n, estimate=est, null_inside=(abs(est - null) < 0.2)))
        return pd.DataFrame(rows)

    good = SC.robustness_reading(curve({"baseline": (1.8, 1.3), "a": (1.5, 1.25), "b": (2.2, 1.4)}))
    assert good.sign_same_as_baseline_in_every_spec.all() and (good.share_intervals_excluding_null == 1.0).all()
    assert good.set_index("estimand").loc["S1_calendar_all_obs_median", "widest_departure_spec"] == "b"
    assert "not sensitive" in good.reading.iloc[0]
    bad = SC.robustness_reading(curve({"baseline": (1.8, 1.3), "a": (-0.4, 0.95)}))
    assert not bad.sign_same_as_baseline_in_every_spec.any() and "sign changes" in bad.reading.iloc[0]


def test_variant_pipelines_refuse_paths_outside_stage2d_and_restore_the_dte_constant(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="stage2d"):
        SC.aligned_from_stage2a("x", str(tmp_path / "elsewhere"), "spot", "pdir")
    with pytest.raises(ValueError, match="stage2d"):
        SC.stage2a_rate_variant("root", str(tmp_path / "elsewhere"), 0.05)
    seen = {}

    def boom(*a, **k):
        seen["dte"] = biv.DTE_MAX
        raise RuntimeError("stop")

    monkeypatch.setattr(biv, "build", boom)
    before = biv.DTE_MAX
    with pytest.raises(RuntimeError):
        SC.aligned_from_stage2a("x", str(tmp_path / "stage2d" / "v"), "spot", "pdir", dte_max=21)
    assert seen["dte"] == 21 and biv.DTE_MAX == before


def test_cached_variant_is_not_rebuilt(tmp_path, monkeypatch):
    d = tmp_path / "stage2d" / "v" / "annualization_audit"
    d.mkdir(parents=True)
    pd.DataFrame(dict(a=[1, 2])).to_csv(d / "aligned_observations.csv", index=False)
    monkeypatch.setattr(biv, "build", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rebuilt")))
    assert len(SC.aligned_from_stage2a("x", str(tmp_path / "stage2d" / "v"), "spot", "pdir")) == 2


# ------------------------------------------------------------------------------------------------ builder
def test_gate2_builder_refuses_outputs_outside_stage2d(tmp_path):
    al, s2c = write_stage2c(tmp_path, gate2_aligned(40, 1, with_f=False))
    with pytest.raises(ValueError, match="stage2d"):
        B2.build(al, s2c, str(tmp_path / "gate2"), reps=20, grid_reps=20, variants_provider=lambda: ({}, {}), use_alt_rv=False)


def test_gate2_builder_end_to_end_is_deterministic_and_writes_only_under_the_output_directory(tmp_path):
    df = gate2_aligned(130, 4)
    al, s2c = write_stage2c(tmp_path, df)
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("frozen design")
    before = sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / "stage2c").rglob("*"))
    o1, o2 = tmp_path / "stage2d" / "g1", tmp_path / "stage2d" / "g2"
    kw = dict(reps=60, grid_reps=40, variants_provider=lambda: ({}, {}), prereg=str(prereg), use_alt_rv=False)
    m1 = B2.build(al, s2c, str(o1), **kw)
    B2.build(al, s2c, str(o2), **kw)
    assert before == sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / "stage2c").rglob("*"))
    for need in ("t5_split_contrasts.csv", "t5_post_stratified_contrast.csv", "t5_iv_quartile_composition.csv", "t5_within_iv_quartile_contrasts.csv", "t5_straddling_expiries.csv",
                 "forking_paths_ledger.csv", "t6_strata_estimates.csv", "t6_heterogeneity_tests.csv", "t6_mincer_zarnowitz.csv", "t6_expiry_day_stratum.csv",
                 "t7_t8_specification_curve.csv", "t7_t8_robustness_reading.csv", "block_length_sensitivity.csv", "run_metadata.json"):
        assert (o1 / need).exists(), need
    for f in sorted(os.listdir(o1)):
        if f.endswith(".csv"):
            pd.testing.assert_frame_equal(pd.read_csv(o1 / f), pd.read_csv(o2 / f))
    assert m1["preregistration_sha256"] and len(m1["preregistration_sha256"]) == 64
    con = pd.read_csv(o1 / "t5_split_contrasts.csv")
    fam = con[con.family_A]
    assert len(fam) == 4 and (fam.p_holm_family_A >= fam.p_two_sided_bootstrap - 1e-12).all()
    het = pd.read_csv(o1 / "t6_heterogeneity_tests.csv")
    assert len(het) == 12 and het.family_B.all() and (het.p_holm_family_B >= het.p_value - 1e-12).all()
    assert "unknown" not in " ".join(het[het.dimension == "recent_rv_regime"].usable_levels)
    bs = pd.read_csv(o1 / "block_length_sensitivity.csv")
    assert set(bs.block) == {2, 8} and set(bs.analysis) == {"dev_vs_holdout", "pooled_baseline"}
    w = lambda b: float(bs[(bs.block == b) & (bs.analysis == "pooled_baseline") & (bs.estimand == "S1_calendar_all_obs_median")].eval("ci_hi - ci_lo").iloc[0])
    assert w(8) >= w(2) * 0.9                                                              # longer blocks do not give (much) narrower intervals
    led = pd.read_csv(o1 / "forking_paths_ledger.csv")
    assert (led.results_seen_when_decided == "yes").sum() >= 3 and len(led) >= 15
    sc = pd.read_csv(o1 / "t7_t8_specification_curve.csv")
    assert (sc.spec == "baseline").sum() == len(E.EST_NAMES)


def test_forking_paths_ledger_declares_choices_made_after_results_were_seen():
    seen = [r[0] for r in B2.LEDGER if r[3] == "yes"]
    assert any("Calendar-basis" in s for s in seen) and any("Population for Stage 2D" in s for s in seen)
    assert all(len(r) == 5 for r in B2.LEDGER)
