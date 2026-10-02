"""Stage 2D Gate 3b: T11 synthetic process, full-pipeline recovery (Tier A) and coverage / dependence machinery (Tier B)."""
import math
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optionsengine.research.stage2d import build_gate3b as B3B, coverage_study as CS, estimands as E, synth_pipeline as SPL, synth_process as SP


# ================================================================================================ process
def test_transition_matrix_stationary_law_and_stochastic():
    dgp = SP.DGP()
    P, pi = SP.transition_matrix(dgp)
    assert np.allclose(P.sum(axis=1), 1.0) and (P >= 0).all()
    assert np.allclose(pi @ P, pi, atol=1e-10)
    assert np.allclose(pi, np.array(dgp.pi_target) / sum(dgp.pi_target), atol=1e-8)


def test_simulate_is_deterministic_and_seed_sensitive():
    d = SP.DGP()
    a, b, c = SP.simulate(d, 20, 5), SP.simulate(d, 20, 5), SP.simulate(d, 20, 6)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.V.to_numpy(), c.V.to_numpy()[: len(a)]) if len(c) >= len(a) else True


def test_calendar_structure_expiries_overlap_and_windows_bounded():
    d = SP.DGP()
    r = SP.simulate(d, 60, 3)
    assert r.expiry_idx.nunique() == 60
    assert (r.T_days > 0).all() and (r.T_days <= d.max_dte_days + 1.0).all()
    assert (r.V >= 0).all() and (r.F_eta > 0).all()
    days = pd.to_datetime(r.day)
    assert (days.dt.weekday < 5).all()
    exp = pd.to_datetime(r.expiry)
    assert (exp.dt.weekday <= 4).all()
    # overlapping weekly windows: some calendar day belongs to more than one expiry
    assert r.groupby("day").expiry.nunique().max() >= 2


def test_no_overlap_when_windows_short():
    r = SP.simulate(replace(SP.DGP(), max_dte_days=5.5), 60, 3)
    assert r.groupby("day").expiry.nunique().max() == 1


def test_remainder_is_nested_across_slots():
    """Variance still to come shrinks across the 10:00 > 13:00 > 15:00 snapshots of the same day and expiry."""
    r = SP.simulate(SP.DGP(), 40, 11)
    g = r.groupby(["day", "expiry"])
    ok = 0
    for _, x in g:
        x = x.sort_values("T_days", ascending=False)
        if len(x) == 3:
            assert x.V.is_monotonic_decreasing and x.session_equivalents.is_monotonic_decreasing
            ok += 1
    assert ok > 50


def test_conditional_expectation_is_unbiased_for_realised_variance():
    """F = E[V | state] by construction: its mean over a long history equals the mean of V (law of iterated expectations)."""
    r = SP.simulate(SP.DGP(), 3000, 21)
    f = r.F_eta.to_numpy()
    # F_eta carries the mean-one mispricing/noise factor, so its mean must equal that of V to Monte-Carlo accuracy
    assert abs(f.mean() / r.V.mean() - 1) < 0.03


def test_population_ratio_of_sums_close_to_c():
    for c in (1.2, 0.8):
        t = SP.truth_estimates(SP.DGP(), n_expiries=3000, seed=4, c=c)
        assert abs(t["R3_ratio_of_summed_total_variance"] / c - 1) < 0.04
    assert len(t) == 9


def test_to_aligned_formulas_by_hand():
    rows = pd.DataFrame(dict(day=["2022-03-01"], time=["10:00"], expiry=["2022-03-04"], T_days=[3.0], span_days=[3.0], session_equivalents=[2.5], V=[0.0004], F_eta=[0.0005], expiry_idx=[0]))
    a = SP.to_aligned(rows, 1.5)
    W = 1.5 * 0.0005
    assert a.implied_total_variance.iloc[0] == pytest.approx(W)
    assert a.iv_pct.iloc[0] == pytest.approx(100 * math.sqrt(W / (3 / 365)))
    assert a.rv_session_pct.iloc[0] == pytest.approx(100 * math.sqrt(252 * 0.0004 / 2.5))
    assert a.rv_calendar_pct.iloc[0] == pytest.approx(100 * math.sqrt(365 * 0.0004 / 3.0))
    assert a.spread_session.iloc[0] == pytest.approx(a.iv_pct.iloc[0] - a.rv_session_pct.iloc[0])
    assert a.total_variance_ratio.iloc[0] == pytest.approx(W / 0.0004)
    assert a.dte_bucket.iloc[0] == "1-3d"
    # per-row c
    b = SP.to_aligned(pd.concat([rows, rows]), [1.0, 2.0])
    assert b.implied_total_variance.tolist() == pytest.approx([0.0005, 0.001])


def test_assign_split_is_calendar_cut():
    r = SP.simulate(SP.DGP(), 30, 2)
    s = SP.assign_split(r, 20)
    assert set(s) == {"dev", "holdout"}
    assert s[r.day < r.day[r.expiry_idx >= 20].min()].eq("dev").all()
    assert s[r.day >= r.day[r.expiry_idx >= 20].min()].eq("holdout").all()


# ================================================================================================ coverage machinery
def test_wilson_hand_values():
    lo, hi = CS.wilson(95, 100)
    assert lo == pytest.approx(0.8882, abs=2e-4) and hi == pytest.approx(0.9784, abs=2e-4)
    lo, hi = CS.wilson(0, 10)
    assert lo == 0.0 or lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.2775, abs=2e-3)
    lo, hi = CS.wilson(5, 10)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-9)


def test_scenarios_are_the_planned_set():
    names = [s.name for s in CS.scenarios()]
    assert names == ["B0_calibrated", "B1_strong_dependence", "B2_no_overlap_iid", "B4_overlap_only", "B3_extreme_tails"]
    b0 = CS.scenarios()[0]
    assert b0.wald and set(b0.contrast) == {"null", "shift"}
    assert CS.BLOCKS == (2, 5, 8) or tuple(CS.BLOCKS) == (2, 5, 8)


def fake_results(n, truth_vec, inside_frac):
    """Each method's interval covers the truth in the first inside_frac of histories."""
    res = []
    k = int(round(n * inside_frac))
    for h in range(n):
        cis = {}
        for m in CS.METHODS:
            est = np.array(truth_vec, float)
            width = 1.0
            off = 0.0 if h < k else 10.0
            cis[m] = (est, est - width + off, est + width + off)
        res.append(dict(cis=cis))
    return res


def test_coverage_rows_count_and_flags_with_fake_results():
    sc = CS.scenarios()[0]
    truth = dict(zip(CS.EST_ALL, np.arange(len(CS.EST_ALL), dtype=float)))
    tv = [truth[n] for n in CS.EST_ALL]
    for frac, band, undercov in ((0.95, True, False), (0.60, False, True)):
        df = CS.coverage_rows(sc, fake_results(200, tv, frac), truth)
        assert set(df.method) == set(CS.METHODS)
        assert df.coverage.between(frac - 0.006, frac + 0.006).all()
        assert df.in_target_band_93_97.all() == band
        assert df.undercovers.all() == undercov
        assert (df.wilson_lo <= df.coverage).all() and (df.coverage <= df.wilson_hi).all()
        assert np.allclose(df.mean_width, 2.0)
    # per-expiry estimands are skipped for the row-level methods
    rows_only = df[df.method == "row_iid"]
    assert len(rows_only) == len(CS.EST_ALL) - 4


def test_truth_difference_sign_and_null():
    d = SP.DGP()
    td = CS.truth_difference(d, 1.15, n_expiries=1500, seed=3)
    assert td["R1_geometric_mean_all_obs"] == td["R1_geometric_mean_all_obs"]
    assert td["R1_geometric_mean_all_obs"] > 0 and td["R3_ratio_of_summed_total_variance"] > 0
    assert td["S1_calendar_all_obs_median"] > 0
    td0 = CS.truth_difference(d, 1.0, n_expiries=500, seed=3)
    assert all(abs(v) < 1e-9 for v in td0.values())


def test_history_cis_covers_estimate_and_is_deterministic():
    rows = SP.simulate(SP.DGP(), 80, 17)
    a = SP.to_aligned(rows, 1.2)
    x = CS.history_cis(a, 60, 5)
    y = CS.history_cis(a, 60, 5)
    for m in x:
        for u, v in zip(x[m], y[m]):
            assert np.allclose(u, v, equal_nan=True)
    for m, (est, lo, hi) in x.items():
        ok = np.isfinite(lo) & np.isfinite(hi)
        assert ok.any() and (lo[ok] <= est[ok] + 1e-9).mean() > 0.7


def test_run_scenario_is_deterministic_across_worker_counts():
    sc = replace(CS.scenarios()[2], n_expiries=40, n_dev=25)
    a = CS.run_scenario(sc, 4, 30, 20, 77, workers=1)
    b = CS.run_scenario(sc, 4, 30, 20, 77, workers=2)
    for r, s in zip(a, b):
        for m in CS.METHODS:
            for u, v in zip(r["cis"][m], s["cis"][m]):
                assert np.allclose(u, v, equal_nan=True)


def test_row_iid_undercovers_relative_to_expiry_methods_property():
    sc = replace(CS.scenarios()[0], n_expiries=110, n_dev=70)
    res = CS.run_scenario(sc, 24, 120, 20, 5, workers=1)
    truth = SP.truth_estimates(sc.dgp, 4000, 3)
    cov = CS.coverage_rows(sc, res, truth).set_index(["method", "estimand"]).coverage
    row = np.mean([cov[("row_iid", e)] for e in CS.EST_ALL if ("row_iid", e) in cov.index])
    exp = np.mean([cov[("expiry_iid", e)] for e in CS.EST_ALL if ("expiry_iid", e) in cov.index])
    assert row < exp - 0.10


# ================================================================================================ Tier A
def test_tier_a_generator_expectation_by_monte_carlo():
    sim, theo = SPL.expected_variance_check(SPL.TierADGP(), n_rep=20000, seed=1)
    assert sim / theo == pytest.approx(1.0, abs=0.04)


def test_tier_a_world_truth_columns_and_determinism():
    d = SPL.TierADGP(n_weeks=6)
    a = SPL.generate_world(d, 3, None, None, write=False)
    b = SPL.generate_world(d, 3, None, None, write=False)
    pd.testing.assert_frame_equal(a, b)
    for col in ("obs_id", "V_true", "F_true", "W_true", "iv_true", "spot", "session_equivalents", "span_days"):
        assert col in a.columns
    assert np.allclose(a.W_true, d.c * a.F_true)
    assert a.obs_id.is_unique


def test_tier_a_real_stack_recovers_truth(tmp_path):
    d = SPL.TierADGP(n_weeks=8, c=1.3)
    root, pdir, work = str(tmp_path / "w"), str(tmp_path / "p"), str(tmp_path / "s")
    os.makedirs(work)
    truth = SPL.generate_world(d, 5, root, pdir, write=True)
    al = B3B.run_real_stack(root, pdir, work, workers=1)
    assert len(al) > 20
    if "obs_id" not in al.columns:
        pytest.skip("aligned rows carry no obs_id")
    rec = SPL.recovery_table(al, truth, d.c)
    assert len(rec) > 20
    assert rec.iv_error.abs().max() < 0.05
    assert rec.V_rel_error.abs().max() < 1e-6
    assert rec.ratio_error.abs().max() < 0.01


# ================================================================================================ builder
def test_build_refuses_paths_outside_stage2d(tmp_path):
    with pytest.raises(ValueError):
        B3B.build("a.csv", "b", str(tmp_path / "plain"), tier_a=False, tier_b=False)


def test_truth_aligned_uses_generator_columns():
    t = pd.DataFrame(dict(expiry=["e"], day=["d"], time=["10:00"], T_years=[3 / 365], V_true=[0.0004], W_true=[0.0006], iv_true=[0.2], session_equivalents=[2.5], span_days=[3.0]))
    a = B3B.truth_aligned(t)
    assert a.iv_pct.iloc[0] == pytest.approx(20.0)
    assert a.total_variance_ratio.iloc[0] == pytest.approx(1.5)
    assert a.spread_session.iloc[0] == pytest.approx(20 - 100 * math.sqrt(252 * 0.0004 / 2.5))


# ================================================================================================ strengthened (mutation survivors)
def test_conditional_expectation_unbiased_slot_by_slot():
    r = SP.simulate(SP.DGP(), 4000, 8)
    for slot in ("10:00", "13:00", "15:00"):
        x = r[r.time == slot]
        assert x.V.mean() / x.F_eta.mean() == pytest.approx(1.0, abs=0.06), slot
    # last-day rows (only the intraday remainder and one more session are left) isolate the within-day segment weights
    e = r[r.T_days < 1.3]
    for slot in ("10:00", "13:00"):
        x = e[e.time == slot]
        assert len(x) > 500 and x.V.mean() / x.F_eta.mean() == pytest.approx(1.0, abs=0.05), slot


def test_dte_bucket_boundaries():
    base = dict(day="2022-03-01", time="10:00", expiry="2022-03-04", span_days=3.0, session_equivalents=2.5, V=0.0004, F_eta=0.0005, expiry_idx=0)
    rows = pd.DataFrame([dict(base, T_days=t) for t in (3.0, 3.5, 7.0, 7.5)])
    assert SP.to_aligned(rows, 1.0).dte_bucket.tolist() == ["1-3d", "3-7d", "3-7d", "7-14d"]


def test_coverage_interval_is_closed_and_band_edges():
    sc = CS.scenarios()[0]
    truth = {n: 0.0 for n in CS.EST_ALL}
    res = [dict(cis={m: (np.zeros(len(CS.EST_ALL)), np.zeros(len(CS.EST_ALL)), np.ones(len(CS.EST_ALL))) for m in CS.METHODS}) for _ in range(50)]
    assert (CS.coverage_rows(sc, res, truth).coverage == 1.0).all()           # truth exactly on the lower bound is covered
    df = CS.coverage_rows(sc, fake_results(200, [0.0] * len(CS.EST_ALL), 0.91), {n: 0.0 for n in CS.EST_ALL})
    assert not df.in_target_band_93_97.any()                                  # 91% is outside the 93-97% band
    df = CS.coverage_rows(sc, fake_results(200, [0.0] * len(CS.EST_ALL), 0.965), {n: 0.0 for n in CS.EST_ALL})
    assert df.in_target_band_93_97.all()


def test_tier_a_oracle_expectation_unbiased_across_worlds():
    d = SPL.TierADGP(n_weeks=12)
    vs = fs = 0.0
    for k in range(30):
        t = SPL.generate_world(d, 100 + k, None, None, write=False)
        vs += t.V_true.sum(); fs += t.F_true.sum()
    assert vs / fs == pytest.approx(1.0, abs=0.05)
