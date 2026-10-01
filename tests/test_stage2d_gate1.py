"""Stage 2D Gate 1 (T1-T4): estimands, cluster/block resampling, dependence diagnostics, ratio inference, influence, synthetic dependence studies."""
import json
import math
import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from optionsengine.research.stage2d import build_gate1 as B, clusters as C, dependence as D, estimands as E, influence as I, ratio_inference as R

IST = timezone(timedelta(hours=5, minutes=30))


# ------------------------------------------------------------------------------------------------ helpers
def frame(spec):
    """spec: {expiry_label: [(spread_session, spread_calendar, W_imp, V), ...]} -> DataFrame with the columns Estimands needs."""
    rows = []
    for lab, rr in spec.items():
        for i, (ss, sc, w, v) in enumerate(rr):
            rows.append(dict(expiry=lab, day=f"{lab}-{i}", spread_session=ss, spread_calendar=sc, implied_total_variance=w, rv_total_variance=v,
                             total_variance_ratio_valid=v > 0))
    return pd.DataFrame(rows)


def many_clusters(n=40, m=3, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    spec = {}
    for c in range(n):
        spec[f"2022-01-{c:04d}"] = [(rng.normal(shift, 1), rng.normal(shift, 1), math.exp(rng.normal(0.2, 0.3)) * 1e-4, 1e-4) for _ in range(m)]
    return frame(spec)


def make_est(df, **kw):
    return E.build(df, **kw)


# ------------------------------------------------------------------------------------------------ estimands (hand values)
def test_estimands_hand_values_and_separate_definitions():
    spec = {"a": [(1.0, 0.5, 2.0, 1.0), (3.0, 1.5, 4.0, 1.0), (5.0, 2.5, 3.0, 1.0)],         # ratios 2, 4, 3
            "b": [(10.0, 9.0, 1.0, 1.0)],                                                       # ratio 1
            "c": [(2.0, 4.0, 8.0, 2.0), (4.0, 6.0, 6.0, 2.0)]}                                  # ratios 4, 3
    est = make_est(frame(spec), min_clusters=1)
    t = dict(zip(E.EST_NAMES, est.point()))
    assert t["S1_session_all_obs_median"] == pytest.approx(3.5)                                   # median of 1,3,5,10,2,4
    assert t["S2_session_median_of_expiry_medians"] == pytest.approx(np.median([3.0, 10.0, 3.0]))
    assert t["S1_calendar_all_obs_median"] == pytest.approx(np.median([0.5, 1.5, 2.5, 9.0, 4.0, 6.0]))
    assert t["S2_calendar_median_of_expiry_medians"] == pytest.approx(np.median([1.5, 9.0, 5.0]))
    r = np.array([2, 4, 3, 1, 4, 3.0])
    assert t["R1_geometric_mean_all_obs"] == pytest.approx(math.exp(np.log(r).mean()))
    per = [np.log([2, 4, 3]).mean(), 0.0, np.log([4, 3]).mean()]
    assert t["R1e_geometric_mean_expiry_weighted"] == pytest.approx(math.exp(np.mean(per)))
    assert t["R2_median_ratio_all_obs"] == pytest.approx(np.median(r))
    assert t["R2e_median_of_expiry_median_ratios"] == pytest.approx(np.median([3.0, 1.0, 3.5]))
    assert t["R3_ratio_of_summed_total_variance"] == pytest.approx((2 + 4 + 3 + 1 + 8 + 6) / (3 + 1 + 4))
    # the all-observation and per-expiry estimands are genuinely different quantities on unbalanced clusters
    assert t["S1_session_all_obs_median"] != t["S2_session_median_of_expiry_medians"]
    assert t["S1_calendar_all_obs_median"] != t["S2_calendar_median_of_expiry_medians"]
    assert t["R3_ratio_of_summed_total_variance"] != t["R1_geometric_mean_all_obs"]


def test_zero_realized_variance_rows_leave_log_and_median_ratio_but_not_the_sums():
    spec = {"a": [(1.0, 1.0, 2.0, 1.0), (1.0, 1.0, 5.0, 0.0)], "b": [(1.0, 1.0, 3.0, 1.0)]}
    t = dict(zip(E.EST_NAMES, make_est(frame(spec), min_clusters=1).point()))
    assert t["R1_geometric_mean_all_obs"] == pytest.approx(math.exp(np.mean(np.log([2.0, 3.0]))))
    assert t["R2_median_ratio_all_obs"] == pytest.approx(2.5)
    assert t["R3_ratio_of_summed_total_variance"] == pytest.approx((2 + 5 + 3) / (1 + 0 + 1))      # documented: sums use every row


def test_row_level_and_date_clusters_have_no_per_expiry_estimands():
    est = make_est(many_clusters(10, 5), row_level=True, min_clusters=10)
    t = est.point()
    assert np.isnan(t[[1, 3, 5, 7]]).all() and np.isfinite(t[[0, 2, 4, 6, 8]]).all()
    by_date = make_est(many_clusters(40, 3, 6), cluster_col="day")                           # clusters are dates: a "median of cluster medians" would be per DATE
    td = by_date.point()
    assert np.isnan(td[[1, 3, 5, 7]]).all() and np.isfinite(td[[0, 2, 4, 6, 8]]).all()


def test_estimands_do_not_depend_on_row_order():
    df = many_clusters(40, 4, 3)
    a = make_est(df).point()
    b = make_est(df.sample(frac=1, random_state=1).reset_index(drop=True)).point()
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


# ------------------------------------------------------------------------------------------------ clusters: determinism and failure modes
def test_bootstrap_is_deterministic_for_a_seed_and_differs_across_seeds():
    est = make_est(many_clusters(40, 3, 1))
    a = C.bootstrap(est.cs, est.evaluate, "moving_block", 4, 60, 7, k=9)
    b = C.bootstrap(est.cs, est.evaluate, "moving_block", 4, 60, 7, k=9)
    c = C.bootstrap(est.cs, est.evaluate, "moving_block", 4, 60, 8, k=9)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_bootstrap_is_invariant_to_input_row_order():
    df = many_clusters(40, 3, 2)
    e1, e2 = make_est(df), make_est(df.sample(frac=1, random_state=5).reset_index(drop=True))
    d1 = C.bootstrap(e1.cs, e1.evaluate, "stationary", 3, 40, 11, k=9)
    d2 = C.bootstrap(e2.cs, e2.evaluate, "stationary", 3, 40, 11, k=9)
    np.testing.assert_allclose(d1, d2, atol=1e-12)


def test_fewer_than_min_clusters_raises_and_exactly_min_is_accepted():
    with pytest.raises(C.ClusterError, match="29 clusters"):
        make_est(many_clusters(29, 3))
    make_est(many_clusters(30, 3))
    with pytest.raises(C.ClusterError):
        C.bootstrap_series(np.arange(10.0), "iid", None, 10, 1)


def test_block_longer_than_cluster_count_or_nonpositive_raises():
    est = make_est(many_clusters(30, 3))
    with pytest.raises(C.ClusterError, match="exceeds"):
        C.bootstrap(est.cs, est.evaluate, "moving_block", 31, 5, 1, k=9)
    with pytest.raises(C.ClusterError, match=">= 1"):
        C.bootstrap(est.cs, est.evaluate, "stationary", 0, 5, 1, k=9)
    with pytest.raises(C.ClusterError, match="unknown"):
        C.bootstrap(est.cs, est.evaluate, "wild", 3, 5, 1, k=9)
    with pytest.raises(C.ClusterError):
        C.ClusterSet([], {"x": np.array([])})


def test_whole_clusters_are_kept_together_in_every_replicate():
    est = make_est(many_clusters(35, 3, 4))
    cs = est.cs
    rng = np.random.default_rng(3)
    for scheme in C.SCHEMES:
        seq = C.draw_clusters(scheme, rng, cs.n_clusters, 4 if scheme != "iid" else None)
        rows = cs.rows_for(seq)
        assert len(rows) == int(cs.counts[seq].sum())
        np.testing.assert_array_equal(cs.cluster_of_row[rows], np.repeat(seq, cs.counts[seq]))      # clusters intact, repeated whole


def test_moving_block_sequences_are_runs_of_consecutive_clusters():
    rng = np.random.default_rng(1)
    seq = C.draw_clusters("moving_block", rng, 50, 5)
    assert len(seq) == 50
    for s in range(0, 50, 5):
        blk = seq[s: s + 5]
        assert all((blk[i + 1] - blk[i]) % 50 == 1 for i in range(len(blk) - 1))


def test_stationary_mean_block_length_matches_the_target():
    rng = np.random.default_rng(2)
    lens = []
    for _ in range(400):
        seq = C.draw_clusters("stationary", rng, 200, 5)
        brk = np.flatnonzero((seq[1:] - seq[:-1]) % 200 != 1)
        lens.append(200 / (len(brk) + 1))
    assert 4.3 < np.mean(lens) < 5.7


def test_row_level_design_needs_enough_rows_like_any_other():
    with pytest.raises(C.ClusterError):
        make_est(many_clusters(5, 3), row_level=True)                                           # 15 row-clusters < 30


# ------------------------------------------------------------------------------------------------ dependence diagnostics
def ar1(n, phi, seed, burn=200):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def test_acvf_acf_hand_values():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    g = D.acvf(x, 2)
    assert g[0] == pytest.approx(1.25)
    d = x - 2.5
    assert g[1] == pytest.approx((d[0] * d[1] + d[1] * d[2] + d[2] * d[3]) / 4)
    assert D.acf(x, 1)[1] == pytest.approx(g[1] / g[0])
    assert D.acf(ar1(4000, 0.7, 1), 1)[1] == pytest.approx(0.7, abs=0.05)


def test_chi2_survival_function_known_quantiles():
    for q, df in ((3.841458820694124, 1), (5.991464547107979, 2), (18.307038053275146, 10), (31.410432844230918, 20)):
        assert D.chi2_sf(q, df) == pytest.approx(0.05, abs=1e-9)
    assert D.chi2_sf(0.0, 5) == 1.0
    assert D.chi2_sf(200.0, 5) < 1e-30
    scipy_stats = pytest.importorskip("scipy.stats")
    for q, df in ((0.7, 3), (12.3, 7), (44.0, 20), (9.1, 1)):
        assert D.chi2_sf(q, df) == pytest.approx(scipy_stats.chi2.sf(q, df), rel=1e-9)


def test_ljung_box_detects_serial_dependence_but_not_noise():
    assert D.ljung_box(ar1(260, 0.0, 3), 10)[1] > 0.05
    assert D.ljung_box(ar1(260, 0.6, 3), 10)[1] < 0.001


def test_icc_balanced_hand_example_and_extremes():
    x = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])
    g = np.array(list("aabbcc"))
    r = D.icc_oneway(x, g)
    assert r["icc"] == pytest.approx(1.0) and r["n_clusters"] == 3
    assert r["design_effect"] == pytest.approx(1 + (2 - 1) * 1.0) and r["n_effective"] == pytest.approx(3.0)
    rng = np.random.default_rng(0)
    noise = D.icc_oneway(rng.normal(size=4000), np.repeat(np.arange(400), 10))
    assert abs(noise["icc"]) < 0.05
    mixed = D.icc_oneway(np.repeat(rng.normal(size=400), 10) + rng.normal(size=4000), np.repeat(np.arange(400), 10))
    assert mixed["icc"] == pytest.approx(0.5, abs=0.06)
    with pytest.raises(ValueError):
        D.icc_oneway(np.arange(5.0), np.zeros(5))


def test_politis_white_block_length_grows_with_dependence():
    iid = D.politis_white(ar1(500, 0.0, 5))
    dep = D.politis_white(ar1(500, 0.8, 5))
    assert iid["b_circular"] < 3.0
    assert dep["b_circular"] > 3 * iid["b_circular"] and dep["b_circular"] <= dep["b_max"]
    assert dep["b_stationary"] > 1.0


def test_newey_west_inflates_under_positive_autocorrelation_only():
    iid = D.newey_west(ar1(2000, 0.0, 6))
    dep = D.newey_west(ar1(2000, 0.7, 6))
    assert 0.6 < iid["variance_inflation"] < 1.5
    assert dep["variance_inflation"] > 2.0
    assert dep["nw_se"] > dep["iid_se"]
    assert iid["lags"] == int(math.floor(4 * (2000 / 100) ** (2 / 9)))


def mk_windows(spec):
    rows = []
    for exp, start, end in spec:
        rows.append(dict(expiry=exp, target_start_ts=start, target_end_ts=end))
    return pd.DataFrame(rows)


def test_window_overlap_hand_values():
    w = mk_windows([("e1", "2025-01-01T10:01:00+05:30", "2025-01-09T15:30:00+05:30"), ("e1", "2025-01-03T10:01:00+05:30", "2025-01-09T15:30:00+05:30"),
                    ("e2", "2025-01-06T10:01:00+05:30", "2025-01-16T15:30:00+05:30"), ("e3", "2025-01-20T10:01:00+05:30", "2025-01-23T15:30:00+05:30")])
    win = D.expiry_windows(w)
    assert list(win.expiry) == ["e1", "e2", "e3"] and int(win.n_rows.iloc[0]) == 2
    ov = D.window_overlap(win)
    e1_len = (datetime(2025, 1, 9, 15, 30, tzinfo=IST) - datetime(2025, 1, 1, 10, 1, tzinfo=IST)).total_seconds() / 86400
    share = ((datetime(2025, 1, 9, 15, 30, tzinfo=IST) - datetime(2025, 1, 6, 10, 1, tzinfo=IST)).total_seconds() / 86400) / e1_len
    assert ov.overlap_with_next_share.iloc[0] == pytest.approx(share)
    assert ov.overlap_with_next_days.iloc[2] == 0.0 and ov.n_overlapping_expiries.tolist() == [1, 1, 0]


def snap_frame():
    rows = []
    start = date(2025, 1, 2)                                                                  # Thursday expiries, weekly
    for k in range(6):
        exp = start + timedelta(days=7 * k)
        for off in (10, 6, 5, 3, 1):
            day = exp - timedelta(days=off)
            if day.weekday() >= 5:
                continue
            for tm in ("10:00", "13:00"):
                h, m = (int(x) for x in tm.split(":"))
                obs = datetime(day.year, day.month, day.day, h, m, tzinfo=IST) + timedelta(seconds=60)
                end = datetime(exp.year, exp.month, exp.day, 15, 30, tzinfo=IST)
                rows.append(dict(expiry=exp.isoformat(), day=day.isoformat(), time=tm, T_days=(end - obs).total_seconds() / 86400,
                                 target_start_ts=obs.isoformat(), target_end_ts=end.isoformat(), spread_session=float(k), spread_calendar=float(k)))
    return pd.DataFrame(rows)


def test_nonoverlap_subsample_rule_is_outcome_free_and_picks_largest_T_up_to_six_days():
    f = snap_frame()
    kept, dropped = D.nonoverlap_subsample(f)
    assert (kept.time == "10:00").all() and (kept.T_days <= 6.0).all() and kept.expiry.is_unique
    for exp, g in kept.groupby("expiry"):
        allowed = f[(f.expiry == exp) & (f.time == "10:00") & (f.T_days <= 6.0)]
        assert g.T_days.iloc[0] == allowed.T_days.max()
    ends = pd.to_datetime(kept.target_end_ts, utc=True).to_numpy()
    starts = pd.to_datetime(kept.target_start_ts, utc=True).to_numpy()
    order = np.argsort(ends)
    assert (starts[order][1:] >= ends[order][:-1]).all()                                    # kept windows do not overlap
    g2 = f.copy()
    g2["spread_session"] = np.random.default_rng(0).normal(size=len(g2))                    # outcomes shuffled -> same selection
    kept2, _ = D.nonoverlap_subsample(g2)
    assert list(kept2.expiry) == list(kept.expiry) and list(kept2.day) == list(kept.day)


def test_nonoverlap_subsample_drops_an_overlapping_window_and_missing_snapshots():
    f = snap_frame()
    extra = f[(f.expiry == f.expiry.iloc[0])].copy()
    extra["expiry"] = (date.fromisoformat(f.expiry.iloc[0]) + timedelta(days=2)).isoformat()                # expiry only 2 days later -> overlapping window
    end = datetime.fromisoformat(extra.expiry.iloc[0] + "T15:30:00+05:30")
    extra["target_end_ts"] = end.isoformat()
    extra["T_days"] = [(end - datetime.fromisoformat(s)).total_seconds() / 86400 for s in extra.target_start_ts]
    kept, dropped = D.nonoverlap_subsample(pd.concat([f, extra], ignore_index=True))
    assert (dropped.reason == "window overlaps previously kept window").sum() >= 1
    f2 = f[~((f.expiry == f.expiry.iloc[0]) & (f.time == "10:00"))]
    _, dropped2 = D.nonoverlap_subsample(f2)
    assert (dropped2.reason.str.startswith("no 10:00 snapshot")).sum() == 1


# ------------------------------------------------------------------------------------------------ ratio inference
def test_sign_test_exact_values():
    r = R.sign_test(np.array([2.0] * 9 + [0.5]), center=1.0)
    assert r["n_positive"] == 9 and r["p_two_sided"] == pytest.approx(2 * 11 / 1024)
    assert R.sign_test(np.array([2.0, 0.5, 2.0, 0.5]), 1.0)["p_two_sided"] == pytest.approx(1.0)
    assert R.sign_test(np.array([1.0, 2.0, 2.0, 2.0]), 1.0)["n"] == 3                         # ties dropped
    with pytest.raises(ValueError):
        R.sign_test(np.ones(4), 1.0)


def test_sign_flip_test_is_calibrated_and_deterministic():
    rng = np.random.default_rng(0)
    null = rng.normal(0, 1, 200)
    a = R.sign_flip_test(null, 1, 4000, seed=5)
    assert a["p_two_sided"] > 0.05
    assert R.sign_flip_test(null, 1, 4000, seed=5) == a
    assert R.sign_flip_test(null, 1, 4000, seed=6)["p_two_sided"] != a["p_two_sided"]
    shifted = R.sign_flip_test(null + 0.6, 1, 4000, seed=5)
    assert shifted["p_two_sided"] < 0.001
    assert R.sign_flip_test(np.full(60, 5.0), 1, 999, seed=1)["p_two_sided"] == pytest.approx(1 / 1000)       # the "+1" permutation-p correction
    with pytest.raises(C.ClusterError):
        R.sign_flip_test(np.arange(20.0), block=5, reps=100)                                  # 4 blocks < 10
    with pytest.raises(C.ClusterError):
        R.sign_flip_test(np.arange(20.0), block=0, reps=100)


def test_block_sign_flip_is_less_confident_than_unit_flip_under_positive_dependence():
    x = ar1(260, 0.7, 11) * 0.3 + 0.12
    p1 = R.sign_flip_test(x, 1, 6000, seed=3)["p_two_sided"]
    p8 = R.sign_flip_test(x, 8, 6000, seed=3)["p_two_sided"]
    assert p8 > p1


def test_hac_t_matches_manual_arithmetic():
    x = ar1(300, 0.4, 2) + 0.2
    h = R.hac_t(x, 4)
    nw = D.newey_west(x, 4)
    assert h["t"] == pytest.approx(x.mean() / nw["nw_se"]) and h["lags"] == 4


# ------------------------------------------------------------------------------------------------ influence
class MeanOfClusterMeans:
    """Stub evaluator (smooth statistic) so the jackknife can be checked against closed forms."""

    def __init__(self, cs):
        self.cs = cs
        self.m = np.array([cs.cols["x"][cs.offsets[i]: cs.offsets[i] + cs.counts[i]].mean() for i in range(cs.n_clusters)])

    def evaluate(self, rows, seq):
        return np.array([self.m[seq].mean()])

    def point(self):
        return self.evaluate(None, np.arange(self.cs.n_clusters))


def stub_cs(n=45, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat([f"c{i:03d}" for i in range(n)], 2)
    return C.ClusterSet(labels, {"x": rng.normal(size=2 * n)}), None


def test_delete_one_jackknife_equals_the_closed_form_for_a_mean():
    cs, _ = stub_cs()
    est = MeanOfClusterMeans(cs)
    theta = est.point()
    n = cs.n_clusters
    loo = np.array([[np.delete(est.m, i).mean()] for i in range(n)])
    js = I.jackknife_se(theta, loo)
    assert js["se"][0] == pytest.approx(est.m.std(ddof=1) / math.sqrt(n), rel=1e-12)
    assert js["bias"][0] == pytest.approx(0.0, abs=1e-12)
    assert I.block_jackknife_se(cs, est, 1)[0] == pytest.approx(est.m.std(ddof=1) / math.sqrt(n), rel=1e-12)
    assert I.block_jackknife_se(cs, est, 5)[0] > 0
    with pytest.raises(C.ClusterError):
        I.block_jackknife_se(cs, est, n)


def test_leave_one_out_matches_manual_recomputation():
    df = many_clusters(32, 3, 9)
    est = make_est(df)
    loo = I.leave_one_out(est.cs, est)
    assert loo.shape == (32, 9)
    lab = est.cs.labels[5]
    sub = df[df.expiry != lab]
    ss = np.median(sub.spread_session)
    assert loo[5, 0] == pytest.approx(ss)
    assert loo[5, 8] == pytest.approx(sub.implied_total_variance.sum() / sub.rv_total_variance.sum())


def test_leave_one_group_out_removes_the_whole_group():
    df = many_clusters(40, 3, 10)
    est = make_est(df)
    years = np.array(["2021"] * 12 + ["2022"] * 28)
    out = I.leave_one_group_out(est.cs, est, years)
    assert out["2021"][0] == 12 and out["2022"][0] == 28
    sub = df[df.expiry.isin(est.cs.labels[12:])]
    assert out["2021"][1][0] == pytest.approx(np.median(sub.spread_session))


def test_topk_removal_moves_the_estimate_more_than_removing_random_clusters():
    df = many_clusters(40, 3, 12)
    df.loc[df.expiry == df.expiry.unique()[7], "implied_total_variance"] *= 200                 # one extreme expiry
    est = make_est(df)
    th = est.point()
    loo = I.leave_one_out(est.cs, est)
    tk = I.topk_removal(est.cs, est, loo, th, [1, 3], random_draws=50, seed=1)
    i = E.EST_NAMES.index("R3_ratio_of_summed_total_variance")
    assert abs(tk[1]["top"][i] - th[i]) > abs(tk[1]["rnd_p50"][i] - th[i])
    assert np.argmax(np.abs(loo[:, i] - th[i])) == 7
    with pytest.raises(C.ClusterError):
        I.topk_removal(est.cs, est, loo, th, [40])
    tk2 = I.topk_removal(est.cs, est, loo, th, [1, 3], random_draws=50, seed=1)
    np.testing.assert_array_equal(tk[3]["rnd_p50"], tk2[3]["rnd_p50"])                          # deterministic


def test_trimmed_and_winsorized_means_hand_values():
    x = np.arange(1.0, 11.0)
    x[-1] = 1000.0
    assert I.trimmed_mean(x, 0.1) == pytest.approx(np.mean(np.arange(2.0, 10.0)))
    assert I.winsorized_mean(x, 0.1) == pytest.approx(np.mean([2, 2, 3, 4, 5, 6, 7, 8, 9, 9]))
    assert I.trimmed_mean(np.array([1.0, np.nan, 3.0]), 0.0) == pytest.approx(2.0)


# ------------------------------------------------------------------------------------------------ synthetic dependence studies
def dependent_dataset(rng, n_clusters, m, phi, sigma_mu, sigma_eps=1.0):
    mu = np.zeros(n_clusters)
    z = rng.normal(size=n_clusters + 50)
    a = np.zeros(n_clusters + 50)
    for t in range(1, n_clusters + 50):
        a[t] = phi * a[t - 1] + z[t] * math.sqrt(1 - phi ** 2)
    mu = sigma_mu * a[50:]
    x = np.repeat(mu, m) + rng.normal(0, sigma_eps, n_clusters * m)
    labels = np.repeat([f"c{i:04d}" for i in range(n_clusters)], m)
    return labels, x


class RowMedian:
    def __init__(self, cs):
        self.x = cs.cols["x"]

    def evaluate(self, rows, seq):
        return np.array([np.median(self.x[rows])])


def coverage(phi, sigma_mu, seed, reps=120, n=60, m=8, boot=120):
    """Coverage of 95% percentile intervals for the population median (0) under row-iid / cluster-iid / block resampling."""
    rng = np.random.default_rng(seed)
    hit = {"row_iid": 0, "cluster_iid": 0, "cluster_block6": 0}
    for r in range(reps):
        labels, x = dependent_dataset(rng, n, m, phi, sigma_mu)
        cs = C.ClusterSet(labels, {"x": x})
        rows = C.ClusterSet(np.char.zfill(np.arange(len(x)).astype(str), 6), {"x": x}, row_level=True)
        for name, c, scheme, b in (("row_iid", rows, "iid", None), ("cluster_iid", cs, "iid", None), ("cluster_block6", cs, "moving_block", 6)):
            d = C.bootstrap(c, RowMedian(c).evaluate, scheme, b, boot, seed + r, k=1)
            lo, hi = C.percentile_ci(d)
            hit[name] += bool(lo[0] <= 0.0 <= hi[0])
    return {k: v / reps for k, v in hit.items()}


def test_row_level_bootstrap_undercovers_when_rows_share_a_cluster_effect_but_cluster_bootstrap_does_not():
    cov = coverage(phi=0.0, sigma_mu=1.2, seed=100)
    assert cov["row_iid"] < 0.80
    assert cov["cluster_iid"] >= 0.88 and cov["cluster_block6"] >= 0.88


def test_block_bootstrap_repairs_coverage_that_iid_cluster_resampling_loses_under_serial_dependence():
    cov = coverage(phi=0.85, sigma_mu=1.2, seed=200)
    assert cov["row_iid"] < 0.70
    assert cov["cluster_iid"] < 0.90
    assert cov["cluster_block6"] > cov["cluster_iid"] + 0.02


# ------------------------------------------------------------------------------------------------ builder: guards, baseline reproduction, determinism
def synthetic_aligned(n_exp=130, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2022, 1, 6)
    for k in range(n_exp):
        exp = start + timedelta(days=7 * k)
        end = datetime(exp.year, exp.month, exp.day, 15, 30, tzinfo=IST)
        mu = rng.normal(2.0, 1.0)
        for off in (12, 10, 8, 6, 5, 4, 3, 2, 1):
            day = exp - timedelta(days=off)
            if day.weekday() >= 5:
                continue
            for tm in ("10:00", "13:00", "15:00"):
                h, m = (int(x) for x in tm.split(":"))
                obs = datetime(day.year, day.month, day.day, h, m, tzinfo=IST) + timedelta(seconds=60)
                T = (end - obs).total_seconds() / (365 * 86400)
                v = abs(rng.normal(2e-4, 5e-5)) * T * 20 + 1e-6
                w = v * math.exp(rng.normal(0.25, 0.3))
                rows.append(dict(horizon="EXP", measure="hybrid", expiry=exp.isoformat(), day=day.isoformat(), time=tm, split="dev", T_days=T * 365,
                                 target_start_ts=obs.isoformat(), target_end_ts=end.isoformat(), spread_session=mu + rng.normal(0, 1.5),
                                 spread_calendar=mu - 0.8 + rng.normal(0, 1.5), implied_total_variance=w, rv_total_variance=v, total_variance_ratio_valid=True))
    return pd.DataFrame(rows)


def write_stage2c(tmp_path, df, perturb=0.0):
    est = E.build(df[(df.horizon == "EXP") & (df.measure == "hybrid")].reset_index(drop=True)).point()
    d = tmp_path / "stage2c" / "annualization_audit"
    d.mkdir(parents=True)
    pd.DataFrame([dict(horizon="EXP", measure="hybrid", dimension="overall", level="all", n_obs=len(df), spread_session_median=est[0] + perturb,
                       spread_calendar_median=est[2])]).to_csv(d / "original_vs_aligned.csv", index=False)
    pd.DataFrame([dict(measure="hybrid", dimension="overall", level="all", ratio_geometric_mean=est[4], ratio_median=est[6])]).to_csv(d / "expiry_total_variance.csv", index=False)
    al = tmp_path / "aligned.csv"
    df.to_csv(al, index=False)
    return str(al), str(tmp_path / "stage2c")


def test_builder_refuses_to_run_when_the_baseline_does_not_reproduce(tmp_path):
    al, s2c = write_stage2c(tmp_path, synthetic_aligned(40), perturb=0.01)
    with pytest.raises(RuntimeError, match="baseline mismatch"):
        B.build(al, s2c, str(tmp_path / "stage2d" / "gate1"), 20, 20)


def test_builder_refuses_output_directories_outside_stage2d(tmp_path):
    al, s2c = write_stage2c(tmp_path, synthetic_aligned(40))
    for bad in (tmp_path / "stage2c" / "annualization_audit", tmp_path / "somewhere"):
        with pytest.raises(ValueError, match="stage2d"):
            B.build(al, s2c, str(bad), 20, 20)


def test_builder_end_to_end_is_deterministic_and_writes_only_under_the_output_directory(tmp_path):
    df = synthetic_aligned(130)
    al, s2c = write_stage2c(tmp_path, df)
    before = sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / "stage2c").rglob("*"))
    out1, out2 = tmp_path / "stage2d" / "g1", tmp_path / "stage2d" / "g2"
    m1 = B.build(al, s2c, str(out1), 60, 40)
    m2 = B.build(al, s2c, str(out2), 60, 40)
    after = sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / "stage2c").rglob("*"))
    assert before == after                                                                      # existing artifacts untouched
    files = sorted(os.listdir(out1))
    for need in ("estimands_point.csv", "ci_methods_comparison.csv", "block_length_sensitivity.csv", "dependence_diagnostics.csv", "window_overlap.csv",
                 "ratio_inference.csv", "influence_summary.csv", "leave_one_year_out.csv", "topk_removal.csv", "trimmed_winsorized.csv", "run_metadata.json"):
        assert need in files
    for f in files:
        if f.endswith(".csv"):
            pd.testing.assert_frame_equal(pd.read_csv(out1 / f), pd.read_csv(out2 / f))      # same seed -> byte-identical tables
    ci = pd.read_csv(out1 / "ci_methods_comparison.csv")
    w = ci[ci.estimand == "S1_session_all_obs_median"].set_index("method").width
    assert w["row_iid_NAIVE_reference"] < w["expiry_iid_cluster"]                                  # naive intervals are narrower
    assert m1["block_length_selected"] == m2["block_length_selected"] >= 1
    # the date-block method reports no per-expiry estimand
    dd = ci[(ci.method == "date_block_stage2c_method") & ci.estimand.str.startswith(("S2", "R1e", "R2e"))]
    assert dd.ci_lo.isna().all()


def test_gate1_inputs_are_only_stage2c_outcome_columns_and_timestamps():
    assert set(E.REQUIRED) == {"spread_session", "spread_calendar", "implied_total_variance", "rv_total_variance", "total_variance_ratio_valid"}


@pytest.mark.skipif(not os.path.exists("research_output/stage2c/annualization_audit/aligned_observations.csv"), reason="local Stage 2C aligned file not present")
def test_real_baseline_reproduces_committed_stage2c_numbers():
    e = B.load_baseline("research_output/stage2c/annualization_audit/aligned_observations.csv", "research_output/stage2c")
    t = dict(zip(E.EST_NAMES, E.build(e).point()))
    assert len(e) == 6490
    assert t["S1_session_all_obs_median"] == pytest.approx(2.651, abs=5e-4) and t["S1_calendar_all_obs_median"] == pytest.approx(1.805, abs=5e-4)
    assert t["R2_median_ratio_all_obs"] == pytest.approx(1.3705, abs=5e-4) and t["R1_geometric_mean_all_obs"] == pytest.approx(1.291, abs=5e-4)
