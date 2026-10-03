"""Stage 2D Gate 3a: T9 selection-on-outcome audit and T10 quote-noise sensitivity."""
import math
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from optionsengine.bsm import OptionType
from optionsengine.research.stage2d import build_gate3a as B3, clusters as C, estimands as E, quote_noise as Q, selection as S
from optionsengine.surface import OptionQuote
from tests.rv_synth import frame
from tests.test_iv_rv import WD, alt as alt_returns, build as build_sessions


# ================================================================================================ T9: universe and covariates
def smile_frame():
    rows = []

    def add(sid, day, time, exp, T, fs, reason, atm, expiry_day=False):
        rows.append(dict(smile_id=sid, day=day, time=time, expiry=exp, split="dev", expiry_day=expiry_day, T_days=T, forward_status=fs, group_reason=reason, atm_iv=atm))

    add("a", "2022-03-01", "10:00", "2022-03-10", 9.2, "ok", "ok", 0.15)                  # observed
    add("b", "2022-03-01", "13:00", "2022-03-10", 9.1, "ok", "ok", 0.15)                  # kept, but no EXP target (L2)
    add("c", "2022-03-02", "10:00", "2022-03-10", 8.2, "low_confidence", "ok", np.nan)    # L1: forward low confidence
    add("d", "2022-03-02", "13:00", "2022-03-10", 8.1, None, "no_spot_bar_at_snapshot", np.nan)
    add("e", "2022-03-03", "10:00", "2022-03-10", 7.2, None, "no_option_bars_at_snapshot", np.nan)
    add("f", "2022-03-03", "13:00", "2022-03-10", 7.1, "ok", "ok", np.nan)                # forward ok but no strict ATM
    add("g", "2022-03-10", "10:00", "2022-03-10", 0.2, "ok", "ok", 0.2, True)             # expiry day: by construction
    add("h", "2022-03-02", "15:00", "2022-03-30", 28.0, "ok", "ok", 0.15)                 # > 14 DTE: outside the universe
    return pd.DataFrame(rows)


def feat_frame():
    days = ["2022-03-01", "2022-03-02", "2022-03-03", "2022-03-10"]
    return pd.DataFrame(dict(day=days, ln_rv20=[2.5, 2.6, 2.7, 2.4], prev_range=[0.01, 0.012, 0.011, 0.009], gap_abs=[0.002, 0.004, 0.001, 0.003], ln_y1=[2.4, 2.9, 2.5, np.nan]))


def test_build_groups_layers_reasons_and_covariate_join():
    g = S.build_groups(smile_frame(), feat_frame(), ["a"], {"b": "incomplete_or_missing_session_in_window"}).set_index("smile_id")
    assert len(g) == 7 and "h" not in g.index                                              # > 14 DTE group excluded
    assert g.loc["a", "layer"] == "observed" and g.loc["b", "layer"] == "L2_no_exp_target" and g.loc["b", "l2_reason"] == "incomplete_or_missing_session_in_window"
    assert g.loc["c", "layer"] == "L1_rejected" and g.loc["c", "l1_reason"] == "forward_low_confidence"
    assert g.loc["d", "l1_reason"] == "no_spot_bar_at_snapshot" and g.loc["e", "l1_reason"] == "no_option_bars_at_snapshot"
    assert g.loc["f", "l1_reason"] == "forward_ok_but_no_strict_atm" and g.loc["f", "layer"] == "L1_rejected"
    assert g.loc["g", "layer"] == "expiry_day_by_construction" and not g.loc["g", "in_u_exp"]
    assert g.loc["a", "ln_rv20"] == 2.5 and g.loc["c", "ln_y1"] == 2.9 and g.loc["a", "year"] == 2022 and g.loc["a", "wd"] == 1


def test_universe_counts_hand_values():
    g = S.build_groups(smile_frame(), feat_frame(), ["a"], {"b": "x"})
    u = B3.universe_counts(g)
    a = u[(u.dimension == "all")].iloc[0]
    assert (a.attempted_groups, a.observed, a.expiry_day_by_construction, a.L1_rejected, a.L2_no_exp_target) == (7, 1, 1, 4, 1)
    assert a.share_missing_of_u_exp == pytest.approx(5 / 6) and a.share_L1_rejected_of_non_expiry_day == pytest.approx(4 / 6)
    assert set(u[u.dimension == "reason"].level) == {"L1_rejected: forward_low_confidence", "L1_rejected: no_spot_bar_at_snapshot", "L1_rejected: no_option_bars_at_snapshot",
                                                      "L1_rejected: forward_ok_but_no_strict_atm", "L2_no_exp_target: x"}


def test_design_matrix_uses_only_known_at_t_columns_and_the_outcome_only_on_request():
    g = S.build_groups(smile_frame(), feat_frame(), ["a"], {})
    X, names = S.design_matrix(g)
    assert names == ["intercept"] + S.COVARIATE_NAMES and X.shape == (len(g), 1 + len(S.COVARIATE_NAMES))
    banned = ("n_quotes", "atm", "forward", "iv", "price", "layer", "kept", "observed", "missing", "reason", "ln_y1")
    assert not any(b in n for n in names for b in banned)
    X2, names2 = S.design_matrix(g, with_outcome=True)
    assert names2[-1] == S.OUTCOME_COLUMN and names2[:-1] == names and X2.shape[1] == X.shape[1] + 1
    assert np.array_equal(X2[:, :-1][np.isfinite(X).all(axis=1)], X[np.isfinite(X).all(axis=1)])


def test_complete_cases_drops_rows_missing_covariates_or_outcome():
    g = S.build_groups(smile_frame(), feat_frame(), [], {})
    assert len(S.complete_cases(g, need_outcome=False)) == 7
    assert len(S.complete_cases(g, need_outcome=True)) == 6                                # the 2022-03-10 session has no outcome (end of data)


def write_spot(tmp_path, specs, days):
    rows, _ = build_sessions(specs)
    spot = tmp_path / "spot.parquet"
    frame(rows).to_parquet(spot)
    pdir = tmp_path / "pcp"
    pdir.mkdir(exist_ok=True)
    for d in days:
        (pdir / f"{d.isoformat()}.csv").write_text("x")
    return str(spot), str(pdir)


def long_days(n):
    out, d = [], date(2026, 9, 14)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_spot_features_are_known_at_t_and_the_outcome_is_strictly_later(tmp_path):
    n = 34
    days = long_days(n)
    rng = np.random.default_rng(0)
    base = [(d, list(rng.normal(0, 0.0007, 375)), float(rng.normal(0, 0.002)) if i else 0.0) for i, d in enumerate(days)]
    spot, pdir = write_spot(tmp_path, base, days)
    f0 = S.load_spot_features(spot, pdir).set_index("day")
    d0 = days[25].isoformat()
    assert np.isfinite(f0.loc[d0, ["ln_rv20", "prev_range", "gap_abs", "ln_y1"]]).all()
    # change every bar of the SNAPSHOT day after its first bar: covariates (known at t) must not move; the open/gap and history must not move either
    mod = list(base)
    r = list(base[25][1])
    for i in range(1, 375):
        r[i] = 0.02 if i % 2 else -0.02
    mod[25] = (days[25], r, base[25][2])
    s2, p2 = write_spot(tmp_path / "m1", mod, days) if (tmp_path / "m1").mkdir() is None else (None, None)
    f1 = S.load_spot_features(s2, p2).set_index("day")
    for c in ("ln_rv20", "prev_range", "gap_abs"):
        assert f1.loc[d0, c] == pytest.approx(f0.loc[d0, c], abs=1e-12)
    assert f1.loc[d0, "ln_y1"] == pytest.approx(f0.loc[d0, "ln_y1"], abs=1e-12)           # the snapshot session itself is not part of the outcome
    # change a LATER session: the outcome moves, the covariates do not
    mod2 = list(base)
    r2 = list(base[27][1])
    r2[100] = 0.05
    mod2[27] = (days[27], r2, base[27][2])
    (tmp_path / "m2").mkdir()
    s3, p3 = write_spot(tmp_path / "m2", mod2, days)
    f2 = S.load_spot_features(s3, p3).set_index("day")
    assert f2.loc[d0, "ln_y1"] != pytest.approx(f0.loc[d0, "ln_y1"], abs=1e-6)
    for c in ("ln_rv20", "prev_range", "gap_abs"):
        assert f2.loc[d0, c] == pytest.approx(f0.loc[d0, c], abs=1e-12)
    # change an EARLIER session: ln_rv20 moves (it uses completed sessions before t), the outcome does not
    mod3 = list(base)
    r3 = list(base[20][1])
    r3[100] = 0.05
    mod3[20] = (days[20], r3, base[20][2])
    (tmp_path / "m3").mkdir()
    s4, p4 = write_spot(tmp_path / "m3", mod3, days)
    f3 = S.load_spot_features(s4, p4).set_index("day")
    assert f3.loc[d0, "ln_rv20"] != pytest.approx(f0.loc[d0, "ln_rv20"], abs=1e-6)
    assert f3.loc[d0, "ln_y1"] == pytest.approx(f0.loc[d0, "ln_y1"], abs=1e-12)


# ================================================================================================ ridge logistic
def test_intercept_only_ridge_logit_is_the_empirical_rate_and_penalty_shrinks():
    y = np.array([1.0] * 30 + [0.0] * 70)
    b = S.ridge_logit(np.ones((100, 1)), y, lam=5.0)
    assert 1 / (1 + math.exp(-b[0])) == pytest.approx(0.3, abs=1e-8)
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(400), rng.normal(size=(400, 3))])
    yy = (rng.random(400) < 1 / (1 + np.exp(-(0.5 * X[:, 1] - 0.8 * X[:, 2])))).astype(float)
    norms = [np.linalg.norm(S.ridge_logit(X, yy, lam)[1:]) for lam in (0.0, 1.0, 10.0, 100.0)]
    assert all(a > b for a, b in zip(norms, norms[1:]))


def test_ridge_logit_matches_a_generic_optimiser_on_the_same_objective():
    opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(300), rng.normal(size=(300, 4))])
    y = (rng.random(300) < 1 / (1 + np.exp(-(X[:, 1] - 0.5 * X[:, 3])))).astype(float)
    lam = 2.0

    def neg(b):
        eta = X @ b
        return -(y @ eta - np.sum(np.logaddexp(0, eta))) + 0.5 * lam * np.sum(b[1:] ** 2)

    ref = opt.minimize(neg, np.zeros(5), method="BFGS", options=dict(gtol=1e-10)).x
    np.testing.assert_allclose(S.ridge_logit(X, y, lam), ref, atol=1e-5)


def test_ridge_logit_stays_finite_under_perfect_separation():
    x = np.concatenate([np.full(50, -1.0), np.full(50, 1.0)])
    X = np.column_stack([np.ones(100), x])
    b = S.ridge_logit(X, (x > 0).astype(float), lam=1.0)
    assert np.isfinite(b).all() and b[1] > 0


def test_standardize_leaves_the_intercept_and_handles_constant_columns():
    X = np.column_stack([np.ones(5), np.arange(5.0), np.full(5, 3.0)])
    Z = S.standardize(X)
    assert (Z[:, 0] == 1).all() and Z[:, 1].mean() == pytest.approx(0) and Z[:, 1].std() == pytest.approx(1) and np.isfinite(Z).all()


# ================================================================================================ outcome-dependence permutation test
def synth_groups(n_days=260, per_day=4, beta_out=0.0, seed=0):
    rng = np.random.default_rng(seed)
    days = [(date(2022, 1, 3) + timedelta(days=i)).isoformat() for i in range(n_days)]
    series = np.zeros(n_days)
    for i in range(1, n_days):
        series[i] = 0.7 * series[i - 1] + rng.normal(0, 0.7)
    rows = []
    for i, d in enumerate(days):
        c = rng.normal(0, 1)
        for k in range(per_day):
            rows.append(dict(day=d, expiry=f"e{(i // 5):03d}", year=2022, time=["10:00", "13:00", "15:00", "10:00"][k], wd=i % 5, T_days=float(5 + k), ln_rv20=c + rng.normal(0, 0.3),
                             prev_range=rng.normal(0.01, 0.002), gap_abs=abs(rng.normal(0, 0.003)), ln_y1=series[i]))
    g = pd.DataFrame(rows)
    p = 1 / (1 + np.exp(-(-2.8 + 0.8 * g.ln_rv20 + beta_out * g.ln_y1)))
    miss = (rng.random(len(g)) < p).astype(float)
    return g, miss, days, series


def test_permutation_test_is_calibrated_without_dependence_and_detects_planted_dependence():
    pvals = []
    for s in range(24):
        g, m, days, ser = synth_groups(180, 3, 0.0, s)
        pvals.append(S.date_shift_permutation_test(g, m, days, ser, draws=60, seed=s)["p_two_sided"])
    assert np.mean(np.array(pvals) < 0.05) <= 0.2 and np.mean(pvals) > 0.3
    hit = 0
    for s in range(8):
        g, m, days, ser = synth_groups(180, 3, 1.6, 100 + s)
        r = S.date_shift_permutation_test(g, m, days, ser, draws=60, seed=s)
        hit += r["p_two_sided"] < 0.05 and r["coefficient"] > 0
    assert hit >= 7


def test_permutation_test_is_deterministic_and_guards_short_series():
    g, m, days, ser = synth_groups(120, 2, 0.5, 3)
    a = S.date_shift_permutation_test(g, m, days, ser, draws=30, seed=7)
    assert a == S.date_shift_permutation_test(g, m, days, ser, draws=30, seed=7)
    assert a != S.date_shift_permutation_test(g, m, days, ser, draws=30, seed=8)
    with pytest.raises(C.ClusterError):
        S.date_shift_permutation_test(g, m, days[:30], ser[:30], draws=10, min_shift=20) if False else S.date_shift_permutation_test(g[g.day.isin(days[:30])].reset_index(drop=True), m[g.day.isin(days[:30]).to_numpy()], days[:30], ser[:30], draws=10, min_shift=20)


def test_permutation_p_value_has_the_plus_one_floor():
    g, m, days, ser = synth_groups(200, 3, 3.0, 21)
    r = S.date_shift_permutation_test(g, m, days, ser, draws=30, seed=1)
    assert r["p_two_sided"] == pytest.approx(1 / 31)                                       # (1 + 0) / (1 + draws): never exactly zero


def test_the_outcome_coefficient_is_the_last_model_term_and_signed_correctly():
    g, m, days, ser = synth_groups(200, 3, 2.0, 11)
    assert S.outcome_coefficient(g, m) > 0
    g2, m2, _, _ = synth_groups(200, 3, -2.0, 11)
    assert S.outcome_coefficient(g2, m2) < 0


# ================================================================================================ balance
def test_standardized_difference_hand_values():
    a, b = np.array([1.0, 3.0]), np.array([0.0, 2.0])
    assert S.standardized_difference(a, b) == pytest.approx(1.0 / math.sqrt(2.0))
    assert math.isnan(S.standardized_difference(np.array([1.0]), b))


def balance_frame(n_exp=60, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_exp):
        for i in range(6):
            flag = (c % 6 == 0) and i < 3
            rows.append(dict(expiry=f"e{c:03d}", x=rng.normal(shift if flag else 0, 1), flag=flag))
    return pd.DataFrame(rows)


def test_balance_table_detects_a_planted_difference_and_is_null_otherwise():
    d = balance_frame(60, 1.5, 1)
    t = S.balance_table(d, d.flag.to_numpy(), ["x"], 5, 300, 1).iloc[0]
    assert t.std_difference > 0.8 and t.ci_lo > 0
    d0 = balance_frame(60, 0.0, 2)
    t0 = S.balance_table(d0, d0.flag.to_numpy(), ["x"], 5, 300, 1).iloc[0]
    assert t0.ci_lo < 0 < t0.ci_hi
    with pytest.raises(C.ClusterError):
        S.balance_table(balance_frame(20), balance_frame(20).flag.to_numpy(), ["x"], 5, 50, 1)


# ================================================================================================ Manski bounds
def test_manski_bounds_and_breakdown_hand_values():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert S.manski_median_bounds(v, 1) == (2.5, 3.5)
    assert S.manski_median_bounds(v, 0) == (3.0, 3.0)
    bd = S.breakdown_share(v, 0.0, "above")
    assert bd["k_needed"] == 5 and bd["share_needed"] == pytest.approx(0.5) and bd["observed_median"] == 3.0
    assert S.breakdown_share(-v, 0.0, "below")["k_needed"] == 5                               # mirror image: observed median below the null
    assert S.breakdown_share(v, 0.0, "below")["k_needed"] == 0
    assert S.breakdown_share(v, 4.0, "above")["k_needed"] == 0                               # already below the null: nothing needed
    assert S.breakdown_share(np.array([-1.0, 1, 2, 3]), 0.0)["k_needed"] < 5
    assert S.required_missing_mean_log_ratio(0.3, 100, 10) == pytest.approx(-3.0)


def test_breakdown_share_matches_the_closed_form_on_random_data():
    rng = np.random.default_rng(0)
    v = rng.normal(1.0, 2.0, 501)
    n0 = int((v <= 0).sum())
    bd = S.breakdown_share(v, 0.0, "above")
    k = bd["k_needed"]
    assert S.median_with_missing(v, k, -np.inf) <= 0 < S.median_with_missing(v, max(k - 1, 0), -np.inf)
    assert bd["share_needed"] == pytest.approx(k / (len(v) + k))
    assert abs(k - (len(v) - 2 * n0)) <= 1                                                   # (n0 + k) / (n + k) >= 1/2


# ================================================================================================ IPW
def ipw_cs(n_cl=120, per=6, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    labels, c, obs, sc = [], [], [], []
    for k in range(n_cl):
        for _ in range(per):
            ci = rng.normal()
            p_obs = 1 / (1 + np.exp(-(1.0 - 1.5 * ci))) if informative else 0.7
            labels.append(f"e{k:03d}")
            c.append(ci)
            obs.append(float(rng.random() < p_obs))
            sc.append(2.0 + 1.5 * ci + rng.normal(0, 0.5))
    c, obs, sc = np.array(c), np.array(obs), np.array(sc)
    X = np.column_stack([np.ones(len(c)), c])
    ratio = np.exp(0.3 + 0.0 * c)
    return C.ClusterSet(labels, {"X": X, "observed": obs, "sc": sc, "ss": sc + 1, "lr": np.log(ratio), "ratio": ratio}), sc


def test_ipw_removes_selection_on_a_known_covariate_and_is_neutral_otherwise():
    cs, sc_all = ipw_cs(informative=True)
    ev = S.IPWEvaluator(cs)
    pt = ev.point()
    assert pt[4] < 1.7                                                                      # unweighted: selection pushes the median spread down (truth 2.0)
    assert pt[0] == pytest.approx(2.0, abs=0.2) and pt[1] == pytest.approx(pt[0] + 1, abs=1e-9)
    cs2, _ = ipw_cs(informative=False)
    pt2 = S.IPWEvaluator(cs2).point()
    assert pt2[0] == pytest.approx(pt2[4], abs=0.15)
    np.testing.assert_allclose(pt2[2], pt2[6], rtol=1e-12)                                  # constant ratio: weighted == unweighted geometric mean


def test_ipw_weights_are_one_without_missing_and_capped_with_it():
    cs, _ = ipw_cs(60, 5, 3, informative=False)
    cs.cols["observed"][:] = 1.0
    w, p = S.IPWEvaluator(cs).weights(np.arange(cs.n_rows))
    np.testing.assert_allclose(w, 1.0, atol=1e-4)
    cs2, _ = ipw_cs(informative=True)
    w2, p2 = S.IPWEvaluator(cs2).weights(np.arange(cs2.n_rows))
    assert w2.max() <= S.WEIGHT_CAP and (w2 >= 1).all()


def test_ipw_bootstrap_is_deterministic_and_guards_few_clusters():
    cs, _ = ipw_cs(40, 5, 5)
    ev = S.IPWEvaluator(cs)
    a = C.bootstrap(cs, ev.evaluate, "moving_block", 5, 20, 3, k=8)
    np.testing.assert_array_equal(a, C.bootstrap(cs, ev.evaluate, "moving_block", 5, 20, 3, k=8))
    with pytest.raises(C.ClusterError):
        ipw_cs(20, 5, 1)


# ================================================================================================ T10: Black-76 and scenarios
def test_black76_known_values_parity_and_vega():
    assert float(Q.b76_price(100.0, 100.0, 1.0, 0.0, 0.2, True)) == pytest.approx(7.965567455405804, rel=1e-12)
    F, K, T, r, s = 17500.0, 17400.0, 0.04, 0.065, 0.14
    c, p = float(Q.b76_price(F, K, T, r, s, True)), float(Q.b76_price(F, K, T, r, s, False))
    assert c - p == pytest.approx(math.exp(-r * T) * (F - K), rel=1e-12)
    h = 1e-5
    fd = (float(Q.b76_price(F, K, T, r, s + h, True)) - float(Q.b76_price(F, K, T, r, s - h, True))) / (2 * h) / 100
    assert float(Q.b76_vega(F, K, T, r, s)) == pytest.approx(fd, rel=1e-6)


@pytest.mark.parametrize("call", [True, False])
def test_implied_vol_round_trip_bounds_and_robust_initial_guess(call):
    rng = np.random.default_rng(0)
    n = 300
    F = rng.uniform(15000, 25000, n)
    K = F * np.exp(rng.uniform(-0.06, 0.06, n))
    T = rng.uniform(0.003, 0.05, n)
    sig = rng.uniform(0.08, 0.4, n)
    px = Q.b76_price(F, K, T, 0.065, sig, np.full(n, call))
    iv = Q.implied_vol(px, F, K, T, 0.065, np.full(n, call))
    intrinsic = math.exp(0) * 0 + np.exp(-0.065 * T) * (np.maximum(F - K, 0) if call else np.maximum(K - F, 0))
    ok = (px - intrinsic) > 1e-4                                                            # deep-ITM options with no time value carry no volatility information
    assert ok.sum() > 100
    np.testing.assert_allclose(iv[ok], sig[ok], atol=1e-7)
    iv_bad_init = Q.implied_vol(px, F, K, T, 0.065, np.full(n, call), init=np.full(n, 3.0))
    np.testing.assert_allclose(iv_bad_init[ok], sig[ok], atol=1e-7)
    below = Q.implied_vol(np.array([0.0]), np.array([100.0]), np.array([100.0]), np.array([0.1]), 0.0, np.array([call]))
    assert np.isnan(below[0])
    assert np.isnan(Q.implied_vol(np.array([1000.0]), np.array([100.0]), np.array([100.0]), np.array([0.1]), 0.0, np.array([call]))[0])


def smile_points(n=40, seed=0):
    """Synthetic points table and aligned rows: OTM puts below F, calls at/above; prices from Black-76 with a known smile."""
    rng = np.random.default_rng(seed)
    pts, rows = [], []
    for i in range(n):
        F = float(rng.uniform(17000, 24000))
        T = float(rng.uniform(2, 13)) / 365
        sid = f"s{i:03d}"
        for j, d in enumerate(range(-6, 6)):
            K = round(F / 50) * 50 + 50 * d
            x = math.log(K / F)
            call = x >= 0
            iv = 0.13 + 0.4 * x * x + 0.05 * (-x if not call else 0)
            px = float(Q.b76_price(F, K, T, 0.065, iv, call))
            pts.append(dict(smile_id=sid, strike=float(K), kind="call" if call else "put", price=round(px, 2), age_min=1.0 + (j % 3), log_moneyness=x, iv=float(iv), used=True, forward=F, T_days=T * 365))
        rows.append(sid)
    points = pd.DataFrame(pts)
    br = Q.bracket_table(points, rows)
    atm = Q.interpolate_atm(br.log_moneyness_lo, br.log_moneyness_hi, br.iv_lo, br.iv_hi)
    al = pd.DataFrame(dict(obs_id=br.smile_id, rv_session_pct=11.0 + rng.normal(0, 1, len(br)), rv_calendar_pct=10.5 + rng.normal(0, 1, len(br)), rv_total_variance=1e-4 * np.ones(len(br)),
                           expiry=[f"2023-01-{1 + k % 28:02d}" for k in range(len(br))], day=[f"2022-12-{1 + k % 28:02d}" for k in range(len(br))], total_variance_ratio_valid=True,
                           T_years=br.T_days / 365))
    al["iv_pct"] = 100 * atm
    al["implied_total_variance"] = atm ** 2 * al.T_years
    al["spread_session"] = al.iv_pct - al.rv_session_pct
    al["spread_calendar"] = al.iv_pct - al.rv_calendar_pct
    return points, br, al, atm


def test_bracket_table_picks_the_adjacent_strikes_and_drops_one_sided_smiles():
    pts, br, al, atm = smile_points(10)
    assert len(br) == 10 and (br.log_moneyness_lo < 0).all() and (br.log_moneyness_hi >= 0).all() and (br.strike_hi - br.strike_lo == 50).all()
    only_calls = pts[pts.kind == "call"]
    assert len(Q.bracket_table(only_calls, br.smile_id)) == 0
    assert Q.interpolate_atm(-0.01, 0.03, 0.10, 0.14) == pytest.approx(0.11)


def test_rowdata_zero_shift_reproduces_the_baseline_and_a_uniform_shift_moves_iv_by_price_over_vega():
    pts, br, al, atm = smile_points(40, 1)
    rd = Q.RowData(al, br)
    iv0 = rd.atm_iv(rd.p_lo, rd.p_hi)
    np.testing.assert_allclose(iv0, atm, atol=1e-5)                                         # prices are rounded to the 0.05 tick
    f = rd.frame(rd.atm_iv(rd.p_lo, rd.p_hi))
    np.testing.assert_allclose(f.spread_calendar, al.spread_calendar, atol=1e-3)
    pl, ph = Q.scenario_prices(rd, "abs", 1.0)
    d_iv = rd.atm_iv(pl, ph) - iv0
    np.testing.assert_allclose(d_iv, 1.0 / rd.vega_atm() / 100, rtol=0.05)                  # first-order: +Rs1 raises the IV by about 1/vega vol points
    assert (d_iv > 0).all()
    pl2, ph2 = Q.scenario_prices(rd, "abs", -1.0)
    assert (rd.atm_iv(pl2, ph2) < iv0).all()


def test_scenario_prices_kinds_floor_determinism_and_errors():
    pts, br, al, atm = smile_points(12, 2)
    rd = Q.RowData(al, br)
    pl, ph = Q.scenario_prices(rd, "rel", 0.1)
    np.testing.assert_allclose(pl, np.maximum(rd.p_lo * 1.1, Q.TICK))
    pl, ph = Q.scenario_prices(rd, "abs", -1e6)
    assert (pl == Q.TICK).all() and (ph == Q.TICK).all()                                     # floored at the tick
    pl, ph = Q.scenario_prices(rd, "opposite", 1.0)
    np.testing.assert_allclose(pl, rd.p_lo + np.where(rd.call_lo, 1.0, -1.0))
    a = Q.scenario_prices(rd, "noise_abs", 1.0, np.random.default_rng(5))
    b = Q.scenario_prices(rd, "noise_abs", 1.0, np.random.default_rng(5))
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], Q.scenario_prices(rd, "noise_abs", 1.0, np.random.default_rng(6))[0])
    with pytest.raises(ValueError):
        Q.scenario_prices(rd, "bid_ask", 1.0)


def test_scenario_frame_drops_unsolved_rows_and_recomputes_both_bases():
    pts, br, al, atm = smile_points(15, 3)
    rd = Q.RowData(al, br)
    iv = rd.atm_iv(rd.p_lo, rd.p_hi)
    iv_bad = iv.copy()
    iv_bad[2] = np.nan
    f = rd.frame(iv_bad)
    assert len(f) == len(al) - 1
    np.testing.assert_allclose(f.iv_pct, 100 * np.delete(iv, 2))
    np.testing.assert_allclose(f.spread_calendar, f.iv_pct - f.rv_calendar_pct)
    np.testing.assert_allclose(f.implied_total_variance, np.delete(iv, 2) ** 2 * np.delete(rd.T, 2))


def test_tipping_bias_identities_and_eiv_correction():
    pts, br, al, atm = smile_points(30, 4)
    rd = Q.RowData(al, br)
    t = Q.tipping_bias(rd, 1.8, 1.44)
    assert t["iv_factor_for_unit_ratio"] == pytest.approx(1 / 1.2) and t["iv_reduction_pct_for_unit_ratio"] == pytest.approx((1 - 1 / 1.2) * 100)
    assert t["price_bias_rs_median"] == pytest.approx(float(np.median(1.8 * rd.vega_atm())))
    assert t["price_bias_pct_of_price_median"] == pytest.approx(float(np.median(1.8 * rd.vega_atm() / rd.price_atm()) * 100))
    assert Q.eiv_corrected_slope(0.8, 4.0, 1.0) == pytest.approx(0.8 / 0.75)
    assert math.isnan(Q.eiv_corrected_slope(0.8, 1.0, 1.0))
    assert Q.eiv_corrected_slope(0.8, 4.0, 0.0) == 0.8


def quote(strike, kind, price, age=1.0):
    return OptionQuote(float(strike), OptionType.CALL if kind == "call" else OptionType.PUT, price, age)


def test_callput_disagreement_reads_both_sides_and_applies_the_filters():
    F, T, r = 18000.0, 0.03, 0.065
    qs = []
    for K in (17900, 18000, 18100):
        for kind in ("call", "put"):
            qs.append(quote(K, kind, round(float(Q.b76_price(F, K, T, r, 0.15, kind == "call")), 2)))
    d = Q.callput_disagreement(qs, F, T, r)
    assert len(d) == 3 and d.diff_vol_points.abs().max() < 0.01                               # same IV on both sides: only tick rounding
    qs2 = [quote(18000, "call", round(float(Q.b76_price(F, 18000, T, r, 0.16, True)), 2)), quote(18000, "put", round(float(Q.b76_price(F, 18000, T, r, 0.15, False)), 2))]
    assert Q.callput_disagreement(qs2, F, T, r).diff_vol_points.iloc[0] == pytest.approx(1.0, abs=0.01)
    assert len(Q.callput_disagreement([quote(18000, "call", 100.0, age=6.0), quote(18000, "put", 100.0)], F, T, r)) == 0       # stale call
    assert len(Q.callput_disagreement([quote(19000, "call", 50.0), quote(19000, "put", 900.0)], F, T, r, moneyness_max=0.01)) == 0   # far from the money
    assert len(Q.callput_disagreement([quote(18000, "call", 100.0)], F, T, r)) == 0           # one-sided


# ================================================================================================ builder guards and end to end
def test_gate3a_builder_refuses_output_outside_stage2d(tmp_path):
    with pytest.raises(ValueError, match="stage2d"):
        B3.build("a", "b", "c", str(tmp_path / "gate3"))


REAL = all(os.path.exists(p) for p in ("research_output/stage2c/annualization_audit/aligned_observations.csv", "research_output/stage2a/full/points.csv", "data/hist1m/NIFTY50_1m.parquet",
                                          "research_output/stage2c/iv_rv_observations.csv"))


@pytest.mark.skipif(not REAL, reason="local Stage 2A/2C artifacts not present")
def test_gate3a_end_to_end_on_real_artifacts_is_deterministic_and_touches_nothing_else(tmp_path):
    import hashlib
    watch = ["research_output/stage2a/full/smiles.csv", "research_output/stage2c/annualization_audit/aligned_observations.csv", "research_output/stage2c/REPORT.md",
             "research_output/stage2d/gate2/REPORT.md"]
    before = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in watch if os.path.exists(p)}
    kw = dict(perm_draws=20, boot_reps=15, mc_draws=3, cp_expiries_per_year=1, cp_snaps_per_expiry=2)
    o1, o2 = tmp_path / "stage2d" / "a", tmp_path / "stage2d" / "b"
    m1 = B3.build("research_output/stage2c/annualization_audit/aligned_observations.csv", "research_output/stage2c", "research_output/stage2a/full", str(o1), **kw)
    B3.build("research_output/stage2c/annualization_audit/aligned_observations.csv", "research_output/stage2c", "research_output/stage2a/full", str(o2), **kw)
    assert before == {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in before}
    names = sorted(f for f in os.listdir(o1) if f.endswith(".csv"))
    for need in ("t9_universe_counts.csv", "t9_balance.csv", "t9_outcome_dependence_test.csv", "t9_ipw_estimates.csv", "t9_manski_bounds.csv", "t9_strike_universe_check.csv",
                 "t9_dte_composition_diagnostic.csv", "t10_atm_reconstruction.csv", "t10_price_bias_scenarios.csv", "t10_random_noise_scenarios.csv", "t10_tipping_bias.csv",
                 "t10_quote_age_gradient.csv", "t10_call_put_disagreement.csv", "t10_eiv_scenarios.csv"):
        assert need in names, need
    for f in names:
        pd.testing.assert_frame_equal(pd.read_csv(o1 / f), pd.read_csv(o2 / f))
    assert (m1["n_observed"], m1["n_l1"], m1["n_l2"]) == (6490, 73, 115)
    rec = pd.read_csv(o1 / "t10_atm_reconstruction.csv").iloc[0]
    assert rec.max_abs_resolved_atm_iv_diff_vs_stored < 1e-8 and rec.n_unsolved == 0
    assert bool(pd.read_csv(o1 / "t9_strike_universe_check.csv").passes.iloc[0])
    diag = pd.read_csv(o1 / "t9_dte_composition_diagnostic.csv")
    assert diag.note.str.contains("diagnostic").all() and not any(c for c in diag.columns if "spread" in c or "ratio" in c)
    sc = pd.read_csv(o1 / "t10_price_bias_scenarios.csv")
    assert sc.label.str.contains("ASSUMPTION").all()
    zero = sc[(sc.scenario.str.startswith("baseline")) & (sc.estimand == "S1_calendar_all_obs_median")].estimate.iloc[0]
    assert zero == pytest.approx(1.804903, abs=1e-4)
