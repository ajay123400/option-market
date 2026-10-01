"""Parity-forward estimator and its quality gate (regression tests for inconsistent parity data)."""
import math

import pytest

from optionsengine import (ForwardConfig, ForwardStatus, InvalidInputError, ParityObservation,
                           estimate_parity_forward)

F_TRUE, R, T, SPOT = 24500.0, 0.065, 7 / 365, 24500.0
STRIKES = [24400, 24450, 24500, 24550, 24600, 24650, 24700, 24350]
BPS = 1e-4 * SPOT       # 2.45 index points


def obs(strikes=STRIKES, noise=None, f=F_TRUE, r=R, t=T):
    """Parity-consistent call/put prices around forward f with optional additive errors (in index points)
    applied to (C - P) -- i.e. exactly the shape of error a stale or wide quote produces."""
    noise = noise or [0.0] * len(strikes)
    out = []
    for k, n in zip(strikes, noise):
        diff = (f - k) * math.exp(-r * t) + n * math.exp(-r * t)      # C - P
        base = 40.0 + abs(f - k) * 0.5
        out.append(ParityObservation(float(k), base + max(diff, 0) + 0.5 * 0, base - min(diff, 0)))
    return out


def est(o, **kw):
    return estimate_parity_forward(o, R, T, SPOT, ForwardConfig(**kw) if kw else ForwardConfig())


def test_consistent_observations_give_exact_forward():
    e = est(obs())
    assert e.status is ForwardStatus.OK and e.ok
    assert e.forward == pytest.approx(F_TRUE, abs=1e-9) and e.candidate_forward == e.forward
    assert e.n_candidates == 6 and e.n_inliers == 6 and e.rejected_strikes == () and e.dispersion < 1e-9


def test_uses_the_nearest_strikes_only():
    # far strikes carry large errors; nearest-6 selection must exclude them
    far = obs(noise=[0, 0, 0, 0, 0, 0, 500.0, -500.0])      # 24700 and 24350 are the 7th/8th nearest
    e = est(far)
    assert e.ok and e.forward == pytest.approx(F_TRUE, abs=1e-9) and e.n_candidates == 6


def test_single_outlier_is_rejected_and_forward_is_unbiased():
    noisy = obs(noise=[0.2, -0.1, 0.0, 0.15, 60.0, -0.2])[:6]       # one stale strike 60 pts off
    e = est(noisy)
    assert e.status is ForwardStatus.OK
    assert e.rejected_strikes == (24600.0,) and e.n_inliers == 5
    assert e.forward == pytest.approx(F_TRUE, abs=0.3)               # outlier did not drag the estimate


def test_outlier_floor_protects_tight_clusters_from_over_rejection():
    # MAD ~ 0.05 pts, but 1-bps floor (2.45 pts) means a 1.0-pt deviation is NOT an outlier
    e = est(obs(noise=[0.05, 0.0, -0.05, 0.0, 1.0, 0.0])[:6])
    assert e.ok and e.rejected_strikes == () and e.n_inliers == 6


def test_dispersion_above_limit_is_low_confidence_with_no_forward():
    # six strikes spread over +/-25 pts: all mutually inconsistent, none an outlier of the others
    spread = [-25.0, -15.0, -5.0, 5.0, 15.0, 25.0]
    e = est(obs(noise=spread)[:6])
    assert e.status is ForwardStatus.LOW_CONFIDENCE and not e.ok
    assert e.forward is None                                          # never hand out an unreliable forward
    assert e.candidate_forward == pytest.approx(F_TRUE, abs=6)       # exposed for diagnostics only
    assert e.dispersion > e.dispersion_limit and "dispersion" in e.reason


def _linear_spread(total_range):
    a = total_range / 2
    return [-a, -0.6 * a, -0.2 * a, 0.2 * a, 0.6 * a, a]          # evenly spread: no point is an outlier of the rest


def test_dispersion_threshold_boundary():
    limit_pts = ForwardConfig().max_dispersion_bps * BPS             # 12.25 pts
    inside = est(obs(noise=_linear_spread(limit_pts * 0.9))[:6])
    just_over = est(obs(noise=_linear_spread(limit_pts * 1.1))[:6])
    assert inside.ok and inside.dispersion == pytest.approx(limit_pts * 0.9, rel=1e-6)
    assert just_over.status is ForwardStatus.LOW_CONFIDENCE and just_over.rejected_strikes == ()


def test_threshold_is_configurable():
    spread = _linear_spread(8.0)                                     # 8 pt range = 3.3 bps
    assert est(obs(noise=spread)[:6]).ok                             # default limit 5 bps
    assert est(obs(noise=spread)[:6], max_dispersion_bps=1.0).status is ForwardStatus.LOW_CONFIDENCE


def test_lone_extreme_points_on_a_tight_cluster_are_removed_as_outliers():
    # documents the outlier rule: with a tight core (MAD ~ 0) deviations above the 1-bp floor are outliers
    noisy = obs(noise=[-7.0, 0, 0, 0, 0, 7.0])[:6]
    e = est(noisy)
    assert e.ok and set(e.rejected_strikes) == {24400.0, 24650.0} and e.n_inliers == 4


def test_too_many_outliers_is_low_confidence():
    bad = obs(noise=[0, 0, 0, 80.0, -90.0, 100.0])[:6]               # half the strikes are junk
    e = est(bad, max_outlier_fraction=0.4)
    assert e.status is ForwardStatus.LOW_CONFIDENCE and e.forward is None


def test_too_few_pairs_is_unavailable():
    for n in (0, 1, 2):
        e = est(obs()[:n])
        assert e.status is ForwardStatus.UNAVAILABLE and e.forward is None and e.candidate_forward is None
        assert "min_pairs" in e.reason


def test_invalid_pairs_are_dropped_and_counted_not_trusted():
    good = obs()[:3]
    junk = [ParityObservation(24500.0, float("nan"), 50.0), ParityObservation(24500.0, 0.0, 50.0),
            ParityObservation(24500.0, 50.0, -1.0), ParityObservation(-5.0, 50.0, 50.0)]
    e = est(good + junk)
    assert e.n_dropped_invalid == 4 and e.n_input == 7 and e.n_candidates == 3 and e.ok
    only_junk = est(junk)
    assert only_junk.status is ForwardStatus.UNAVAILABLE


def test_three_candidates_cannot_identify_an_outlier_so_dispersion_decides():
    e = est(obs(noise=[0.0, 40.0, 0.0])[:3])        # 24400, 24450, 24500 with one 40-pt error
    assert e.n_candidates == 3 and e.rejected_strikes == ()          # no outlier removal with < 4 candidates
    assert e.status is ForwardStatus.LOW_CONFIDENCE                  # ... but the dispersion gate catches it


def test_systematic_stale_bias_is_flagged_when_not_uniform():
    # ITM strikes stale (lag by 30 pts), ATM strikes fresh -> inconsistent set
    e = est(obs(noise=[30.0, 30.0, 0.0, 0.0, -30.0, -30.0])[:6])
    assert e.status is ForwardStatus.LOW_CONFIDENCE and e.forward is None


def test_result_is_deterministic_and_order_independent():
    o = obs(noise=[0.3, -0.2, 0.1, 0.0, 0.2, -0.1])[:6]
    a, b = est(o), est(list(reversed(o)))
    assert a == b


def test_forward_depends_on_r_only_through_discounting():
    o = obs()
    f1 = estimate_parity_forward(o, 0.055, T, SPOT).forward
    f2 = estimate_parity_forward(o, 0.075, T, SPOT).forward
    assert abs(f1 - f2) < 0.2      # +/-1pp of r moves a one-week forward by well under a point


@pytest.mark.parametrize("r,t,spot", [(float("nan"), T, SPOT), (R, 0.0, SPOT), (R, -1.0, SPOT), (R, T, 0.0),
                                      (R, T, float("inf"))])
def test_invalid_scalars_raise(r, t, spot):
    with pytest.raises(InvalidInputError):
        estimate_parity_forward(obs(), r, t, spot)


@pytest.mark.parametrize("kw", [dict(min_pairs=1), dict(n_nearest=2, min_pairs=3), dict(outlier_mad_k=0),
                                dict(max_dispersion_bps=0), dict(max_outlier_fraction=1.5)])
def test_invalid_config_rejected(kw):
    with pytest.raises(InvalidInputError):
        ForwardConfig(**kw)


def test_forward_feeds_iv_pipeline_consistently():
    """End to end: a gated forward -> carry yield -> BSM IV recovers the generating vol."""
    from optionsengine import bsm_price, implied_carry_yield, implied_volatility
    e = est(obs())
    q = implied_carry_yield(SPOT, e.forward, T, R)
    K = 24700.0
    p = bsm_price(SPOT, K, T, R, q, 0.17, "call")
    assert implied_volatility(p, SPOT, K, T, R, q, "call").iv == pytest.approx(0.17, abs=1e-8)
