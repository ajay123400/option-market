import math

import numpy as np
import pandas as pd
import pytest

from optionsengine.research import iv_rv_stats as st


def frame(iv, rv, days=None, expiries=None):
    n = len(iv)
    df = pd.DataFrame(dict(iv_pct=iv, future_rv_pct=rv, day=days or [f"2025-01-{(i % 28) + 1:02d}" for i in range(n)],
                           expiry=expiries or ["2025-02-06"] * n, time="10:00"))
    df["iv_minus_rv"] = df.iv_pct - df.future_rv_pct
    df["iv_over_rv"] = np.where(df.future_rv_pct > 0, df.iv_pct / df.future_rv_pct.replace(0, np.nan), np.nan)
    df["abs_err"] = df.iv_minus_rv.abs()
    df["sq_err"] = df.iv_minus_rv ** 2
    return df


def test_describe_hand_values():
    g = frame([12.0, 14.0, 10.0, 16.0], [10.0, 15.0, 10.0, 8.0], days=["a", "a", "b", "c"], expiries=["e1", "e1", "e2", "e2"])
    r = st.describe(g)
    assert r["n_obs"] == 4 and r["unique_dates"] == 3 and r["unique_expiries"] == 2
    assert r["snapshots_per_date_max"] == 2 and r["snapshots_per_date_median"] == 1.0
    assert r["iv_mean"] == 13.0 and r["rv_mean"] == 10.75 and r["iv_median"] == 13.0 and r["rv_median"] == 10.0
    assert r["diff_mean"] == pytest.approx(2.25) and r["diff_median"] == pytest.approx(1.0)          # diffs 2,-1,0,8
    assert r["frac_iv_gt_rv"] == 0.5 and r["frac_iv_lt_rv"] == 0.25 and r["frac_iv_eq_rv"] == 0.25
    assert r["mae"] == pytest.approx(2.75) and r["mse"] == pytest.approx((4 + 1 + 0 + 64) / 4) and r["rmse"] == pytest.approx(math.sqrt(17.25))
    assert r["diff_std"] == pytest.approx(float(np.std([2, -1, 0, 8], ddof=1)))
    assert r["diff_p50"] == pytest.approx(1.0) and r["diff_p99"] == pytest.approx(float(np.percentile([2, -1, 0, 8], 99)))


def test_ratio_is_not_computed_when_future_rv_is_zero():
    g = frame([12.0, 14.0, 10.0], [0.0, 7.0, 5.0])
    assert math.isnan(g.iv_over_rv.iloc[0])
    r = st.describe(g)
    assert r["ratio_n_valid"] == 2 and r["ratio_median"] == pytest.approx(np.median([2.0, 2.0])) and r["ratio_mean"] == pytest.approx(2.0)
    z = st.describe(frame([12.0, 14.0], [0.0, 0.0]))
    assert z["ratio_n_valid"] == 0 and math.isnan(z["ratio_median"])


def test_correlations_hand_values_and_degenerate_cases():
    g = frame([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0])
    r = st.describe(g)
    assert r["corr_pearson"] == pytest.approx(1.0) and r["corr_spearman"] == pytest.approx(1.0)
    c = st.describe(frame([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))
    assert math.isnan(c["corr_pearson"])


def test_percentiles_and_empty_group():
    r = st.describe(frame([10.0], [4.0]))
    assert r["n_obs"] == 1 and r["diff_p50"] == 6.0 and math.isnan(r["diff_std"])
    assert st.describe(frame([], []))["n_obs"] == 0


def test_block_bootstrap_is_deterministic_and_respects_dates():
    rng = np.random.default_rng(0)
    n = 600
    days = np.array([f"d{i // 3:04d}" for i in range(n)])                 # 200 dates, 3 rows each
    vals = rng.normal(2.0, 1.0, n)
    a = st.block_bootstrap(vals, days, {"mean": np.mean}, 5, reps=200, seed=1)
    b = st.block_bootstrap(vals, days, {"mean": np.mean}, 5, reps=200, seed=1)
    assert a == b
    lo, hi = a["mean"]
    assert lo < vals.mean() < hi and 0 < hi - lo < 0.6


def test_block_bootstrap_widens_interval_for_serially_dependent_data():
    rng = np.random.default_rng(2)
    nd = 300
    base = np.cumsum(rng.normal(0, 0.3, nd))                             # strongly autocorrelated date-level signal
    days = np.repeat([f"d{i:04d}" for i in range(nd)], 2)
    vals = np.repeat(base, 2) + rng.normal(0, 0.05, nd * 2)
    iid_like = st.block_bootstrap(vals, days, {"mean": np.mean}, 1, reps=300, seed=3)["mean"]
    blocked = st.block_bootstrap(vals, days, {"mean": np.mean}, 20, reps=300, seed=3)["mean"]
    assert (blocked[1] - blocked[0]) > 1.5 * (iid_like[1] - iid_like[0])   # ignoring dependence would understate uncertainty


def test_bootstrap_returns_nan_for_tiny_samples_and_ci_columns_named():
    assert math.isnan(st.block_bootstrap(np.arange(5.0), np.array(list("abcde")), {"m": np.mean}, 3)["m"][0])
    r = st.describe(frame(list(np.linspace(10, 15, 80)), list(np.linspace(8, 12, 80))), ci_horizon="F5", reps=50)
    assert "diff_median_ci_lo" in r and "ci_method" in r and "block=5" in r["ci_method"]


def test_block_lengths():
    assert [st.block_length(h) for h in ("F1", "F5", "F10", "F20", "EXP")] == [5, 5, 10, 20, 10]


def test_summarize_groups_and_never_ranks():
    g = frame([12.0, 14.0, 13.0, 15.0], [10.0, 10.0, 10.0, 10.0])
    g["horizon"] = ["F1", "F1", "F5", "F5"]
    out = st.summarize(g, ["horizon"])
    assert list(out.horizon) == ["F1", "F5"] and "rank" not in " ".join(out.columns).lower() and "best" not in " ".join(out.columns).lower()
