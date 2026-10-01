"""Hand-countable checks for the Stage 2A report tables."""
import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from optionsengine.research import report  # noqa: E402


def smiles():
    rows = [
        # smile_id, split, forward_status, T_days, atm_iv, atm_loose_iv, rr25, expiry_day
        ("a", "dev", "ok", 10.0, 0.10, 0.10, -0.01, False),
        ("b", "dev", "ok", 10.0, 0.14, 0.14, -0.02, False),
        ("c", "dev", "ok", 10.0, 0.20, 0.20, np.nan, False),
        ("d", "dev", "ok", 0.5, np.nan, 0.30, np.nan, True),
        ("e", "dev", "low_confidence", 10.0, np.nan, np.nan, np.nan, False),
        ("f", "holdout", "ok", 10.0, 0.25, 0.25, -0.03, False),
        ("g", "holdout", "unavailable", 10.0, np.nan, np.nan, np.nan, False),
    ]
    d = pd.DataFrame(rows, columns=["smile_id", "split", "forward_status", "T_days", "atm_iv", "atm_loose_iv", "rr25", "expiry_day"])
    d["group_reason"] = d.forward_status.map(lambda x: "forward_unavailable" if x == "unavailable" else None)
    d["n_itm_side_excluded"] = 5
    d["n_strikes_without_otm_quote"] = 1
    return d


def test_forward_status_rates_hand_counts():
    t = report.forward_status_rates(smiles())
    assert t.loc["dev", "ok"] == 4 and t.loc["dev", "low_confidence"] == 1 and t.loc["dev", "total"] == 5
    assert t.loc["dev", "ok_share"] == 0.8 and t.loc["holdout", "unavailable"] == 1


def test_smile_coverage_counts_only_primary_smiles():
    t = report.smile_coverage(smiles())
    row = t.loc[("dev", "7-14d")]
    assert row.smiles == 3 and row.atm == 3 and row.rr_eligible == 3 and row.rr_bf == 2
    assert t.loc[("dev", "<=1d")].atm == 0 and t.loc[("dev", "<=1d")].atm_loose == 1      # loose ATM labelled separately


def test_regime_cutoffs_use_development_only():
    s = smiles()
    cuts = report.regime_cutoffs(s)
    dev_vals = [0.10, 0.14, 0.20]
    assert cuts["low_high"] == pytest.approx(float(np.quantile(dev_vals, 1 / 3)))
    s.loc[s.smile_id == "f", "atm_iv"] = 9.99                      # a wild holdout value must not move the cut-offs
    assert report.regime_cutoffs(s) == cuts


def test_point_exclusion_accounting_adds_up():
    s = smiles().iloc[:1]
    p = pd.DataFrame({"split": ["dev"] * 4, "used": [True, False, False, False],
                      "exclusion": [None, "stale", "iv_failed:below_lower_bound", "resolution_limited"]})
    t = report.point_exclusions(s, p)
    r = t.loc["dev"]
    assert r.otm_points == 4 and r.used == 1 and r["excl:stale"] == 1 and r["excl:iv_failed"] == 1
    assert r["excl:resolution_limited"] == 1 and r.itm_side_not_used == 5 and r.strikes_without_otm_quote == 1


def test_bucket_boundaries():
    d = report.add_buckets(pd.DataFrame({"T_days": [0.5, 1.0, 1.0001, 7.0, 30.0, 31.0], "m": [0.0, 0.005, 0.0051, 0.01, 0.12, 0.5]}), m_col="m")
    assert list(d.T_bucket.astype(str)) == ["<=1d", "<=1d", "1-3d", "3-7d", "14-30d", ">30d"]
    assert list(d.m_bucket.astype(str)) == ["<0.5%", "<0.5%", "0.5-1%", "0.5-1%", "7-12%", ">12%"]


def test_forward_by_tenor_hand_counts():
    t = report.forward_by_tenor(smiles())
    row = t.loc["7-14d"]
    assert row.ok == 4 and row.low_confidence == 1 and row.unavailable == 1 and row.total == 6
    assert t.loc["<=1d"].ok == 1
