"""End-to-end Stage 2B driver on a tiny synthetic dataset (hand-checkable)."""
import hashlib
import json
import math
import os
from datetime import date

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from optionsengine.research import build_rv as b  # noqa: E402
from tests.rv_synth import alt_returns, frame, path_from_returns, session_rows  # noqa: E402

DAYS = [date(2026, 9, 14), date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18), date(2026, 9, 21), date(2026, 9, 22)]


def make(tmp_path, with_sat=True, hole=False):
    rows, price = [], 100.0
    for k, d in enumerate(DAYS):
        closes = path_from_returns(price, alt_returns(375, 0.0005 + 0.0001 * k))
        opens = [price] + closes[:-1]
        c2 = list(closes)
        if hole and k == 3:
            c2[120] = None
        rows += session_rows(d, c2, opens=opens)
        price = closes[-1]
    if with_sat:
        rows += session_rows(date(2026, 9, 19), [100.0] * 45)
    rows = sorted(rows)
    spot = tmp_path / "spot.parquet"
    frame(rows).astype({"ts": "int64"}).to_parquet(spot)
    pdir = tmp_path / "participant"
    pdir.mkdir(exist_ok=True)
    for d in DAYS:
        (pdir / f"{d.isoformat()}.csv").write_text("x")
    return str(spot), str(pdir)


def test_outputs_schema_and_values(tmp_path):
    spot, pdir = make(tmp_path)
    out = tmp_path / "out"
    meta = b.build(spot, str(out), pdir, with_sensitivity=False)
    assert meta["n_sessions"] == 8 and meta["regular_bars"] == 375
    dq = pd.read_csv(out / "data_quality.csv"); daily = pd.read_csv(out / "daily_rv.csv"); roll = pd.read_csv(out / "rolling_rv.csv")
    assert list(dq.columns) == list(b.DQ_COLUMNS)
    assert dq.session_type.value_counts().to_dict() == {"regular": 7, "special_weekend": 1}
    assert dq[dq.session_type == "regular"].complete.all()
    r = daily[daily.session_date == "2026-09-14"].iloc[0]
    assert r.intraday_vol_ann_pct == pytest.approx(100 * 0.0005 * math.sqrt(375 * 252), rel=1e-9)
    assert set(roll.method) == {"strict", "coverage_qualified"} and set(roll.measure) == {"close_to_close", "intraday", "hybrid"}
    assert set(roll.window_sessions) == {5, 10, 20}
    assert json.load(open(out / "run_metadata.json"))["annualization_days"] == 252


def test_incomplete_session_is_recorded_and_excluded_from_strict(tmp_path):
    spot, pdir = make(tmp_path, hole=True)
    out = tmp_path / "out"
    b.build(spot, str(out), pdir, with_sensitivity=False)
    dq = pd.read_csv(out / "data_quality.csv").set_index("session_date")
    assert dq.loc["2026-09-17", "n_missing_minutes"] == 1 and not dq.loc["2026-09-17", "complete"]
    assert "missing_minutes" in dq.loc["2026-09-17", "exclusion_reasons"]
    daily = pd.read_csv(out / "daily_rv.csv").set_index("session_date")
    assert math.isnan(daily.loc["2026-09-17", "intraday_variance"]) and daily.loc["2026-09-17", "n_valid_returns"] == 373
    roll = pd.read_csv(out / "rolling_rv.csv")
    w = roll[(roll.measure == "intraday") & (roll.window_sessions == 5) & (roll.method == "strict") & (roll.session_date == "2026-09-21")].iloc[0]
    assert w.n_obs == 4 and not w.complete and math.isnan(w.rv_ann_pct)


def test_source_files_unchanged_and_runs_are_deterministic(tmp_path):
    spot, pdir = make(tmp_path)
    h = hashlib.sha256(open(spot, "rb").read()).hexdigest()
    b.build(spot, str(tmp_path / "o1"), pdir, with_sensitivity=True)
    b.build(spot, str(tmp_path / "o2"), pdir, with_sensitivity=True)
    assert hashlib.sha256(open(spot, "rb").read()).hexdigest() == h
    for f in ("daily_rv.csv", "rolling_rv.csv", "data_quality.csv", "sensitivity.csv"):
        assert open(tmp_path / "o1" / f).read() == open(tmp_path / "o2" / f).read()


def test_sensitivity_table_shape_and_reference_row(tmp_path):
    spot, pdir = make(tmp_path)
    b.build(spot, str(tmp_path / "o"), pdir, with_sensitivity=True)
    s = pd.read_csv(tmp_path / "o" / "sensitivity.csv")
    assert {"min_session_coverage", "min_window_fraction", "measure", "window_sessions", "share_with_value"} <= set(s.columns)
    ref = s[(s.min_session_coverage == 1.0) & (s.min_window_fraction == 1.0)]
    assert (ref.median_abs_diff_vs_strict_volpts.dropna() == 0).all()           # coverage 1.0 / fraction 1.0 == strict
