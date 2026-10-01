"""End-to-end Stage 2C on a tiny synthetic Stage 2A + spot dataset (hand-checkable)."""
import hashlib
import json
import math
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from optionsengine.research import build_iv_rv as b  # noqa: E402
from tests.rv_synth import frame, path_from_returns, session_rows  # noqa: E402

# 24 weekdays: Dec 2024 (development) -> Jan 2025 (holdout)
DAYS = []
d = date(2024, 12, 9)
while len(DAYS) < 24:
    if d.weekday() < 5:
        DAYS.append(d)
    d += timedelta(days=1)
VOL = {i: 0.0004 + 0.00005 * (i % 5) for i in range(24)}
CONST_IDX = {6}                                    # one session with constant prices => zero RV


def make_spot(tmp_path):
    rows, price = [], 100.0
    for i, day in enumerate(DAYS):
        rets = [0.0] * 375 if i in CONST_IDX else [VOL[i] if k % 2 == 0 else -VOL[i] for k in range(375)]
        closes = path_from_returns(price, rets)
        rows += session_rows(day, closes, opens=[price] + closes[:-1])
        price = closes[-1]
    p = tmp_path / "spot.parquet"
    frame(sorted(rows)).astype({"ts": "int64"}).to_parquet(p)
    pdir = tmp_path / "participant"; pdir.mkdir()
    for day in DAYS:
        (pdir / f"{day.isoformat()}.csv").write_text("x")
    return str(p), str(pdir)


def spot_close(spot_path, day, hhmm):
    df = pd.read_parquet(spot_path)
    h, m = (int(x) for x in hhmm.split(":"))
    ts = int(pd.Timestamp(f"{day.isoformat()} {hhmm}", tz="Asia/Kolkata").timestamp())
    return float(df[df.ts == ts].close.iloc[0])


def smile_row(sid, day, time, expiry, split, status="ok", t_days=5.0, atm=0.15, loose=None, exp_day=False, spot=100.0):
    return dict(smile_id=sid, day=day.isoformat(), time=time, expiry=expiry.isoformat(), split=split, expiry_day=exp_day, spot=spot,
                T_days=t_days, n_quotes=50, group_reason=None, forward_status=status, primary=(status == "ok"), forward_used=spot,
                atm_iv=atm, atm_k_low=99.0, atm_k_high=101.0, atm_reason=None, atm_loose_iv=loose if loose is not None else atm,
                atm_loose_unreliable_inputs=0.0 if loose is None else 1.0)


def make_stage2a(tmp_path, spot_path, holdout_iv=0.15):
    rows = []
    rows.append(smile_row("p1", DAYS[1], "10:00", DAYS[8], "dev", t_days=7.2, atm=0.14))          # primary, 7-14d
    rows.append(smile_row("p2", DAYS[2], "13:00", DAYS[8], "dev", t_days=6.0, atm=0.16))          # primary, 3-7d
    rows.append(smile_row("p3", DAYS[3], "15:00", DAYS[8], "dev", t_days=5.0, atm=0.18))
    rows.append(smile_row("p4", DAYS[4], "10:00", DAYS[8], "dev", t_days=4.0, atm=0.20))
    rows.append(smile_row("hold", DAYS[16], "10:00", DAYS[22], "holdout", t_days=6.0, atm=holdout_iv))
    rows.append(smile_row("far", DAYS[1], "10:00", DAYS[20], "dev", t_days=15.5, atm=0.14))      # >14 DTE: counted, not analysed
    rows.append(smile_row("lowc", DAYS[1], "13:00", DAYS[8], "dev", status="low_confidence", t_days=7.0, atm=np.nan, loose=np.nan))
    rows.append(smile_row("expd", DAYS[8], "10:00", DAYS[8], "dev", t_days=0.2, atm=0.30, exp_day=True))
    rows.append(smile_row("zero", DAYS[5], "10:00", DAYS[9], "dev", t_days=3.5, atm=0.12))        # next session (6) has constant prices
    rows.append(smile_row("wk", DAYS[0], "10:00", DAYS[3], "dev", t_days=3.2, atm=0.13))             # Mon -> Thu: no weekend inside
    loose = smile_row("loose", DAYS[3], "13:00", DAYS[9], "dev", t_days=6.0, atm=np.nan, loose=0.25)
    rows.append(loose)
    sm = pd.DataFrame(rows)
    sm["atm_iv"] = sm.atm_iv.astype(float)
    pts = []
    for sid in sm.smile_id:
        pts.append(dict(smile_id=sid, strike=99.0, kind="put", log_moneyness=-0.01, iv=0.15, iv_low=0.149, iv_high=0.151,
                        reliable=True, resolution_limited=False, used=True, exclusion=None, forward_status="ok", T_days=5.0, time="10:00",
                        day="2024-12-10", split="dev"))
        wide = 0.02 if sid == "p4" else 0.0005                  # p4: +/-2 vol points tick uncertainty => resolution-sensitive
        pts.append(dict(smile_id=sid, strike=101.0, kind="call", log_moneyness=0.01, iv=0.15, iv_low=0.15 - wide, iv_high=0.15 + wide,
                        reliable=True, resolution_limited=False, used=True, exclusion=None, forward_status="ok", T_days=5.0, time="10:00",
                        day="2024-12-10", split="dev"))
    d = tmp_path / "s2a"; d.mkdir()
    sm.to_csv(d / "smiles.csv", index=False)
    pd.DataFrame(pts).to_csv(d / "points.csv", index=False)
    return str(d)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return run(tmp_path_factory.mktemp("iv_rv"))


def run(tmp_path, **kw):
    spot, pdir = make_spot(tmp_path)
    # spot used by the smiles must equal the spot file at the snapshot bar: patch smiles spot after building
    s2a = make_stage2a(tmp_path, spot, **kw)
    sm = pd.read_csv(os.path.join(s2a, "smiles.csv"))
    sm["spot"] = [spot_close(spot, date.fromisoformat(r.day), r.time) for r in sm.itertuples()]
    sm.to_csv(os.path.join(s2a, "smiles.csv"), index=False)
    out = tmp_path / "out"
    meta = b.build(s2a, spot, str(out), pdir, reps=60, reps_secondary=30)
    return meta, out, spot


def test_populations_universe_and_filters(ctx):
    meta, out, _ = ctx
    long = pd.read_csv(out / "iv_rv_observations.csv")
    obs = long.drop_duplicates("obs_id").set_index("obs_id")
    assert set(obs.index[obs.population == "primary"]) == {"p1", "p2", "p3", "p4", "hold", "expd", "zero", "wk"}
    assert list(obs.index[obs.population == "loose"]) == ["loose"]                      # kept separate, never mixed
    assert "far" not in obs.index and "lowc" not in obs.index                          # >14 DTE and LOW_CONFIDENCE excluded
    assert meta["n_primary"] == 8 and meta["n_loose"] == 1 and meta["spot_mismatch_rows"] == 0
    cov = pd.read_csv(out / "data_coverage.csv")
    row = lambda name: cov[(cov.stage.str.startswith(name)) & (cov.split == "all")].iloc[0]
    assert row("S0").n_obs == 11 and row("S2").n_obs == 10 and row("S3").n_obs == 9 and row("S4 primary").n_obs == 8 and row("S4b").n_obs == 1
    assert row("DIAG forward OK and >14").n_obs == 1                                    # the 'far' row is counted, not analysed
    prim = pd.read_csv(out / "summary_by_horizon.csv")
    assert set(prim.population) == {"primary", "loose"}
    assert prim[prim.population == "loose"].split_group.tolist() == ["all"] * len(prim[prim.population == "loose"])


def test_iv_minus_rv_values_and_target_metadata(ctx):
    _, out, _ = ctx
    long = pd.read_csv(out / "iv_rv_observations.csv")
    r = long[(long.obs_id == "p1") & (long.horizon == "F1") & (long.measure == "intraday")].iloc[0]
    expected_rv = 100 * math.sqrt(252 * 375 * VOL[2] ** 2)                              # session index 2 follows snapshot day index 1
    assert r.future_rv_pct == pytest.approx(expected_rv, rel=1e-9)
    assert r.iv_pct == pytest.approx(14.0) and r.iv_minus_rv == pytest.approx(14.0 - expected_rv, rel=1e-9)
    assert r.iv_over_rv == pytest.approx(14.0 / expected_rv) and r.abs_err == pytest.approx(abs(r.iv_minus_rv)) and r.sq_err == pytest.approx(r.iv_minus_rv ** 2)
    assert not r.includes_overnight and r.n_valid_sessions == 1 and bool(r.complete)
    assert r.target_start_ts.startswith(DAYS[2].isoformat() + "T09:15") and r.target_end_ts.startswith(DAYS[2].isoformat() + "T15:30")
    assert r.observation_ts_ist == f"{DAYS[1].isoformat()}T10:01:00+05:30"
    h = long[(long.obs_id == "p1") & (long.horizon == "F1") & (long.measure == "hybrid")].iloc[0]
    assert bool(h.includes_overnight) and h.n_returns == 376


def test_zero_future_rv_has_no_ratio(ctx):
    _, out, _ = ctx
    long = pd.read_csv(out / "iv_rv_observations.csv")
    z = long[(long.obs_id == "zero") & (long.horizon == "F1") & (long.measure == "intraday")].iloc[0]
    assert z.future_rv_pct == 0.0 and z.iv_minus_rv == pytest.approx(12.0) and math.isnan(z.iv_over_rv) and not bool(z.ratio_valid)
    summ = pd.read_csv(out / "summary_by_horizon.csv")
    assert (summ.ratio_n_valid <= summ.n_obs).all()


def test_expiry_aligned_and_expiry_day_handling(ctx):
    _, out, _ = ctx
    long = pd.read_csv(out / "iv_rv_observations.csv")
    e = long[(long.obs_id == "expd") & (long.horizon == "EXP")]
    assert (~e.target_available).all() and (e.unavailable_reason == "expiry_day_partial_session_only").all()
    p = long[(long.obs_id == "p2") & (long.horizon == "EXP") & (long.measure == "intraday")].iloc[0]
    partial_var = 149 * VOL[2] ** 2                                                     # 13:00 snapshot on day index 2
    later = sum(0.0 if i in CONST_IDX else 375 * VOL[i] ** 2 for i in range(3, 9))     # session 6 has constant prices
    assert bool(p.target_available) and p.rv_total_variance == pytest.approx(partial_var + later, rel=1e-9)
    assert bool(p.includes_partial_first_session) and p.n_valid_sessions == 6
    assert p.iv_total_variance == pytest.approx(0.16 ** 2 * 6.0 / 365, rel=1e-12)
    assert p.total_variance_diff == pytest.approx(p.iv_total_variance - p.rv_total_variance, rel=1e-12)
    assert p.rv_calendar_basis_pct == pytest.approx(100 * math.sqrt(p.rv_total_variance / (6.0 / 365)), rel=1e-12)
    assert p.iv_minus_rv_calendar_basis == pytest.approx(p.iv_pct - p.rv_calendar_basis_pct, rel=1e-12)
    tvs = pd.read_csv(out / "summary_expiry_total_variance.csv")
    assert {"iv_total_variance_median", "rv_total_variance_median", "ratio_median"} <= set(tvs.columns)


def test_weekend_flag_and_time_basis_ratio(ctx):
    _, out, _ = ctx
    long = pd.read_csv(out / "iv_rv_observations.csv")
    wk = long[long.obs_id == "wk"].iloc[0]
    p1 = long[long.obs_id == "p1"].iloc[0]
    assert not bool(wk.snapshot_to_expiry_has_weekend) and bool(p1.snapshot_to_expiry_has_weekend)       # Mon->Thu vs Tue->Thu next week
    e = long[(long.obs_id == "p2") & (long.horizon == "EXP") & (long.measure == "hybrid")].iloc[0]
    assert e.time_basis_ratio == pytest.approx((6.0 / 365) / ((149 / 375 + 6) / 252), rel=1e-12)
    summ = pd.read_csv(out / "summary_by_iv_regime.csv")
    assert "snapshot_to_expiry_has_weekend" in set(summ.dimension)


def test_resolution_sensitive_flag_and_halfwidth(ctx):
    _, out, _ = ctx
    long = pd.read_csv(out / "iv_rv_observations.csv").drop_duplicates("obs_id").set_index("obs_id")
    # bracketing points sit at x = -0.01 / +0.01, so the interpolation weight is 0.5: half-widths (0.10, 2.00) -> 1.05 and (0.10, 0.05) -> 0.075 vol points
    assert long.loc["p4", "atm_halfwidth_volpts"] == pytest.approx(1.05, abs=1e-9) and bool(long.loc["p4", "resolution_sensitive"])
    assert long.loc["p1", "atm_halfwidth_volpts"] == pytest.approx(0.075, abs=1e-9) and not bool(long.loc["p1", "resolution_sensitive"])


def test_regime_cutoffs_use_development_observations_only(tmp_path):
    _, out1, _ = run(tmp_path, holdout_iv=0.15)
    c1 = json.load(open(out1 / "regime_cutoffs.json"))
    import shutil
    shutil.rmtree(tmp_path / "out"); shutil.rmtree(tmp_path / "s2a"); shutil.rmtree(tmp_path / "participant")
    os.remove(tmp_path / "spot.parquet")
    _, out2, _ = run(tmp_path, holdout_iv=0.90)                                        # a wild holdout IV must not move any cut-off
    c2 = json.load(open(out2 / "regime_cutoffs.json"))
    assert c1 == c2
    assert len(c1["iv_quartiles_pct"]) == 3 and "F1|hybrid" in c1["future_rv_quartiles_pct"]
    long = pd.read_csv(out2 / "iv_rv_observations.csv")
    assert long[long.obs_id == "hold"].iv_quartile.iloc[0] == "Q4 high"


def test_outputs_exist_are_deterministic_and_do_not_touch_sources(tmp_path):
    _, out, spot = run(tmp_path)
    for f in ("iv_rv_observations.csv", "summary_by_horizon.csv", "summary_by_dte.csv", "summary_by_snapshot_time.csv",
              "summary_by_iv_regime.csv", "sample_observations.csv", "data_coverage.csv", "target_availability.csv",
              "resolution_data_loss.csv", "summary_resolution_limited.csv", "dev_vs_holdout.csv", "run_metadata.json"):
        assert (out / f).exists(), f
    h0 = hashlib.sha256(open(spot, "rb").read()).hexdigest()
    first = {f: open(out / f).read() for f in ("iv_rv_observations.csv", "summary_by_horizon.csv", "summary_by_dte.csv")}
    s2a = str(tmp_path / "s2a")
    pdir = str(tmp_path / "participant")
    out2 = tmp_path / "out2"
    b.build(s2a, spot, str(out2), pdir, reps=60, reps_secondary=30)
    assert hashlib.sha256(open(spot, "rb").read()).hexdigest() == h0
    for f, text in first.items():
        assert open(out2 / f).read() == text


def test_summaries_carry_no_ranking_or_pnl_columns(ctx):
    _, out, _ = ctx
    for f in ("summary_by_horizon.csv", "summary_by_dte.csv", "summary_by_snapshot_time.csv", "summary_by_iv_regime.csv"):
        cols = " ".join(pd.read_csv(out / f).columns).lower()
        assert not any(w in cols for w in ("rank", "best", "pnl", "profit", "signal", "edge", "sharpe"))
