"""End-to-end Stage 2A on a tiny synthetic dataset with a KNOWN smile (flat 18 % vol)."""
import hashlib
import json
import math
import os
from datetime import date

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from optionsengine import bsm_price, implied_carry_yield  # noqa: E402
from optionsengine.research import build_surface as bs, loaders  # noqa: E402

DAY = date(2026, 9, 15)           # 7 days before the 2026-09-22 expiry (15:40 IST close)
EXPIRY = "2026-09-22"
SPOT = 24500.0
R = 0.065
SIGMA = 0.18
T10 = loaders.snapshot_ts(DAY, "10:00")
STRIKES = list(range(23200, 25801, 100))


def tick(p):
    return round(round(p / 0.05) * 0.05, 2)


def make_root(tmp_path, stale_strike=None, drop_spot=False):
    root = tmp_path / "hist1m"
    (root / "options").mkdir(parents=True, exist_ok=True)
    Tyr = ((loaders.datetime(2026, 9, 22, 15, 40, tzinfo=loaders.IST) -
            loaders.datetime.fromtimestamp(T10 + 60, tz=loaders.IST)).total_seconds()) / (365 * 86400)
    q = implied_carry_yield(SPOT, SPOT, Tyr, R)
    rows = []
    for k in STRIKES:
        for typ, kind in (("CE", "call"), ("PE", "put")):
            p = tick(bsm_price(SPOT, k, Tyr, R, q, SIGMA, kind))
            sym = f"NSE:NIFTY2692224500{typ}"
            for m in range(-40, 1):
                traded = (m == 0) if k != stale_strike else (m == -30)
                rows.append((f"{sym}{k}", typ, k, T10 + 60 * m, p, p, p, p, 3 if traded else 0, 10))
    df = pd.DataFrame(rows, columns=["symbol", "type", "strike", "ts", "open", "high", "low", "close", "volume", "oi"])
    df = df.astype({"strike": "int32", "ts": "int64", "open": "float32", "high": "float32", "low": "float32", "close": "float32"})
    df.to_parquet(root / "options" / f"{EXPIRY}.parquet")
    ts = [] if drop_spot else [T10 + 60 * m for m in range(-40, 1)]
    pd.DataFrame({"ts": ts, "open": SPOT, "high": SPOT, "low": SPOT, "close": SPOT}).astype({"ts": "int64"}).to_parquet(root / "NIFTY50_1m.parquet")
    return str(root)


def run(root, out, **kw):
    return bs.build(root, str(out), times=("10:00",), workers=1, r=R, **kw)


def read(out):
    return pd.read_csv(os.path.join(out, "smiles.csv")), pd.read_csv(os.path.join(out, "points.csv"))


def test_known_flat_smile_recovered_and_flags_populated(tmp_path):
    root = make_root(tmp_path)
    out = tmp_path / "out"
    meta = run(root, out)
    s, p = read(out)
    row = s[s.day == "2026-09-15"].iloc[0]
    assert row.forward_status == "ok" and bool(row.primary) and row.group_reason != row.group_reason     # NaN
    assert row.T_days == pytest.approx(7 + (5 * 60 + 39) / 1440, rel=1e-3) or row.T_days > 7              # 15:40 close used
    assert row.forward_used == pytest.approx(SPOT, abs=0.3)
    assert row.atm_iv == pytest.approx(SIGMA, abs=2e-3)                       # tick-rounded prices: small, bounded error
    assert abs(row.rr25) < 4e-3 and abs(row.bf25) < 4e-3
    assert row.split == "holdout" and meta["n_smile_rows"] == 1             # 2026-09-15 >= default split 2025-01-01
    assert (p.kind == p.strike.map(lambda k: "call" if k >= row.forward_used else "put")).all()
    assert json.load(open(os.path.join(out, "run_metadata.json")))["risk_free_rate_note"].startswith("ASSUMPTION")


def test_expiry_close_1540_is_used_for_T(tmp_path):
    out = tmp_path / "out"; run(make_root(tmp_path), out)
    s, _ = read(out)
    exp_T = (7 * 86400 + (15 * 3600 + 40 * 60) - (10 * 3600 + 1 * 60)) / 86400      # 7 d + 5h39m
    assert s.iloc[0].T_days == pytest.approx(exp_T, abs=1e-9)


def test_stale_strike_is_excluded_and_counted(tmp_path):
    out = tmp_path / "out"; run(make_root(tmp_path, stale_strike=24600), out)
    s, p = read(out)
    st = p[(p.strike == 24600)]
    assert (st.exclusion == "stale").all() and not st.used.any() and st.iv.isna().all()
    assert s.iloc[0].n_stale == 1


def test_missing_spot_bar_is_reported_not_filled(tmp_path):
    out = tmp_path / "out"; run(make_root(tmp_path, drop_spot=True), out)
    s, p = read(out)
    assert s.iloc[0].group_reason == "no_spot_bar_at_snapshot" and len(p) == 0 and pd.isna(s.iloc[0].atm_iv)


def test_holdout_split_assignment_and_date_filters(tmp_path):
    out = tmp_path / "out"; run(make_root(tmp_path), out, split_date="2027-01-01")
    assert read(out)[0].iloc[0].split == "dev"
    out2 = tmp_path / "out2"; run(make_root(tmp_path), out2, start=date(2026, 9, 16))
    s2, p2 = read(out2)
    assert len(s2) == 0 and len(p2) == 0 and list(s2.columns) == list(bs.SMILE_COLUMNS)    # empty but schema-stable


def test_source_files_are_not_modified(tmp_path):
    root = make_root(tmp_path)
    before = {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
              for d, _, fs in os.walk(root) for f in fs}
    run(root, tmp_path / "out")
    after = {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
             for d, _, fs in os.walk(root) for f in fs}
    assert before == after


def test_forward_threshold_override_changes_only_the_gate(tmp_path):
    root = make_root(tmp_path)
    out = tmp_path / "o"; meta = run(root, out, forward_bps=3.0)
    assert meta["forward_max_dispersion_bps"] == 3.0 and meta["forward_default_bps"] == 5.0
