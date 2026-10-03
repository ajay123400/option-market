"""Read-only loaders for the recorded NIFTY 1-minute datasets (`data/hist1m`).

Timestamp semantics (verified in the Phase 1 audit): `ts` is epoch seconds
(UTC); bars are labelled by their START, so a bar's close is only known at
`ts + 60 s`. A snapshot "at bar t0" is therefore observed at t0 + 60 s, and
only bars with ts <= t0 may be used (no look-ahead). Option bars that have no
trade in the minute are copies of the previous close with volume 0; the age
of a quote is measured from the last bar that actually traded.

Source files are never written to.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..bsm import OptionType
from ..surface import OptionQuote

IST = timezone(timedelta(hours=5, minutes=30))
IST_OFFSET_SECONDS = 19800
BAR_SECONDS = 60


def snapshot_ts(day: date, hhmm: str) -> int:
    """Epoch seconds of the bar that STARTS at `hhmm` IST on `day`."""
    h, m = (int(x) for x in hhmm.split(":"))
    return int(datetime(day.year, day.month, day.day, h, m, tzinfo=IST).timestamp())


def ist_date(ts: int) -> date:
    return (datetime.fromtimestamp(ts, tz=IST)).date()


def load_spot_closes(path: str) -> Dict[int, float]:
    """ts -> close for every 1-minute spot bar (exact-minute lookups only)."""
    df = pd.read_parquet(path, columns=["ts", "close"])
    return dict(zip(df["ts"].to_numpy(dtype="int64").tolist(), df["close"].to_numpy(dtype="float64").tolist()))


def list_expiry_files(root: str) -> List[Tuple[str, str]]:
    d = os.path.join(root, "options")
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".parquet"):
            out.append((fn[:-8], os.path.join(d, fn)))
    return out


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sort and add `lts` = ts of the last bar (<= this bar) that had volume > 0,
    forward-filled per contract. Strictly backward-looking: a row's `lts` never
    depends on later rows (exercised by the truncation test on real data)."""
    df = df.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)
    lts = df["ts"].where(df["volume"] > 0)
    df["lts"] = lts.groupby(df["symbol"]).ffill()
    return df


def load_expiry_frame(path: str) -> pd.DataFrame:
    """One expiry file prepared with `prepare_frame`."""
    return prepare_frame(pd.read_parquet(path, columns=["symbol", "type", "strike", "ts", "close", "volume"]))


def trading_days(df: pd.DataFrame) -> List[date]:
    days = (df["ts"].to_numpy(dtype="int64") + IST_OFFSET_SECONDS) // 86400
    return [date(1970, 1, 1) + timedelta(days=int(d)) for d in np.unique(days)]


def quotes_at(df: pd.DataFrame, t0: int) -> List[OptionQuote]:
    """All listed contracts' quotes at bar `t0` (bar-start epoch seconds).
    Price = that bar's close rounded to 2 dp (float32 -> tick grid); age is an
    upper bound in minutes since the last trading bar (None if never traded)."""
    snap = df[df["ts"] == t0]
    out: List[OptionQuote] = []
    for strike, typ, close, lts in zip(snap["strike"].to_numpy(), snap["type"].to_numpy(),
                                       snap["close"].to_numpy(dtype="float64"), snap["lts"].to_numpy(dtype="float64")):
        age = None if np.isnan(lts) else (t0 + BAR_SECONDS - lts) / 60.0
        out.append(OptionQuote(float(strike), OptionType.CALL if typ == "CE" else OptionType.PUT,
                               round(float(close), 2), age))
    return out
