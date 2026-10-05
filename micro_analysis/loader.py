"""Read-only loading of recorder databases into pandas."""
import hashlib
import os
import sqlite3
from typing import List, Tuple

import pandas as pd


def load_db(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    uri = "file:" + os.path.abspath(path).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        cycles = pd.read_sql_query("SELECT * FROM cycles ORDER BY cycle_id", con)
        quotes = pd.read_sql_query("SELECT * FROM quotes ORDER BY cycle_id, symbol", con)
    finally:
        con.close()
    day = os.path.basename(path).replace("micro_", "").replace(".sqlite", "")
    for df in (cycles, quotes):
        df["day"] = day
    return cycles, quotes


def load_many(paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cs, qs = zip(*(load_db(p) for p in paths))
    return pd.concat(cs, ignore_index=True), pd.concat(qs, ignore_index=True)


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
