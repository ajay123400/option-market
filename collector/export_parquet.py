"""Offline export of a daily recorder database to Parquet (run on the research side; the recorder itself needs no pyarrow).

    python -m collector.export_parquet E:\\nifty_microstructure\\micro_20261005.sqlite [--out DIR]

Writes <out>/quotes_YYYYMMDD.parquet and cycles_YYYYMMDD.parquet (zstd) and a SHA-256 manifest. Read-only on the database.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys


def export(db_path: str, out_dir: str = None) -> dict:
    import pandas as pd
    out_dir = out_dir or os.path.dirname(os.path.abspath(db_path))
    os.makedirs(out_dir, exist_ok=True)
    uri = "file:" + os.path.abspath(db_path).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        stamp = os.path.basename(db_path).replace("micro_", "").replace(".sqlite", "")
        res = {}
        for t in ("quotes", "cycles"):
            df = pd.read_sql_query(f"SELECT * FROM {t} ORDER BY " + ("cycle_id, symbol" if t == "quotes" else "cycle_id"), con)
            p = os.path.join(out_dir, f"{t}_{stamp}.parquet")
            df.to_parquet(p, compression="zstd", index=False)
            with open(p, "rb") as f:
                res[t] = dict(file=os.path.basename(p), rows=len(df), sha256=hashlib.sha256(f.read()).hexdigest())
    finally:
        con.close()
    with open(os.path.join(out_dir, f"export_manifest_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    print(json.dumps(export(a.db, a.out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
