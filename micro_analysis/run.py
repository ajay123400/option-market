"""CLI: analyse one or more recorder databases and write the descriptive tables.

    python -m micro_analysis.run E:\\nifty_microstructure\\micro_20261005.sqlite [more.sqlite ...] --out E:\\nifty_microstructure\\analysis_20261005

Outputs (CSV + run_metadata.json): row_metrics, atm_by_cycle_expiry, primary_instants, liquidity_summary, last_trade_bias, last_trade_bias_by_day, cost_scenarios, exclusions.
Read-only on the databases. Descriptive statistics only: not a signal, not evidence of tradability.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from . import summaries as S
from .analyze import analyze_frames
from .config import AnalysisConfig
from .loader import load_many, sha256


def run(dbs, out_dir, cfg: AnalysisConfig = AnalysisConfig()) -> dict:
    if not dbs:
        raise ValueError("no databases given")
    cycles, quotes = load_many(dbs)
    res = analyze_frames(cycles, quotes, cfg)
    rows, atm, ex = res["rows"], res["atm"], res["exclusions"]
    os.makedirs(out_dir, exist_ok=True)
    tables = dict(row_metrics=rows, atm_by_cycle_expiry=atm, primary_instants=S.primary_instants(atm), liquidity_summary=S.liquidity_summary(rows), last_trade_bias=S.last_trade_bias(rows),
                  last_trade_bias_by_day=S.last_trade_bias_by_day(rows), cost_scenarios=S.cost_scenarios(atm, cfg), exclusions=S.exclusion_summary(ex, rows, atm))
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    meta = dict(inputs=[dict(path=p, sha256=sha256(p)) for p in dbs], config=cfg.as_dict(), n_cycles=int(len(cycles)), n_option_rows_in=int((quotes["kind"] == "option").sum()), n_row_metrics=int(len(rows)),
                n_atm_records=int(len(atm)), tables={k: int(len(v)) for k, v in tables.items()}, numpy=np.__version__, pandas=pd.__version__, generated=datetime.now().isoformat(timespec="seconds"),
                note="descriptive statistics of recorded quotes; scenarios are arithmetic on recorded bid/ask, not forecasts or trading rules")
    with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dbs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, default=AnalysisConfig().rate)
    a = ap.parse_args(argv)
    meta = run(a.dbs, a.out, AnalysisConfig(rate=a.rate))
    print(json.dumps({k: meta[k] for k in ("n_cycles", "n_option_rows_in", "n_row_metrics", "n_atm_records", "tables")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
