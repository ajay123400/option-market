"""Write small, committed sample outputs from a full Stage 2A run.
    python research_output/stage2a/validation/make_samples.py research_output/stage2a/full research_output/stage2a/sample"""
import json
import os
import shutil
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
s = pd.read_csv(f"{src}/smiles.csv"); p = pd.read_csv(f"{src}/points.csv")
pick = []


def first(df, label):
    if len(df):
        r = df.iloc[0]
        pick.append((label, r.smile_id))


ok = s[(s.forward_status == "ok") & s.atm_iv.notna()]
first(ok[(ok.day >= "2023-03-01") & (ok.day < "2024-01-01") & ok.rr25.notna() & ok.T_days.between(6, 8) & (ok.time == "13:00")], "normal weekly smile, ATM + RR/BF (development)")
first(ok[(ok.day >= "2025-03-03") & ok.rr25.notna() & ok.T_days.between(6, 8) & (ok.time == "13:00")], "normal weekly smile, ATM + RR/BF (holdout)")
first(ok[(ok.day == "2024-06-04") & (ok.time == "10:00") & (ok.T_days > 1)], "election-result day, very high IV")
first(s[s.expiry_day & (s.time == "15:00") & s.atm_iv.notna() & (s.day >= "2025-01-01")], "expiry day 15:00 (many resolution-limited points)")
first(s[(s.forward_status == "low_confidence") & (s.day >= "2025-01-01")], "LOW_CONFIDENCE forward: points labelled, no metrics")
first(s[(s.forward_status == "unavailable")], "forward unavailable: no smile")
sel = [i for _, i in pick]
labels = dict((i, l) for l, i in pick)
ss = s[s.smile_id.isin(sel)].copy(); ss.insert(0, "sample_note", ss.smile_id.map(labels))
ss.to_csv(f"{dst}/smiles_sample.csv", index=False)
p[p.smile_id.isin(sel)].to_csv(f"{dst}/points_sample.csv", index=False)
# 200-row head of the full tables for schema reference
s.head(200).to_csv(f"{dst}/smiles_head200.csv", index=False)
if os.path.isdir(f"{src}/report"):
    shutil.copytree(f"{src}/report", f"{dst}/report", dirs_exist_ok=True)
shutil.copy(f"{src}/run_metadata.json", f"{dst}/run_metadata.json")
print(json.dumps({"samples": labels}, indent=1))
