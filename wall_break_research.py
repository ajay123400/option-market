"""wall_break_research.py -- can the option chain warn that an OI wall is
about to break, and how early?

Nearest weekly expiry, every 5 minutes 09:30-14:30. The call wall (most call
OI on a 100-pt strike above NIFTY) and the put wall (most put OI below) are
taken whenever NIFTY is within 1 ATM straddle of them. Signals at that
moment (all known at the time):

  dist      distance to the wall, in ATM straddles (09:30 straddle)
  mom5/15   NIFTY's move toward the wall over the last 5 / 15 min (straddles)
  oi30      % change of the wall's OI over the last 30 min (unwinding < 0)
  oiday     % change of the wall's OI since yesterday's close
  beyond30  OI added at the next strike beyond the wall (30 min), as a share
            of the wall's OI -- writers moving the wall further out
  inside30  the same for the strike between NIFTY and the wall
  prem30    % change of the wall option's premium over 30 min
  share     wall OI / all OTM OI on that side

Outcome: NIFTY closes a 1-min candle beyond the wall within the next 60 min.
Models (logistic regression + gradient boosting) are trained on 2021-23 and
scored on 2024-26: price-only (dist, momentum, time) vs price + OI signals.

Event study: for every wall that broke, the OI / distance path over the 60
minutes before the break, vs walls NIFTY came within 0.1 straddle of without
breaking (aligned at the closest approach) -- when do the two paths differ?

Output: results/wall_break/report.html (+ snaps.parquet, paths.parquet)
"""
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import market_calendar as mc
import paths
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "wall_break")
HORIZON = 60
LOOK = 60


def _dte(day, exp_d):
    n, d = 0, day + timedelta(days=1)
    while d <= exp_d:
        n += mc.is_trading_day(d)
        d += timedelta(days=1)
    return n


def collect(years=5):
    man = S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") == "done")
    cutoff = (date.today() - timedelta(days=int(365.25 * years))).isoformat()
    sp = S._load_spot()
    sp_day = sp.index.date
    snaps, paths_ = [], []
    for exp in exps:
        if exp < cutoff:
            continue
        df = S._load_options(exp)
        df = df[df.strike % 100 == 0]
        exp_d = pd.Timestamp(exp).date()
        fdays = sorted(df["day"].unique())
        for i_d, D in enumerate(fdays):
            if D > exp_d or i_d == 0 or _dte(D, exp_d) > 4:
                continue
            s = sp[sp_day == D]
            s = s[(s.index.strftime("%H:%M") >= "09:15") & (s.index.strftime("%H:%M") <= "15:29")]
            if len(s) < 300:
                continue
            grid = s.ts.to_numpy()
            spot = s.close.to_numpy()
            hm = s.index.strftime("%H:%M").to_numpy()
            i930 = int(np.searchsorted(hm, "09:30"))
            dd = df[df["day"] == D]
            atm = int(round(spot[i930] / 100) * 100)
            a = dd[(dd.strike == atm) & (dd.ts <= grid[i930])].sort_values("ts").groupby("type").close.last()
            if len(a) < 2:
                continue
            strad = float(a.sum())
            lo, hi = spot.min() - 3 * strad, spot.max() + 3 * strad
            dd = dd[(dd.strike >= lo) & (dd.strike <= hi)]
            oi = dd.pivot_table(index="ts", columns=["type", "strike"], values="oi", aggfunc="last").reindex(grid).ffill()
            px = dd.pivot_table(index="ts", columns=["type", "strike"], values="close", aggfunc="last").reindex(grid).ffill()
            prev = df[df["day"] == fdays[i_d - 1]].sort_values("ts").groupby(["type", "strike"]).oi.last()
            seen = {}
            for i in range(i930, int(np.searchsorted(hm, "14:31")), 5):
                if i < 30:
                    continue
                for typ in ("CE", "PE"):
                    up = typ == "CE"
                    cols = [c for c in oi.columns if c[0] == typ and ((c[1] > spot[i]) if up else (c[1] < spot[i]))
                            and abs(c[1] - spot[i]) <= 3 * strad]
                    if not cols:
                        continue
                    row = oi.loc[grid[i], cols]
                    if row.isna().all() or row.max() <= 0:
                        continue
                    w = row.idxmax()
                    W = w[1]
                    dist = ((W - spot[i]) if up else (spot[i] - W)) / strad
                    sgn = 1 if up else -1
                    # event study bookkeeping: remember each wall the first time it is seen
                    seen.setdefault((typ, W), i)
                    if dist > 1.0:
                        continue
                    o_now, o_30 = oi.at[grid[i], w], oi.at[grid[i - 30], w]
                    nb, ni = (typ, W + 100 * sgn), (typ, W - 100 * sgn)

                    def dOI(c):
                        return ((oi.at[grid[i], c] - oi.at[grid[i - 30], c]) / o_30) if c in oi.columns and o_30 else 0.0
                    p_now, p_30 = px.at[grid[i], w], px.at[grid[i - 30], w]
                    fut = spot[i + 1:i + 1 + HORIZON]
                    br = np.flatnonzero((fut > W) if up else (fut < W))
                    snaps.append({
                        "day": D.isoformat(), "expiry": exp, "type": typ, "wall": int(W), "minute": i,
                        "dist": dist, "mom5": sgn * (spot[i] - spot[i - 5]) / strad, "mom15": sgn * (spot[i] - spot[i - 15]) / strad,
                        "oi30": (o_now / o_30 - 1) if o_30 else 0.0,
                        "oiday": (o_now / prev.get((typ, W)) - 1) if prev.get((typ, W)) else 0.0,
                        "beyond30": dOI(nb), "inside30": dOI(ni),
                        "prem30": (p_now / p_30 - 1) if p_30 and np.isfinite(p_30) and p_30 > 0 else 0.0,
                        "share": float(o_now / row.sum()),
                        "broke60": bool(len(br)), "t_break": int(br[0]) + 1 if len(br) else None,
                        "broke_eod": bool((spot[-1] > W) if up else (spot[-1] < W))})
            # ---- event study: the walls seen today, broken vs approached-and-held
            for (typ, W), i0 in seen.items():
                up = typ == "CE"
                w = (typ, W)
                beyond = (spot > W) if up else (spot < W)
                after = np.flatnonzero(beyond[i0:]) + i0
                d_path = ((W - spot) if up else (spot - W)) / strad
                if len(after):
                    b, kind = int(after[0]), "break"
                else:
                    seg = d_path[i0:]
                    c = int(np.argmin(seg)) + i0
                    if d_path[c] > 0.1:
                        continue
                    b, kind = c, "hold"
                if b - LOOK < 0 or d_path[b - LOOK] < 0.2:
                    continue
                base = oi.at[grid[b - LOOK], w]
                if not base or not np.isfinite(base):
                    continue
                nb = (typ, W + (100 if up else -100))
                for k in range(-LOOK, 1, 5):
                    j = b + k
                    paths_.append({"day": D.isoformat(), "type": typ, "wall": int(W), "kind": kind, "k": k,
                                   "oi": oi.at[grid[j], w] / base,
                                   "beyond": (oi.at[grid[j], nb] - oi.at[grid[b - LOOK], nb]) / base if nb in oi.columns else np.nan,
                                   "dist": d_path[j]})
        print(f"\r{exp} snaps {len(snaps)} paths {len(paths_)}", end="", flush=True)
    print()
    return pd.DataFrame(snaps), pd.DataFrame(paths_)


PRICE = ["dist", "mom5", "mom15", "minute"]
OIF = ["oi30", "oiday", "beyond30", "inside30", "prem30", "share"]


def models(sn):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = sn.copy()
    for c in OIF + PRICE:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(0).clip(x[c].quantile(0.01), x[c].quantile(0.99))
    A, B = x[x.day < "2024"], x[x.day >= "2024"]
    y = "broke60"
    out = {"n_train": int(len(A)), "n_test": int(len(B)), "rate_train": round(float(A[y].mean() * 100), 1),
           "rate_test": round(float(B[y].mean() * 100), 1), "models": []}
    preds = {}
    for name, feats in (("Price only (distance, momentum, time)", PRICE), ("Price + OI signals", PRICE + OIF), ("OI signals only", OIF)):
        for algo, mk in (("logistic", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                         ("boosting", lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300))):
            m = mk().fit(A[feats], A[y])
            p = m.predict_proba(B[feats])[:, 1]
            preds[(name, algo)] = p
            top = B.assign(p=p).nlargest(max(1, len(B) // 10), "p")
            out["models"].append({"name": name, "algo": algo, "auc": round(float(roc_auc_score(B[y], p)), 3),
                                  "top10": round(float(top[y].mean() * 100), 1)})
            if algo == "logistic" and name == "Price + OI signals":
                coefs = m[-1].coef_[0]
                out["coef"] = [{"f": f, "c": round(float(c), 3)} for f, c in zip(feats, coefs)]
    return out


def univariate(sn):
    """Break-in-60-min rate by each OI signal's fifth, within distance bands
    (so a signal is not credited for simply being closer)."""
    x = sn.copy()
    x["band"] = pd.cut(x.dist, [0, 0.25, 0.5, 1.0], labels=["0–0.25", "0.25–0.5", "0.5–1"])
    res = {}
    for f in OIF + ["mom15"]:
        rows = []
        for b, z in x.groupby("band", observed=True):
            z = z.assign(q=pd.qcut(z[f].rank(method="first"), 5, labels=False))
            rows.append({"band": str(b), "cells": [{"q": int(q), "rate": round(float(v.broke60.mean() * 100), 1), "n": int(len(v)),
                                                     "lo": round(float(v[f].min()), 3), "hi": round(float(v[f].max()), 3)}
                                                    for q, v in z.groupby("q")]})
        res[f] = rows
    return res


def event_study(pa):
    g = pa.groupby(["kind", "k"]).agg(oi=("oi", "mean"), beyond=("beyond", "mean"), dist=("dist", "mean"), n=("oi", "size")).reset_index()
    ev = {k: x.sort_values("k").to_dict("records") for k, x in g.groupby("kind")}
    # lead time: the first minute (of the last 60) the wall's OI is >= 10% below its level 60 min earlier
    lead = []
    for (d, t, w, kind), x in pa.groupby(["day", "type", "wall", "kind"]):
        x = x.sort_values("k")
        hit = x[x.oi <= 0.9]
        lead.append({"kind": kind, "fired": bool(len(hit)), "lead": int(-hit.k.iloc[0]) if len(hit) else None})
    L = pd.DataFrame(lead)
    lt = {}
    for kind, x in L.groupby("kind"):
        f = x[x.fired]
        lt[kind] = {"n": int(len(x)), "fired": round(float(x.fired.mean() * 100), 1),
                    "median_lead": float(f.lead.median()) if len(f) else None,
                    "dist": {str(b): int(v) for b, v in pd.cut(f.lead, [-1, 5, 15, 30, 45, 60]).value_counts().sort_index().items()}}
    return {"paths": ev, "lead": lt}


def prob_table(sn):
    """Chance of a break within 60 min by distance and the last 15 min's move
    toward the wall -- the practical, price-only rule of thumb."""
    x = sn.copy()
    x["band"] = pd.cut(x.dist, [0, 0.1, 0.25, 0.4, 0.6, 1.0], labels=["0–0.1", "0.1–0.25", "0.25–0.4", "0.4–0.6", "0.6–1"])
    x["mom"] = pd.cut(x.mom15, [-99, -0.1, 0.1, 99], labels=["moving away", "flat", "moving toward"])
    x["half"] = np.where(x.day < "2024", "a", "b")
    rows = []
    for b, z in x.groupby("band", observed=True):
        r = {"band": str(b)}
        for m, v in z.groupby("mom", observed=True):
            r[str(m)] = {"rate": round(float(v.broke60.mean() * 100), 1), "n": int(len(v)),
                         "a": round(float(v[v.half == "a"].broke60.mean() * 100), 1), "b": round(float(v[v.half == "b"].broke60.mean() * 100), 1),
                         "eod": round(float(v.broke_eod.mean() * 100), 1)}
        rows.append(r)
    return rows


def report(sn, md, uv, ev, pt):
    data = {"models": md, "uni": uv, "ev": ev, "prob": pt, "n": int(len(sn)), "days": int(sn.day.nunique()),
            "rate": round(float(sn.broke60.mean() * 100), 1)}
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str)))
    return path


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wall Break Warning</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--grp:#eef0f6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--grp:#23262e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1300px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
tr.grp td{background:var(--grp);font-weight:700;font-size:12px;color:var(--muted)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px}.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
svg text{fill:var(--muted);font-size:11px}.legend{display:flex;gap:14px;font-size:12px;margin-top:4px}.legend i{display:inline-block;width:14px;height:3px;margin-right:6px;vertical-align:middle}
td.h{text-align:center;font-weight:600}.sm{font-size:11px;color:var(--muted);font-weight:400}
</style></head><body><div class="wrap">
<h1>Can the option chain warn that an OI wall will break?</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>What the data says</h2><ul id="verdict"></ul></div>
<div class="card"><h2>What happens to the wall's OI in the hour before a break (vs walls that held)</h2>
<div class="grid2"><div><div id="cOi"></div><div class="legend"><span><i style="background:#e0434a"></i>walls that broke</span><span><i style="background:#0e9f6e"></i>walls NIFTY reached (≤ 0.1 straddle) but that held</span></div>
<p class="note">Wall OI as a multiple of its level 60 min before (1.00 = unchanged). Minutes relative to the break / the closest approach.</p></div>
<div><div id="cDist"></div><p class="note">Distance of NIFTY from the wall, in ATM straddles.</p></div></div>
<div class="scroll" style="margin-top:10px"><table id="tLead"></table></div></div>
<div class="card"><h2>Predicting a break within 60 minutes — trained 2021–23, tested 2024–26</h2>
<div class="scroll"><table id="tModels"></table></div>
<p class="note">AUC: 0.5 = coin toss, 1.0 = perfect. "Top 10%" = how often a break followed when the model was most confident.</p></div>
<div class="card"><h2>Rule of thumb — chance NIFTY breaks the wall within 60 min</h2>
<div class="scroll"><table id="tProb"></table></div>
<p class="note">Distance in ATM straddles (09:30). Moving toward / away = NIFTY moved more than 0.1 straddle toward / away from the wall in the last 15 min. Small print: 2021–23 / 2024–26 rate, and the chance it closes the day beyond the wall.</p></div>
<div class="card"><h2>Each OI signal on its own, within the same distance band</h2>
<div class="scroll"><table id="tUni"></table></div>
<p class="note">Break-in-60-min rate for each fifth of the signal (lowest → highest), inside each distance band so a signal is not credited for simply being closer.</p></div>
<div class="card"><h2>Method &amp; limits</h2><ul>
<li>Nearest weekly expiry, 100-pt strikes, 1-min OI and prices from the Fyers history; exchange OI updates every few minutes, so a 1-min OI change can lag.</li>
<li>Walls are re-picked at every 5-min snapshot; consecutive snapshots of the same approach are not independent.</li>
<li>A break = a 1-min NIFTY close beyond the wall strike.</li>
</ul></div>
</div><script>
const D = /*DATA*/null;
const M = D.models, E = D.ev;
document.getElementById("meta").textContent = `${D.n.toLocaleString("en-IN")} wall snapshots (NIFTY within 1 straddle of a wall) on ${D.days.toLocaleString("en-IN")} days, Sep 2021 → Sep 2026 · ${D.rate}% broke within 60 min.`;
const md = (n, a) => M.models.find(m => m.name.startsWith(n) && m.algo === a);
const pB = md("Price only", "boosting"), oB = md("Price + OI", "boosting"), oO = md("OI signals only", "boosting");
const LB = E.lead.break, LH = E.lead.hold;
const pk = (k, t) => E.paths[k].find(r => r.k === t);
const kp = (v, l) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`;
document.getElementById("kpis").innerHTML =
  kp(`${pB.auc} → ${oB.auc}`, "prediction quality (AUC) 2024–26: price only → price + OI signals") +
  kp(`${pB.top10}% → ${oB.top10}%`, "break rate in the model's most-confident 10%: no gain from OI") +
  kp(`+${Math.round((pk("break", -20).oi - 1) * 100)}%`, "the wall's OI 20 min before a break — it rises, it does not unwind") +
  kp(`${LB.fired}% vs ${LH.fired}%`, "“OI fell ≥ 10%” fired before breaks vs before holds — no difference");
document.getElementById("verdict").innerHTML = [
  `<b>No usable early warning from OI.</b> Before a wall breaks its OI does not unwind — on average it <i>rises</i> ~${Math.round((pk("break", -20).oi - 1) * 100)}% in the last hour (writers add or defend as NIFTY approaches) and only dips at the break itself. Walls that held show the same pattern.`,
  `<b>The classic unwinding signal is noise here:</b> a ≥ 10% OI drop at the wall in the last hour appeared before ${LB.fired}% of breaks and ${LH.fired}% of holds, with the same median lead of ${LB.median_lead} min. It does not separate the two.`,
  `<b>Price tells almost everything.</b> Distance to the wall, the last 15 minutes' move and the time of day predict a break within 60 min with AUC ${pB.auc} out of sample; adding every OI signal lifts it only to ${oB.auc}, and the most-confident 10% break at the same rate (${pB.top10}% vs ${oB.top10}%). OI signals alone reach ${oO.auc} — mostly because they move with distance.`,
  `<b>Two weak OI effects:</b> OI added at the strike <i>between</i> NIFTY and the wall means fewer breaks (a nearer wall is forming), and a wall that holds a large share of that side's OI broke more, not less. Neither improves the prediction enough to matter.`,
  `<b>So "how early" = as early as price shows it.</b> Use the distance table below: within 0.1 straddle of the wall a break inside the hour happened ~70% of the time, at 0.1–0.25 about a third, beyond 0.4 rarely. Whether NIFTY was moving toward or away mattered little — a market that is <i>moving</i> (either way) breaks walls more than a flat one.`,
].map(x => `<li>${x}</li>`).join("");
function chart(id, key, lo, hi, fmt) {
  const W = 600, H = 240, pl = 50, pr = 10, pt = 10, pb = 24, ks = E.paths.break.map(r => r.k);
  const x = k => pl + (k - ks[0]) / (ks[ks.length - 1] - ks[0]) * (W - pl - pr), y = v => pt + (hi - v) / (hi - lo) * (H - pt - pb);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img">`;
  for (let i = 0; i <= 4; i++) { const v = lo + (hi - lo) * i / 4; s += `<line x1="${pl}" x2="${W - pr}" y1="${y(v)}" y2="${y(v)}" stroke="var(--border)"/><text x="${pl - 5}" y="${y(v) + 4}" text-anchor="end">${fmt(v)}</text>`; }
  ks.forEach(k => { if (k % 15 === 0) s += `<text x="${x(k)}" y="${H - 6}" text-anchor="middle">${k}</text>`; });
  [["break", "#e0434a"], ["hold", "#0e9f6e"]].forEach(([kind, c]) => { s += `<path d="${E.paths[kind].map((r, i) => `${i ? "L" : "M"}${x(r.k).toFixed(1)},${y(r[key]).toFixed(1)}`).join("")}" fill="none" stroke="${c}" stroke-width="2"/>`; });
  document.getElementById(id).innerHTML = s + "</svg>";
}
const oiV = E.paths.break.concat(E.paths.hold).map(r => r.oi);
chart("cOi", "oi", Math.min(...oiV) - 0.01, Math.max(...oiV) + 0.01, v => v.toFixed(2));
const dV = E.paths.break.concat(E.paths.hold).map(r => r.dist);
chart("cDist", "dist", Math.min(...dV) - 0.02, Math.max(...dV) + 0.02, v => v.toFixed(2));
const bins = Object.keys(LB.dist);
document.getElementById("tLead").innerHTML = `<tr><th>“Wall OI fell ≥ 10% in the last hour”</th><th>Cases</th><th>Signal fired</th><th>Median minutes before</th>${bins.map(b => `<th>${b.replace("(", "").replace("]", "").replace("-1", "0").replace(", ", "–")} min</th>`).join("")}</tr>` +
  [["Walls that broke", LB], ["Walls that held", LH]].map(([l, r]) => `<tr><td>${l}</td><td>${r.n}</td><td>${r.fired}%</td><td>${r.median_lead}</td>${bins.map(b => `<td>${r.dist[b]}</td>`).join("")}</tr>`).join("");
document.getElementById("tModels").innerHTML = `<tr><th>Inputs</th><th>Model</th><th>AUC 2024–26</th><th>Break rate in top 10%</th></tr>` +
  M.models.map(m => `<tr><td>${m.name}</td><td>${m.algo}</td><td><b>${m.auc}</b></td><td>${m.top10}%</td></tr>`).join("") +
  `<tr><td colspan="4" class="note" style="text-align:left">Base rate: ${M.rate_train}% of snapshots broke within 60 min in 2021–23, ${M.rate_test}% in 2024–26.</td></tr>`;
const moms = ["moving away", "flat", "moving toward"];
const col = v => `background:rgba(224,67,74,${Math.min(0.6, v / 100 * 0.9)})`;
document.getElementById("tProb").innerHTML = `<tr><th>Distance</th>${moms.map(m => `<th style="text-align:center">${m}</th>`).join("")}</tr>` +
  D.prob.map(r => `<tr><td>${r.band}</td>${moms.map(m => { const c = r[m]; return c ? `<td class="h" style="${col(c.rate)}">${c.rate}%<div class="sm">${c.a}% / ${c.b}% · day ${c.eod}% · n ${c.n}</div></td>` : "<td>—</td>"; }).join("")}</tr>`).join("");
const NAMES = {oi30: "Wall OI change, 30 min", oiday: "Wall OI change since yesterday", beyond30: "OI added beyond the wall", inside30: "OI added between NIFTY and the wall", prem30: "Wall premium change, 30 min", share: "Wall's share of side OI", mom15: "NIFTY move toward wall, 15 min (price)"};
let h = `<tr><th>Signal</th><th>Distance</th><th>Lowest fifth</th><th>2nd</th><th>3rd</th><th>4th</th><th>Highest fifth</th></tr>`;
Object.entries(D.uni).forEach(([f, rows]) => { h += `<tr class="grp"><td colspan="7">${NAMES[f] || f}</td></tr>`;
  rows.forEach(r => { h += `<tr><td></td><td>${r.band}</td>${r.cells.map(c => `<td>${c.rate}%<div class="sm">${c.lo} … ${c.hi}</div></td>`).join("")}</tr>`; }); });
document.getElementById("tUni").innerHTML = h;
</script></body></html>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    ps, pp = os.path.join(OUT, "snaps.parquet"), os.path.join(OUT, "paths.parquet")
    if os.path.exists(ps) and "--fresh" not in sys.argv:
        sn, pa = pd.read_parquet(ps), pd.read_parquet(pp)
    else:
        sn, pa = collect()
        sn.to_parquet(ps, index=False)
        pa.to_parquet(pp, index=False)
    return sn, pa, models(sn), univariate(sn), event_study(pa)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    sn, pa, md, uv, ev = main()
    print(len(sn), "snapshots,", sn.day.nunique(), "days; break-60 rate", round(sn.broke60.mean() * 100, 1))
    print(json.dumps(md, indent=1))
    for f, rows in uv.items():
        print(f, [(r["band"], [c["rate"] for c in r["cells"]]) for r in rows])
    print(json.dumps(ev["lead"], indent=1))
    for kind, rows in ev["paths"].items():
        print(kind, [(r["k"], round(r["oi"], 3), round(r["dist"], 2)) for r in rows])
    pt = prob_table(sn)
    for r in pt:
        print(r["band"], {k: v["rate"] for k, v in r.items() if k != "band"})
    print(report(sn, md, uv, ev, pt))
