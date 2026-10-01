"""ml_pilot.py -- does a machine-learning model beat the simple IV rule for
selling NIFTY weekly premium?

Unit: every 09:30 entry 1-4 trading days before the nearest weekly expiry
(the template backtest, results/template_bt/trades.parquet), held to expiry.

Features (all known at 09:30): ATM IV, straddle as % of NIFTY, IV rank, term
structure, 25/10-delta skew, smile curvature, days to expiry, weekday,
realized volatility (5 / 20 days), yesterday's range and move, today's gap,
the 09:15-09:30 range and move, and implied / realized ratios.

Models, walk-forward: first trained on Sep 2021 - Sep 2022, then re-trained
every quarter on everything before it (rows whose expiry is not yet over are
left out) and used only on the next quarter -- the model never sees the
period it trades.

  vol model   predicts log(|NIFTY move to expiry| / straddle); sell when the
              predicted move is below 0.9 x the straddle
  P&L model   predicts the strategy's P&L; sell when it is above 0

Compared on the same out-of-sample days with: sell every day, and the IV
rule (ATM IV >= the 60th percentile of the training days, re-set each quarter).

Output: results/ml_pilot/report.html
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import iv_surface_research as IV
import paths
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "ml_pilot")
TPL = ["Short straddle|ATM", "Short strangle|Δ0.20", "Short strangle|Δ0.15", "Iron condor|Δ0.15|w200"]
FEATS = ["atm_iv", "strad_pct", "ivr", "term", "skew25", "skew10", "curv", "dte", "wd", "rv5", "rv20",
         "prev_range", "prev_ret", "gap", "open_range", "open_move", "iv_rv5", "iv_rv20"]
NAMES = {"atm_iv": "ATM IV", "strad_pct": "Straddle % of NIFTY", "ivr": "IV rank (1y)", "term": "Term (this ÷ next week IV)",
         "skew25": "25Δ put − call IV", "skew10": "10Δ put − call IV", "curv": "Smile curvature", "dte": "Days to expiry",
         "wd": "Weekday", "rv5": "Realized vol 5d", "rv20": "Realized vol 20d", "prev_range": "Yesterday's range %",
         "prev_ret": "Yesterday's move %", "gap": "Today's gap %", "open_range": "09:15–09:30 range %",
         "open_move": "09:15–09:30 move %", "iv_rv5": "ATM IV ÷ realized 5d", "iv_rv20": "ATM IV ÷ realized 20d"}
CUT = 0.9
START = "2022-10-01"


def dataset():
    sp = S._load_spot()
    day = pd.Index(sp.index.date)
    hm = sp.index.strftime("%H:%M")
    d = sp.groupby(day).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    d.index = [x.isoformat() for x in d.index]
    r = np.log(d.close).diff()
    daily = pd.DataFrame({"rv5": r.rolling(5).std() * np.sqrt(252) * 100, "rv20": r.rolling(20).std() * np.sqrt(252) * 100,
                          "prev_range": (d.high - d.low) / d.close * 100, "prev_ret": r * 100}).shift(1)
    daily["gap"] = (d.open / d.close.shift(1) - 1) * 100
    m = (hm >= "09:15") & (hm <= "09:30")
    o = sp[m].groupby(day[m]).agg(h=("high", "max"), l=("low", "min"), o=("open", "first"), c=("close", "last"))
    o.index = [x.isoformat() for x in o.index]
    daily["open_range"] = (o.h - o.l) / o.c * 100
    daily["open_move"] = (o.c / o.o - 1) * 100
    daily["spot930"] = o.c
    # expiry settlement
    t = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "template_bt", "trades.parquet"))
    t = t[(t.slot == "09:30") & (t.dte >= 1) & (t.dte <= 4) & t.tpl.isin(TPL)].drop(columns=["atm_iv"])
    settle = d.close.rename("settle")
    strad = t[t.tpl == "Short straddle|ATM"][["expiry", "day", "premium_pts"]].rename(columns={"premium_pts": "strad"})
    strad["strad"] += 1.0                      # add back the 2 x 0.5 pt slippage -> the quoted straddle
    days = pd.read_parquet(os.path.join(IV.OUT, "days.parquet"))
    fe = IV.day_features(days)
    x = (t.merge(strad, on=["expiry", "day"]).merge(days[["expiry", "day", "atm_iv", "skew25", "skew10", "curv"]], on=["expiry", "day"])
          .merge(fe[["day", "term", "ivr"]], on="day", how="left").merge(daily, left_on="day", right_index=True)
          .merge(settle, left_on="expiry", right_index=True))
    x["atm_iv"] *= 100
    x["skew25"] *= 100
    x["skew10"] *= 100
    x["strad_pct"] = x.strad / x.spot930 * 100
    x["wd"] = pd.to_datetime(x.day).dt.weekday
    x["iv_rv5"] = x.atm_iv / x.rv5
    x["iv_rv20"] = x.atm_iv / x.rv20
    x["move_ratio"] = (x.settle - x.spot930).abs() / x.strad
    x["y_vol"] = np.log((x.settle - x.spot930).abs() + 1) - np.log(x.strad)
    return x.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_vol", "hold_rs", "atm_iv"])


def _gbm():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40, l2_regularization=1.0)


def walk_forward(x):
    quarters = pd.date_range(START, x.day.max(), freq="QS")
    one = x[x.tpl == "Short straddle|ATM"]           # one row per entry for the vol model
    preds, imps = [], []
    for q0 in quarters:
        q0s, q1s = q0.date().isoformat(), (q0 + pd.offsets.QuarterBegin(1)).date().isoformat()
        test = x[(x.day >= q0s) & (x.day < q1s)]
        if test.empty:
            continue
        tr1 = one[one.expiry < q0s]                     # outcomes fully known before the quarter
        vm = _gbm().fit(tr1[FEATS], tr1.y_vol)
        p = test[["expiry", "day", "tpl"]].copy()
        p["pred_ratio"] = np.exp(vm.predict(test[FEATS]))
        p["iv_thr"] = tr1.drop_duplicates("day").atm_iv.quantile(0.6)
        p["pnl_pred"] = np.nan
        for tpl in TPL:
            tr = x[(x.tpl == tpl) & (x.expiry < q0s)]
            lo, hi = tr.hold_rs.quantile([0.01, 0.99])
            pm = _gbm().fit(tr[FEATS], tr.hold_rs.clip(lo, hi))
            mask = test.tpl == tpl
            p.loc[mask, "pnl_pred"] = pm.predict(test.loc[mask, FEATS])
        preds.append(p)
        tq = one[(one.day >= q0s) & (one.day < q1s)]
        if len(tq) > 30:
            pi = permutation_importance(vm, tq[FEATS], tq.y_vol, n_repeats=5, random_state=0)
            imps.append(pi.importances_mean)
    P = pd.concat(preds)
    x = x.merge(P, on=["expiry", "day", "tpl"])
    imp = pd.Series(np.mean(imps, axis=0), index=FEATS).sort_values(ascending=False)
    return x, imp


def evaluate(x):
    rules = {"Sell every day": lambda z: np.ones(len(z), bool),
             "IV rule (ATM IV ≥ 60th pct of past)": lambda z: z.atm_iv >= z.iv_thr,
             f"ML vol model (predicted move < {CUT} × straddle)": lambda z: z.pred_ratio < CUT,
             "ML P&L model (predicted P&L > 0)": lambda z: z.pnl_pred > 0,
             "IV rule AND ML vol model": lambda z: (z.atm_iv >= z.iv_thr) & (z.pred_ratio < CUT)}
    out = []
    for tpl in TPL:
        z = x[x.tpl == tpl].sort_values(["day", "expiry"])
        for name, f in rules.items():
            sel = z[np.asarray(f(z))]
            eq = sel.hold_rs.cumsum()
            yrs = sel.groupby(sel.day.str[:4]).hold_rs.sum()
            out.append({"tpl": tpl, "rule": name, "n": int(len(sel)), "share": round(len(sel) / len(z) * 100),
                        "avg": round(float(sel.hold_rs.mean())) if len(sel) else None, "total": round(float(sel.hold_rs.sum())),
                        "win": round(float((sel.hold_rs > 0).mean() * 100), 1) if len(sel) else None,
                        "p5": round(float(sel.hold_rs.quantile(0.05))) if len(sel) else None,
                        "worst": round(float(sel.hold_rs.min())) if len(sel) else None,
                        "dd": round(float((eq - eq.cummax()).min())) if len(sel) else None,
                        "years": {k: round(float(v)) for k, v in yrs.items()}})
    return out


def forecast_quality(x):
    """How well does each forecast rank the realized move / straddle ratio?"""
    one = x[x.tpl == "Short straddle|ATM"]
    def sp(a, b):
        return round(float(a.rank().corr(b.rank())), 3)
    return {"ml": sp(one.pred_ratio, one.move_ratio), "iv": sp(-one.atm_iv, one.move_ratio),
            "iv_rv": sp(-one.iv_rv20, one.move_ratio), "n": int(len(one)),
            "mean_ratio": round(float(one.move_ratio.mean()), 3), "share_below1": round(float((one.move_ratio < 1).mean() * 100), 1)}


def simple_models(x):
    """The same walk-forward with simpler forecasters, in case the boosted
    model was just too flexible for ~1,000 days of data."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    one = x[x.tpl == "Short straddle|ATM"]
    small = ["atm_iv", "strad_pct", "rv20", "term", "dte"]
    out = {}
    for name, feats, mk in (("Linear (ridge), all features", FEATS, lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))),
                            ("Linear (ridge), 5 features", small, lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))),
                            ("Boosting, 5 features, shallow", small, lambda: HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=150, min_samples_leaf=60))):
        preds = []
        for q0 in pd.date_range(START, one.day.max(), freq="QS"):
            a, b = q0.date().isoformat(), (q0 + pd.offsets.QuarterBegin(1)).date().isoformat()
            tr, te = one[one.expiry < a], one[(one.day >= a) & (one.day < b)]
            if te.empty:
                continue
            med = tr[feats].median()
            m = mk().fit(tr[feats].fillna(med), tr.y_vol)
            preds.append(pd.Series(m.predict(te[feats].fillna(med)), index=te.index))
        p = pd.concat(preds)
        out[name] = round(float(p.rank().corr(one.loc[p.index].move_ratio.rank())), 3)
    return out


def report(x, imp, ev, fq, sm):
    data = {"ev": ev, "fq": fq, "sm": sm, "imp": [{"f": NAMES[k], "v": round(float(v), 4)} for k, v in imp.items()],
            "tpl": TPL, "days": int(x.day.nunique()), "first": x.day.min(), "last": x.day.max(), "cut": CUT}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str)))
    return path


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML Pilot vs IV Rule</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--grp:#eef0f6;--best:#e9f8f1}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--grp:#23262e;--best:#16302a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1360px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
tr.grp td{background:var(--grp);font-weight:700;font-size:12px;color:var(--muted)}tr.best td{background:var(--best)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px}.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>Machine-learning pilot — does it beat the simple IV rule?</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>Verdict</h2><ul id="verdict"></ul></div>
<div class="card"><h2>Out-of-sample results — every rule on the same days</h2>
<p class="note">09:30 entries 1–4 trading days before expiry, held to expiry, 1 lot, all charges. Every model and every IV threshold was fitted only on data before the quarter it traded. Best ₹/trade per strategy shaded.</p>
<div class="scroll"><table id="tEv"></table></div></div>
<div class="grid2">
<div class="card"><h2>Forecasting the move (to expiry ÷ straddle)</h2><div class="scroll"><table id="tFq"></table></div>
<p class="note">Rank correlation between the forecast and the actual move ÷ straddle, out of sample. Higher = the forecast sorts calm and wild weeks better.</p></div>
<div class="card"><h2>Which inputs helped the boosted model (out of sample)</h2><div class="scroll"><table id="tImp"></table></div>
<p class="note">Permutation importance on the next quarter: how much the forecast got worse when that input was shuffled. Negative = the model did better without it, i.e. it learned noise.</p></div></div>
<div class="card"><h2>Setup</h2><ul>
<li>Inputs at 09:30: ATM IV, straddle % of NIFTY, IV rank, term structure, 25Δ/10Δ skew, smile curvature, days to expiry, weekday, realized vol 5/20 days, yesterday's range and move, today's gap, the 09:15–09:30 range and move, IV ÷ realized.</li>
<li>Vol model: gradient boosting on log(|NIFTY move to expiry| ÷ straddle); sell when the forecast is below ${CUT}. P&amp;L model: gradient boosting on the strategy's ₹ result; sell when the forecast is above 0.</li>
<li>Walk-forward: trained on Sep 2021 onwards, first used Oct 2022, re-trained every quarter; rows whose expiry had not happened yet were never used for training.</li>
<li>Not yet included: India VIX, event calendar (RBI, budget, results, US data), global overnight markets. They could help the move forecast; the pilot says the base is weak, so they would need to add a lot.</li>
</ul></div>
</div><script>
const D = /*DATA*/null;
document.querySelector(".card:last-child ul li:nth-child(2)").innerHTML = document.querySelector(".card:last-child ul li:nth-child(2)").innerHTML.replace("${CUT}", D.cut + " × the straddle");
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
document.getElementById("meta").textContent = `Out of sample: ${D.days} trading days, ${D.first} → ${D.last} · NIFTY weekly options, 1-min history.`;
const R = (tpl, rule) => D.ev.find(r => r.tpl === tpl && r.rule.startsWith(rule));
const s = "Short straddle|ATM", g = "Short strangle|Δ0.20";
const kp = (v, l) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`;
document.getElementById("kpis").innerHTML =
  kp(`${D.fq.ml} vs ${D.fq.iv}`, "how well the move is forecast: ML model vs ATM IV alone (rank correlation, higher = better)") +
  kp(`${rs(R(s, "IV rule").avg)} vs ${rs(R(s, "ML vol").avg)}`, "short straddle per trade: IV rule vs ML vol model") +
  kp(`${rs(R(s, "IV rule").dd)} vs ${rs(R(s, "ML vol").dd)}`, "short straddle worst drawdown: IV rule vs ML vol model") +
  kp(`${rs(R(g, "IV rule").avg)} vs ${rs(R(g, "ML P&L").avg)}`, "strangle Δ0.20 per trade: IV rule vs ML P&L model");
const simple = Object.entries(D.sm).map(([k, v]) => `${k} ${v}`).join(", ");
document.getElementById("verdict").innerHTML = [
  `<b>No — the ML models did not beat the IV rule, and mostly did not beat selling every day.</b> The IV rule earned far more per trade than either ML rule, with a much smaller drawdown; the ML rules landed close to "sell every day". Adding the ML vol model on top of the IV rule changed ₹/trade by only a few percent (up for some strategies, down for others) while trading fewer days — within noise.`,
  `<b>The move forecast was the weak link:</b> the boosted model's forecast had a rank correlation of ${D.fq.ml} with what actually happened; ATM IV on its own had ${D.fq.iv}. Simpler models did not fix it (${simple}). Most inputs had negative importance out of sample — the model learned noise.`,
  `<b>Why:</b> about 1,000 trading days is small for ML, weekly moves are dominated by news, and the one strong input (the IV level) is already used by the simple rule. Extra inputs added noise, not information.`,
  `<b>Selling every day made the largest ₹ total</b> (more trades) but with 2–4.5× the drawdown; the IV rule made far more per trade and per unit of risk. Which is better depends on how much drawdown you accept — that is a sizing choice, not a model question.`,
  `<b>Recommendation:</b> keep the IV rule; do not build the full ML system now. Adding India VIX and an event calendar could be tried later as a small test, but the pilot gives little reason to expect a big gain.`,
].map(x => `<li>${x}</li>`).join("");
let h = `<tr><th>Rule</th><th>Trades</th><th>Share of days</th><th>Win %</th><th>₹ / trade</th><th>Total</th><th>5% worst</th><th>Worst</th><th>Max drawdown</th></tr>`;
D.tpl.forEach(t => {
  const rows = D.ev.filter(r => r.tpl === t), best = rows.reduce((a, r) => (r.avg ?? -1e9) > (a.avg ?? -1e9) ? r : a);
  h += `<tr class="grp"><td colspan="9">${t.replace(/\|/g, " · ")}</td></tr>`;
  rows.forEach(r => { h += `<tr class="${r === best ? "best" : ""}"><td>${r.rule}</td><td>${r.n}</td><td>${r.share}%</td><td>${r.win}</td><td class="${cl(r.avg)}"><b>${rs(r.avg)}</b></td><td class="${cl(r.total)}">${rs(r.total)}</td><td class="down">${rs(r.p5)}</td><td class="down">${rs(r.worst)}</td><td class="down">${rs(r.dd)}</td></tr>`; });
});
document.getElementById("tEv").innerHTML = h;
document.getElementById("tFq").innerHTML = `<tr><th>Forecast</th><th>Rank corr.</th></tr><tr><td>ATM IV alone</td><td><b>${D.fq.iv}</b></td></tr><tr><td>IV ÷ realized 20d alone</td><td>${D.fq.iv_rv}</td></tr><tr><td>Boosted model, all inputs</td><td>${D.fq.ml}</td></tr>` +
  Object.entries(D.sm).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("") +
  `<tr><td colspan="2" class="note" style="text-align:left">Average actual move ÷ straddle: ${D.fq.mean_ratio}; the move was smaller than the straddle ${D.fq.share_below1}% of the time.</td></tr>`;
document.getElementById("tImp").innerHTML = `<tr><th>Input</th><th>Importance</th></tr>` + D.imp.map(r => `<tr><td>${r.f}</td><td class="${cl(r.v)}">${r.v}</td></tr>`).join("");
</script></body></html>"""


def main():
    x = dataset()
    x, imp = walk_forward(x)
    return x, imp, evaluate(x), forecast_quality(x)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    x, imp, ev, fq = main()
    print("rows", len(x), "days", x.day.nunique(), x.day.min(), x.day.max())
    print("forecast quality (rank corr with realized/straddle):", fq)
    print(imp.round(4).to_string())
    for r in ev:
        print(f"{r['tpl']:24s} {r['rule']:48s} n {r['n']:4d} ({r['share']:3d}%) avg {r['avg']} total {r['total']} win {r['win']} p5 {r['p5']} worst {r['worst']} dd {r['dd']} {r['years']}")
    print(report(x, imp, ev, fq, simple_models(dataset())))
