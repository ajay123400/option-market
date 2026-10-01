"""fii_research.py -- does NSE's participant-wise open interest (FII / Client
/ Pro positions, end of day) say anything about NIFTY's next day / week, or
about when to sell premium?

Signals are computed from day t's file and matched with what happened AFTER
t (next day, next 5 days; strategy entries from the next trading day), so
nothing is known before it was published. Every signal is checked separately
on 2021-23 and 2024-26: a real effect has the same sign in both.

Output: results/fii/report.html
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import paths
import participant_oi as P
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "fii")
WHO = {"FII": "FII", "Client": "Client (retail + HNI)", "Pro": "Pro (broker prop)"}
NAMES = {
    "fut_long": "Index futures: long % of positions",
    "fut_net_chg": "Index futures: 1-day change in net long (share of OI)",
    "opt_bias": "Index options: bullish tilt (net calls − net puts, share)",
    "opt_bias_chg": "Index options: 1-day change in bullish tilt",
    "put_write": "Index puts: net written (short − long, share)",
    "call_write": "Index calls: net written (short − long, share)",
}
OUTCOMES = {"ret1": "Next day close-to-close %", "oc1": "Next day open-to-close %", "gap1": "Next day gap %",
            "ret5": "Next 5 days %", "abs1": "Next day size of move %"}
TPL = ["Short strangle|Δ0.15", "Short strangle|Δ0.20", "Short straddle|ATM", "Short put|Δ0.20", "Short call|Δ0.20"]


def features():
    df = pd.read_parquet(os.path.join(P.DIR, "all.parquet"))
    f = []
    for who in WHO:
        x = df[df.client == who].sort_values("day").set_index("day")
        fl, fs = x.fut_idx_long, x.fut_idx_short
        cl, cs, pl, ps = x.idx_call_long, x.idx_call_short, x.idx_put_long, x.idx_put_short
        opt = cl + cs + pl + ps
        g = pd.DataFrame(index=x.index)
        g["fut_long"] = fl / (fl + fs)
        g["fut_net_chg"] = (fl - fs).diff() / (fl + fs)
        g["opt_bias"] = ((cl - cs) - (pl - ps)) / opt
        g["opt_bias_chg"] = g.opt_bias.diff()
        g["put_write"] = (ps - pl) / (ps + pl)
        g["call_write"] = (cs - cl) / (cs + cl)
        g.columns = [f"{who}.{c}" for c in g.columns]
        f.append(g)
    return pd.concat(f, axis=1).replace([np.inf, -np.inf], np.nan)


def outcomes():
    sp = S._load_spot()
    d = sp.groupby(sp.index.date).agg(open=("open", "first"), close=("close", "last"))
    d.index = [x.isoformat() for x in d.index]
    o = pd.DataFrame(index=d.index)
    o["ret1"] = (d.close.shift(-1) / d.close - 1) * 100
    o["oc1"] = (d.close.shift(-1) / d.open.shift(-1) - 1) * 100
    o["gap1"] = (d.open.shift(-1) / d.close - 1) * 100
    o["ret5"] = (d.close.shift(-5) / d.close - 1) * 100
    o["abs1"] = o.ret1.abs()
    o["next_day"] = pd.Series(d.index, index=d.index).shift(-1)
    return o


def _ic(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return None, None, 0
    r = a[m].rank().corr(b[m].rank())
    n = int(m.sum())
    t = r * math.sqrt((n - 2) / max(1e-9, 1 - r * r))
    return round(float(r), 3), round(float(t), 2), n


def ic_table(F, O):
    X = F.join(O, how="inner")
    A, B = X[X.index < "2024"], X[X.index >= "2024"]
    rows = []
    for c in F.columns:
        r = {"f": c}
        for oc in OUTCOMES:
            ra, ta, na = _ic(A[c], A[oc])
            rb, tb, nb = _ic(B[c], B[oc])
            r[oc] = {"a": ra, "ta": ta, "b": rb, "tb": tb, "n": na + nb,
                     "stable": bool(ra is not None and rb is not None and np.sign(ra) == np.sign(rb) and abs(ta) >= 2 and abs(tb) >= 2)}
        rows.append(r)
    return rows


def quintiles(F, O, cols):
    X = F.join(O, how="inner")
    out = {}
    for c in cols:
        x = X.dropna(subset=[c, "ret1"])
        x = x.assign(q=pd.qcut(x[c].rank(method="first"), 5, labels=False), half=np.where(x.index < "2024", "a", "b"))
        out[c] = [{"q": int(q), "lo": round(float(v[c].min()), 3), "hi": round(float(v[c].max()), 3), "n": int(len(v)),
                   "ret1": round(float(v.ret1.mean()), 3), "up": round(float((v.ret1 > 0).mean() * 100), 1),
                   "ret5": round(float(v.ret5.mean()), 3), "abs1": round(float(v.abs1.mean()), 3),
                   "a": round(float(v[v.half == "a"].ret1.mean()), 3), "b": round(float(v[v.half == "b"].ret1.mean()), 3)}
                  for q, v in x.groupby("q")]
    return out


def strategy_filter(F, O):
    """Template trades entered the NEXT trading day (09:30, 1-4 days to
    expiry); threshold = top 40% of 2021-23 days, applied to both periods."""
    t = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "template_bt", "trades.parquet"))
    t = t[(t.slot == "09:30") & (t.dte >= 1) & (t.dte <= 4) & t.tpl.isin(TPL)]
    # signal from the previous trading day's file
    prev = O[["next_day"]].dropna().reset_index().rename(columns={"index": "sig_day", "next_day": "day"})
    t = t.merge(prev, on="day").merge(F, left_on="sig_day", right_index=True)
    t["half"] = np.where(t.expiry < "2024", "2021–23", "2024–26")
    res = []
    for c in F.columns:
        thr = t[t.half == "2021–23"].drop_duplicates("day")[c].quantile(0.6)
        for tpl in TPL:
            row = {"f": c, "tpl": tpl}
            for h, x in t[t.tpl == tpl].groupby("half"):
                hi = x[c] >= thr
                row[h] = {"hi": round(float(x[hi].hold_rs.mean())), "lo": round(float(x[~hi].hold_rs.mean())),
                          "nh": int(hi.sum()), "nl": int((~hi).sum())}
            a, b = row.get("2021–23"), row.get("2024–26")
            row["stable"] = bool(a and b and np.sign(a["hi"] - a["lo"]) == np.sign(b["hi"] - b["lo"])
                                 and abs(a["hi"] - a["lo"]) > 300 and abs(b["hi"] - b["lo"]) > 300)
            res.append(row)
    return res


LAGS = list(range(20, 261, 20))


def _eff(t, Fx, c, tpl):
    x = t[t.tpl == tpl].merge(Fx[[c]], left_on="sig_day", right_index=True).dropna(subset=[c])
    A, B = x[x.expiry < "2024"], x[x.expiry >= "2024"]
    thr = A.drop_duplicates("day")[c].quantile(0.6)
    return (float(A[A[c] >= thr].hold_rs.mean() - A[A[c] < thr].hold_rs.mean()),
            float(B[B[c] >= thr].hold_rs.mean() - B[B[c] < thr].hold_rs.mean()), int((B[c] >= thr).sum()))


def placebo(F, O):
    """Each (signal, strategy): the high-vs-rest difference with the real
    signal, and with the same signal shifted 20..260 trading days (it cannot
    know anything about those trades). A real effect should beat its placebos."""
    t = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "template_bt", "trades.parquet"))
    t = t[(t.slot == "09:30") & (t.dte >= 1) & (t.dte <= 4) & t.tpl.isin(TPL)]
    prev = O[["next_day"]].dropna().reset_index().rename(columns={"index": "sig_day", "next_day": "day"})
    t = t.merge(prev, on="day")
    rows, same_real, same_plc = [], 0, []
    plc_counts = {L: 0 for L in LAGS}
    for c in F.columns:
        for tpl in TPL:
            a, b, nb = _eff(t, F, c, tpl)
            same = np.sign(a) == np.sign(b) and min(abs(a), abs(b)) > 300
            same_real += same
            pl = [_eff(t, F.shift(L), c, tpl) for L in LAGS]
            for L, (x, y, _) in zip(LAGS, pl):
                plc_counts[L] += np.sign(x) == np.sign(y) and min(abs(x), abs(y)) > 300
            m = min(abs(a), abs(b)) if np.sign(a) == np.sign(b) else 0
            k = sum(1 for x, y, _ in pl if np.sign(x) == np.sign(y) and min(abs(x), abs(y)) >= m) if m else None
            rows.append({"f": c, "tpl": tpl, "a": round(a), "b": round(b), "nb": nb, "beat": k == 0})
    return {"rows": rows, "real": int(same_real), "placebo": {str(L): int(v) for L, v in plc_counts.items()},
            "beat_all": int(sum(r["beat"] for r in rows)), "total": len(rows)}


def build_report():
    F, O = features(), outcomes()
    sp = S._load_spot()
    d = sp.groupby(sp.index.date).agg(open=("open", "first"), close=("close", "last"))
    d.index = [x.isoformat() for x in d.index]
    O["o5"] = (d.close.shift(-5) / d.open.shift(-1) - 1) * 100
    OUTCOMES["o5"] = "Next day open → 5th day close % (tradeable)"
    ic = ic_table(F, O)
    pb = placebo(F, O)
    x = F[["FII.fut_long"]].dropna()
    thr = float(x[x.index < "2024"]["FII.fut_long"].quantile(0.6))
    hi = x[(x.index >= "2024") & (x["FII.fut_long"] >= thr)]
    drift = {w: {y: round(float(v) * 100, 1) for y, v in F[f"{w}.fut_long"].groupby(F.index.str[:4]).mean().items()} for w in WHO}
    data = {"ic": ic, "outcomes": OUTCOMES, "names": NAMES, "who": WHO, "placebo": pb,
            "fii_hi_months": hi.index.str[:7].value_counts().sort_index().to_dict(), "fii_thr": round(thr * 100, 1),
            "drift": drift, "days": int(len(F)), "first": F.index.min(), "last": F.index.max()}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str)))
    return path, data


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FII / Client / Pro Positions</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--grp:#eef0f6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--grp:#23262e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1360px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
tr.grp td{background:var(--grp);font-weight:700;font-size:12px;color:var(--muted)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px}.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
td.c{text-align:center}.sm{font-size:11px;color:var(--muted)}
</style></head><body><div class="wrap">
<h1>FII / Client / Pro positions — do they predict NIFTY or help sell premium?</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>What the data says</h2><ul id="verdict"></ul></div>
<div class="card"><h2>Signal → what happened next (rank correlation, 2021–23 | 2024–26)</h2>
<p class="note">Each cell: correlation in 2021–23 and 2024–26 with its t-stat. Shaded = |t| ≥ 2 in <b>both</b> periods with the same sign. The file is published after the close, so only "open → 5th day" and "open-to-close" can actually be traded; the gap happens before you can act.</p>
<div class="scroll"><table id="tIc"></table></div></div>
<div class="card"><h2>Strategy filter — real signal vs placebo</h2>
<p class="note">For each signal × strategy (short strangle Δ0.15 / Δ0.20, short straddle, short put, short call; 09:30 entries 1–4 days before expiry, the day after the file): top 40% of days (threshold from 2021–23) vs the rest. "Consistent" = the same-sign difference of more than ₹300 a trade in both periods. Placebo = the same signal moved 20–260 trading days away, so it cannot know anything about those trades.</p>
<div class="scroll"><table id="tPl"></table></div>
<h2 style="margin-top:16px">The ones that beat all their placebos</h2><div class="scroll"><table id="tBeat"></table></div></div>
<div class="card"><h2>Why fixed thresholds break — the positions drift for years</h2><div class="scroll"><table id="tDrift"></table></div>
<p class="note" id="hiNote"></p></div>
<div class="card"><h2>Limits</h2><ul>
<li>NSE's file is end of day and covers <b>all</b> index derivatives (NIFTY, BANKNIFTY, FINNIFTY, …), not NIFTY alone, and not by strike — it cannot say who wrote a given OI wall.</li>
<li>Contract counts change with lot sizes; the signals here are shares (long ÷ long+short), which are not affected.</li>
<li>Option positions collapse at every expiry, so the options signals jump on expiry days.</li>
</ul></div>
</div><script>
const D = /*DATA*/null;
document.getElementById("meta").textContent = `${D.days.toLocaleString("en-IN")} trading days of NSE participant-wise OI (${D.first} → ${D.last}), matched with NIFTY and the 5-year template backtest.`;
const P = D.placebo;
const pv = Object.values(P.placebo), pmin = Math.min(...pv), pmax = Math.max(...pv);
const stable = (o) => D.ic.filter(r => r[o].stable).length;
const kp = (v, l) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`;
document.getElementById("kpis").innerHTML =
  kp(`${stable("oc1")} / ${D.ic.length}`, "signals that predict the next day's open-to-close move in both periods") +
  kp(`${stable("o5")} / ${D.ic.length}`, "signals that predict the tradeable next-5-day move in both periods") +
  kp(`${P.real} vs ${pmin}–${pmax}`, `"consistent" strategy filters: real signals vs placebo signals (of ${P.total})`) +
  kp(`${D.drift.FII["2021"]}% → ${D.drift.FII["2026"]}%`, "FII index-futures long share, 2021 → 2026 — the levels drift");
const gapN = stable("gap1");
document.getElementById("verdict").innerHTML = [
  `<b>No tradeable direction signal.</b> None of the ${D.ic.length} signals predicted the next day's open-to-close move in both periods, and ${stable("o5") ? stable("o5") : "none"} predicted the move from the next open to the 5th day's close. FII's options tilt came closest (5-day correlation ~0.1, clear in 2021–23, borderline in 2024–26) — too weak to trade on its own.`,
  `<b>They do "predict" the overnight gap</b> (${gapN} signal${gapN === 1 ? "" : "s"} consistent, several strong in 2024–26) — but the file comes out after the close, so that gap cannot be captured; it mostly reflects the same day's flows continuing overnight.`,
  `<b>As a strategy filter they look good only by chance.</b> ${P.real} of ${P.total} signal × strategy pairs looked consistent in both periods — and the same signals shifted by months (which cannot know anything) gave ${pmin}–${pmax}. ${P.beat_all} pairs beat all 13 of their placebos, about what chance alone gives across ${P.total} tries (~6).`,
  `<b>The strongest-looking one is a single regime.</b> "FII index futures mostly long → sell no straddle" worked in both periods, but in 2024–26 every such day fell in Jan–Oct 2024; FII's long share then slid to ~${D.drift.FII["2026"]}% and never crossed the ${D.fii_thr}% threshold again. Client long share is its mirror image. That is one market phase, not a daily signal.`,
  `<b>Some options signals just repeat the IV level</b> (e.g. Pro call writing moves with ATM IV) — the IV filter from the earlier study already covers them.`,
  `<b>Use:</b> good background on who holds what (FII net short futures for two years is worth knowing), but no rule from it survived the checks. The IV level remains the only filter that held up out of sample.`,
].map(x => `<li>${x}</li>`).join("");
const outs = Object.keys(D.outcomes);
let h = `<tr><th>Signal</th>${outs.map(o => `<th class="c">${D.outcomes[o]}</th>`).join("")}</tr>`;
Object.keys(D.who).forEach(w => {
  h += `<tr class="grp"><td colspan="${outs.length + 1}">${D.who[w]}</td></tr>`;
  D.ic.filter(r => r.f.startsWith(w + ".")).forEach(r => {
    h += `<tr><td>${D.names[r.f.split(".")[1]]}</td>${outs.map(o => { const c = r[o]; return `<td class="c" style="${c.stable ? "background:rgba(79,95,224,.18)" : ""}">${c.a} <span class="sm">(${c.ta})</span> | ${c.b} <span class="sm">(${c.tb})</span></td>`; }).join("")}</tr>`;
  });
});
document.getElementById("tIc").innerHTML = h;
document.getElementById("tPl").innerHTML = `<tr><th>Signal set</th><th>Consistent pairs (of ${P.total})</th></tr><tr><td><b>Real signals</b></td><td><b>${P.real}</b></td></tr>` +
  Object.entries(P.placebo).map(([L, v]) => `<tr><td>Placebo: shifted ${L} trading days</td><td>${v}</td></tr>`).join("");
const rs = v => (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
document.getElementById("tBeat").innerHTML = `<tr><th>Signal</th><th>Strategy</th><th>High − rest, 2021–23</th><th>High − rest, 2024–26</th><th>2024–26 high days</th></tr>` +
  P.rows.filter(r => r.beat).map(r => `<tr><td>${D.who[r.f.split(".")[0]]} · ${D.names[r.f.split(".")[1]]}</td><td>${r.tpl.replace(/\|/g, " · ")}</td><td class="${r.a > 0 ? "up" : "down"}">${rs(r.a)}</td><td class="${r.b > 0 ? "up" : "down"}">${rs(r.b)}</td><td>${r.nb}</td></tr>`).join("");
const yrs = Object.keys(D.drift.FII);
document.getElementById("tDrift").innerHTML = `<tr><th>Index futures long share</th>${yrs.map(y => `<th>${y}</th>`).join("")}</tr>` +
  Object.entries(D.drift).map(([w, v]) => `<tr><td>${D.who[w]}</td>${yrs.map(y => `<td>${v[y]}%</td>`).join("")}</tr>`).join("");
document.getElementById("hiNote").textContent = `Days in 2024–26 with FII long share ≥ ${D.fii_thr}% (the 2021–23 top-40% threshold), by month: ` + Object.entries(D.fii_hi_months).map(([m, n]) => `${m}: ${n}`).join(", ") + ".";
</script></body></html>"""


def main():
    F, O = features(), outcomes()
    ic = ic_table(F, O)
    stable = [r["f"] for r in ic if any(r[o]["stable"] for o in OUTCOMES)]
    key = sorted(set(stable + ["FII.fut_long", "FII.fut_net_chg", "Client.fut_long", "FII.opt_bias", "Client.opt_bias"]))
    return F, O, ic, quintiles(F, O, key), strategy_filter(F, O), stable


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    F, O, ic, qt, sf, stable = main()
    print(F.index.min(), F.index.max(), len(F))
    for r in ic:
        print(f"{r['f']:22s} " + " | ".join(f"{o}: {r[o]['a']}({r[o]['ta']}) {r[o]['b']}({r[o]['tb']}){' *' if r[o]['stable'] else ''}" for o in OUTCOMES))
    print("stable:", stable)
    for r in sf:
        if r["stable"]:
            print("FILTER", r)
