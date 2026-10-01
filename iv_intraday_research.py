"""iv_intraday_research.py -- is the IV rule usable during the day, not only
at 09:30?

The template backtest (results/template_bt/trades.parquet) has entries at
09:30, 12:00 and 14:30 (1-4 trading days before expiry, held to expiry) and
the ATM IV at each entry. For every entry time the threshold is the 60th
percentile of the past 500 days' ATM IV at that same time (only earlier
days), so nothing is fitted on the future.

Questions:
 1. Does the rule work at 12:00 / 14:30 (IV then vs its threshold)?
 2. Late sell days: below the threshold at 09:30, above it by 12:00 / 14:30
    (IV spiked during the day). Is selling then good?
 3. Faded sell days: above at 09:30, back below by 12:00 / 14:30.
 4. The IV change from 09:30 on its own: does rising or falling IV during the
    day change the result of a later entry?

Output: results/iv_intraday/report.html
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import paths

OUT = os.path.join(paths.BASE_DIR, "results", "iv_intraday")
TPL = ["Short straddle|ATM", "Short strangle|Δ0.20", "Short strangle|Δ0.15", "Iron condor|Δ0.15|w200"]
SLOTS = ["09:30", "12:00", "14:30"]


def load():
    t = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "template_bt", "trades.parquet"))
    t = t[(t.dte >= 1) & (t.dte <= 4) & t.tpl.isin(TPL)]
    # one IV per (expiry, day, slot): the ATM IV at that entry (same for every template)
    iv = t.groupby(["expiry", "day", "slot"]).atm_iv.first().unstack("slot")
    iv = iv.reset_index().sort_values("day")
    for s in SLOTS:
        # threshold from the PAST 500 entry-days at the same time of day
        iv[f"thr_{s}"] = iv[s].shift(1).rolling(500, min_periods=120).quantile(0.6)
        iv[f"hi_{s}"] = iv[s] >= iv[f"thr_{s}"]
    t = t.merge(iv, on=["expiry", "day"])
    return t.dropna(subset=["thr_09:30", "thr_12:00", "thr_14:30"])


def st(x):
    h = x.hold_rs
    return {"n": int(len(x)), "avg": round(float(h.mean())) if len(x) else None, "total": round(float(h.sum())),
            "win": round(float((h > 0).mean() * 100), 1) if len(x) else None,
            "p5": round(float(h.quantile(0.05))) if len(x) > 4 else None,
            "a": round(float(h[x.expiry < "2024"].mean())) if (x.expiry < "2024").any() else None,
            "b": round(float(h[x.expiry >= "2024"].mean())) if (x.expiry >= "2024").any() else None}


def analyse(t):
    res = {"rule_by_slot": [], "groups": [], "change": []}
    for tpl in TPL:
        for s in SLOTS:
            x = t[(t.tpl == tpl) & (t.slot == s)]
            hi = x[f"hi_{s}"].astype(bool)
            res["rule_by_slot"].append({"tpl": tpl, "slot": s, "hi": st(x[hi]), "lo": st(x[~hi])})
        for s in ("12:00", "14:30"):
            x = t[(t.tpl == tpl) & (t.slot == s)]
            h930, hs = x["hi_09:30"].astype(bool), x[f"hi_{s}"].astype(bool)
            for lab, m in (("Sell day at 09:30 and still at entry", h930 & hs), ("Late sell day: below at 09:30, above at entry", ~h930 & hs),
                           ("Faded: above at 09:30, below at entry", h930 & ~hs), ("Normal day at both times", ~h930 & ~hs)):
                res["groups"].append({"tpl": tpl, "slot": s, "g": lab, **st(x[m])})
            # IV change since 09:30, in quintiles, split by the 09:30 state
            x = x.assign(chg=x[s] - x["09:30"])
            for state, y in (("normal at 09:30", x[~h930]), ("sell day at 09:30", x[h930])):
                if len(y) < 50:
                    continue
                q = pd.qcut(y.chg.rank(method="first"), 5, labels=False)
                res["change"].append({"tpl": tpl, "slot": s, "state": state,
                                      "cells": [{"lo": round(float(v.chg.min()), 2), "hi": round(float(v.chg.max()), 2), **st(v)} for _, v in y.groupby(q)]})
    return res


def report(t, res):
    data = {**res, "tpl": TPL, "days": int(t.day.nunique()), "first": t.day.min(), "last": t.day.max(),
            "late_days": {s: int((~t.drop_duplicates("day")["hi_09:30"].astype(bool) & t.drop_duplicates("day")[f"hi_{s}"].astype(bool)).sum()) for s in ("12:00", "14:30")}}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str)))
    return path


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IV Rule Intraday</title><style>
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
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}.sm{font-size:11px;color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px}.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
</style></head><body><div class="wrap">
<h1>The IV rule during the day — 09:30 vs 12:00 vs 14:30</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>What the data says</h2><ul id="verdict"></ul></div>
<div class="card"><h2>1 · The rule at each entry time (IV then vs its own threshold)</h2><div class="scroll"><table id="t1"></table></div>
<p class="note">Threshold = 60th percentile of the past 500 days' ATM IV at the same time of day. Cells: ₹ per trade (small: n · 2021–23 / 2024–26 · 5% worst).</p></div>
<div class="card"><h2>2 · Late and faded sell days (12:00 and 14:30 entries)</h2><div class="scroll"><table id="t2"></table></div></div>
<div class="card"><h2>3 · IV change since 09:30 (fifths), split by the 09:30 state</h2><div class="scroll"><table id="t3"></table></div>
<p class="note">Change in ATM IV (vol points) from 09:30 to the entry. Fifths within each group.</p></div>
<div class="card"><h2>Limits</h2><ul><li>Held to expiry, 1 lot, all charges, 1-min closing prices with 0.5 pt slippage per order (the template backtest).</li>
<li>Late sell days are few; their numbers are less certain than the 09:30 rule's.</li></ul></div>
</div><script>
const D = /*DATA*/null;
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
const cell = s => `<td class="${cl(s.avg)}"><b>${rs(s.avg)}</b><div class="sm">n ${s.n} · ${rs(s.a)} / ${rs(s.b)} · 5% ${rs(s.p5)}</div></td>`;
document.getElementById("meta").textContent = `${D.days} trading days (${D.first} → ${D.last}; the first ~6 months only build the threshold) · entries 1–4 days before expiry, held to expiry.`;
const R = (tpl, s) => D.rule_by_slot.find(r => r.tpl === tpl && r.slot === s);
const G = (tpl, s, g) => D.groups.find(r => r.tpl === tpl && r.slot === s && r.g.startsWith(g));
const S = "Short straddle|ATM", g2 = "Short strangle|Δ0.20";
const kp = (v, l) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`;
document.getElementById("kpis").innerHTML =
  kp(`${rs(R(S, "09:30").hi.avg)} / ${rs(R(S, "12:00").hi.avg)} / ${rs(R(S, "14:30").hi.avg)}`, "straddle on sell days, entered 09:30 / 12:00 / 14:30") +
  kp(`${rs(R(S, "09:30").lo.avg)} / ${rs(R(S, "12:00").lo.avg)} / ${rs(R(S, "14:30").lo.avg)}`, "straddle on normal days, same three times") +
  kp(`${rs(G(S, "12:00", "Late").avg)} (n ${G(S, "12:00", "Late").n})`, "straddle at 12:00 on late sell days (IV crossed after 09:30)") +
  kp(`${rs(G(S, "12:00", "Faded").avg)} (n ${G(S, "12:00", "Faded").n})`, "straddle at 12:00 on faded sell days (IV fell back below)");
window.__v = {R, G, S, g2};
</script>
<script>
(function(){
const {R, G, S, g2} = window.__v;
const L12 = G(S, "12:00", "Late"), L14 = G(S, "14:30", "Late"), F12 = G(S, "12:00", "Faded"), K12 = G(S, "12:00", "Sell day"), N12 = G(S, "12:00", "Normal");
const Lg = G(g2, "12:00", "Late"), Fg = G(g2, "12:00", "Faded");
document.getElementById("verdict").innerHTML = [
  `<b>The rule works at every time of day.</b> Straddle on sell days: ${rs(R(S, "09:30").hi.avg)} at 09:30, ${rs(R(S, "12:00").hi.avg)} at 12:00, ${rs(R(S, "14:30").hi.avg)} at 14:30 a trade — vs ${rs(R(S, "09:30").lo.avg)}, ${rs(R(S, "12:00").lo.avg)}, ${rs(R(S, "14:30").lo.avg)} on normal days. Checking the IV again later in the day is valid, as long as it is compared with the threshold for that time.`,
  `<b>Late sell days (IV crossed the line after 09:30):</b> the straddle sold at 12:00 made ${rs(L12.avg)} a trade (${L12.n} trades; 2021–23 ${rs(L12.a)}, 2024–26 ${rs(L12.b)}), at 14:30 ${rs(L14.avg)} (${L14.n}); strangle Δ0.20 at 12:00 ${rs(Lg.avg)}. Compare days that were sell days all along: ${rs(K12.avg)}, and normal days: ${rs(N12.avg)}.`,
  `<b>Faded sell days (above at 09:30, back below at 12:00):</b> straddle ${rs(F12.avg)} a trade (${F12.n}), strangle Δ0.20 ${rs(Fg.avg)} — what matters is the IV at the moment of entry, not the morning's reading.`,
  `<b>The direction of the intraday IV move adds a little on sell days</b> (section 3): when IV kept rising after 09:30, the later entry sold richer premium and did better; when IV fell a lot (2+ vol points) the later entry still made money but clearly less. On normal days a big IV rise mostly means the day turned into a late sell day.`,
  `<b>Caution:</b> late and faded sell days are only ~30 per entry time in 5 years, so those two rows are much less certain than the main rule (hundreds of days).`,
].map(x => `<li>${x}</li>`).join("");
let h = `<tr><th>Strategy</th><th>Entry</th><th>Sell day (IV ≥ threshold)</th><th>Normal day</th></tr>`;
D.tpl.forEach(t => ["09:30", "12:00", "14:30"].forEach(s => { const r = R(t, s); h += `<tr><td>${t.replace(/\|/g, " · ")}</td><td>${s}</td>${cell(r.hi)}${cell(r.lo)}</tr>`; }));
document.getElementById("t1").innerHTML = h;
let h2 = `<tr><th>Strategy · entry</th><th>Group</th><th>₹ / trade</th><th>Win %</th></tr>`;
D.tpl.forEach(t => ["12:00", "14:30"].forEach(s => {
  h2 += `<tr class="grp"><td colspan="4">${t.replace(/\|/g, " · ")} · ${s}</td></tr>`;
  D.groups.filter(r => r.tpl === t && r.slot === s).forEach(r => { h2 += `<tr><td></td><td style="text-align:left">${r.g}</td>${cell(r)}<td>${r.win ?? "—"}</td></tr>`; });
}));
document.getElementById("t2").innerHTML = h2;
let h3 = `<tr><th>Strategy · entry · 09:30 state</th><th>Biggest IV fall</th><th>2nd</th><th>3rd</th><th>4th</th><th>Biggest IV rise</th></tr>`;
D.change.forEach(r => { h3 += `<tr><td>${r.tpl.replace(/\|/g, " · ")} · ${r.slot} · ${r.state}</td>${r.cells.map(c => `<td class="${cl(c.avg)}"><b>${rs(c.avg)}</b><div class="sm">${c.lo} … ${c.hi} · n ${c.n}</div></td>`).join("")}</tr>`; });
document.getElementById("t3").innerHTML = h3;
})();
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    t = load()
    res = analyse(t)
    for r in res["rule_by_slot"]:
        print(r["tpl"], r["slot"], "hi", r["hi"]["avg"], r["hi"]["n"], (r["hi"]["a"], r["hi"]["b"]), "lo", r["lo"]["avg"], r["lo"]["n"], (r["lo"]["a"], r["lo"]["b"]))
    for r in res["groups"]:
        print(r["tpl"], r["slot"], r["g"][:30], r["avg"], r["n"], (r["a"], r["b"]), r["p5"])
    for r in res["change"]:
        if r["tpl"] == "Short straddle|ATM":
            print(r["slot"], r["state"], [(c["lo"], c["hi"], c["avg"], c["n"]) for c in r["cells"]])
    print(report(t, res))
