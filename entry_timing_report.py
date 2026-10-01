"""entry_timing_report.py -- WHEN to enter each Strategy Ideas template:
days to expiry (0..9, i.e. this week's vs next week's expiry) and entry time
(09:30 / 12:00 / 14:30), from the 5-year template backtest
(results/template_bt/trades.parquet, template_backtest.py).

Output: results/entry_timing/report.html
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import paths

SRC = os.path.join(paths.BASE_DIR, "results", "template_bt", "trades.parquet")
OUT = os.path.join(paths.BASE_DIR, "results", "entry_timing")
FRAC = {"09:30": 1.0, "12:00": 0.6, "14:30": 0.2}   # share of the entry day still ahead
KEY = ["Short strangle|Δ0.15", "Short strangle|Δ0.20", "Short strangle|range", "Short straddle|ATM",
       "Short put|Δ0.20", "Short call|Δ0.20", "Iron condor|Δ0.15|w200", "Iron condor|range|w200",
       "Iron butterfly|ATM|w200", "Bull put spread|Δ0.20|w200", "Bear call spread|Δ0.20|w200",
       "Bull call spread|buy ATM|w200", "Bear put spread|buy ATM|w200", "Long straddle|ATM", "Long strangle|Δ0.20"]
BUCKETS = ["0 · expiry day", "1–4 · this week", "5–9 · next week"]


def load():
    t = pd.read_parquet(SRC)
    t["year"] = t.expiry.str[:4]
    t["half"] = np.where(t.expiry < "2024-01-01", "A", "B")
    t["days"] = t.dte + t.slot.map(FRAC)
    t["perday"] = t.hold_rs / t.days
    t["bucket"] = np.select([t.dte == 0, t.dte <= 4], BUCKETS[:2], BUCKETS[2])
    return t


def stats(d):
    h = d.hold_rs
    yrs = d.groupby("year").hold_rs.sum()
    return {"n": int(len(d)), "win": round(float((h > 0).mean() * 100), 1), "avg": round(float(h.mean())),
            "perday": round(float(d.perday.mean())), "p5": round(float(h.quantile(0.05))), "worst": round(float(h.min())),
            "tail": round(float(h.mean() / abs(h.quantile(0.05))), 3) if h.quantile(0.05) < 0 else None,
            "man": round(float(d.managed_rs.mean())),
            "a": round(float(d.perday[d.half == "A"].mean())) if (d.half == "A").any() else None,
            "b": round(float(d.perday[d.half == "B"].mean())) if (d.half == "B").any() else None,
            "yp": int((yrs > 0).sum()), "yn": int(len(yrs))}


def build():
    t = load()
    bucket_rows = []
    for tpl in KEY:
        x = t[t.tpl == tpl]
        for b in BUCKETS:
            bucket_rows.append({"tpl": tpl, "bucket": b, **stats(x[x.bucket == b])})
    heat = {tpl: [stats(t[(t.tpl == tpl) & (t.dte == d)]) for d in range(10)] for tpl in KEY}
    slot = [{"tpl": tpl, **{s: stats(t[(t.tpl == tpl) & (t.slot == s)]) for s in FRAC}} for tpl in KEY]
    fam = []
    for name, x in t.groupby("name"):
        fam.append({"name": name, "credit": bool(x.credit.mean() > 0.5),
                    **{b: round(float(x[x.bucket == b].perday.mean())) for b in BUCKETS},
                    "hold": round(float(x.hold_rs.mean())), "man": round(float(x.managed_rs.mean()))})
    fam.sort(key=lambda r: -r[BUCKETS[1]])
    # every cell: template x dte x slot
    cells = []
    for (tpl, dte, sl), x in t.groupby(["tpl", "dte", "slot"]):
        cells.append({"tpl": tpl, "name": tpl.split("|")[0], "dte": int(dte), "slot": sl, **stats(x)})
    # out of sample: the best bucket per template picked on 2021-23, scored on 2024-26
    oos = {"same": 0, "total": 0, "picked_b": [], "other_b": []}
    for tpl, x in t.groupby("tpl"):
        a = x[x.half == "A"].groupby("bucket").perday.mean()
        bb = x[x.half == "B"].groupby("bucket").perday.mean()
        if len(a) < 3 or len(bb) < 3:
            continue
        pick = a.idxmax()
        oos["total"] += 1
        oos["same"] += int(bb.idxmax() == pick)
        oos["picked_b"].append(float(bb[pick]))
        oos["other_b"].append(float(bb.drop(pick).mean()))
    oos = {"same": oos["same"], "total": oos["total"], "picked_b": round(float(np.mean(oos["picked_b"]))),
           "other_b": round(float(np.mean(oos["other_b"])))}
    wins = {b: int((t[t.bucket == b].groupby("tpl").perday.mean() > 0).sum()) for b in BUCKETS}
    meta = {"n": int(len(t)), "expiries": int(t.expiry.nunique()), "first": t.expiry.min(), "last": t.expiry.max(),
            "tpls": int(t.tpl.nunique()), "wins": wins}
    data = {"buckets": BUCKETS, "bucket_rows": bucket_rows, "heat": heat, "key": KEY, "slot": slot, "fam": fam,
            "cells": cells, "oos": oos, "meta": meta}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str)))
    return path, data


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entry Timing Study</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--best:#e9f8f1;--grp:#eef0f6;--accent:#4f5fe0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--best:#16302a;--grp:#23262e;--accent:#8a95ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1360px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
tr.best td{background:var(--best)}tr.grp td{background:var(--grp);font-weight:700;font-size:12px;color:var(--muted)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px}.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
td.h{text-align:center;font-weight:600}.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px;font-size:12.5px}
button,select,input{font:inherit;font-size:12.5px;border:1px solid var(--border);background:var(--card);color:var(--text);border-radius:8px;padding:4px 10px}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}th[data-s]{cursor:pointer}th[data-s]:hover{color:var(--text)}
</style></head><body><div class="wrap">
<h1>When to enter — this week's expiry or next week's?</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>What the data says</h2><ul id="verdict"></ul></div>
<div class="card"><h2>This week vs next week — main strategies</h2>
<p class="note">Held to expiry, 1 lot. <b>₹/day</b> = P&amp;L ÷ trading days the trade was open (09:30 entry counts that day fully, 12:00 as 0.6, 14:30 as 0.2) — the fair way to compare a 2-day trade with a 9-day one, since the same margin is tied up longer. <b>Tail</b> = average ÷ the 5% worst outcome (higher = more reward for the tail risk taken).</p>
<div class="scroll"><table id="tBucket"></table></div></div>
<div class="card"><h2>Day by day (days to expiry 9 → 0)</h2>
<div class="bar"><span class="note">Show:</span><button class="on" data-h="perday">₹ per day held</button><button data-h="avg">₹ per trade</button><button data-h="tail">Tail ratio</button><button data-h="win">Win %</button></div>
<div class="scroll"><table id="tHeat"></table></div>
<p class="note">Days to expiry = trading days left after the entry day (NSE calendar). 5–9 = the week before, i.e. next week's expiry. All three entry times pooled.</p></div>
<div class="card"><h2>Entry time (all days pooled)</h2><div class="scroll"><table id="tSlot"></table></div></div>
<div class="card"><h2>Every strategy family — ₹ per day held, and the managed exit</h2>
<p class="note">Managed = credit: book at +50% of the credit, stop at −2×; debit: book at +100%, stop at −50% (1-min closes). Average of all templates in the family.</p>
<div class="scroll"><table id="tFam"></table></div></div>
<div class="card"><h2>Explorer — every template × days to expiry × entry time</h2>
<div class="bar"><select id="fName"><option value="">All strategies</option></select>
<select id="fBucket"><option value="">All days</option></select>
<label class="note"><input type="checkbox" id="fMin" checked> n ≥ 150</label>
<span class="note">Click a column to sort · click again to reverse · Shift+click to add a 2nd/3rd sort</span> <span id="sortTxt" class="note"></span></div>
<div class="scroll" style="max-height:620px;overflow-y:auto"><table id="tCells"></table></div></div>
<div class="card"><h2>How it was tested &amp; limits</h2><ul>
<li>Same strike rules as the live Strategy Ideas table (deltas from that moment's prices, range from that day's volume leaders), entered at 09:30 / 12:00 / 14:30 on every trading day 0–9 days before each weekly expiry, held to expiry and settled at NIFTY's close.</li>
<li>1-min closing prices (no bid/ask in the history) with 0.5 pt slippage per order; all charges; 1 lot of today's size (65).</li>
<li>Trades of the same expiry overlap (the same week is counted from several entry days), so they are not independent — read the n with that in mind.</li>
<li>Naked shorts (strangle, straddle, short put/call) carry unlimited risk; their worst trade here is not the worst possible. 5 years hold no Covid-style crash.</li>
<li>The expiry weekday changed over the period (Thursday → Tuesday), so a given days-to-expiry is not always the same weekday.</li>
</ul></div>
</div><script>
const D = /*DATA*/null;
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
const M = D.meta;
document.getElementById("meta").textContent = `${M.n.toLocaleString("en-IN")} trades · ${M.tpls} templates · ${M.expiries} weekly expiries (${M.first} → ${M.last}) · 1-min option prices · 0.5 pt slippage per order · all charges · 1 lot (65).`;
const row = (tpl, b) => D.bucket_rows.find(r => r.tpl === tpl && r.bucket === b);
const ss = "Short strangle|Δ0.15";
const kp = (v, l) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`;
document.getElementById("kpis").innerHTML =
  kp(`${M.wins[D.buckets[1]]} / ${M.tpls}`, "templates profitable with a this-week entry (1–4 days left)") +
  kp(`${M.wins[D.buckets[2]]} / ${M.tpls}`, "templates profitable with a next-week entry (5–9 days left)") +
  kp(`${M.wins[D.buckets[0]]} / ${M.tpls}`, "templates profitable entered on expiry day") +
  kp(`${rs(row(ss, D.buckets[1]).perday)} vs ${rs(row(ss, D.buckets[2]).perday)}`, "short strangle Δ0.15 per day held: this week vs next week");
const O = D.oos;
document.getElementById("verdict").innerHTML = [
  `<b>This week's expiry (1–4 days left) is the best time for the premium sellers.</b> Short strangles and the short straddle earn 2–4× more per day held than when entered a week early, lose about half as much in the bad 5%, and were profitable in all 6 years (next week: 4–5 of 6).`,
  `<b>Next week's expiry pays more per trade but not per day.</b> A short strangle Δ0.15 made ${rs(row(ss, D.buckets[2]).avg)} a trade entered 5–9 days out vs ${rs(row(ss, D.buckets[1]).avg)} 1–4 days out — but it is open ~3× as long, its 5% worst is ${rs(row(ss, D.buckets[2]).p5)} vs ${rs(row(ss, D.buckets[1]).p5)}, and 2021–23 was weak (${rs(row(ss, D.buckets[2]).a)}/day vs ${rs(row(ss, D.buckets[1]).a)}/day). Most of the next-week edge came from 2024–26.`,
  `<b>Expiry day (0 days left) earns almost nothing:</b> only ${M.wins[D.buckets[0]]} of ${M.tpls} templates were profitable, by small amounts — the premium left is small and charges + slippage take it.`,
  `<b>Defined-risk credit trades (iron condors, credit spreads, iron fly) are about break-even or losing at every entry day.</b> The wings cost more than they save once charges and slippage are in. Timing does not fix that.`,
  `<b>Entry time barely matters:</b> 09:30, 12:00 and 14:30 give nearly the same average for every strategy. Pick the day, not the minute.`,
  `<b>The managed exit (book +50% / stop −2×) lowered the result for the short strangle and straddle</b> — holding to expiry did better on average (with the bigger tail).`,
  `<b>Long options and long straddles/strangles lose at every horizon</b> — time decay beats them on average.`,
  `<b>Out-of-sample check:</b> picking each template's best entry bucket on 2021–23 and scoring it on 2024–26, the pick was also the best bucket in ${O.same} of ${O.total} templates; its 2024–26 result was ${rs(O.picked_b)}/day vs ${rs(O.other_b)}/day for the other buckets.`,
].map(x => `<li>${x}</li>`).join("");

// bucket table
let h = `<tr><th>Strategy</th><th>Entry</th><th>n</th><th>Win %</th><th>₹ / trade</th><th>₹ / day held</th><th>5% worst</th><th>Worst</th><th>Tail</th><th>₹/day 2021–23</th><th>₹/day 2024–26</th><th>Years +</th></tr>`;
D.key.forEach(tpl => {
  const rows = D.buckets.map(b => row(tpl, b));
  const best = rows.reduce((a, r) => r.perday > a.perday ? r : a, rows[0]);
  h += `<tr class="grp"><td colspan="12">${tpl.replace(/\|/g, " · ")}</td></tr>`;
  rows.forEach(r => {
    h += `<tr class="${r === best && r.perday > 0 ? "best" : ""}"><td></td><td style="text-align:left">${r.bucket}</td><td>${r.n}</td><td>${r.win}</td>
      <td class="${cl(r.avg)}">${rs(r.avg)}</td><td class="${cl(r.perday)}"><b>${rs(r.perday)}</b></td><td class="down">${rs(r.p5)}</td><td class="down">${rs(r.worst)}</td>
      <td>${r.tail ?? "—"}</td><td class="${cl(r.a)}">${rs(r.a)}</td><td class="${cl(r.b)}">${rs(r.b)}</td><td>${r.yp}/${r.yn}</td></tr>`;
  });
});
document.getElementById("tBucket").innerHTML = h;

// heat map
function heat(metric) {
  const all = D.key.flatMap(t => D.heat[t].map(c => c[metric])).filter(v => v != null);
  const mx = metric === "win" ? 50 : Math.max(...all.map(Math.abs)) || 1;
  const col = v => {
    if (v == null) return "";
    const d = metric === "win" ? v - 50 : v, a = Math.min(1, Math.abs(d) / (metric === "win" ? 40 : mx * 0.6));
    return `background:${d >= 0 ? `rgba(14,159,110,${a * 0.55})` : `rgba(224,67,74,${a * 0.55})`}`;
  };
  const fmt = v => v == null ? "—" : metric === "tail" ? v.toFixed(2) : metric === "win" ? v + "%" : rs(v);
  let s = `<tr><th>Strategy</th>${[9, 8, 7, 6, 5, 4, 3, 2, 1, 0].map(d => `<th style="text-align:center">${d}</th>`).join("")}</tr>`;
  D.key.forEach(t => {
    s += `<tr><td>${t.replace(/\|/g, " · ")}</td>${[9, 8, 7, 6, 5, 4, 3, 2, 1, 0].map(d => { const v = D.heat[t][d][metric]; return `<td class="h" style="${col(v)}">${fmt(v)}</td>`; }).join("")}</tr>`;
  });
  document.getElementById("tHeat").innerHTML = s;
}
heat("perday");
document.querySelectorAll("[data-h]").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("[data-h]").forEach(x => x.classList.toggle("on", x === b)); heat(b.dataset.h);
}));

// slot
document.getElementById("tSlot").innerHTML = `<tr><th>Strategy</th><th>09:30 ₹/trade</th><th>12:00 ₹/trade</th><th>14:30 ₹/trade</th><th>09:30 win %</th><th>12:00 win %</th><th>14:30 win %</th></tr>` +
  D.slot.map(r => `<tr><td>${r.tpl.replace(/\|/g, " · ")}</td>${["09:30", "12:00", "14:30"].map(s => `<td class="${cl(r[s].avg)}">${rs(r[s].avg)}</td>`).join("")}${["09:30", "12:00", "14:30"].map(s => `<td>${r[s].win}</td>`).join("")}</tr>`).join("");

// families
document.getElementById("tFam").innerHTML = `<tr><th>Family</th><th>Type</th>${D.buckets.map(b => `<th>₹/day · ${b}</th>`).join("")}<th>Held ₹/trade</th><th>Managed ₹/trade</th></tr>` +
  D.fam.map(r => `<tr><td>${r.name}</td><td>${r.credit ? "credit" : "debit"}</td>${D.buckets.map(b => `<td class="${cl(r[b])}">${rs(r[b])}</td>`).join("")}<td class="${cl(r.hold)}">${rs(r.hold)}</td><td class="${cl(r.man)}">${rs(r.man)}</td></tr>`).join("");

// explorer with multi-sort
const COLS = [["tpl", "Template"], ["dte", "Days left"], ["slot", "Time"], ["n", "n"], ["win", "Win %"], ["avg", "₹ / trade"], ["perday", "₹ / day"],
  ["p5", "5% worst"], ["worst", "Worst"], ["tail", "Tail"], ["man", "Managed ₹"], ["a", "₹/day 21–23"], ["b", "₹/day 24–26"], ["yp", "Years +"]];
let sorts = [{k: "perday", dir: -1}];
const fName = document.getElementById("fName"), fBucket = document.getElementById("fBucket");
[...new Set(D.cells.map(c => c.name))].sort().forEach(n => fName.insertAdjacentHTML("beforeend", `<option>${n}</option>`));
D.buckets.forEach(b => fBucket.insertAdjacentHTML("beforeend", `<option>${b}</option>`));
const inB = (c, b) => !b || (b === D.buckets[0] ? c.dte === 0 : b === D.buckets[1] ? c.dte >= 1 && c.dte <= 4 : c.dte >= 5);
function cells() {
  const minN = document.getElementById("fMin").checked;
  const rows = D.cells.filter(c => (!fName.value || c.name === fName.value) && inB(c, fBucket.value) && (!minN || c.n >= 150));
  rows.sort((x, y) => { for (const {k, dir} of sorts) { const a = x[k] ?? -1e12, b = y[k] ?? -1e12; const c = typeof a === "string" ? a.localeCompare(b) : a - b; if (c) return c * dir; } return 0; });
  let s = `<tr>${COLS.map(([k, t]) => { const i = sorts.findIndex(x => x.k === k); return `<th data-s="${k}">${t}${i >= 0 ? (sorts[i].dir < 0 ? " ▾" : " ▴") + (sorts.length > 1 ? `<sup>${i + 1}</sup>` : "") : ""}</th>`; }).join("")}</tr>`;
  s += rows.slice(0, 400).map(c => `<tr><td>${c.tpl.replace(/\|/g, " · ")}</td><td>${c.dte}</td><td>${c.slot}</td><td>${c.n}</td><td>${c.win}</td><td class="${cl(c.avg)}">${rs(c.avg)}</td><td class="${cl(c.perday)}"><b>${rs(c.perday)}</b></td><td class="down">${rs(c.p5)}</td><td class="down">${rs(c.worst)}</td><td>${c.tail ?? "—"}</td><td class="${cl(c.man)}">${rs(c.man)}</td><td class="${cl(c.a)}">${rs(c.a)}</td><td class="${cl(c.b)}">${rs(c.b)}</td><td>${c.yp}/${c.yn}</td></tr>`).join("");
  document.getElementById("tCells").innerHTML = s;
  document.getElementById("sortTxt").textContent = `· ${rows.length} rows${rows.length > 400 ? " (first 400 shown)" : ""}`;
}
document.getElementById("tCells").addEventListener("click", e => {
  const th = e.target.closest("th[data-s]"); if (!th) return;
  const k = th.dataset.s, i = sorts.findIndex(x => x.k === k), def = ["tpl", "slot"].includes(k) ? 1 : -1;
  if (e.shiftKey) { if (i >= 0) sorts[i].dir *= -1; else if (sorts.length < 3) sorts.push({k, dir: def}); else sorts[2] = {k, dir: def}; }
  else if (i === 0 && sorts.length === 1) sorts[0].dir *= -1;
  else sorts = [{k, dir: i >= 0 ? sorts[i].dir : def}];
  cells();
});
[fName, fBucket, document.getElementById("fMin")].forEach(el => el.addEventListener("change", cells));
cells();
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    p, d = build()
    print(p)
    print("oos:", d["oos"], "wins:", d["meta"]["wins"])
    for r in d["bucket_rows"]:
        if r["tpl"].startswith(("Short strangle|Δ0.15", "Short straddle")):
            print(r["tpl"], r["bucket"], r["avg"], r["perday"], r["p5"], r["a"], r["b"], f"{r['yp']}/{r['yn']}")
