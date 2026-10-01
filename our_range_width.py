"""our_range_width.py -- what size range does the volume-leader method give?

Sources (already computed):
  results/positional_range_research/rows.csv   every day's fresh 09:20 range
  results/intraday_range_bt/imp_all_orig/days_detail.json
                                               every intraday range recalculation
                                               (leader changes) after 09:20
Width = UPPER - LOWER = (CE leader - PE leader) + CE leader's first-5-min high
                                                + PE leader's first-5-min high.
Output: results/our_range_width/report.html
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import paths

OUT = os.path.join(paths.BASE_DIR, "results", "our_range_width")
BINS = [0, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800, 10000]
LABS = ["< 150", "150–200", "200–250", "250–300", "300–350", "350–400", "400–450", "450–500", "500–600", "600–800", "800+"]
DTE = {0: "Expiry day", 1: "1 day before", 2: "2 days before", 3: "3 days before", 4: "4 days before"}


def _dist(w):
    c = pd.cut(w, BINS, labels=LABS, right=False).value_counts().reindex(LABS).fillna(0).astype(int)
    n = len(w)
    cum = c.cumsum()
    return [{"bin": b, "n": int(c[b]), "pct": round(c[b] / n * 100, 1), "cum": round(cum[b] / n * 100, 1)} for b in LABS]


def _q(w):
    return {k: round(float(w.quantile(q))) for k, q in (("p10", .1), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9))} | \
           {"min": round(float(w.min())), "max": round(float(w.max())), "mean": round(float(w.mean())), "n": int(len(w))}


def build():
    r = pd.read_csv(os.path.join(paths.BASE_DIR, "results", "positional_range_research", "rows.csv"))
    r["year"] = r.day.str[:4]
    r["weekday"] = pd.to_datetime(r.day).dt.strftime("%a")
    r["pct"] = r.width / r.spot_920 * 100
    r["gap"] = r.ce_leader - r.pe_leader
    r["highs"] = r.width - r.gap
    r["ld"] = np.where(r.gap == 0, "Same strike (straddle)", np.where(r.gap > 0, "CE leader above PE leader", "CE leader below PE leader"))

    # later-in-the-day ranges (after a leader change), from the intraday replay
    later = []
    f = os.path.join(paths.BASE_DIR, "results", "intraday_range_bt", "imp_all_orig", "days_detail.json")
    if os.path.isfile(f):
        for d in json.load(open(f, encoding="utf-8")):
            for e in d["events"]:
                if e["kind"] == "LEADER":
                    later.append({"day": d["day"], "time": e["time"], "width": e["upper"] - e["lower"],
                                  "slot": "09:20–10:30" if e["time"] < "10:30" else "10:30–12:30" if e["time"] < "12:30" else "12:30–15:15"})
    L = pd.DataFrame(later)

    data = {
        "all": _q(r.width), "dist": _dist(r.width),
        "day1": _q(r[r.day_no == 1].width), "dist_day1": _dist(r[r.day_no == 1].width),
        "pct": _q(r.pct.round(2) * 100),  # basis points, rounded later in JS
        "dte": [{"k": DTE.get(k, k), **_q(g.width)} for k, g in r.groupby("dte") if len(g) > 20],
        "year": [{"k": y, **_q(g.width), "pct": round(float(g.pct.median()), 2)} for y, g in r.groupby("year")],
        "weekday": [{"k": w, **_q(g.width)} for w in ("Mon", "Tue", "Wed", "Thu", "Fri") for g in [r[r.weekday == w]] if len(g)],
        "leaders": [{"k": k, "share": round(len(g) / len(r) * 100, 1), **_q(g.width)} for k, g in r.groupby("ld")],
        "gap": [{"k": int(k), "n": int(len(g)), "share": round(len(g) / len(r) * 100, 1), "w": round(float(g.width.median()))}
                for k, g in r.groupby("gap") if len(g) >= 5],
        "highs": _q(r.highs),
        "later": [{"k": s, **_q(g.width)} for s, g in L.groupby("slot")] if len(L) else [],
        "later_all": _q(L.width) if len(L) else None,
        "span": [r.day.min(), r.day.max()], "expiries": int(r.expiry.nunique()),
        "narrow": r.nsmallest(5, "width")[["day", "width", "ce_leader", "pe_leader"]].round(0).to_dict("records"),
        "wide": r.nlargest(5, "width")[["day", "width", "ce_leader", "pe_leader"]].round(0).to_dict("records"),
    }
    data["pct"] = {k: (round(v / 100, 2) if k != "n" else v) for k, v in data["pct"].items()}
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "report.html")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(TEMPLATE.replace("/*DATA*/", json.dumps(data, default=str)))
    return p, data


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Our Range Width</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--accent:#4f5fe0;--soft:#eceefc}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--soft:#252a3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 14px}.kpi .k{font-size:11.5px;color:var(--muted);font-weight:600;text-transform:uppercase}
.kpi .v{font-size:21px;font-weight:800;font-variant-numeric:tabular-nums}.kpi .s{font-size:12px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.scroll{overflow-x:auto}.note{color:var(--muted);font-size:12.5px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.bar{display:inline-block;height:10px;background:var(--accent);border-radius:3px;vertical-align:middle;opacity:.75}
ul{margin:0;padding-left:18px}li{margin:4px 0}
</style></head><body><div class="wrap">
<h1>What size range does our method give?</h1><p class="note" id="sub"></p>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>In short</h2><ul id="short"></ul></div>
<div class="grid2">
 <div class="card"><h2>Width distribution — every day's 09:20 range</h2><div class="scroll"><table id="tDist"></table></div></div>
 <div class="card"><h2>Width distribution — Day-1 (entry) range only</h2><div class="scroll"><table id="tDist1"></table></div></div>
</div>
<div class="card"><h2>Percentiles (points)</h2><div class="scroll"><table id="tPct"></table></div>
<p class="note">10% = 1 range in 10 is narrower than this; 90% = 1 in 10 is wider.</p></div>
<div class="grid2">
 <div class="card"><h2>By year</h2><div class="scroll"><table id="tYear"></table></div></div>
 <div class="card"><h2>What the width is made of</h2><div class="scroll"><table id="tLd"></table></div>
 <p class="note">Width = (CE leader − PE leader) + CE leader's first-5-min high + PE leader's first-5-min high.</p>
 <div class="scroll" style="margin-top:10px"><table id="tGap"></table></div></div>
</div>
<div class="grid2">
 <div class="card"><h2>Narrowest 09:20 ranges</h2><div class="scroll"><table id="tNar"></table></div></div>
 <div class="card"><h2>Widest 09:20 ranges</h2><div class="scroll"><table id="tWide"></table></div></div>
</div>
</div><script>
const D = /*DATA*/;
const f0 = v => v == null ? "—" : Math.round(v).toLocaleString("en-IN");
document.getElementById("sub").textContent = `${D.expiries} weekly expiries · ${D.span[0]} → ${D.span[1]} · ${D.all.n} daily 09:20 ranges · ${D.later_all ? D.later_all.n + " later-in-the-day ranges (after a leader change)" : ""}`;
const k = (l, v, s) => `<div class="kpi"><div class="k">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
document.getElementById("kpis").innerHTML = [
  k("Typical 09:20 range", f0(D.all.p50) + " pts", `median · middle half ${f0(D.all.p25)}–${f0(D.all.p75)}`),
  k("As % of NIFTY", D.pct.p50 + "%", `median · middle half ${D.pct.p25}–${D.pct.p75}%`),
  k("Day-1 (entry) range", f0(D.day1.p50) + " pts", `median · 1 in 10 under ${f0(D.day1.p10)}, 1 in 10 over ${f0(D.day1.p90)}`),
  k("Expiry-day range", f0((D.dte.find(x => x.k === "Expiry day") || {}).p50) + " pts", "median — ranges shrink toward expiry"),
  D.later_all ? k("After a leader change", f0(D.later_all.p50) + " pts", "median width of ranges recalculated later in the day") : ""].join("");
const e = D.dte.find(x => x.k === "Expiry day"), f4 = D.dte.find(x => x.k === "4 days before"), top = [...D.dist].sort((a, b) => b.n - a.n).slice(0, 3);
const sameL = D.leaders.find(x => x.k.startsWith("Same"));
document.getElementById("short").innerHTML = [
  `The most common 09:20 range widths are <b>${top.map(t => t.bin).join(", ")}</b> points (together ${top.reduce((a, t) => a + t.pct, 0).toFixed(0)}% of days).`,
  `Half of all 09:20 ranges are between <b>${f0(D.all.p25)} and ${f0(D.all.p75)}</b> points wide (median ${f0(D.all.p50)}, about ${D.pct.p50}% of NIFTY).`,
  `The range <b>shrinks toward expiry</b>: median ${f4 ? f0(f4.p50) : "—"} four days before, ${e ? f0(e.p50) : "—"} on expiry day — it tracks the option premium, which decays.`,
  sameL ? `In <b>${sameL.share}%</b> of days both leaders are the <b>same strike</b>; then the width is just the two first-candle highs (median ${f0(sameL.p50)}).` : "",
  D.later_all ? `Ranges recalculated later in the day (after a leader change) have a median width of <b>${f0(D.later_all.p50)}</b>.` : ""].filter(Boolean).map(x => `<li>${x}</li>`).join("");
const dist = (id, rows) => document.getElementById(id).innerHTML = `<tr><th>Width (pts)</th><th>Days</th><th>Share</th><th></th><th>Narrower or equal</th></tr>` +
  rows.map(r => `<tr><td>${r.bin}</td><td>${r.n}</td><td>${r.pct}%</td><td style="text-align:left;width:40%"><span class="bar" style="width:${r.pct * 3}%"></span></td><td>${r.cum}%</td></tr>`).join("");
dist("tDist", D.dist); dist("tDist1", D.dist_day1);
const pr = (lab, q, extra = "") => `<tr><td>${lab}</td><td>${q.n}</td>${["p10", "p25", "p50", "p75", "p90", "min", "max"].map(c => `<td>${f0(q[c])}</td>`).join("")}${extra}</tr>`;
const ph = (extra = "") => `<tr><th></th><th>Count</th><th>10%</th><th>25%</th><th>Median</th><th>75%</th><th>90%</th><th>Min</th><th>Max</th>${extra}</tr>`;
document.getElementById("tPct").innerHTML = ph() + pr("Every day, 09:20", D.all) + D.dte.map(x => pr("09:20 · " + x.k, x)).join("") +
  D.weekday.map(x => pr("09:20 · " + x.k, x)).join("") + D.later.map(x => pr("After leader change · " + x.k, x)).join("");
document.getElementById("tYear").innerHTML = `<tr><th>Year</th><th>Days</th><th>25%</th><th>Median</th><th>75%</th><th>Median % of NIFTY</th></tr>` +
  D.year.map(x => `<tr><td>${x.k}</td><td>${x.n}</td><td>${f0(x.p25)}</td><td><b>${f0(x.p50)}</b></td><td>${f0(x.p75)}</td><td>${x.pct}%</td></tr>`).join("");
document.getElementById("tLd").innerHTML = `<tr><th>Leaders at 09:20</th><th>Share</th><th>Median width</th><th>25%</th><th>75%</th></tr>` +
  D.leaders.map(x => `<tr><td>${x.k}</td><td>${x.share}%</td><td><b>${f0(x.p50)}</b></td><td>${f0(x.p25)}</td><td>${f0(x.p75)}</td></tr>`).join("") +
  `<tr><td>Sum of the two first-candle highs</td><td></td><td><b>${f0(D.highs.p50)}</b></td><td>${f0(D.highs.p25)}</td><td>${f0(D.highs.p75)}</td></tr>`;
document.getElementById("tGap").innerHTML = `<tr><th>CE leader − PE leader</th><th>Days</th><th>Share</th><th>Median width</th></tr>` +
  D.gap.map(x => `<tr><td>${x.k > 0 ? "+" : ""}${x.k}</td><td>${x.n}</td><td>${x.share}%</td><td>${f0(x.w)}</td></tr>`).join("");
const lst = (id, rows) => document.getElementById(id).innerHTML = `<tr><th>Day</th><th>Width</th><th>CE leader</th><th>PE leader</th></tr>` +
  rows.map(r => `<tr><td>${r.day}</td><td>${f0(r.width)}</td><td>${r.ce_leader}</td><td>${r.pe_leader}</td></tr>`).join("");
lst("tNar", D.narrow); lst("tWide", D.wide);
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    p, d = build()
    print(p)
    for kk in ("all", "day1", "pct", "highs", "later_all"):
        print(kk, d[kk])
    print("dist", [(x["bin"], x["pct"]) for x in d["dist"]])
    print("dte", [(x["k"], x["p50"]) for x in d["dte"]])
    print("year", [(x["k"], x["p50"], x["pct"]) for x in d["year"]])
    print("leaders", d["leaders"])
    print("gap", d["gap"])
    print("later", [(x["k"], x["p50"], x["n"]) for x in d["later"]])
    print("weekday", [(x["k"], x["p50"]) for x in d["weekday"]])
