"""nifty_range_stats.py -- how far does NIFTY actually move? Daily, calendar
week (Mon-Fri) and expiry cycle (first day after the previous weekly expiry
-> expiry close), from the 1-min NIFTY 50 history in data/hist1m/.

Output: results/nifty_range_stats/{daily.csv, weekly.csv, cycles.csv, report.html}
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import paths
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "nifty_range_stats")
QS = [0.25, 0.5, 0.75, 0.9, 0.95]
BANDS = [100, 150, 200, 250, 300, 400, 500, 600, 800]


def _spot():
    s = S._load_spot().copy()
    s["day"] = s.index.date
    s["hm"] = s.index.hour * 60 + s.index.minute
    return s[(s.hm >= 555) & (s.hm <= 930)]  # 09:15 - 15:30


def daily(s):
    g = s.groupby("day")
    d = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})
    p920 = s[s.hm == 560].groupby("day").open.first()  # the 09:20 bar's open = price at 09:20
    d["p920"] = p920
    d.index = pd.to_datetime(d.index)
    d["prev_close"] = d.close.shift(1)
    d["range"] = d.high - d.low
    d["range_pct"] = d.range / d.open * 100
    d["gap"] = d.open - d.prev_close
    d["oc"] = d.close - d.open
    d["from920_up"] = d.high - d.p920
    d["from920_dn"] = d.p920 - d.low
    d["weekday"] = d.index.strftime("%a")
    d["year"] = d.index.year.astype(str)
    return d


def _period(s, groups, label):
    rows = []
    for key, x in s.groupby(groups):
        if x.empty:
            continue
        days = sorted(set(x.day))
        first = x[x.day == days[0]]
        p920 = first[first.hm == 560].open
        ref = float(p920.iloc[0]) if len(p920) else float(first.open.iloc[0])
        after = x[(x.day > days[0]) | (x.hm >= 560)]
        hi, lo, cl = float(after.high.max()), float(after.low.min()), float(x.close.iloc[-1])
        rows.append({label: key if not isinstance(key, tuple) else key[0], "start": days[0].isoformat(), "end": days[-1].isoformat(),
                     "days": len(days), "open": float(first.open.iloc[0]), "ref920": ref, "high": float(x.high.max()),
                     "low": float(x.low.min()), "close": cl, "range": float(x.high.max() - x.low.min()),
                     "move": cl - ref, "up": hi - ref, "down": ref - lo})
    d = pd.DataFrame(rows)
    d["range_pct"] = d.range / d.open * 100
    d["abs_move"] = d.move.abs()
    d["year"] = d.start.str[:4]
    return d


def cycles(s):
    exps = sorted(e for e in S._manifest())  # every weekly expiry (incl. the one with no option data)
    edges = [pd.Timestamp(e).date() for e in exps]
    out = np.full(len(s), None, dtype=object)
    dv = s.day.values
    for prev, e in zip(edges[:-1], edges[1:]):
        m = (dv > prev) & (dv <= e)
        out[m] = e.isoformat()
    s = s.assign(expiry=out).dropna(subset=["expiry"])
    d = _period(s, "expiry", "expiry")
    d["expiry_wd"] = pd.to_datetime(d.expiry).dt.strftime("%a")
    return d


def weeks(s):
    wk = pd.to_datetime(s.day).dt.to_period("W-SUN").astype(str).values
    return _period(s.assign(week=wk), "week", "week")


def _q(x):
    return {f"{int(q * 100)}": round(float(x.quantile(q)), 1) for q in QS} | {"max": round(float(x.max()), 1),
                                                                                 "mean": round(float(x.mean()), 1)}


def _bands(d):
    """P(|close - 09:20 price| <= X) and P(never traded beyond +/-X) for each X."""
    return [{"x": b, "close_within": round(float((d.abs_move <= b).mean() * 100), 1),
             "never_beyond": round(float(((d.up <= b) & (d.down <= b)).mean() * 100), 1),
             "up_beyond": round(float((d.up > b).mean() * 100), 1),
             "down_beyond": round(float((d.down > b).mean() * 100), 1)} for b in BANDS]


def build():
    s = _spot()
    D, W, C = daily(s), weeks(s), cycles(s)
    D = D.dropna(subset=["prev_close"])
    os.makedirs(OUT, exist_ok=True)
    D.to_csv(os.path.join(OUT, "daily.csv"))
    W.to_csv(os.path.join(OUT, "weekly.csv"), index=False)
    C.to_csv(os.path.join(OUT, "cycles.csv"), index=False)

    # our Day-1 leader range width, for comparison (positional_range_research.py)
    lr = None
    f = os.path.join(paths.BASE_DIR, "results", "positional_range_research", "rows.csv")
    if os.path.isfile(f):
        r = pd.read_csv(f)
        r = r[r.day_no == 1][["expiry", "width", "upper", "lower"]]
        j = C.merge(r, on="expiry")
        lr = {"n": len(j), "width_med": round(float(j.width.median())), "range_med": round(float(j.range.median())),
              "range_gt_width": round(float((j.range > j.width).mean() * 100), 1),
              "half_width_med": round(float((j.width / 2).median())),
              "move_within_half": round(float((j.abs_move <= j.width / 2).mean() * 100), 1)}

    by_year = lambda d, col: {y: round(float(g[col].median()), 1) for y, g in d.groupby("year")}
    data = {
        "span": [s.day.min().isoformat(), s.day.max().isoformat()], "n": {"days": len(D), "weeks": len(W), "cycles": len(C)},
        "daily": {"range": _q(D.range), "range_pct": _q(D.range_pct), "gap": _q(D.gap.abs()), "oc": _q(D.oc.abs()),
                  "up920": _q(D.from920_up.dropna()), "dn920": _q(D.from920_dn.dropna()),
                  "year_range": by_year(D, "range"), "year_pct": by_year(D, "range_pct"),
                  "weekday": {w: {"range": round(float(g.range.median()), 1), "oc": round(float(g.oc.abs().median()), 1),
                                  "gap": round(float(g.gap.abs().median()), 1), "n": len(g)}
                              for w, g in D.groupby("weekday") if len(g) > 20},
                  "bands": [{"x": b, "range_le": round(float((D.range <= b).mean() * 100), 1),
                             "close_within": round(float(((D.close - D.p920).abs() <= b).mean() * 100), 1)} for b in BANDS[:7]],
                  "big": {"gap_gt_200": int((D.gap.abs() > 200).sum()), "range_gt_500": int((D.range > 500).sum())}},
        "cycle": {"range": _q(C.range), "range_pct": _q(C.range_pct), "move": _q(C.abs_move), "up": _q(C.up), "down": _q(C.down),
                  "year_range": by_year(C, "range"), "year_pct": by_year(C, "range_pct"), "year_move": by_year(C, "abs_move"),
                  "bands": _bands(C), "era": {w: {"n": len(g), "range": round(float(g.range.median()), 1), "move": round(float(g.abs_move.median()), 1),
                                                   "days": round(float(g.days.mean()), 1)} for w, g in C.groupby("expiry_wd") if len(g) > 3},
                  "up_close": round(float((C.move > 0).mean() * 100), 1)},
        "week": {"range": _q(W.range), "range_pct": _q(W.range_pct), "move": _q(W.abs_move), "year_range": by_year(W, "range"),
                 "bands": _bands(W)},
        "leader": lr,
        "worst_cycles": C.nlargest(8, "range")[["expiry", "start", "range", "move", "up", "down"]].round(0).to_dict("records"),
    }
    page = TEMPLATE.replace("/*DATA*/", json.dumps(data, default=str))
    p = os.path.join(OUT, "report.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(page)
    return p, data


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NIFTY Range Statistics</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--accent:#4f5fe0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}h3{font-size:15px;margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 14px}.kpi .k{font-size:11.5px;color:var(--muted);font-weight:600;text-transform:uppercase}
.kpi .v{font-size:21px;font-weight:800;font-variant-numeric:tabular-nums}.kpi .s{font-size:12px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.scroll{overflow-x:auto}.note{color:var(--muted);font-size:12.5px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.bar{display:inline-block;height:8px;background:var(--accent);border-radius:4px;vertical-align:middle;margin-left:6px;opacity:.7}
.tabs{display:flex;gap:6px;margin-bottom:14px}.tabs button{font:inherit;border:1px solid var(--border);background:var(--card);color:var(--text);border-radius:999px;padding:6px 14px;cursor:pointer;font-weight:600}
.tabs button.on{background:var(--text);color:var(--card)}ul{margin:0;padding-left:18px}li{margin:4px 0}
</style></head><body><div class="wrap">
<h1>How far does NIFTY move?</h1><p class="note" id="sub"></p>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>In short</h2><ul id="short"></ul></div>
<div class="tabs" id="tabs"><button data-t="cycle" class="on">Expiry cycle (Day 1 → expiry)</button><button data-t="week">Calendar week (Mon → Fri)</button><button data-t="daily">Daily</button></div>
<div id="body"></div>
</div><script>
const D = /*DATA*/;
const f0 = v => v == null ? "—" : Math.round(v).toLocaleString("en-IN");
const f1 = v => v == null ? "—" : Number(v).toFixed(1);
document.getElementById("sub").textContent = `NIFTY 50 spot, 1-min data · ${D.span[0]} → ${D.span[1]} · ${D.n.days} days · ${D.n.weeks} calendar weeks · ${D.n.cycles} expiry cycles`;
const k = (l, v, s) => `<div class="kpi"><div class="k">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
const C = D.cycle, Dy = D.daily, Wk = D.week, L = D.leader;
document.getElementById("kpis").innerHTML = [
  k("Expiry cycle high–low", f0(C.range["50"]) + " pts", `median · 75%: ${f0(C.range["75"])} · 90%: ${f0(C.range["90"])}`),
  k("Cycle: 09:20 Day 1 → expiry close", f0(C.move["50"]) + " pts", `median move · 75%: ${f0(C.move["75"])} · 90%: ${f0(C.move["90"])}`),
  k("Calendar week high–low", f0(Wk.range["50"]) + " pts", `median · 90%: ${f0(Wk.range["90"])}`),
  k("Daily high–low", f0(Dy.range["50"]) + " pts", `median · ${f1(Dy.range_pct["50"])}% of NIFTY · 90%: ${f0(Dy.range["90"])}`),
  k("Daily gap (open vs prev close)", f0(Dy.gap["50"]) + " pts", `median · 90%: ${f0(Dy.gap["90"])} · ${Dy.big.gap_gt_200} gaps > 200`),
  L ? k("Our Day-1 leader range", f0(L.width_med) + " pts", `median width · NIFTY's cycle high–low beat it in ${L.range_gt_width}% of weeks`) : ""].join("");
const short = [
  `An expiry cycle's high–low is typically <b>${f0(C.range["50"])} points</b> (${f1(C.range_pct["50"])}% of NIFTY); 1 cycle in 10 is wider than <b>${f0(C.range["90"])}</b>.`,
  `From Day 1 09:20 to the expiry close NIFTY ends a median <b>${f0(C.move["50"])} points</b> away — but during the cycle it typically travels <b>${f0(C.up["50"])} up</b> and <b>${f0(C.down["50"])} down</b> from the 09:20 price.`,
  `NIFTY closed the cycle within ±300 of its Day-1 09:20 price in <b>${C.bands.find(b => b.x === 300).close_within}%</b> of cycles, but stayed within ±300 the whole way in only <b>${C.bands.find(b => b.x === 300).never_beyond}%</b>.`,
  L ? `Our Day-1 leader range (median <b>${f0(L.width_med)}</b> wide, ±${f0(L.half_width_med)}) is narrower than NIFTY's median cycle high–low (${f0(L.range_med)}); the close ended within half the range width of the 09:20 price in ${L.move_within_half}% of cycles.` : "",
  `A normal day moves <b>${f0(Dy.range["50"])} points</b> high–low; from 09:20 the day typically extends ${f0(Dy.up920["50"])} up and ${f0(Dy.dn920["50"])} down.`];
document.getElementById("short").innerHTML = short.filter(Boolean).map(x => `<li>${x}</li>`).join("");

const qrow = (lab, q) => `<tr><td>${lab}</td>${["25", "50", "75", "90", "95", "max", "mean"].map(c => `<td>${f0(q[c])}</td>`).join("")}</tr>`;
const qhead = `<tr><th></th><th>25%</th><th>Median</th><th>75%</th><th>90%</th><th>95%</th><th>Max</th><th>Mean</th></tr>`;
const yrTable = (cols) => { const years = Object.keys(cols[0][1]);
  return `<tr><th>Year</th>${cols.map(c => `<th>${c[0]}</th>`).join("")}</tr>` + years.map(y => `<tr><td>${y}</td>${cols.map(c => `<td>${c[2] ? f1(c[1][y]) + "%" : f0(c[1][y])}</td>`).join("")}</tr>`).join(""); };
const bandTable = (b) => `<tr><th>±X from 09:20</th><th>Closed within ±X</th><th>Never went beyond ±X</th><th>Went above +X</th><th>Went below −X</th></tr>` +
  b.map(r => `<tr><td>${r.x}</td><td>${r.close_within}%<span class="bar" style="width:${r.close_within}px"></span></td><td>${r.never_beyond}%<span class="bar" style="width:${r.never_beyond}px"></span></td><td>${r.up_beyond}%</td><td>${r.down_beyond}%</td></tr>`).join("");
function render(t) {
  let h = "";
  if (t === "cycle") {
    h = `<div class="card"><h2>Expiry cycle — first day after the previous expiry → expiry close (points)</h2><div class="scroll"><table>${qhead}
      ${qrow("High – low", C.range)}${qrow("|Close − Day-1 09:20 price|", C.move)}${qrow("Max rise above 09:20 price", C.up)}${qrow("Max fall below 09:20 price", C.down)}</table></div>
      <p class="note">Range as % of NIFTY: median ${f1(C.range_pct["50"])}%, 90% ${f1(C.range_pct["90"])}%. NIFTY closed the cycle higher than its 09:20 price in ${C.up_close}% of cycles.</p></div>
      <div class="card"><h2>Probability table — Day-1 09:20 price ± X points</h2><div class="scroll"><table>${bandTable(C.bands)}</table></div>
      <p class="note">"Closed within" = at the expiry close. "Never went beyond" = never traded outside ±X from 09:20 on Day 1 until expiry — what an unadjusted short strangle at ±X needs.</p></div>
      <div class="grid2"><div class="card"><h2>Median by year</h2><div class="scroll"><table>${yrTable([["High–low", C.year_range], ["High–low %", C.year_pct, 1], ["|Move|", C.year_move]])}</table></div></div>
      <div class="card"><h2>Thursday vs Tuesday expiries</h2><div class="scroll"><table><tr><th>Expiry day</th><th>Cycles</th><th>Trading days</th><th>High–low</th><th>|Move|</th></tr>
      ${Object.entries(C.era).map(([w, v]) => `<tr><td>${w}</td><td>${v.n}</td><td>${v.days}</td><td>${f0(v.range)}</td><td>${f0(v.move)}</td></tr>`).join("")}</table></div>
      <p class="note">Mon/Wed rows = holiday-shifted expiries.</p></div></div>
      <div class="card"><h2>Widest cycles</h2><div class="scroll"><table><tr><th>Expiry</th><th>Day 1</th><th>High–low</th><th>Close − 09:20</th><th>Max up</th><th>Max down</th></tr>
      ${D.worst_cycles.map(r => `<tr><td>${r.expiry}</td><td>${r.start}</td><td>${f0(r.range)}</td><td>${f0(r.move)}</td><td>${f0(r.up)}</td><td>${f0(r.down)}</td></tr>`).join("")}</table></div></div>`;
  } else if (t === "week") {
    h = `<div class="card"><h2>Calendar week — Monday 09:15 → Friday close (points)</h2><div class="scroll"><table>${qhead}
      ${qrow("High – low", Wk.range)}${qrow("|Close − Monday 09:20 price|", Wk.move)}</table></div>
      <p class="note">Range as % of NIFTY: median ${f1(Wk.range_pct["50"])}%, 90% ${f1(Wk.range_pct["90"])}%.</p></div>
      <div class="card"><h2>Probability table — Monday 09:20 price ± X points</h2><div class="scroll"><table>${bandTable(Wk.bands)}</table></div></div>
      <div class="card"><h2>Median high–low by year</h2><div class="scroll"><table>${yrTable([["High–low", Wk.year_range]])}</table></div></div>`;
  } else {
    const wd = ["Mon", "Tue", "Wed", "Thu", "Fri"].filter(w => Dy.weekday[w]);
    h = `<div class="card"><h2>Daily (points)</h2><div class="scroll"><table>${qhead}
      ${qrow("High – low", Dy.range)}${qrow("Gap |open − previous close|", Dy.gap)}${qrow("|Close − open|", Dy.oc)}${qrow("Rise above 09:20 price", Dy.up920)}${qrow("Fall below 09:20 price", Dy.dn920)}</table></div>
      <p class="note">Range as % of NIFTY: median ${f1(Dy.range_pct["50"])}%, 90% ${f1(Dy.range_pct["90"])}%. Days with a gap over 200 pts: ${Dy.big.gap_gt_200}; days with a range over 500: ${Dy.big.range_gt_500}.</p></div>
      <div class="grid2"><div class="card"><h2>By weekday (median)</h2><div class="scroll"><table><tr><th>Day</th><th>Days</th><th>High–low</th><th>|Close − open|</th><th>|Gap|</th></tr>
      ${wd.map(w => `<tr><td>${w}</td><td>${Dy.weekday[w].n}</td><td>${f0(Dy.weekday[w].range)}</td><td>${f0(Dy.weekday[w].oc)}</td><td>${f0(Dy.weekday[w].gap)}</td></tr>`).join("")}</table></div></div>
      <div class="card"><h2>Median by year</h2><div class="scroll"><table>${yrTable([["High–low", Dy.year_range], ["High–low %", Dy.year_pct, 1]])}</table></div></div></div>
      <div class="card"><h2>How often a day stays small</h2><div class="scroll"><table><tr><th>X points</th><th>Day's high–low ≤ X</th><th>Close within ±X of 09:20</th></tr>
      ${Dy.bands.map(b => `<tr><td>${b.x}</td><td>${b.range_le}%</td><td>${b.close_within}%</td></tr>`).join("")}</table></div></div>`;
  }
  document.getElementById("body").innerHTML = h;
}
document.getElementById("tabs").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return;
  document.querySelectorAll("#tabs button").forEach(x => x.classList.toggle("on", x === b)); render(b.dataset.t); });
render("cycle");
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    p, d = build()
    print(p)
    print(json.dumps({k: d[k] for k in ("n", "leader")}, default=str))
    print("cycle", d["cycle"]["range"], d["cycle"]["move"], d["cycle"]["up"], d["cycle"]["down"])
    print("cycle bands", d["cycle"]["bands"])
    print("cycle era", d["cycle"]["era"], "years", d["cycle"]["year_range"], d["cycle"]["year_pct"])
    print("week", d["week"]["range"], d["week"]["move"])
    print("daily", d["daily"]["range"], d["daily"]["gap"], d["daily"]["oc"], d["daily"]["weekday"], d["daily"]["year_range"], d["daily"]["big"])
