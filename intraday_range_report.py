"""intraday_range_report.py -- turns results/intraday_range_bt/ into one
self-contained HTML report (no external files; opens offline in any browser).

Usage:  python intraday_range_report.py [variant]   -> results/intraday_range_bt/<variant>/report.html
"""
import json
import os
import sys

import pandas as pd

import paths

ROOT = os.path.join(paths.BASE_DIR, "results", "intraday_range_bt")


def _rules(c):
    entry = ("Sell the volume-leader strikes themselves: the highest-volume 100-pt CE strike and PE strike at 09:20 "
             "(e.g. leaders 23200 CE / 23200 PE -> sell 23200 CE + 23200 PE)." if c["entry"] == "leader" else
             "Sell the 50-pt strike nearest UPPER (CE) and nearest LOWER (PE). UPPER = CE leader + PE leader's first-candle high, "
             "LOWER = PE leader - CE leader's first-candle high.")
    roll = ("Leaders re-checked every minute; range up -> book PE, sell PE nearest new LOWER; range down -> book CE, sell CE nearest new UPPER."
            if c.get("roll") == "range" else "None - the entry strikes are held.")
    sl = int(round(c["sl_frac"] * 100))
    re = (f"{c['reentries']} - after a stop, sell the leader strikes of that minute again with a fresh {sl}% stop (not after {c['last_entry']})."
          if c["reentries"] else "None - done for the day after a stop.")
    return [["Leaders (09:20)", "Highest-volume 100-pt CE and PE strikes after the first 5-min candle (09:15-09:20), current-week expiry, expiry day included."],
            ["Entry", "09:20 - " + entry + " 1 lot each."],
            ["Adjustment", roll],
            ["Stop loss", f"Combined: exit both legs when the loss reaches {sl}% of the premium collected (e.g. collected 200 -> exit when worth {200 + 2 * sl})."],
            ["Re-entry", re],
            ["Exit", f"{c['exit_time']}. Fills at the next minute's open with {c['slippage']} pt slippage per order. Charges: Rs 20/order, STT, exchange, SEBI, stamp, GST."]]


def build(variant):
    DIR = os.path.join(ROOT, variant)
    summ = json.load(open(os.path.join(DIR, "summary.json"), encoding="utf-8"))
    days = json.load(open(os.path.join(DIR, "days_detail.json"), encoding="utf-8"))
    skipped = json.load(open(os.path.join(DIR, "skipped.json"), encoding="utf-8"))
    trades = pd.read_csv(os.path.join(DIR, "trades.csv"))
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "events"} for r in days])
    d["month"] = d.day.str[:7]
    d["year"] = d.day.str[:4]
    d["cum"] = d.net_rs.cumsum()
    d["peak"] = d.cum.cummax().clip(lower=0)
    d["dd"] = d.cum - d.peak

    # monthly grid
    m = d.groupby("month").net_rs.sum()
    years = sorted(d.year.unique())
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    grid = {y: {i: m.get(f"{y}-{i:02d}") for i in range(1, 13)} for y in years}
    ymax = max(1.0, float(m.abs().max()))

    # exit reasons / legs
    exits = d.exit.value_counts().to_dict()
    by_leg = trades.groupby("type").pnl_pts.agg(["count", "sum", "mean"]).round(2).to_dict("index")

    # points per lot by year (lot sizes changed over the years -- points compare fairly)
    pts_year = d.groupby("year").pnl_pts.sum().round(1).to_dict()

    compare = []
    for v in sorted(os.listdir(ROOT)):
        f = os.path.join(ROOT, v, "summary.json")
        if os.path.isfile(f):
            c = json.load(open(f, encoding="utf-8"))
            compare.append({"key": v, "label": c["config"].get("label", v), "net_rs": c["net_rs"], "gross_rs": c["gross_rs"],
                            "charges": c["charges"], "points": c["points"], "win_pct": c["win_pct"], "days": c["days"],
                            "max_dd": c["max_drawdown_rs"], "avg_premium": c.get("avg_premium"),
                            "stops": c["stop_loss_days"], "exp": c["by_expiry_day"].get("expiry day", {}).get("net_rs")})
    compare.sort(key=lambda x: x["key"] != variant)
    data = {
        "title": summ["config"].get("label", variant), "rules": _rules(summ["config"]), "compare": compare,
        "variant": variant,
        "summary": summ, "exits": exits, "by_leg": by_leg, "pts_year": pts_year,
        "equity": [{"d": r.day, "c": round(r.cum, 0), "dd": round(r.dd, 0)} for r in d.itertuples()],
        "grid": grid, "ymax": ymax, "years": years, "months": months,
        "days": days, "skipped": skipped,
    }
    page = TEMPLATE.replace("/*DATA*/", json.dumps(data, default=str).replace("</", "<\\/"))
    out = os.path.join(DIR, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intraday Backtest Report</title>
<style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--accent:#4f5fe0;--soft:#eceefc}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--soft:#252a3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 12px}
.sub{color:var(--muted);margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.kpi .k{font-size:11.5px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.kpi .v{font-size:20px;font-weight:800;margin-top:2px;font-variant-numeric:tabular-nums}
.up{color:var(--up)}.down{color:var(--down)}
.rules{display:grid;grid-template-columns:120px 1fr;gap:6px 14px;font-size:13.5px}.rules b{color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:7px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:700;font-size:11.5px;text-transform:uppercase}
.scroll{overflow-x:auto}
.heat td{text-align:center;font-size:12px;border-radius:4px}
svg text{fill:var(--muted);font-size:11px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.tabs button,select,input{font:inherit;border:1px solid var(--border);background:var(--card);color:var(--text);border-radius:999px;padding:5px 12px;cursor:pointer}
.tabs button.on{background:var(--text);color:var(--card)}
details{border-bottom:1px solid var(--border)}summary{cursor:pointer;padding:8px 4px;display:grid;grid-template-columns:110px 50px 1fr 90px 90px 110px;gap:8px;align-items:center;font-variant-numeric:tabular-nums}
summary::-webkit-details-marker{display:none}
.ev{font-size:12.5px;padding:4px 10px 12px 18px;color:var(--muted)}.ev div{padding:2px 0}.ev b{color:var(--text)}
.pill{display:inline-block;border-radius:999px;padding:1px 8px;font-size:11.5px;font-weight:700;background:var(--soft)}
.note{font-size:12.5px;color:var(--muted)}
@media(max-width:700px){summary{grid-template-columns:90px 1fr 80px;}.hm{display:none}}
</style></head><body><div class="wrap">
<h1 id="ttl"></h1>
<div class="sub" id="sub"></div>

<div class="card"><h2>Rules tested</h2><div class="rules" id="rules"></div></div>
<div class="card"><h2>Comparison: same days, same prices</h2><div class="scroll"><table id="tCmp"></table></div>
<p class="note">Points = per-lot points over all days (fair across years: the NIFTY lot was 50, 25, 75, then 65). Net is after slippage and all charges, 1 lot.</p></div>

<div class="kpis" id="kpis"></div>

<div class="card" style="margin-top:16px"><h2>Equity curve (net ₹, 1 lot) &amp; drawdown</h2><div id="eq"></div></div>

<div class="grid2">
  <div class="card"><h2>By year</h2><div class="scroll"><table id="tYear"></table></div><p class="note">Points = per-lot points (fair across years — the NIFTY lot was 50 → 25 → 75 → 65).</p></div>
  <div class="card"><h2>By weekday · expiry day</h2><div class="scroll"><table id="tWd"></table></div></div>
</div>
<div class="card"><h2>Monthly net ₹</h2><div class="scroll"><table class="heat" id="heat"></table></div></div>
<div class="grid2">
  <div class="card"><h2>Exits &amp; legs</h2><div class="scroll"><table id="tExit"></table></div></div>
  <div class="card"><h2>Best / worst days</h2><div class="scroll"><table id="tBW"></table></div></div>
</div>

<div class="card"><h2>Every day</h2>
  <div class="tabs" id="dayTabs">
    <select id="fYear"></select>
    <button data-f="all" class="on">All</button><button data-f="win">Wins</button><button data-f="loss">Losses</button>
    <button data-f="sl">Stop-loss days</button><button data-f="re">Re-entry days</button><button data-f="exp">Expiry days</button>
  </div>
  <div id="dayList"></div><div class="note" id="dayMore"></div>
</div>
<p class="note" id="skip"></p>
</div>
<script>
const D = /*DATA*/;
const S = D.summary;
const esc = t => String(t).replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
document.getElementById("ttl").textContent = D.title;
document.getElementById("rules").innerHTML = D.rules.map(([a, b]) => `<b>${esc(a)}</b><span>${esc(b)}</span>`).join("");
const rs = v => (v < 0 ? "-" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cls = v => v > 0 ? "up" : v < 0 ? "down" : "";
document.getElementById("sub").textContent = `${S.from} → ${S.to} · ${S.days} trading days · NIFTY weekly options · 1-min data`;
const k = (l, v, c = "") => `<div class="kpi"><div class="k">${l}</div><div class="v ${c}">${v}</div></div>`;
document.getElementById("kpis").innerHTML = [
  k("Net P&L", rs(S.net_rs), cls(S.net_rs)), k("Gross P&L", rs(S.gross_rs), cls(S.gross_rs)), k("Charges", rs(S.charges)),
  k("Points / lot", S.points.toLocaleString("en-IN"), cls(S.points)), k("Win rate", S.win_pct + "%"),
  k("Avg win", rs(S.avg_win), "up"), k("Avg loss", rs(S.avg_loss), "down"),
  k("Max drawdown", rs(S.max_drawdown_rs), "down"), k("Stop-loss days", S.stop_loss_days + " / " + S.days),
  k("Avg premium", S.avg_premium), S.reentry_days ? k("Re-entry days", S.reentry_days) : k("Avg rolls / day", S.avg_rolls)].join("");

// equity + drawdown svg
(function () {
  const e = D.equity, W = 1200, H = 300, H2 = 90, pl = 70, pr = 10, pt = 10;
  const n = e.length, maxC = Math.max(0, ...e.map(x => x.c)), minC = Math.min(0, ...e.map(x => x.c)), minD = Math.min(...e.map(x => x.dd), -1);
  const x = i => pl + i / (n - 1) * (W - pl - pr), y = v => pt + (maxC - v) / (maxC - minC || 1) * (H - pt - 20);
  const y2 = v => H + 10 + (v / minD) * H2;
  let p = e.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(r.c).toFixed(1)}`).join("");
  let dd = `M${x(0)},${H + 10}` + e.map((r, i) => `L${x(i).toFixed(1)},${y2(r.dd).toFixed(1)}`).join("") + `L${x(n - 1)},${H + 10}Z`;
  const ticks = [maxC, (maxC + minC) / 2, minC, 0];
  const yrs = []; e.forEach((r, i) => { if (i === 0 || r.d.slice(0, 4) !== e[i - 1].d.slice(0, 4)) yrs.push([i, r.d.slice(0, 4)]); });
  document.getElementById("eq").innerHTML = `<svg viewBox="0 0 ${W} ${H + H2 + 30}" width="100%" role="img" aria-label="Equity curve">
    ${ticks.map(t => `<line x1="${pl}" x2="${W - pr}" y1="${y(t)}" y2="${y(t)}" stroke="var(--border)"/><text x="${pl - 6}" y="${y(t) + 4}" text-anchor="end">${rs(t)}</text>`).join("")}
    ${yrs.map(([i, t]) => `<line x1="${x(i)}" x2="${x(i)}" y1="${pt}" y2="${H + H2 + 10}" stroke="var(--border)" stroke-dasharray="3,3"/><text x="${x(i) + 4}" y="${H + H2 + 24}">${t}</text>`).join("")}
    <path d="${p}" fill="none" stroke="var(--accent)" stroke-width="1.8"/>
    <path d="${dd}" fill="rgba(224,67,74,.25)" stroke="var(--down)" stroke-width="1"/>
    <text x="${pl - 6}" y="${y2(minD) + 4}" text-anchor="end">${rs(minD)}</text></svg>`;
})();

function tbl(id, head, rows) {
  document.getElementById(id).innerHTML = `<tr>${head.map(h => `<th>${h}</th>`).join("")}</tr>` + rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("");
}
tbl("tYear", ["Year", "Days", "Win %", "Points/lot", "Net ₹"], Object.entries(S.by_year).map(([y, v]) =>
  [y, v.days, v.win_pct + "%", `<span class="${cls(v.pts)}">${v.pts}</span>`, `<b class="${cls(v.net_rs)}">${rs(v.net_rs)}</b>`]));
const wdOrder = ["Mon", "Tue", "Wed", "Thu", "Fri"];
tbl("tWd", ["Day", "Days", "Win %", "Points/lot", "Net ₹"], [
  ...wdOrder.filter(w => S.by_weekday[w]).map(w => [w, S.by_weekday[w].days, S.by_weekday[w].win_pct + "%", S.by_weekday[w].pts, `<b class="${cls(S.by_weekday[w].net_rs)}">${rs(S.by_weekday[w].net_rs)}</b>`]),
  ...Object.entries(S.by_expiry_day).map(([w, v]) => [`<b>${w}</b>`, v.days, v.win_pct + "%", v.pts, `<b class="${cls(v.net_rs)}">${rs(v.net_rs)}</b>`])]);
// heatmap
(function () {
  let h = `<tr><th>Year</th>${D.months.map(m => `<th>${m}</th>`).join("")}<th>Total</th></tr>`;
  D.years.forEach(y => {
    let tot = 0;
    h += `<tr><td><b>${y}</b></td>` + D.months.map((m, i) => {
      const v = D.grid[y][i + 1]; if (v == null) return "<td></td>"; tot += v;
      const a = Math.min(1, Math.abs(v) / D.ymax) * .55 + .08;
      return `<td style="background:${v >= 0 ? `rgba(14,159,110,${a})` : `rgba(224,67,74,${a})`}">${Math.round(v / 1000)}k</td>`;
    }).join("") + `<td><b class="${cls(tot)}">${rs(tot)}</b></td></tr>`;
  });
  document.getElementById("heat").innerHTML = h;
})();
tbl("tExit", ["", "Count", "Points total", "Points avg"], [
  ...Object.entries(D.exits).map(([e, c]) => [e, c, "", ""]),
  ...Object.entries(D.by_leg).map(([t, v]) => [`${t} legs (incl. rolls)`, v.count, v.sum, v.mean])]);
const sorted = [...D.days].sort((a, b) => b.net_rs - a.net_rs);
tbl("tBW", ["Day", "Exit", "Rolls", "Net ₹"], [...sorted.slice(0, 5), ...sorted.slice(-5)].map(r =>
  [`${r.day} ${r.weekday}`, r.exit, r.rolls, `<b class="${cls(r.net_rs)}">${rs(r.net_rs)}</b>`]));

// day list
const fy = document.getElementById("fYear");
fy.innerHTML = `<option value="">All years</option>` + D.years.map(y => `<option>${y}</option>`).join("");
let filt = "all", shown = 150;
function evLine(e) {
  const r = `range ${Math.round(e.lower)} – ${Math.round(e.upper)}`;
  if (e.kind === "ENTRY" || e.kind === "RE-ENTRY") return `<div><b>${e.time} ${e.kind}</b> · leaders CE ${e.ce_leader} / PE ${e.pe_leader}${e.upper ? ` · ${r}` : ""} · sold <b>${e.ce} CE</b> + <b>${e.pe} PE</b> · premium ${e.premium}</div>`;
  if (e.kind === "LEADER") return `<div><b>${e.time} LEADER</b> ${e.prev_leaders.join("/")} → ${e.ce_leader}/${e.pe_leader} · ${r} · shift <span class="${cls(e.shift)}">${e.shift > 0 ? "+" : ""}${e.shift}</span>${e.roll ? ` · <b>roll ${e.roll}</b>` : " · no roll"}</div>`;
  if (e.kind === "STOP") return `<div><b class="down">${e.time} STOP LOSS</b> at ${e.pnl_pts} pts</div>`;
  return "";
}
tbl("tCmp", ["Variant", "Days", "Win %", "Avg premium", "Points/lot", "Gross", "Charges", "Net", "Max DD", "Stop days", "Expiry-day net"],
  D.compare.map(c => [`${c.key === D.variant ? "&#9654; " : ""}${esc(c.label)}`, c.days, c.win_pct + "%", c.avg_premium ?? "", c.points,
    rs(c.gross_rs), rs(c.charges), `<b class="${cls(c.net_rs)}">${rs(c.net_rs)}</b>`, rs(c.max_dd), c.stops, c.exp == null ? "" : rs(c.exp)]));
function renderDays() {
  const y = fy.value;
  const rows = D.days.filter(r => (!y || r.day.startsWith(y)) && (filt === "all" || (filt === "win" && r.net_rs > 0) ||
    (filt === "loss" && r.net_rs <= 0) || (filt === "sl" && r.stops > 0) || (filt === "re" && r.trades > 1) || (filt === "exp" && r.expiry_day)));
  document.getElementById("dayList").innerHTML = rows.slice(-shown).reverse().map(r => `<details><summary>
      <span>${r.day}</span><span>${r.weekday}${r.expiry_day ? " <span class=pill>exp</span>" : ""}</span>
      <span class="hm">${r.range_920 ? `range ${Math.round(r.range_920[0])} – ${Math.round(r.range_920[1])} · ` : ""}prem ${r.premium0} · ${r.trades > 1 ? "re-entered" : r.rolls + " roll"}</span>
      <span class="hm">${r.exit}</span><span class="${cls(r.pnl_pts)}">${r.pnl_pts} pts</span><b class="${cls(r.net_rs)}">${rs(r.net_rs)}</b></summary>
      <div class="ev">${r.events.map(evLine).join("")}<div>lot ${r.lot} · gross ${rs(r.pnl_rs)} · charges ${rs(r.charges)} · expiry ${r.expiry}</div></div></details>`).join("");
  document.getElementById("dayMore").innerHTML = rows.length > shown ? `Showing latest ${shown} of ${rows.length} — <a href="#" id="more">show all</a>` : `${rows.length} days`;
  const m = document.getElementById("more"); if (m) m.onclick = ev => { ev.preventDefault(); shown = 1e9; renderDays(); };
}
document.getElementById("dayTabs").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  document.querySelectorAll("#dayTabs button").forEach(x => x.classList.toggle("on", x === b)); filt = b.dataset.f; renderDays();
});
fy.addEventListener("change", renderDays);
renderDays();
document.getElementById("skip").textContent = `${D.skipped.length} days skipped (no data at 09:15 / no leader trade / missing strike price).`;
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print(build(sys.argv[1] if len(sys.argv) > 1 else "leader_sl30_re1"))
