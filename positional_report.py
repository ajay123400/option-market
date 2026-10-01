"""positional_report.py -- self-contained HTML report for positional_bt.py
(results/positional_bt/<variant>/ -> report.html)."""
import json
import os
import sys

import numpy as np
import pandas as pd

import paths

ROOT = os.path.join(paths.BASE_DIR, "results", "positional_bt")


def _grp(d, col):
    out = []
    for k, g in d.groupby(col, observed=True):
        out.append({"k": str(k), "n": int(len(g)), "net": round(float(g.net_rs.sum())), "pts": round(float(g.pnl_pts.sum()), 1),
                    "avg": round(float(g.pnl_pts.mean()), 1), "win": round(float((g.net_rs > 0).mean() * 100), 1)})
    return out


def build(variant="user_rules"):
    D = os.path.join(ROOT, variant)
    s = json.load(open(os.path.join(D, "summary.json"), encoding="utf-8"))
    weeks = json.load(open(os.path.join(D, "weeks_detail.json"), encoding="utf-8"))
    d = pd.DataFrame([{k: v for k, v in w.items() if k != "events"} for w in weeks])
    d["width"] = [next(e["upper"] - e["lower"] for e in w["events"] if e["kind"] == "RANGE") for w in weeks]
    d["year"] = d.expiry.str[:4]
    d["period"] = np.where(d.expiry < "2024-01-01", "2021–23 (in-sample)", "2024–26 (out-of-sample)")
    d["era"] = np.where(d.expiry >= "2025-09-01", "Tuesday expiries (Sep 2025 →)", "Thursday expiries (→ Aug 2025)")
    q = pd.qcut(d.width, 4)
    d["wq"] = [f"{iv.left:.0f} – {iv.right:.0f}" for iv in q]
    d["rollg"] = d.rolls.clip(upper=6).map(lambda r: "6+" if r >= 6 else str(r))
    eq = d.net_rs.cumsum()
    cfg = s.get("config", {})
    hedge = cfg.get("hedge_pts")
    risk = ((f"Hedge: buy a CE {hedge} pts above the sold call and a PE {hedge} pts below the sold put (moved with the leg on a roll). "
             if hedge else "No hedge. ")
            + (f"Stop: total P&L <= -{int(cfg['sl_frac'] * 100)}% of the entry premium. " if cfg.get("sl_frac") else "No stop loss. ")
            + (f"Target: net P&L after charges >= +{int(cfg['target_frac'] * 100)}% of the entry "
               + ("net credit." if hedge else "premium.") if cfg.get("target_frac") else "No target: held to expiry."))
    compare = []
    for v in sorted(os.listdir(ROOT)):
        f = os.path.join(ROOT, v, "summary.json")
        if not os.path.isfile(f):
            continue
        c = json.load(open(f, encoding="utf-8"))
        wk = pd.read_csv(os.path.join(ROOT, v, "weeks.csv"))
        ins, outs = wk[wk.expiry < "2024-01-01"], wk[wk.expiry >= "2024-01-01"]
        compare.append({"key": v, "label": c["config"].get("label", v), "weeks": c["weeks"], "win": c["win_pct"],
                        "pts": c["points"], "net": c["net_rs"], "charges": c["charges"], "dd": c["max_drawdown_rs"],
                        "worst": c["worst"]["net_rs"], "in": round(float(ins.net_rs.sum())), "out": round(float(outs.net_rs.sum())),
                        "prem": c.get("avg_premium")})
    compare.sort(key=lambda r: (r["key"] != variant, r["key"]))
    data = {
        "risk": risk, "title": cfg.get("label", variant), "variant": variant, "compare": compare,
        "s": s, "equity": [[e, round(c)] for e, c in zip(d.expiry, eq)],
        "dd": [[e, round(v)] for e, v in zip(d.expiry, eq - eq.cummax().clip(lower=0))],
        "year": _grp(d, "year"), "period": _grp(d, "period"), "exit": _grp(d, "exit"), "rolls": _grp(d, "rollg"),
        "width": sorted(_grp(d, "wq"), key=lambda r: float(r["k"].split(" – ")[0])), "era": _grp(d, "era"),
        "weeks": weeks,
    }
    page = TEMPLATE.replace("/*DATA*/", json.dumps(data, default=str).replace("</", "<\\/"))
    out = os.path.join(D, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Positional Range Backtest</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--accent:#4f5fe0;--soft:#eceefc}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--soft:#252a3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 12px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 14px}.kpi .k{font-size:11.5px;color:var(--muted);font-weight:600;text-transform:uppercase}
.kpi .v{font-size:20px;font-weight:800;font-variant-numeric:tabular-nums}
.rules{display:grid;grid-template-columns:110px 1fr;gap:6px 14px;font-size:13.5px}.rules b{color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}.scroll{overflow-x:auto}
svg text{fill:var(--muted);font-size:11px}.note{color:var(--muted);font-size:12.5px}
details{border-bottom:1px solid var(--border)}summary{cursor:pointer;padding:8px 4px;display:grid;grid-template-columns:120px 1fr 110px 90px 110px;gap:8px;font-variant-numeric:tabular-nums}
summary::-webkit-details-marker{display:none}.ev{font-size:12.5px;padding:4px 10px 12px 18px;color:var(--muted)}.ev div{padding:2px 0}.ev b{color:var(--text)}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}.tabs button,select{font:inherit;border:1px solid var(--border);background:var(--card);color:var(--text);border-radius:999px;padding:5px 12px;cursor:pointer}
.tabs button.on{background:var(--text);color:var(--card)}
@media(max-width:700px){summary{grid-template-columns:100px 1fr 90px}.hm{display:none}}
</style></head><body><div class="wrap">
<h1 id="ttl">Positional range method — 5-year backtest</h1><p class="note" id="sub"></p>
<div class="card"><h2>Rules tested</h2><div class="rules">
<b>Week</b><span>Day 1 = first trading day after the previous expiry. Held to expiry, settled at intrinsic vs NIFTY's expiry close — unless stop or target hits first.</span>
<b>Leaders</b><span>Highest-volume 100-pt CE and PE strikes on each day's own volume, checked every 5-min candle close from 09:20.</span>
<b>Range</b><span>UPPER = CE leader + PE leader's first 5-min high · LOWER = PE leader − CE leader's first 5-min high (that day's candle). Recalculated only when a leader changes.</span>
<b>Entry</b><span>Day 1, 09:20: sell the 50-pt strike nearest UPPER (CE) and nearest LOWER (PE), 1 lot each, no hedge.</span>
<b>Adjust</b><span>Only on a range change and only toward the market: call rolls down to the strike nearest the new UPPER; put rolls up to the strike nearest the new LOWER. Never away. Limit: down/up to a straddle and at most 100 points past it. Rolls allowed on expiry day.</span>
<b>Risk</b><span id="risk"></span>
<b>Fills</b><span>Checked on 5-min closes; fills at the next minute's open, 0.5 pt slippage per order, all charges (₹20/order, STT, exchange, SEBI, stamp, GST).</span>
</div></div>
<div class="card"><h2>All variants: same weeks, same prices</h2><div class="scroll"><table id="tCmp"></table></div>
<p class="note">In-sample = expiries 2021–23, out-of-sample = 2024–26. Premium = entry credit in points (net of the hedge cost for hedged variants).</p></div>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>Equity (net ₹, 1 lot) &amp; drawdown</h2><div id="eq"></div></div>
<div class="grid2">
 <div class="card"><h2>By year</h2><div class="scroll"><table id="tYear"></table></div></div>
 <div class="card"><h2>In-sample vs out-of-sample · expiry era</h2><div class="scroll"><table id="tPer"></table></div></div>
 <div class="card"><h2>How weeks ended</h2><div class="scroll"><table id="tExit"></table></div></div>
 <div class="card"><h2>By number of rolls</h2><div class="scroll"><table id="tRoll"></table></div></div>
</div>
<div class="card"><h2>By Day-1 range width (quartiles, points)</h2><div class="scroll"><table id="tWidth"></table></div></div>
<div class="card"><h2>Every week</h2><div class="tabs" id="tabs"><select id="fy"></select>
 <button data-f="all" class="on">All</button><button data-f="win">Wins</button><button data-f="loss">Losses</button>
 <button data-f="STOP LOSS">Stop loss</button><button data-f="TARGET">Target</button><button data-f="EXPIRY">Held to expiry</button></div>
 <div id="wl"></div></div>
</div><script>
const D = /*DATA*/, S = D.s;
const rs = v => (v < 0 ? "-" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cls = v => v > 0 ? "up" : v < 0 ? "down" : "";
document.getElementById("sub").textContent = `${S.weeks} weekly expiries · ${S.from} → ${S.to} · NIFTY options · 1-min data · 1 lot (lot size of that time)`;
document.getElementById("ttl").textContent = "Positional range method · " + D.title;
document.getElementById("risk").textContent = D.risk;
const k = (l, v, c = "") => `<div class="kpi"><div class="k">${l}</div><div class="v ${c}">${v}</div></div>`;
document.getElementById("kpis").innerHTML = [k("Net P&L", rs(S.net_rs), cls(S.net_rs)), k("Gross", rs(S.gross_rs), cls(S.gross_rs)),
  k("Charges", rs(S.charges)), k("Points / lot", S.points, cls(S.points)), k("Win rate", S.win_pct + "%"),
  k("Avg win", rs(S.avg_win), "up"), k("Avg loss", rs(S.avg_loss), "down"), k("Max drawdown", rs(S.max_drawdown_rs), "down"),
  k("Worst week", rs(S.worst.net_rs) + `<div class="note">${S.worst.expiry}</div>`, "down"), k("Avg rolls / week", S.avg_rolls)].join("");
(function () {
  const e = D.equity, dd = D.dd, W = 1200, H = 280, H2 = 90, pl = 72, pr = 10, pt = 10, n = e.length;
  const mx = Math.max(0, ...e.map(x => x[1])), mn = Math.min(0, ...e.map(x => x[1])), md = Math.min(-1, ...dd.map(x => x[1]));
  const x = i => pl + i / (n - 1) * (W - pl - pr), y = v => pt + (mx - v) / (mx - mn || 1) * (H - pt - 20), y2 = v => H + 10 + v / md * H2;
  const yrs = []; e.forEach((r, i) => { if (!i || r[0].slice(0, 4) !== e[i - 1][0].slice(0, 4)) yrs.push([i, r[0].slice(0, 4)]); });
  document.getElementById("eq").innerHTML = `<svg viewBox="0 0 ${W} ${H + H2 + 30}" width="100%" role="img" aria-label="Equity">
   ${[mx, 0, mn].map(t => `<line x1="${pl}" x2="${W - pr}" y1="${y(t)}" y2="${y(t)}" stroke="var(--border)"/><text x="${pl - 6}" y="${y(t) + 4}" text-anchor="end">${rs(t)}</text>`).join("")}
   ${yrs.map(([i, t]) => `<line x1="${x(i)}" x2="${x(i)}" y1="${pt}" y2="${H + H2 + 10}" stroke="var(--border)" stroke-dasharray="3,3"/><text x="${x(i) + 4}" y="${H + H2 + 24}">${t}</text>`).join("")}
   <path d="${e.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(r[1]).toFixed(1)}`).join("")}" fill="none" stroke="var(--accent)" stroke-width="1.8"/>
   <path d="M${x(0)},${H + 10}${dd.map((r, i) => `L${x(i).toFixed(1)},${y2(r[1]).toFixed(1)}`).join("")}L${x(n - 1)},${H + 10}Z" fill="rgba(224,67,74,.25)" stroke="var(--down)"/>
   <text x="${pl - 6}" y="${y2(md) + 4}" text-anchor="end">${rs(md)}</text></svg>`;
})();
const tbl = (id, head, rows) => document.getElementById(id).innerHTML = `<tr>${head.map(h => `<th>${h}</th>`).join("")}</tr>` + rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("");
const std = r => [r.k, r.n, r.win + "%", `<span class="${cls(r.pts)}">${r.pts}</span>`, r.avg, `<b class="${cls(r.net)}">${rs(r.net)}</b>`];
const H6 = ["", "Weeks", "Win %", "Points", "Avg pts", "Net ₹"];
tbl("tYear", H6, D.year.map(std)); tbl("tPer", H6, [...D.period, ...D.era].map(std)); tbl("tExit", H6, D.exit.map(std));
tbl("tRoll", ["Rolls", "Weeks", "Win %", "Points", "Avg pts", "Net ₹"], D.rolls.map(std)); tbl("tWidth", ["Width", "Weeks", "Win %", "Points", "Avg pts", "Net ₹"], D.width.map(std));
tbl("tCmp", ["Variant", "Weeks", "Win %", "Avg premium", "Points", "Charges", "2021–23", "2024–26", "Net ₹", "Max DD", "Worst week"],
  D.compare.map(c => [`${c.key === D.variant ? "&#9654; " : ""}${c.label}`, c.weeks, c.win + "%", c.prem, `<span class="${cls(c.pts)}">${c.pts}</span>`,
    rs(c.charges), `<span class="${cls(c.in)}">${rs(c.in)}</span>`, `<span class="${cls(c.out)}">${rs(c.out)}</span>`,
    `<b class="${cls(c.net)}">${rs(c.net)}</b>`, rs(c.dd), `<span class="down">${rs(c.worst)}</span>`]));
const fy = document.getElementById("fy"); let filt = "all";
fy.innerHTML = `<option value="">All years</option>` + D.year.map(r => `<option>${r.k}</option>`).join("");
function ev(e) {
  const t = `<b>${e.time}</b>`;
  if (e.kind === "RANGE") return `<div>${t} range ${Math.round(e.lower)} – ${Math.round(e.upper)} · leaders ${e.ce_leader} CE / ${e.pe_leader} PE${e.spot ? ` · NIFTY ${e.spot}` : ""}${e.rolls ? ` · <b>${e.rolls.join(" · ")}</b> (P&L ${e.total_pts} pts)` : ""}</div>`;
  if (e.kind === "ENTRY") return `<div>${t} <b>SELL ${e.ce} CE @${e.ce_price} + ${e.pe} PE @${e.pe_price}</b>${e.ceh ? ` · <b>BUY ${e.ceh} CE @${e.ceh_price} + ${e.peh} PE @${e.peh_price}</b>` : ""} · ${e.ceh ? "net credit" : "premium"} ${e.premium}${e.stop != null ? ` · stop ${e.stop}` : ""}${e.target != null ? ` · target ${e.target} pts` : " · no target"}</div>`;
  if (e.kind === "EXPIRY") return `<div>${t} <b>settled at expiry</b> vs NIFTY ${e.settle}</div>`;
  return `<div>${t} <b class="${e.kind === "TARGET" ? "up" : "down"}">${e.kind}</b>${e.width ? ` (range ${Math.round(e.width)} wide)` : ""} · ${e.total_pts} pts · net ${rs(e.net_rs)}</div>`;
}
function render() {
  const y = fy.value;
  const w = D.weeks.filter(r => (!y || r.expiry.startsWith(y)) && (filt === "all" || (filt === "win" && r.net_rs > 0) || (filt === "loss" && r.net_rs <= 0) || r.exit === filt));
  document.getElementById("wl").innerHTML = w.slice().reverse().map(r => `<details><summary><span>${r.expiry}</span>
    <span class="hm">premium ${r.premium0} · ${r.rolls} roll${r.rolls === 1 ? "" : "s"} · ${r.range_changes} range changes</span><span>${r.exit}</span>
    <span class="${cls(r.pnl_pts)}">${r.pnl_pts} pts</span><b class="${cls(r.net_rs)}">${rs(r.net_rs)}</b></summary>
    <div class="ev">${r.events.map(ev).join("")}<div>lot ${r.lot} · gross ${rs(r.pnl_rs)} · charges ${rs(r.charges)}</div></div></details>`).join("") || `<p class="note">No weeks.</p>`;
}
document.getElementById("tabs").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return;
  document.querySelectorAll("#tabs button").forEach(x => x.classList.toggle("on", x === b)); filt = b.dataset.f; render(); });
fy.addEventListener("change", render); render();
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    for v in (sys.argv[1:] or ["user_rules"]):
        print(build(v))
