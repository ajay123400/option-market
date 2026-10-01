"""ashish_report.py -- HTML report for ashish_rules_bt.py (results/ashish_rules/results.pkl)."""
import json
import os
import pickle
import sys

import ashish_rules_bt as A

TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Volatility Seller Rules</title><style>
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
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px;font-size:12.5px}
button{font:inherit;font-size:12.5px;border:1px solid var(--border);background:var(--card);color:var(--text);border-radius:8px;padding:4px 10px;cursor:pointer}
button.on{background:#4f5fe0;color:#fff;border-color:#4f5fe0}
</style></head><body><div class="wrap">
<h1>A volatility seller's rules (tastytrade style) — tested on NIFTY weekly options</h1>
<p class="note" style="margin-bottom:16px">Rules from the Groww "Trading ki Baat" interview with Ashish Gupta, 5 years of 1-min NIFTY option data (Sep 2021 → Sep 2026), 1 lot (65), 0.5 pt slippage per order, all charges. Positions marked every 15 minutes. "Green" = the IV rule passes at 09:30.</p>
<div class="card"><h2>What the tests say</h2><ul id="verdict"></ul></div>
<div class="card"><h2>1 + 2 · Exits and the delta-roll adjustment</h2>
<div class="bar"><button data-s="A" class="on">Set A: current week, 1–4 days left</button><button data-s="B">Set B: next week's expiry, entered with 1 day left in the current one</button>
<span class="note">·</span><button data-g="1" class="on">Green days</button><button data-g="0">Normal days</button></div>
<div class="scroll"><table id="tEx"></table></div>
<p class="note">Target: straddle +20% / strangle +35% of the credit. Stop: loss of 2× the credit. Delta roll (16Δ strangle only): when the two short deltas differ by more than 0.25, the untested short is rolled to the strike that halves the gap (never past ATM), no extra lots. Best ₹/trade shaded.</p></div>
<div class="card"><h2>3 · Weekly tail hedge: buy one far-OTM put every week</h2><div class="scroll"><table id="tHg"></table></div>
<h2 style="margin-top:16px">Green-day straddle: its 10 worst weeks, with one 4Δ put per short lot</h2><div class="scroll"><table id="tWorst"></table></div><p class="note" id="hgNote"></p></div>
<div class="card"><h2>4 · Filters: IV rule vs IV percentile ≥ 70 vs IV ÷ realized vol</h2><div class="scroll"><table id="tFl"></table></div>
<p class="note">09:30 entries 1–4 days before expiry, held to expiry (template backtest). Thresholds use only past days (500-day rolling). HV = realized volatility of NIFTY's daily closes over the past 10 / 30 days.</p></div>
</div><script>
const D = /*DATA*/null;
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
const S = (set, strat, green, v) => D.summ.find(r => r.set === set && r.strat === strat && r.green === green && r.v === v);
const F = (tpl, rule) => D.filters.find(r => r.tpl === tpl && r.rule.startsWith(rule));
const st = "Short straddle ATM", sg = "Short strangle 16Δ";
const a = v => S("A", st, true, v), ag = v => S("A", sg, true, v), b = v => S("B", st, true, v), bg = v => S("B", sg, true, v);
const H4 = D.hedge["4"], T = D.tot;
document.getElementById("verdict").innerHTML = [
  `<b>Profit targets (20% / 35%) cut the result on NIFTY weeklies.</b> Green-day straddle 1–4 days out: ${rs(a("hold").avg)} held vs ${rs(a("target").avg)} with the 20% target (win rate up to ${a("target").win}% but half the money). Strangle: ${rs(ag("hold").avg)} vs ${rs(ag("target").avg)}. Exception: on the 8-day next-week strangle the 35% target kept ${rs(bg("target").avg)} of ${rs(bg("hold").avg)} and made the 5% worst trade ${rs(bg("target").p5)}.`,
  `<b>The 2× credit stop rarely helps.</b> On 1–4-day straddles it almost never triggered before expiry; on strangles it lowered the average (${rs(ag("hold").avg)} → ${rs(ag("stop2x").avg)}) without a better tail.`,
  `<b>The delta-roll adjustment hurt on 1–4-day weeklies</b> (${rs(ag("hold").avg)} → ${rs(ag("roll").avg)}, ~${ag("roll").rolls} rolls a trade, drawdown ${rs(ag("hold").dd)} → ${rs(ag("roll").dd)}): gamma near expiry flips the deltas back and forth. On the 8-day next-week strangle it did better (${rs(bg("roll").avg)} vs hold ${rs(bg("hold").avg)}, drawdown ${rs(bg("roll").dd)} vs ${rs(bg("hold").dd)}) — closer to the 30–45-day trades it was designed for — but the IV-normal exit was safer still (5% worst ${rs(bg("ivexit").p5)}, drawdown ${rs(bg("ivexit").dd)}).`,
  `<b>Best exit for the next-week trade remains "IV turns normal"</b>: straddle ${rs(b("ivexit").avg)} vs ${rs(b("hold").avg)} held, drawdown ${rs(b("ivexit").dd)} vs ${rs(b("hold").dd)}. On 1–4-day trades it changes little (IV seldom normalises that fast).`,
  `<b>The weekly far-OTM put hedge costs ~${H4.cost_pct_notional_yr}% of notional a year and paid in only ${H4.paid_weeks} of ${H4.weeks} weeks.</b> It did not help the green-day straddle's worst weeks (those were 3–4% moves, not crashes): total ${rs(T.unhedged)} → ${rs(T.hedged)}, worst week ${rs(T.worst_unhedged)} → ${rs(T.worst_hedged)}. It is insurance against a Covid-style crash, which these 5 years did not contain — a cost you pay for the tail you cannot test.`,
  `<b>IV percentile ≥ 70 beat the 60th-percentile rule per trade in both periods</b> (straddle ${rs(F("Short straddle|ATM", "IVP").avg)} vs ${rs(F("Short straddle|ATM", "IV rule (").avg)}; 2021–23 ${rs(F("Short straddle|ATM", "IVP").a)} vs ${rs(F("Short straddle|ATM", "IV rule (").a)}, 2024–26 ${rs(F("Short straddle|ATM", "IVP").b)} vs ${rs(F("Short straddle|ATM", "IV rule (").b)}) on fewer days (${F("Short straddle|ATM", "IVP").share}% vs ${F("Short straddle|ATM", "IV rule (").share}%).`,
  `<b>IV ÷ HV30 adds a safety layer, not a replacement.</b> Alone it is weaker than the IV rule (${rs(F("Short straddle|ATM", "IV/HV30 high").avg)}); on top of it the straddle's 5% worst improved (${rs(F("Short straddle|ATM", "IV rule AND IV/HV30").p5)} vs ${rs(F("Short straddle|ATM", "IV rule (").p5)}), but the gain came in 2024–26 only. IV ÷ HV10 added nothing.`,
].map(x => `<li>${x}</li>`).join("");
let setSel = "A", gSel = true;
function ex() {
  let h = `<tr><th>Exit / adjustment</th><th>Trades</th><th>Win %</th><th>₹ / trade</th><th>5% worst</th><th>Worst</th><th>Max DD</th><th>Held (cal. days)</th><th>Closed early</th><th>Rolls / trade</th><th>2021–23</th><th>2024–26</th></tr>`;
  [st, sg].forEach(s => {
    const rows = D.summ.filter(r => r.set === setSel && r.strat === s && r.green === gSel);
    if (!rows.length) return;
    const best = rows.reduce((x, r) => r.avg > x.avg ? r : x);
    h += `<tr class="grp"><td colspan="12">${s}</td></tr>`;
    rows.forEach(r => { h += `<tr class="${r === best ? "best" : ""}"><td>${D.label[r.v]}</td><td>${r.n}</td><td>${r.win}</td><td class="${cl(r.avg)}"><b>${rs(r.avg)}</b></td><td class="down">${rs(r.p5)}</td><td class="down">${rs(r.worst)}</td><td class="down">${rs(r.dd)}</td><td>${r.held}</td><td>${r.early}%</td><td>${r.rolls || "—"}</td><td class="${cl(r.a)}">${rs(r.a)}</td><td class="${cl(r.b)}">${rs(r.b)}</td></tr>`; });
  });
  document.getElementById("tEx").innerHTML = h;
}
document.querySelectorAll("[data-s]").forEach(x => x.addEventListener("click", () => { setSel = x.dataset.s; document.querySelectorAll("[data-s]").forEach(y => y.classList.toggle("on", y === x)); ex(); }));
document.querySelectorAll("[data-g]").forEach(x => x.addEventListener("click", () => { gSel = x.dataset.g === "1"; document.querySelectorAll("[data-g]").forEach(y => y.classList.toggle("on", y === x)); ex(); }));
ex();
const yrs = [...new Set(Object.values(D.hedge).flatMap(h => Object.keys(h.per_year)))].sort();
document.getElementById("tHg").innerHTML = `<tr><th>Put delta</th><th>Weeks</th><th>Avg cost / week</th><th>5-yr total</th><th>Cost % of notional / yr</th><th>Weeks it paid</th>${yrs.map(y => `<th>${y}</th>`).join("")}<th>Best weeks (NIFTY move, P&amp;L)</th></tr>` +
  Object.entries(D.hedge).map(([d, h]) => `<tr><td>${d}Δ</td><td>${h.weeks}</td><td>${rs(h.avg_cost)}</td><td class="${cl(h.total)}">${rs(h.total)}</td><td>${h.cost_pct_notional_yr}%</td><td>${h.paid_weeks}</td>${yrs.map(y => `<td class="${cl(h.per_year[y])}">${rs(h.per_year[y])}</td>`).join("")}<td style="text-align:left">${h.best.filter(x => x["d" + d + "_rs"] > 0).map(x => `${x.week} (${x.move}%, ${rs(x["d" + d + "_rs"])})`).join(", ")}</td></tr>`).join("");
document.getElementById("tWorst").innerHTML = `<tr><th>Expiry week</th><th>Short lots</th><th>NIFTY move</th><th>Straddle P&amp;L</th><th>Hedge P&amp;L (1 put)</th><th>With hedge</th></tr>` +
  D.worst.map(w => `<tr><td>${w.expiry}</td><td>${w.lots}</td><td>${w.move == null ? "—" : w.move + "%"}</td><td class="down">${rs(w.pnl)}</td><td class="${cl(w.d4_rs)}">${rs(w.d4_rs)}</td><td class="down">${rs(w.hedged)}</td></tr>`).join("");
document.getElementById("hgNote").textContent = `Whole period, green-day straddles: ${rs(T.unhedged)} without the hedge, ${rs(T.hedged)} with one 4Δ put per short lot each week. Worst week ${rs(T.worst_unhedged)} → ${rs(T.worst_hedged)}.`;
let fh = `<tr><th>Rule</th><th>Trades</th><th>Share of days</th><th>₹ / trade</th><th>5% worst</th><th>Total</th><th>2021–23</th><th>2024–26</th></tr>`;
[...new Set(D.filters.map(r => r.tpl))].forEach(t => {
  fh += `<tr class="grp"><td colspan="8">${t.replace(/\|/g, " · ")}</td></tr>`;
  D.filters.filter(r => r.tpl === t).forEach(r => { fh += `<tr><td>${r.rule}</td><td>${r.n}</td><td>${r.share}%</td><td class="${cl(r.avg)}"><b>${rs(r.avg)}</b></td><td class="down">${rs(r.p5)}</td><td class="${cl(r.total)}">${rs(r.total)}</td><td class="${cl(r.a)}">${rs(r.a)}</td><td class="${cl(r.b)}">${rs(r.b)}</td></tr>`; });
});
document.getElementById("tFl").innerHTML = fh;
</script></body></html>"""

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    R = pickle.load(open(os.path.join(A.OUT, "results.pkl"), "rb"))
    R["label"] = A.LABEL
    R["hedge"] = {str(k): v for k, v in R["hedge"].items()}
    p = os.path.join(A.OUT, "report.html")
    open(p, "w", encoding="utf-8").write(TEMPLATE.replace("/*DATA*/null", json.dumps(R, ensure_ascii=False, default=str)))
    print(p)
