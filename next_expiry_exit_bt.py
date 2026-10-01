"""next_expiry_exit_bt.py -- on a green day (IV rule passes at 09:30) when
the current weekly expiry has only 1 trading day left, sell the NEXT weekly
expiry instead, and test exits based on IV going back to normal.

Entry: 09:30, next expiry, short straddle (ATM) / short strangle (0.20 and
0.15 delta, strikes picked from that moment's prices like the Ideas table).
"IV normal" = the rule's own reading (results/iv_rule/history.parquet: ATM IV
of the nearest expiry with 1-4 days left vs the 60th-percentile threshold of
that time of day). The current expiry's expiry day has no reading (no expiry
with 1-4 days left), so no IV check happens that day.

Exits compared:
  hold        held to expiry, settled at NIFTY's expiry close
  iv0930      buy back at the first 09:30 check where IV is normal
  ivany       buy back at the first check (09:30 / 12:00 / 14:30) where IV is normal
  tp50        buy back when the position is up 50% of the premium (1-min closes)
  iv0930+tp50 whichever of the two comes first

1-min closes, 0.5 pt slippage per order, all charges, 1 lot (65).
Output: results/next_expiry_exit/report.html
"""
import json
import math
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

import charges
import manual_trades
import paths
import sell_rules as SR
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "next_expiry_exit")
LOT = manual_trades.LOT_SIZE
SLIP = 0.5
STRATS = {"Short straddle ATM": None, "Short strangle Δ0.20": 0.20, "Short strangle Δ0.15": 0.15}
EXITS = ["hold", "iv0930", "ivany", "tp50", "iv0930+tp50"]
LABEL = {"hold": "Hold to expiry", "iv0930": "Exit when IV normal at 09:30", "ivany": "Exit when IV normal at any check (09:30/12:00/14:30)",
         "tp50": "Exit at +50% of premium", "iv0930+tp50": "IV normal at 09:30 OR +50%, whichever first"}


def _ts(day, hm):
    return int(pd.Timestamp(f"{day} {hm}", tz=S.IST).timestamp())


def pick_legs(q, spot, T, target):
    """q {(k, 'CE'|'PE'): price} -> [(type, strike)] for the strategy."""
    fwd, _, _ = SR.measure(q, spot, T)
    if not fwd:
        return None
    atm = int(round(fwd / 50) * 50)
    if target is None:
        return [("CE", atm), ("PE", atm)] if (atm, "CE") in q and (atm, "PE") in q else None
    legs = []
    for t in ("CE", "PE"):
        best = None
        for (k, ty), p in q.items():
            if ty != t or (t == "CE" and k <= fwd) or (t == "PE" and k >= fwd):
                continue
            iv, d = SR._iv_delta(p, fwd, k, T, t == "CE")
            if d is None:
                continue
            gap = abs(abs(d) - target)
            if best is None or gap < best[0]:
                best = (gap, k)
        if best is None:
            return None
        legs.append((t, best[1]))
    return legs


def collect():
    H = SR.history()
    h = H.dropna(subset=["thr"]).copy()
    h["normal"] = h.atm_iv < h.thr
    state = {(r.day, r.slot): r.normal for r in h.itertuples()}
    d930 = h[h.slot == "09:30"]
    entries = d930[(d930.dte == 1)][["day", "expiry", "normal"]]
    man = S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") in ("done", "live"))
    sp = S._load_spot()
    st, sc = sp["ts"].to_numpy(), sp["close"].to_numpy()
    cache, rows = {}, []
    for e in entries.itertuples():
        nxt = [x for x in exps if x > e.expiry]
        if not nxt:
            continue
        E2 = nxt[0]
        if E2 not in cache:
            df = S._load_options(E2)
            mins = df["date"].dt.hour * 60 + df["date"].dt.minute
            cache = {E2: df[mins <= 15 * 60 + 40].sort_values("ts")}
        df = cache[E2]
        e2d = pd.Timestamp(E2).date()
        if (pd.Timestamp(E2) - pd.Timestamp(e.day)).days > 12:
            continue
        t_ent = _ts(e.day, "09:30")
        day_rows = df[(df["day"] == pd.Timestamp(e.day).date()) & (df.ts <= t_ent)]
        if day_rows.empty:
            continue
        last = day_rows.groupby(["strike", "type"]).close.last()
        j = np.searchsorted(st, t_ent, side="right") - 1
        spot = float(sc[j])
        q = {(int(k), t): float(v) for (k, t), v in last.items() if abs(k - spot) <= 1600}
        T = (SR._exp_close_ts(E2) - t_ent) / (365 * 86400)
        j_exp = np.searchsorted(st, _ts(E2, "23:59"), side="right") - 1
        settle = float(sc[j_exp])
        for name, target in STRATS.items():
            legs = pick_legs(q, spot, T, target)
            if not legs or any((k, t) not in q for t, k in legs):
                continue
            sold = sum(q[(k, t)] for t, k in legs) - SLIP * len(legs)
            fut = df[(df.ts > t_ent)]
            series = []
            for t, k in legs:
                x = fut[(fut.type == t) & (fut.strike == k)]
                series.append(pd.Series(x.close.to_numpy(), index=x.ts.to_numpy()))
            path = pd.concat(series, axis=1).sort_index().ffill()
            path = path.fillna(pd.Series([q[(k, t)] for t, k in legs], index=path.columns))
            comb = path.sum(axis=1)
            ts_arr, c_arr = comb.index.to_numpy(), comb.to_numpy()
            intr = sum(max(0.0, settle - k) if t == "CE" else max(0.0, k - settle) for t, k in legs)

            def value_at(tsv):
                i = np.searchsorted(ts_arr, tsv, side="right") - 1
                return None if i < 0 else float(c_arr[i])
            # IV checks after entry, in time order
            checks = []
            dd = pd.Timestamp(e.day).date()
            while dd <= e2d:
                for slot in SR.SLOTS:
                    tv = _ts(dd.isoformat(), slot)
                    if tv > t_ent and (dd.isoformat(), slot) in state and tv < SR._exp_close_ts(E2):
                        checks.append((tv, slot, state[(dd.isoformat(), slot)]))
                dd += timedelta(days=1)
            tp_hit = np.flatnonzero(c_arr <= sold * 0.5 - SLIP * len(legs)) if len(c_arr) else []
            tp_t = int(ts_arr[tp_hit[0]]) if len(tp_hit) else None
            first930 = next((tv for tv, s, n in checks if s == "09:30" and n), None)
            firstany = next((tv for tv, s, n in checks if n), None)
            res = {}
            for ex in EXITS:
                t_exit = {"hold": None, "iv0930": first930, "ivany": firstany, "tp50": tp_t,
                          "iv0930+tp50": min([x for x in (first930, tp_t) if x], default=None)}[ex]
                if t_exit is None:
                    pts = sold - intr
                    ch = sum(charges.order_charges("SELL", q[(k, t)], LOT) for t, k in legs)
                    held = (SR._exp_close_ts(E2) - t_ent) / 86400
                    why = "expiry"
                else:
                    # IV exits act on the reading at the check -> fill one minute later
                    fill_t = t_exit if ex == "tp50" or (ex == "iv0930+tp50" and t_exit == tp_t) else t_exit + 60
                    buy = value_at(fill_t) + SLIP * len(legs)
                    pts = sold - buy
                    per = buy / len(legs)
                    ch = sum(charges.order_charges("SELL", q[(k, t)], LOT) + charges.order_charges("BUY", per, LOT) for t, k in legs)
                    held = (t_exit - t_ent) / 86400
                    why = "exit"
                res[ex] = {"rs": pts * LOT - ch, "held": held, "why": why}
            rows.append({"day": e.day, "cur_expiry": e.expiry, "expiry": E2, "green": not e.normal, "strat": name,
                         "prem_rs": sold * LOT, **{f"{ex}_rs": res[ex]["rs"] for ex in EXITS},
                         **{f"{ex}_held": res[ex]["held"] for ex in EXITS}, **{f"{ex}_why": res[ex]["why"] for ex in EXITS}})
    return pd.DataFrame(rows)


def summarize(d):
    out = []
    for strat in STRATS:
        for green in (True, False):
            x = d[(d.strat == strat) & (d.green == green)]
            for ex in EXITS:
                v = x[f"{ex}_rs"]
                eq = v.cumsum()
                out.append({"strat": strat, "green": green, "exit": ex, "n": int(len(x)), "avg": round(float(v.mean())),
                            "total": round(float(v.sum())), "win": round(float((v > 0).mean() * 100), 1),
                            "p5": round(float(v.quantile(0.05))), "worst": round(float(v.min())),
                            "held": round(float(x[f"{ex}_held"].mean()), 1),
                            "per_day": round(float((v / x[f"{ex}_held"].clip(lower=0.25)).mean())),
                            "exited": round(float((x[f"{ex}_why"] == "exit").mean() * 100)),
                            "a": round(float(v[x.day < "2024"].mean())), "b": round(float(v[x.day >= "2024"].mean())),
                            "dd": round(float((eq - eq.cummax()).min()))})
    return out


def report(d, summ):
    data = {"summ": summ, "label": LABEL, "exits": EXITS, "strats": list(STRATS), "n_days": int(d.day.nunique()),
            "first": d.day.min(), "last": d.day.max()}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str)))
    return path


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Next Expiry on Green Days</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--grp:#eef0f6;--best:#e9f8f1}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--grp:#23262e;--best:#16302a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1300px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
tr.grp td{background:var(--grp);font-weight:700;font-size:12px;color:var(--muted)}tr.best td{background:var(--best)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}
.rules{display:grid;grid-template-columns:130px 1fr;gap:6px 14px;font-size:13.5px}.rules b{color:var(--muted)}
</style></head><body><div class="wrap">
<h1>Green day, 1 day left — sell the next expiry, exit when IV turns normal?</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="card"><h2>Rules tested</h2><div class="rules">
<b>Entry</b><span>09:30 on days when the current weekly expiry has 1 trading day left and the IV rule is green; sell the NEXT weekly expiry (straddle at ATM, strangles by delta). The same entries on normal days shown for comparison.</span>
<b>"IV normal"</b><span>The rule's own reading: ATM IV of the nearest expiry with 1–4 days left below the 60th-percentile threshold for that time of day. No reading exists on the current expiry's expiry day, so no IV exit that day.</span>
<b>Exits</b><span id="exList"></span>
<b>Costs</b><span>1-min closes, 0.5 pt slippage per order, all charges, 1 lot (65). IV exits fill at the next minute's close after the check.</span></div></div>
<div class="card"><h2>What the test says</h2><ul id="verdict"></ul></div>
<div class="card"><h2>Results</h2><div class="scroll"><table id="t"></table></div>
<p class="note">Held = average days in the trade. ₹/day = P&amp;L ÷ days held. Exited = share of trades closed before expiry by the rule. Best ₹/trade per strategy on green days shaded.</p></div>
</div><script>
const D = /*DATA*/null;
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
document.getElementById("meta").textContent = `${D.n_days} entry days (${D.first} → ${D.last}) with 1 trading day left in the current weekly expiry.`;
document.getElementById("exList").innerHTML = D.exits.map(e => D.label[e]).join(" · ");
const G = (s, green, ex) => D.summ.find(r => r.strat === s && r.green === green && r.exit === ex);
const s0 = "Short straddle ATM", h = G(s0, true, "hold"), i9 = G(s0, true, "iv0930"), ia = G(s0, true, "ivany"), tp = G(s0, true, "tp50"), co = G(s0, true, "iv0930+tp50"), nh = G(s0, false, "hold");
const best = D.strats.map(s => D.exits.map(e => G(s, true, e)).reduce((a, r) => r.avg > a.avg ? r : a));
document.getElementById("verdict").innerHTML = [
  `<b>Selling the next expiry on green days made money; on normal days it did not.</b> Straddle held to expiry: ${rs(h.avg)} a trade on green days (${h.n}; 2021–23 ${rs(h.a)}, 2024–26 ${rs(h.b)}) vs ${rs(nh.avg)} on normal days.`,
  `<b>Exiting when IV turns normal:</b> at the 09:30 check ${rs(i9.avg)} a trade (${i9.exited}% exited early, held ${i9.held} days, 5% worst ${rs(i9.p5)}); at any check ${rs(ia.avg)} (${ia.exited}% exited, ${ia.held} days, 5% worst ${rs(ia.p5)}) — vs holding ${rs(h.avg)} (${h.held} days, 5% worst ${rs(h.p5)}).`,
  `<b>Per day held:</b> hold ${rs(h.per_day)}, IV exit at 09:30 ${rs(i9.per_day)}, at any check ${rs(ia.per_day)}, +50% target ${rs(tp.per_day)}, combined ${rs(co.per_day)}.`,
  `<b>Best exit per trade on green days:</b> ` + best.map(r => `${r.strat}: ${D.label[r.exit].toLowerCase()} (${rs(r.avg)})`).join("; ") + ".",
].map(x => `<li>${x}</li>`).join("");
let t = `<tr><th>Exit</th><th>Trades</th><th>Win %</th><th>₹ / trade</th><th>₹ / day</th><th>Held (calendar days)</th><th>Exited early</th><th>5% worst</th><th>Worst</th><th>Max DD</th><th>2021–23</th><th>2024–26</th></tr>`;
D.strats.forEach(s => [true, false].forEach(g => {
  t += `<tr class="grp"><td colspan="12">${s} · ${g ? "green day" : "normal day (comparison)"}</td></tr>`;
  const rows = D.exits.map(e => G(s, g, e)), b = rows.reduce((a, r) => r.avg > a.avg ? r : a);
  rows.forEach(r => { t += `<tr class="${g && r === b ? "best" : ""}"><td>${D.label[r.exit]}</td><td>${r.n}</td><td>${r.win}</td><td class="${cl(r.avg)}"><b>${rs(r.avg)}</b></td><td class="${cl(r.per_day)}">${rs(r.per_day)}</td><td>${r.held}</td><td>${r.exited}%</td><td class="down">${rs(r.p5)}</td><td class="down">${rs(r.worst)}</td><td class="down">${rs(r.dd)}</td><td class="${cl(r.a)}">${rs(r.a)}</td><td class="${cl(r.b)}">${rs(r.b)}</td></tr>`; });
}));
document.getElementById("t").innerHTML = t;
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    d = collect()
    os.makedirs(OUT, exist_ok=True)
    d.to_parquet(os.path.join(OUT, "trades.parquet"), index=False)
    summ = summarize(d)
    for r in summ:
        print(f"{r['strat']:22s} {'GREEN' if r['green'] else 'normal':6s} {r['exit']:12s} n {r['n']:3d} avg {r['avg']:6d} /day {r['per_day']:6d} held {r['held']:4.1f} exited {r['exited']:3d}% p5 {r['p5']:7d} worst {r['worst']:7d} dd {r['dd']:8d} A {r['a']:6d} B {r['b']:6d}")
    print(report(d, summ))
