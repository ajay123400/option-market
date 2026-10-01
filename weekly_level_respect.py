"""weekly_level_respect.py -- do the week's range levels act as support /
resistance on the FOLLOWING days (multi-day), and does expiry pin to them?

Levels: every UPPER / LOWER the positional method produced during a week
(Day-1 09:20 and every recalculation after a leader change), rounded to 5 pts
and de-duplicated. Each level is tested on NIFTY's 1-min bars from the NEXT
trading day's open until the expiry close -- only if price was on the correct
side of it at the end of the day it was created (resistance above, support below).

Controls, same count and same side:
  random   the creation-day close +/- a distance taken from another week's level
  shifted  our level +/- 75 pts (is it THE level, or just "somewhere around here"?)

Touch / reject / break / fake / retest definitions as in level_respect.py.
Pin: distance of the expiry close from the nearest level of each set.
Output: results/weekly_level_respect/{events.csv, report.html}
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import paths
import simulator as S
from level_respect import test_level

OUT = os.path.join(paths.BASE_DIR, "results", "weekly_level_respect")
SHIFT = 75


def load():
    return (pd.read_csv(os.path.join(OUT, "events.csv")), pd.read_csv(os.path.join(OUT, "pins.csv")))


def run():
    weeks = json.load(open(os.path.join(paths.BASE_DIR, "results", "positional_bt", "nohedge_nosl", "weeks_detail.json"), encoding="utf-8"))
    sp = S._load_spot()
    sp = sp.assign(day=sp.index.date, hm=sp.index.hour * 60 + sp.index.minute)
    sp = sp[(sp.hm >= 555) & (sp.hm <= 930)]
    sp["five"] = ((sp.hm - 555 + 1) % 5 == 0)
    day_close = sp.groupby("day").close.last()
    rng = np.random.default_rng(11)
    # pool of level distances (from creation-day close) for the random control
    pool = []
    levels = []
    for w in weeks:
        exp = pd.Timestamp(w["expiry"]).date()
        seen = set()
        for e in w["events"]:
            if e["kind"] != "RANGE":
                continue
            cday = pd.Timestamp(e["time"][:10]).date()
            if cday >= exp or cday not in day_close.index:
                continue
            c = float(day_close[cday])
            for side, L in ((1, e["upper"]), (-1, e["lower"])):
                key = (side, round(L / 5) * 5)
                if key in seen:
                    continue
                seen.add(key)
                d = (L - c) * side
                if d < 20:
                    continue  # already broken by the creation day's close
                levels.append({"expiry": w["expiry"], "created": cday.isoformat(), "side": side, "level": float(L), "dist": d, "close": c})
                pool.append(d)
    pool = np.array(pool)
    events, pins = [], []
    by_exp = {}
    for lv in levels:
        by_exp.setdefault(lv["expiry"], []).append(lv)
    for exp, lvls in by_exp.items():
        e_day = pd.Timestamp(exp).date()
        exp_close = float(day_close[e_day]) if e_day in day_close.index else None
        sets = {"Our week's levels": [], "Random (same distance, other week)": [], f"Shifted ±{SHIFT} from ours": []}
        for lv in lvls:
            c0 = pd.Timestamp(lv["created"]).date()
            bars = sp[(sp.day > c0) & (sp.day <= e_day)]
            if len(bars) < 30:
                continue
            ctrl = [("Our week's levels", lv["level"]),
                    ("Random (same distance, other week)", lv["close"] + lv["side"] * float(rng.choice(pool))),
                    (f"Shifted ±{SHIFT} from ours", lv["level"] + lv["side"] * SHIFT * (1 if rng.random() < .5 else -1))]
            for name, L in ctrl:
                if (L - lv["close"]) * lv["side"] < 20:
                    continue
                res = test_level(bars, float(L), lv["side"])
                ndays = bars.day.nunique()
                events.append({"expiry": exp, "created": lv["created"], "days_after": ndays, "kind": name,
                               "side": "upper / resistance" if lv["side"] > 0 else "lower / support",
                               "level": round(L, 2), "dist": round((L - lv["close"]) * lv["side"], 1), **res})
                sets[name].append(L)
        if exp_close is not None:
            for name, L in sets.items():
                if L:
                    pins.append({"expiry": exp, "kind": name, "n": len(L), "pin": float(np.min(np.abs(np.array(L) - exp_close)))})
    ev, pn = pd.DataFrame(events), pd.DataFrame(pins)
    os.makedirs(OUT, exist_ok=True)
    ev.to_csv(os.path.join(OUT, "events.csv"), index=False)
    pn.to_csv(os.path.join(OUT, "pins.csv"), index=False)
    return ev, pn


def _stats(g):
    t = g[g.touched]
    n = len(t)
    br = t.outcome == "broke"
    p = (t.outcome == "rejected").mean() if n else 0
    return {"levels": len(g), "dist": round(float(g.dist.median())), "touched": round(len(t) / len(g) * 100, 1), "n": n,
            "rejected": round(p * 100, 1), "ci": round(1.96 * np.sqrt(p * (1 - p) / n) * 100, 1) if n else None,
            "broke": round(br.mean() * 100, 1) if n else None,
            "fake": round((br & t.fake).sum() / max(1, br.sum()) * 100, 1),
            "touches": round(float(t.touches.mean()), 2) if n else None, "multi": round((t.touches >= 2).mean() * 100, 1) if n else None,
            "held": round((t.closed_beyond.astype(str) == "False").mean() * 100, 1) if n else None}


def build(ev, pn):
    kinds = list(dict.fromkeys(ev.kind))
    summary = [{"kind": k, "side": "both", **_stats(ev[ev.kind == k])} for k in kinds] + \
              [{"kind": k, "side": s, **_stats(g)} for (k, s), g in ev.groupby(["kind", "side"])]
    # same-distance buckets: ours vs random
    ev = ev.copy()
    ev["db"] = pd.cut(ev.dist, [20, 100, 200, 300, 500, 5000], labels=["20–100", "100–200", "200–300", "300–500", "500+"])
    dm = [{"kind": k, "bucket": str(b), **_stats(g)} for (k, b), g in ev.groupby(["kind", "db"], observed=True) if len(g) >= 10]
    pin = []
    for k, g in pn.groupby("kind"):
        pin.append({"kind": k, "weeks": len(g), "levels": round(float(g.n.mean()), 1), "median": round(float(g.pin.median()), 1),
                    "within50": round((g.pin <= 50).mean() * 100, 1), "within100": round((g.pin <= 100).mean() * 100, 1)})
    data = {"summary": summary, "dm": dm, "pin": pin, "weeks": int(ev.expiry.nunique()), "shift": SHIFT,
            "span": [ev.expiry.min(), ev.expiry.max()]}
    p = os.path.join(OUT, "report.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/", json.dumps(data, default=str)))
    return p, data


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weekly Level Respect</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--ours:#eef0fd}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--ours:#23284a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}tr.ours td{background:var(--ours);font-weight:600}.scroll{overflow-x:auto}.note{color:var(--muted);font-size:12.5px}
ul{margin:0;padding-left:18px}li{margin:4px 0}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>Do the week's range levels hold on the following days?</h1><p class="note" id="sub"></p>
<div class="card"><h2>Method</h2><ul>
<li><b>Our levels:</b> every UPPER / LOWER the method produced during the week (Day-1 09:20 and each leader-change recalculation), tested from the <b>next day's open to the expiry close</b> — only if NIFTY was still on the right side of it at that day's close.</li>
<li><b>Random:</b> same side and the same kind of distance from the creation-day close, taken from another week. <b>Shifted:</b> our level moved ±<span id="sh"></span> pts — if the exact level matters, these should do clearly worse.</li>
<li><b>Touched</b> = within 10 pts · <b>Rejected</b> = moves 60 pts back before breaking · <b>Broke</b> = 5-min close 20+ pts beyond · <b>Fake</b> = broke, then came back 20+ inside later (before expiry) · <b>Held</b> = the expiry close is still on the right side.</li>
</ul></div>
<div class="card"><h2>In short</h2><ul id="short"></ul></div>
<div class="card"><h2>After NIFTY reaches a level (following days, until expiry)</h2><div class="scroll"><table id="t1"></table></div>
<p class="note">± = 95% confidence range of the rejection rate. If our row's range overlaps the controls', the difference is not significant.</p></div>
<div class="grid2">
<div class="card"><h2>Same distance, like for like</h2><div class="scroll"><table id="t2"></table></div></div>
<div class="card"><h2>Expiry pin: how close did the expiry close finish to the nearest level?</h2><div class="scroll"><table id="t3"></table></div>
<p class="note">Same number of levels in each set per week. A clear pin effect would show our levels much closer than the controls.</p></div>
</div>
</div><script>
const D = /*DATA*/, f = v => v == null ? "—" : v;
document.getElementById("sub").textContent = `${D.weeks} weekly expiries · ${D.span[0]} → ${D.span[1]} · NIFTY 50 1-min bars`;
document.getElementById("sh").textContent = D.shift;
const both = D.summary.filter(r => r.side === "both");
const o = both.find(r => r.kind.startsWith("Our")), r0 = both.find(r => r.kind.startsWith("Random")), s0 = both.find(r => r.kind.startsWith("Shifted"));
const po = D.pin.find(r => r.kind.startsWith("Our")), pr = D.pin.find(r => r.kind.startsWith("Random")), ps = D.pin.find(r => r.kind.startsWith("Shifted"));
document.getElementById("short").innerHTML = [
  `Once NIFTY reaches one of <b>our week's levels</b> on a later day it is <b>rejected ${o.rejected}% (±${o.ci})</b> of the time, vs <b>${r0.rejected}% (±${r0.ci})</b> for random levels and <b>${s0.rejected}% (±${s0.ci})</b> for levels ${D.shift} pts away from ours.`,
  `Breaks: ours ${o.broke}% · random ${r0.broke}% · shifted ${s0.broke}%. Fake breaks (came back inside before expiry): ours ${o.fake}% · random ${r0.fake}% · shifted ${s0.fake}%.`,
  `Levels still held at expiry once touched: ours ${o.held}% · random ${r0.held}% · shifted ${s0.held}%.`,
  po && pr ? `Expiry pin: the expiry close finished a median <b>${po.median} pts</b> from our nearest level, vs ${pr.median} (random) and ${ps ? ps.median : "—"} (shifted).` : ""].filter(Boolean).map(x => `<li>${x}</li>`).join("");
document.getElementById("t1").innerHTML = `<tr><th>Levels</th><th>Side</th><th>Count</th><th>Median distance</th><th>Touched</th><th>Rejected</th><th>Broke</th><th>…fake</th><th>Avg touches</th><th>2+ touches</th><th>Held at expiry</th></tr>` +
  D.summary.map(r => `<tr${r.kind.startsWith("Our") ? ' class="ours"' : ""}><td>${r.kind}</td><td>${r.side}</td><td>${r.levels}</td><td>${r.dist}</td><td>${r.touched}%</td><td>${r.rejected}% <span class="note">±${r.ci}</span></td><td>${f(r.broke)}%</td><td>${r.fake}%</td><td>${f(r.touches)}</td><td>${f(r.multi)}%</td><td>${f(r.held)}%</td></tr>`).join("");
const bk = [...new Set(D.dm.map(x => x.bucket))], ks = [...new Set(D.dm.map(x => x.kind))];
document.getElementById("t2").innerHTML = `<tr><th>Distance</th>${ks.map(k => `<th>${k.split(" (")[0].replace("Our week's levels", "Ours")}: rejected</th>`).join("")}</tr>` +
  bk.map(b => `<tr><td>${b}</td>${ks.map(k => { const x = D.dm.find(y => y.bucket === b && y.kind === k); return `<td>${x ? x.rejected + "% <span class='note'>(" + x.n + ")</span>" : "—"}</td>`; }).join("")}</tr>`).join("");
document.getElementById("t3").innerHTML = `<tr><th>Levels</th><th>Weeks</th><th>Levels / week</th><th>Median distance</th><th>Within 50</th><th>Within 100</th></tr>` +
  D.pin.map(r => `<tr${r.kind.startsWith("Our") ? ' class="ours"' : ""}><td>${r.kind}</td><td>${r.weeks}</td><td>${r.levels}</td><td>${r.median}</td><td>${r.within50}%</td><td>${r.within100}%</td></tr>`).join("");
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    ev, pn = load() if "--report" in sys.argv else run()
    p, d = build(ev, pn)
    print(p)
    for r in d["summary"]:
        print(r)
    print(d["pin"])
    print([(x["kind"][:6], x["bucket"], x["rejected"], x["n"]) for x in d["dm"]])
