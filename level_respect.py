"""level_respect.py -- does NIFTY respect the 09:20 volume-leader range?

For every trading day, levels above (resistance) and below (support) NIFTY's
09:20 price are tested on the rest of the day's 1-min NIFTY bars:

  ours      the day's fresh 09:20 range: UPPER / LOWER
  random    the same distances from 09:20 as our levels, but taken from a
            different (shuffled) day -- the baseline "any level this far away"
  prev day  previous day's high / low
  OI wall   strike with the most CE OI above / PE OI below spot at 09:19

A level is TOUCHED when price comes within 10 pts of it. After the first touch:
  rejected  price moves 60 pts back (away from the level) before breaking it
  broke     a 5-min candle closes 20+ pts beyond the level
  -> fake   broke, then came back 20+ pts inside the level the same day
  neither   none of the above by the close
Retests = separate touches (price must move 40+ pts away between touches).

Output: results/level_respect/{events.csv, report.html}
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import paths
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "level_respect")
TOUCH, BREAK, REJECT, AWAY = 10, 20, 60, 40


def _levels_from_oi(exp, day, spot):
    df = S._load_options(exp)
    x = df[(df["day"] == day) & (df["date"].dt.hour * 60 + df["date"].dt.minute <= 9 * 60 + 19)]
    if x.empty:
        return None, None
    oi = x.sort_values("ts").groupby(["type", "strike"]).oi.last().reset_index()
    ce = oi[(oi.type == "CE") & (oi.strike > spot + 20) & (oi.strike % 100 == 0)]
    pe = oi[(oi.type == "PE") & (oi.strike < spot - 20) & (oi.strike % 100 == 0)]
    up = float(ce.loc[ce.oi.idxmax(), "strike"]) if len(ce) and ce.oi.max() > 0 else None
    dn = float(pe.loc[pe.oi.idxmax(), "strike"]) if len(pe) and pe.oi.max() > 0 else None
    return up, dn


def test_level(bars, L, side):
    """bars: the day's 1-min NIFTY bars after 09:20 (DataFrame with high/low/close, 5-min flag).
    side = +1 resistance above, -1 support below. Returns a dict of outcomes."""
    h, l, c, five = bars.high.to_numpy(), bars.low.to_numpy(), bars.close.to_numpy(), bars.five.to_numpy()
    near = (h >= L - TOUCH) if side > 0 else (l <= L + TOUCH)
    idx = np.flatnonzero(near)
    if not len(idx):
        return {"touched": False}
    t0 = idx[0]
    res = {"touched": True, "touch_min": int(t0), "outcome": "neither", "fake": False, "beyond_max": 0.0, "back_max": 0.0}
    broke_at = None
    for i in range(t0, len(c)):
        beyond = (h[i] - L) if side > 0 else (L - l[i])
        back = (L - l[i]) if side > 0 else (h[i] - L)
        res["beyond_max"] = max(res["beyond_max"], beyond)
        if broke_at is None:
            res["back_max"] = max(res["back_max"], back)
            if back >= REJECT:
                res["outcome"] = "rejected"
                break
            if five[i] and ((c[i] - L) if side > 0 else (L - c[i])) >= BREAK:
                res["outcome"] = "broke"
                broke_at = i
        else:
            inside = (L - c[i]) if side > 0 else (c[i] - L)
            if inside >= BREAK:
                res["fake"] = True
                break
    # retests: separate touches with a 40-pt move away in between
    touches, armed = 0, True
    for i in range(t0, len(c)):
        dist = (L - h[i]) if side > 0 else (l[i] - L)
        if armed and dist <= TOUCH:
            touches += 1
            armed = False
        elif not armed and dist >= AWAY:
            armed = True
    res["touches"] = touches
    res["closed_beyond"] = bool(((c[-1] - L) if side > 0 else (L - c[-1])) > 0)
    return res


def run(verbose=True):
    rows = pd.read_csv(os.path.join(paths.BASE_DIR, "results", "positional_range_research", "rows.csv"))
    rows = rows.dropna(subset=["spot_920"])
    sp = S._load_spot()
    sp = sp.assign(day=sp.index.date, hm=sp.index.hour * 60 + sp.index.minute)
    sp = sp[(sp.hm >= 555) & (sp.hm <= 930)]
    daily = sp.groupby("day").agg(high=("high", "max"), low=("low", "min"))
    prev = daily.shift(1)
    rng = np.random.default_rng(7)
    dist_up = (rows.upper - rows.spot_920).to_numpy()
    dist_dn = (rows.spot_920 - rows.lower).to_numpy()
    perm = rng.permutation(len(rows))
    events = []
    by_day = {d: g for d, g in sp.groupby("day")}
    for n, r in enumerate(rows.itertuples()):
        day = pd.Timestamp(r.day).date()
        g = by_day.get(day)
        if g is None:
            continue
        bars = g[g.hm >= 561].copy()   # from the 09:21 bar on
        if len(bars) < 30:
            continue
        bars["five"] = ((bars.hm - 555 + 1) % 5 == 0)
        s0 = float(r.spot_920)
        cand = [("Our range", r.upper, r.lower),
                ("Random (same distance, other day)", s0 + abs(dist_up[perm[n]]), s0 - abs(dist_dn[perm[n]]))]
        if day in prev.index and pd.notna(prev.loc[day, "high"]):
            cand.append(("Previous day high / low", prev.loc[day, "high"], prev.loc[day, "low"]))
        oi_up, oi_dn = _levels_from_oi(r.expiry, day, s0)
        cand.append(("OI wall (max CE / PE OI)", oi_up, oi_dn))
        for name, up, dn in cand:
            for side, L in ((1, up), (-1, dn)):
                if L is None or pd.isna(L):
                    continue
                d = (L - s0) * side
                if d < 20:
                    continue  # already at / through the level at 09:20
                res = test_level(bars, float(L), side)
                events.append({"day": r.day, "expiry": r.expiry, "dte": r.dte, "kind": name,
                               "side": "upper / resistance" if side > 0 else "lower / support",
                               "level": round(float(L), 2), "dist": round(d, 1), **res})
        if verbose and n % 100 == 0:
            print(f"\r{n}/{len(rows)}", end="", flush=True)
    if verbose:
        print()
    ev = pd.DataFrame(events)
    os.makedirs(OUT, exist_ok=True)
    ev.to_csv(os.path.join(OUT, "events.csv"), index=False)
    return ev


def summarize(ev):
    out = []
    for (k, s), g in ev.groupby(["kind", "side"]):
        t = g[g.touched]
        n = len(t)
        out.append({"kind": k, "side": s, "levels": len(g), "dist": round(float(g.dist.median())),
                    "touched": round(len(t) / len(g) * 100, 1),
                    "rejected": round((t.outcome == "rejected").mean() * 100, 1) if n else None,
                    "broke": round((t.outcome == "broke").mean() * 100, 1) if n else None,
                    "fake": round(((t.outcome == "broke") & t.fake).sum() / max(1, (t.outcome == "broke").sum()) * 100, 1) if n else None,
                    "neither": round((t.outcome == "neither").mean() * 100, 1) if n else None,
                    "touches": round(float(t.touches.mean()), 2) if n else None,
                    "multi": round((t.touches >= 2).mean() * 100, 1) if n else None,
                    "closed_beyond": round(t.closed_beyond.mean() * 100, 1) if n else None,
                    "beyond_med": round(float(t[t.outcome == "broke"].beyond_max.median()), 0) if (t.outcome == "broke").any() else None})
    return out


def build(ev):
    s = summarize(ev)
    # distance-matched comparison: our range vs random, by distance bucket (touched levels only)
    ev = ev.copy()
    ev["db"] = pd.cut(ev.dist, [20, 100, 150, 200, 300, 2000], labels=["20–100", "100–150", "150–200", "200–300", "300+"])
    dm = []
    for (k, b), g in ev[ev.kind.isin(["Our range", "Random (same distance, other day)"]) & ev.touched].groupby(["kind", "db"], observed=True):
        dm.append({"kind": k, "bucket": str(b), "n": len(g), "rejected": round((g.outcome == "rejected").mean() * 100, 1),
                   "broke": round((g.outcome == "broke").mean() * 100, 1)})
    ours = ev[(ev.kind == "Our range")]
    by_dte = []
    for d, g in ours.groupby("dte"):
        t = g[g.touched]
        if len(t) > 20:
            by_dte.append({"dte": int(d), "n": len(g), "touched": round(len(t) / len(g) * 100, 1),
                           "rejected": round((t.outcome == "rejected").mean() * 100, 1), "broke": round((t.outcome == "broke").mean() * 100, 1)})
    data = {"summary": s, "dm": dm, "dte": by_dte, "days": int(ev.day.nunique()),
            "span": [ev.day.min(), ev.day.max()], "params": {"touch": TOUCH, "break": BREAK, "reject": REJECT, "away": AWAY}}
    p = os.path.join(OUT, "report.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/", json.dumps(data, default=str)))
    return p, data


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Range Respect Test</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--accent:#4f5fe0;--ours:#eef0fd}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--ours:#23284a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}tr.ours td{background:var(--ours);font-weight:600}.scroll{overflow-x:auto}.note{color:var(--muted);font-size:12.5px}
.rules{display:grid;grid-template-columns:120px 1fr;gap:6px 14px;font-size:13.5px}.rules b{color:var(--muted)}ul{margin:0;padding-left:18px}li{margin:4px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>Does NIFTY respect our range?</h1><p class="note" id="sub"></p>
<div class="card"><h2>How it was measured</h2><div class="rules" id="rules"></div></div>
<div class="card"><h2>In short</h2><ul id="short"></ul></div>
<div class="card"><h2>What happens after NIFTY reaches a level (same day)</h2><div class="scroll"><table id="t1"></table></div>
<p class="note">Rejected / broke / neither are shares of the touched levels. "Fake" = share of the breaks that came back 20+ pts inside the level the same day.</p></div>
<div class="grid2">
<div class="card"><h2>Our range vs random levels, same distance</h2><div class="scroll"><table id="t2"></table></div>
<p class="note">Touched levels only, grouped by distance from the 09:20 price — a fair like-for-like check.</p></div>
<div class="card"><h2>Our range by days to expiry</h2><div class="scroll"><table id="t3"></table></div></div>
</div>
</div><script>
const D = /*DATA*/, P = D.params;
document.getElementById("sub").textContent = `${D.days} trading days · ${D.span[0]} → ${D.span[1]} · NIFTY 50 1-min bars after 09:20 · levels fixed at 09:20 each day`;
document.getElementById("rules").innerHTML = `
<b>Levels</b><span><b>Our range</b> = the day's 09:20 UPPER / LOWER. <b>Random</b> = the same distances from 09:20 as our levels but taken from another day (baseline for "any level that far away"). <b>Previous day</b> high / low. <b>OI wall</b> = strike with the most CE OI above / PE OI below at 09:19.</span>
<b>Touched</b><span>price comes within ${P.touch} pts of the level after 09:20.</span>
<b>Rejected</b><span>after the touch, price moves ${P.reject} pts back before breaking.</span>
<b>Broke</b><span>a 5-min candle closes ${P.break}+ pts beyond the level.</span>
<b>Retests</b><span>separate touches of the level, with a ${P.away}+ pt move away in between.</span>`;
const S = D.summary, f = v => v == null ? "—" : v;
const get = (k, s) => S.find(x => x.kind === k && x.side.startsWith(s));
const ou = get("Our range", "upper"), ol = get("Our range", "lower"), ru = get("Random", "upper"), rl = get("Random", "lower");
const rj = (a, b) => ((a.rejected + b.rejected) / 2).toFixed(0), br = (a, b) => ((a.broke + b.broke) / 2).toFixed(0);
document.getElementById("short").innerHTML = [
  `When NIFTY reaches <b>our UPPER / LOWER</b> the same day, it is <b>rejected ~${rj(ou, ol)}%</b> of the time and <b>breaks ~${br(ou, ol)}%</b>.`,
  `A <b>random level at the same distance</b> is rejected ~${rj(ru, rl)}% and broken ~${br(ru, rl)}% — compare the two to see whether the range adds anything beyond distance.`,
  `Our levels are reached on ${((ou.touched + ol.touched) / 2).toFixed(0)}% of days (median ${ou.dist} pts above / ${ol.dist} below the 09:20 price); once reached they are touched ${((ou.touches + ol.touches) / 2).toFixed(1)} times on average.`,
  `Of the breaks of our levels, ${((ou.fake + ol.fake) / 2).toFixed(0)}% turned out fake (came back inside the same day).`].map(x => `<li>${x}</li>`).join("");
document.getElementById("t1").innerHTML = `<tr><th>Level</th><th>Side</th><th>Days</th><th>Median distance</th><th>Touched</th><th>Rejected</th><th>Broke</th><th>…of which fake</th><th>Neither</th><th>Avg touches</th><th>2+ touches</th><th>Closed beyond</th><th>Median overshoot on a break</th></tr>` +
  S.map(r => `<tr${r.kind === "Our range" ? ' class="ours"' : ""}><td>${r.kind}</td><td>${r.side}</td><td>${r.levels}</td><td>${r.dist}</td><td>${r.touched}%</td><td>${f(r.rejected)}%</td><td>${f(r.broke)}%</td><td>${f(r.fake)}%</td><td>${f(r.neither)}%</td><td>${f(r.touches)}</td><td>${f(r.multi)}%</td><td>${f(r.closed_beyond)}%</td><td>${f(r.beyond_med)}</td></tr>`).join("");
const buckets = [...new Set(D.dm.map(x => x.bucket))];
document.getElementById("t2").innerHTML = `<tr><th>Distance</th><th>Ours: n</th><th>Ours: rejected</th><th>Ours: broke</th><th>Random: n</th><th>Random: rejected</th><th>Random: broke</th></tr>` +
  buckets.map(b => { const o = D.dm.find(x => x.bucket === b && x.kind === "Our range") || {}, r = D.dm.find(x => x.bucket === b && x.kind !== "Our range") || {};
    return `<tr><td>${b}</td><td>${f(o.n)}</td><td>${f(o.rejected)}%</td><td>${f(o.broke)}%</td><td>${f(r.n)}</td><td>${f(r.rejected)}%</td><td>${f(r.broke)}%</td></tr>`; }).join("");
document.getElementById("t3").innerHTML = `<tr><th>Days to expiry</th><th>Levels</th><th>Touched</th><th>Rejected</th><th>Broke</th></tr>` +
  D.dte.map(r => `<tr><td>${r.dte === 0 ? "Expiry day" : r.dte}</td><td>${r.n}</td><td>${r.touched}%</td><td>${r.rejected}%</td><td>${r.broke}%</td></tr>`).join("");
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    ev = run()
    p, d = build(ev)
    print(p)
    for r in d["summary"]:
        print(r)
    print(d["dm"])
    print(d["dte"])
