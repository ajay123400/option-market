"""volume_levels_research.py -- do high-volume 1-min futures bars mark NIFTY
levels that later act as support / resistance?

Idea (the user's, from the Futures chart): the minute with an unusually large
NIFTY-futures volume pins a price; when NIFTY comes back to that price days
later it tends to stop there (e.g. 28 Sep 2026 14:27 ~22,800 -> resistance on
30 Sep; 29 Sep 10:07 ~22,585 -> resistance after the 1 Oct gap-down).

Test, Oct 2021 -> Oct 2026, continuous near-month futures volume + NIFTY spot:

  volume levels   (a) each session's highest-volume minute, (b) every minute
                  whose volume is >= 8x the median of the previous ~1 session
                  of minutes; the first minute (09:15) and the last 5 minutes
                  are skipped. Level = NIFTY spot's close at that minute.
  control levels  5 random other minutes of the same session (same day, same
                  market regime) -- what any price level does.
  return          within the next 10 sessions NIFTY first moves >= 0.3% away
                  from the level, then trades back through / onto it (touch)
  outcome         after the touch, does NIFTY first move 0.3% back on the side
                  it came from (HELD) or 0.3% through the level (BROKE)?
                  Undecided by the horizon = left out.

A volume level "works" only if it holds clearly more often than control
levels -- in both halves of the sample, as support and as resistance.

Output: results/volume_levels/report.html (+ levels.parquet)
"""
import json
import math
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

import futures_history as F
import paths
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "volume_levels")
AWAY = 0.003        # must first move this far from the level (fraction)
DECIDE = 0.003      # after the touch: this far back = held, this far through = broke
HORIZON_SESSIONS = 10
RATIO = 8.0
N_CONTROL = 5
START, END = date(2021, 10, 1), date(2026, 10, 9)


def load():
    fut = F.near_month(START, END)
    ft = pd.to_datetime(fut.ts, unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    fut = fut.assign(day=ft.dt.date, hm=ft.dt.hour * 60 + ft.dt.minute)
    close_m = np.where(fut.day >= F.F_O_CLOSE_CHANGE, 15 * 60 + 39, 15 * 60 + 29)
    fut = fut[(fut.hm >= 9 * 60 + 16) & (fut.hm <= close_m - 5)].sort_values("ts").reset_index(drop=True)
    sp = S._load_spot()[["ts", "high", "low", "close"]].sort_values("ts").reset_index(drop=True)
    st = pd.to_datetime(sp.ts, unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    sp["day"] = st.dt.date
    return fut, sp


def pick_levels(fut, sp):
    """Volume and control levels: [{ts, day, kind, ratio, level}]"""
    med = fut.volume.rolling(375, min_periods=200).median().shift(1)
    fut = fut.assign(ratio=fut.volume / med)
    spot_close = pd.Series(sp.close.values, index=sp.ts.values)
    rng = np.random.default_rng(11)
    rows = []
    for day, g in fut.groupby("day"):
        g = g.dropna(subset=["ratio"])
        if g.empty:
            continue
        top = g.loc[g.volume.idxmax()]
        picks = {int(top.ts): ("day-top", float(top.ratio))}
        for r in g[g.ratio >= RATIO].itertuples():
            picks.setdefault(int(r.ts), ("spike", float(r.ratio)))
        for ts, (kind, ratio) in picks.items():
            if ts in spot_close.index:
                rows.append({"ts": ts, "day": day, "kind": kind, "ratio": ratio, "level": float(spot_close[ts])})
        others = g[~g.ts.isin(list(picks))]
        for ts in rng.choice(others.ts.values, size=min(N_CONTROL, len(others)), replace=False):
            if int(ts) in spot_close.index:
                rows.append({"ts": int(ts), "day": day, "kind": "control", "ratio": float(g.set_index("ts").ratio.get(ts, np.nan)), "level": float(spot_close[int(ts)])})
    return pd.DataFrame(rows)


def outcomes(levels, sp):
    """Return / touch / held-or-broke for every level, scanning NIFTY's 1-min path."""
    ts, hi, lo, cl = sp.ts.to_numpy(), sp.high.to_numpy(), sp.low.to_numpy(), sp.close.to_numpy()
    days = sp.day.to_numpy()
    day_start = {}
    for i, d in enumerate(days):
        day_start.setdefault(d, i)
    uniq = list(day_start)
    pos = {d: k for k, d in enumerate(uniq)}
    out = []
    for r in levels.itertuples():
        i0 = np.searchsorted(ts, r.ts, side="right")              # from the next minute
        k = pos.get(r.day)
        if k is None or i0 >= len(ts):
            out.append(None); continue
        end_day = uniq[min(len(uniq) - 1, k + HORIZON_SESSIONS)]
        i1 = (day_start[uniq[k + HORIZON_SESSIONS + 1]] if k + HORIZON_SESSIONS + 1 < len(uniq) else len(ts))
        L = r.level
        up, dn = L * (1 + AWAY), L * (1 - AWAY)
        h, l = hi[i0:i1], lo[i0:i1]
        # 1) first move away
        a_up = np.argmax(h >= up) if (h >= up).any() else None
        a_dn = np.argmax(l <= dn) if (l <= dn).any() else None
        if a_up is None and a_dn is None:
            out.append({"res": "never left"}); continue
        if a_dn is None or (a_up is not None and a_up < a_dn):
            side, j = "above", a_up                                   # price went above -> later tests it as SUPPORT
        else:
            side, j = "below", a_dn                                   # price went below -> later tests it as RESISTANCE
        # 2) first touch back
        hh, ll = h[j:], l[j:]
        t_idx = (np.argmax(ll <= L) if (ll <= L).any() else None) if side == "above" else (np.argmax(hh >= L) if (hh >= L).any() else None)
        if t_idx is None:
            out.append({"res": "no return", "side": side}); continue
        t = j + t_idx
        # 3) held or broke after the touch
        hh, ll = h[t:], l[t:]
        if side == "above":      # support test: held = back up AWAY, broke = down through
            held = np.argmax(hh >= L * (1 + DECIDE)) if (hh >= L * (1 + DECIDE)).any() else None
            broke = np.argmax(ll <= L * (1 - DECIDE)) if (ll <= L * (1 - DECIDE)).any() else None
        else:                    # resistance test
            held = np.argmax(ll <= L * (1 - DECIDE)) if (ll <= L * (1 - DECIDE)).any() else None
            broke = np.argmax(hh >= L * (1 + DECIDE)) if (hh >= L * (1 + DECIDE)).any() else None
        if held is None and broke is None:
            res = "undecided"
        elif broke is None or (held is not None and held < broke):
            res = "held"
        elif held is None or broke < held:
            res = "broke"
        else:
            res = "same bar"
        touch_day = days[i0 + t]
        out.append({"res": res, "side": side, "touch_sessions": int(pos[touch_day] - k)})
    o = pd.DataFrame([x or {"res": "n/a"} for x in out])
    return pd.concat([levels.reset_index(drop=True), o], axis=1)


def _rate(x):
    d = x[x.res.isin(["held", "broke"])]
    n = len(d)
    return n, (float((d.res == "held").mean() * 100) if n else None)


def _z(n1, p1, n2, p2):
    if not n1 or not n2 or p1 is None or p2 is None:
        return None
    a, b = p1 / 100, p2 / 100
    p = (a * n1 + b * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return round((a - b) / se, 2) if se > 0 else None


def analyse(d):
    d = d.copy()
    d["half"] = np.where(pd.to_datetime(d.day) < pd.Timestamp("2024-01-01"), "2021–23", "2024–26")
    d["role"] = d.side.map({"above": "support", "below": "resistance"})
    rows = []
    groups = [("All", d)] + [(f"{s}", d[d.role == s]) for s in ("support", "resistance")] + [(h, d[d.half == h]) for h in ("2021–23", "2024–26")]
    for name, x in groups:
        c = x[x.kind == "control"]
        nc, pc = _rate(c)
        for kind in ("day-top", "spike"):
            v = x[x.kind == kind]
            nv, pv = _rate(v)
            rows.append({"group": name, "kind": kind, "n": nv, "held": round(pv, 1) if pv is not None else None,
                         "ctrl_n": nc, "ctrl_held": round(pc, 1) if pc is not None else None, "z": _z(nv, pv, nc, pc)})
    # spike strength buckets
    sp = d[d.kind == "spike"].copy()
    sp["bucket"] = pd.cut(sp.ratio, [RATIO, 12, 20, 1e9], labels=["8–12×", "12–20×", "20×+"]).astype(str)
    c = d[d.kind == "control"]
    nc, pc = _rate(c)
    buckets = []
    for b, x in sp.groupby("bucket"):
        nv, pv = _rate(x)
        buckets.append({"bucket": b, "n": nv, "held": round(pv, 1) if pv is not None else None, "z": _z(nv, pv, nc, pc)})
    flow = d.groupby("kind").res.value_counts(normalize=True).mul(100).round(1).unstack(fill_value=0).reset_index().to_dict("records")
    return {"rows": rows, "buckets": buckets, "flow": flow, "n_days": int(d.day.nunique()), "first": str(d.day.min()), "last": str(d.day.max()),
            "n_levels": {k: int(v) for k, v in d.kind.value_counts().items()}, "params": {"away": AWAY, "decide": DECIDE, "horizon": HORIZON_SESSIONS, "ratio": RATIO}}


def main():
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "levels.parquet")
    if os.path.exists(p) and "--fresh" not in sys.argv:
        d = pd.read_parquet(p)
    else:
        fut, sp = load()
        lv = pick_levels(fut, sp)
        d = outcomes(lv, sp)
        d.to_parquet(p, index=False)
    return d, analyse(d)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    d, a = main()
    print(json.dumps(a, indent=1, default=str))


def sensitivity(scales=(0.0015, 0.003, 0.005)):
    """Re-score the saved levels at several move sizes (the levels themselves do not change)."""
    global AWAY, DECIDE
    lv = pd.read_parquet(os.path.join(OUT, "levels.parquet"))[["ts", "day", "kind", "ratio", "level"]]
    _, sp = load()
    out = {}
    for s in scales:
        AWAY = DECIDE = s
        out[f"{s * 100:.2f}%"] = analyse(outcomes(lv, sp))
    AWAY = DECIDE = 0.003
    return out


def report(res):
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(res, ensure_ascii=False, default=str)))
    return path


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Volume Levels Study</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--grp:#eef0f6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--grp:#23262e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
tr.grp td{background:var(--grp);font-weight:700;font-size:12px;color:var(--muted)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}
</style></head><body><div class="wrap">
<h1>Do high-volume futures minutes mark support / resistance?</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="card"><h2>What the data says</h2><ul>
<li><b>For the big turns the idea is about — no.</b> At a 0.3% (≈ 65 pt) and 0.5% (≈ 110 pt) scale, levels from high-volume minutes held as often as levels from random minutes of the same day (≈ 42–44% each), as support and as resistance, in both halves.</li>
<li><b>At a small scale there is a slight stickiness.</b> At 0.15% (≈ 35 pt), the session's single highest-volume minute held 51.5% of the time vs 46.7% for random levels (z ≈ 3), similar in both halves and on both sides. Price pauses a little there — a few tens of points — but that is too small and too close to a coin flip to trade after costs.</li>
<li><b>Bigger spikes are not better:</b> 20×-volume minutes behaved like 8–12× ones.</li>
<li><b>Generally, a re-tested level breaks more often than it holds</b> (≈ 43% held vs ≈ 47% broke at 0.3%), whatever made it — so a single nice-looking example on a chart says little.</li>
<li><b>Practical use:</b> a high-volume minute is fine to mark on the chart as context, not as a level to sell against or fade.</li>
</ul></div>
<div class="card"><h2>Held % — volume levels vs random levels of the same day</h2><div class="scroll"><table id="t"></table></div>
<p class="note">Held = after NIFTY came back to the level, it moved the chosen distance back on the side it came from before moving that far through the level. z = two-proportion test vs the random levels; |z| below ~2 is not reliable.</p></div>
<div class="card"><h2>Method &amp; limits</h2><ul>
<li>Continuous near-month NIFTY futures volume (Fyers expired-F&amp;O 1-min data + Arrow for listed contracts), NIFTY spot 1-min closes, Oct 2021 → Oct 2026.</li>
<li>Volume levels: each session's highest-volume minute ("day-top") and every minute with volume ≥ 8× the median of the previous ~375 minutes ("spike"); 09:15 and the last 5 minutes skipped. Random levels: 5 other minutes of the same session.</li>
<li>Return: within 10 sessions NIFTY first moves the chosen distance away, then trades back to the level. Levels never returned to are left out.</li>
</ul></div>
</div><script>
const D = /*DATA*/null, S = Object.keys(D);
const a = D[S[0]];
document.getElementById("meta").textContent = `${a.n_days.toLocaleString("en-IN")} sessions (${a.first} → ${a.last}) · ${a.n_levels["day-top"]} day-top levels, ${a.n_levels.spike} spike levels, ${a.n_levels.control} random levels.`;
let h = `<tr><th>Group</th><th>Level</th><th>n</th><th>Held</th><th>Random n</th><th>Random held</th><th>z</th></tr>`;
S.forEach(s => { h += `<tr class="grp"><td colspan="7">Move size ${s} (away and decide)</td></tr>`;
  D[s].rows.forEach(r => { h += `<tr><td>${r.group}</td><td>${r.kind}</td><td>${r.n}</td><td><b>${r.held}%</b></td><td>${r.ctrl_n}</td><td>${r.ctrl_held}%</td><td class="${r.z >= 2 ? "up" : r.z <= -2 ? "down" : ""}">${r.z}</td></tr>`; }); });
document.getElementById("t").innerHTML = h;
</script></body></html>"""
