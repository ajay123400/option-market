"""positional_compare_report.py -- one HTML page comparing the positional
variants (hedges, stops) from positional_bt.py.

Output: results/positional_bt/compare_report.html
"""
import json
import os
import sys

import pandas as pd

import paths

ROOT = os.path.join(paths.BASE_DIR, "results", "positional_bt")
SHOW = ["nohedge_nosl", "nohedge_dstop3x", "nohedge_dstop2x", "user_rules", "fixhedge200_nosl", "fixhedge500_nosl", "hedge500_nosl"]
COLORS = ["#0e9f6e", "#4f5fe0", "#1aa3b8", "#8a8f9c", "#e08a1e", "#c2419b", "#e0434a"]
GROUPS = [("No hedge", ["nohedge_nosl", "nohedge_dstop3x", "nohedge_dstop2x", "user_rules"]),
          ("Fixed hedge (bought at entry, never moved) · no SL", ["fixhedge200_nosl", "fixhedge300_nosl", "fixhedge400_nosl", "fixhedge500_nosl"]),
          ("Hedge moved with every roll · no SL (previous test)", ["hedge200_nosl", "hedge300_nosl", "hedge400_nosl", "hedge500_nosl"])]


def _rs(v):
    return ("-" if v < 0 else "") + "₹" + f"{abs(v):,.0f}"


def _c(v, fmt=_rs, b=False):
    cls = "up" if v > 0 else "down" if v < 0 else ""
    return f'<td class="{cls}">{"<b>" if b else ""}{fmt(v)}{"</b>" if b else ""}</td>'


WIDTH_GROUPS = [("Range-width filter · no hedge · no SL", ["w400_late", "w400_late_exp", "w400_day1", "w350_late", "w350_late_exp", "w350_day1"]),
                ("Reference", ["nohedge_nosl", "user_rules", "nohedge_dstop3x", "fixhedge200_nosl"])]
WIDTH_SHOW = ["w400_late", "w350_late", "w400_day1", "w350_day1", "nohedge_nosl", "user_rules", "fixhedge200_nosl"]
WIDTH_TEXT = {
    "title": "Positional range method — enter only on a wide range",
    "rules": """<b>Width filter</b><span>No entry while the range is narrow. <i>Day-1 only</i>: trade the week only if the Day-1 09:20 range is at least 350 / 400 pts wide. <i>Late entry</i>: otherwise keep watching; enter on the first leader change (Day 1 or any later day) whose new range is at least 350 / 400 wide — up to the day before expiry (or, "incl. expiry day", on expiry day too). No hedge, no stop loss.</span>""",
    "bullets": "{{AUTO}}",
}


TGT_GROUPS = [("Target 70% vs no target (hold every week to expiry) · no hedge · no SL",
               ["w400_late_notgt", "w400_late", "w350_late_notgt", "w350_late", "w400_day1_notgt", "w400_day1",
                "nohedge_nosl_notgt", "nohedge_nosl"])]
TGT_SHOW = ["w400_late_notgt", "w400_late", "w350_late_notgt", "w350_late", "nohedge_nosl_notgt", "nohedge_nosl"]
TGT_TEXT = {"title": "Positional range method — with vs without the 70% target",
            "rules": WIDTH_TEXT["rules"] + "<b>Target</b><span>Either exit at +70% of the entry premium (net of charges), or <b>no target</b>: every week is held to expiry and settled vs NIFTY's close (rolls unchanged).</span>",
            "extra": "{{TGT}}"}


NAR_GROUPS = [(f"{lab} · no hedge · no SL", [k, k + "_x200", k + "_x250"]) for k, lab in (
    ("w400_late", "≥ 400 late entry · target 70%"), ("w400_late_notgt", "≥ 400 late entry · no target"),
    ("w350_late", "≥ 350 late entry · target 70%"), ("w350_late_notgt", "≥ 350 late entry · no target"),
    ("nohedge_nosl", "Every week · target 70%"), ("nohedge_nosl_notgt", "Every week · no target"))]
NAR_SHOW = ["w400_late_notgt", "w400_late_notgt_x200", "w400_late", "w400_late_x200", "nohedge_nosl", "nohedge_nosl_x200"]
NAR_TEXT = {"title": "Positional range method — exit when the range shrinks to 200",
            "rules": WIDTH_TEXT["rules"] + "<b>Narrow exit</b><span>After entry, close everything as soon as a recalculated range (a leader change) is 200 points wide or less (250 tested as a check).</span>",
            "extra": "{{NAR}}"}


def _nar_bullets(data, weeks):
    b = []
    for base in ("w400_late_notgt", "w400_late", "nohedge_nosl"):
        a, x = data.get(base), data.get(base + "_x200")
        if not a or not x:
            continue
        diff = x["net_rs"] - a["net_rs"]
        b.append(f"<li><b>{a['config']['label']}</b>: {_rs(a['net_rs'])} → with the 200 exit <b class='{'up' if diff > 0 else 'down'}'>{_rs(x['net_rs'])}</b> "
                 f"({'+' if diff > 0 else ''}{_rs(diff)}); max drawdown {_rs(a['max_drawdown_rs'])} → {_rs(x['max_drawdown_rs'])}.</li>")
    w = weeks.get("w400_late_notgt_x200"), weeks.get("w400_late_notgt")
    if w[0] is not None and w[1] is not None:
        j = w[0][w[0].exit == "NARROW RANGE"].set_index("expiry").join(w[1].set_index("expiry"), rsuffix="_hold")
        b.append(f"<li><b>Why it hurts with the width filter:</b> the range naturally shrinks toward expiry (median ~213 on expiry day), so most "
                 f"200-exits happen on expiry morning and give up the last day's decay. In the ≥ 400 no-target version the exit fired in {len(j)} weeks; "
                 f"holding those weeks instead would have been better in {int((j.pnl_pts_hold > j.pnl_pts).sum())} of them.</li>")
    b.append("<li><b>Why it helps without the filter:</b> trading every week includes the narrow, losing weeks; the 200 exit gets out of many of them early "
             "— it acts like a crude width filter after entry. With the ≥ 400 entry filter already in place, it mostly just cuts winners short.</li>")
    b.append("<li>250 exits far more often and is worse everywhere — the result is sensitive to the threshold.</li>")
    return "".join(b)


TGT2_GROUPS = [(f"{lab} · no hedge · no SL", [k, k + "_t80", k + "_t90", k + "_t70pre", k + "_notgt"]) for k, lab in (
    ("w400_late", "≥ 400 late entry"), ("w350_late", "≥ 350 late entry"))]
TGT2_SHOW = ["w400_late_t90", "w400_late_t80", "w400_late", "w400_late_t70pre", "w400_late_notgt"]
TGT2_TEXT = {"title": "Positional range method — which profit target?",
             "rules": WIDTH_TEXT["rules"] + "<b>Targets</b><span>Exit when net P&amp;L after charges reaches 70% / 80% / 90% of the entry premium; "
                      "or 70% only until the day before expiry, then hold on expiry day; or no target (hold every week to expiry).</span>",
             "extra": "{{TGT2}}"}


def _tgt2_bullets(data, weeks):
    b = []
    for k in ("w400_late", "w350_late"):
        rows = []
        for suf, lab in (("", "70%"), ("_t80", "80%"), ("_t90", "90%"), ("_t70pre", "70% till Monday"), ("_notgt", "none")):
            w = weeks.get(k + suf)
            if w is None:
                continue
            rows.append((lab, w.net_rs.sum(), w[w.expiry < "2024-01-01"].net_rs.sum(), w[w.expiry >= "2024-01-01"].net_rs.sum()))
        if not rows:
            continue
        best = max(rows, key=lambda r: r[1])
        best_in = max(rows, key=lambda r: r[2])
        b.append(f"<li><b>{'≥ 400' if k == 'w400_late' else '≥ 350'} late entry:</b> best overall is target <b>{best[0]}</b> ({_rs(best[1])}); "
                 f"best in 2021–23 alone is <b>{best_in[0]}</b> ({_rs(best_in[2])}).</li>")
    b.append("<li><b>90% keeps most of the expiry-day decay</b> but still books the week when it is almost fully won — it beats 70% in both periods "
             "and has the lowest drawdown in the ≥ 400 set. \"70% until Monday, then hold\" was the best in 2021–23 but lagged in 2024–26; "
             "no target was the best only in 2024–26.</li>")
    return "".join(b)


BE_GROUPS = [("≥ 400 late entry · target 90% · no hedge · no SL", ["w400_late_t90", "beB50_w400", "beB100_w400", "beA50_w400", "beA100_w400", "beC_w400"]),
             ("Every week · target 90% · no hedge · no SL", ["nohedge_nosl_t90", "beB50_all", "beB100_all", "beA50_all", "beA100_all", "beC_all"])]
BE_SHOW = ["w400_late_t90", "beB50_w400", "beA100_w400", "beC_w400", "beB50_all", "nohedge_nosl_t90"]
BE_TEXT = {"title": "Positional range method — breakeven-based adjustments (A / B / C)",
           "rules": WIDTH_TEXT["rules"] + """<b>A · centred</b><span>Sell a straddle at the 50-pt strike nearest the range middle; when a recalculated range's middle is 50+ (or 100+) pts from the straddle, move both legs to the new middle.</span>
<b>B · guard</b><span>Current strikes (nearest UPPER / LOWER) and the usual roll toward the market; in addition, when a range edge comes within 50 (or 100) pts of that side's breakeven, roll that leg AWAY until the breakeven is 50 (100) pts beyond the edge again. Breakeven counts booked P&amp;L.</span>
<b>C · balanced</b><span>Sell the CE/PE pair whose breakevens are equally far outside the range; re-balance on every range change.</span>
<b>Exit</b><span>Target 90% of the entry premium (net of charges), otherwise held to expiry. No stop, no hedge.</span>""",
           "extra": "{{BE}}"}


def _be_bullets(data, weeks):
    b = []
    rows = []
    for k in ("w400_late_t90", "beB50_w400", "beB100_w400", "beA50_w400", "beA100_w400", "beC_w400"):
        w, s_ = weeks.get(k), data.get(k)
        if w is None:
            continue
        rows.append((k, " · ".join(s_["config"]["label"].split(" · ")[:2]), w[w.expiry < "2024-01-01"].net_rs.sum(),
                     w[w.expiry >= "2024-01-01"].net_rs.sum(), s_["net_rs"], s_["charges"], s_["avg_rolls"], s_["max_drawdown_rs"]))
    if rows:
        bi = max(rows, key=lambda r: r[2])
        bt = max(rows, key=lambda r: r[4])
        b.append(f"<li>With the ≥ 400 filter, the best in 2021–23 alone was <b>{bi[1]}</b> ({_rs(bi[2])}); its 2024–26 result: "
                 f"<b class='{'up' if bi[3] > 0 else 'down'}'>{_rs(bi[3])}</b>. Best over the whole period: <b>{bt[1]}</b> ({_rs(bt[4])}).</li>")
        b.append("<li>Rolls per week / charges: " + " · ".join(f"{r[1]}: {r[6]} rolls, {_rs(r[5])}" for r in rows) + ".</li>")
    tw = []
    for k in ("w400_late_t90", "beB50_w400", "beB100_w400", "beA50_w400", "beA100_w400", "beC_w400"):
        w = weeks.get(k)
        if w is not None:
            x = w[w.expiry == "2025-06-26"]
            if len(x):
                tw.append(f"{' · '.join(data[k]['config']['label'].split(' · ')[:2])}: {_rs(float(x.net_rs.iloc[0]))}")
    if tw:
        b.append("<li><b>The June 2025 trend week</b> (expiry 26 Jun 2025, NIFTY 24,860 at entry → 25,530 at expiry): " + " · ".join(tw) + ".</li>")
    return "".join(b)


def _tgt_bullets(weeks):
    a, b = weeks.get("w400_late"), weeks.get("w400_late_notgt")
    if a is None or b is None:
        return ""
    j = a.set_index("expiry").join(b.set_index("expiry"), rsuffix="_nt")
    t = j[j.exit == "TARGET"]
    return (f"<li><b>What holding does to the target weeks</b> (≥ 400 late entry): {len(t)} weeks hit the 70% target. Held to expiry instead, "
            f"{int((t.net_rs_nt > t.net_rs).sum())} ended better and {int((t.net_rs_nt < t.net_rs).sum())} worse; "
            f"{int((t.net_rs_nt < 0).sum())} of them turned into a loss. Net effect {_rs(t.net_rs_nt.sum() - t.net_rs.sum())}.</li>")


def _auto_bullets(data, weeks):
    b = []
    def g(v):
        return data.get(v), weeks.get(v)
    s0, w0 = g("nohedge_nosl")
    for v in ("w400_late", "w350_late", "w400_day1"):
        s, w = g(v)
        if not s:
            continue
        ins, outs = w[w.expiry < "2024-01-01"].net_rs.sum(), w[w.expiry >= "2024-01-01"].net_rs.sum()
        b.append(f"<li><b>{s['config']['label']}</b>: {s['weeks']} weeks traded, net <b class='up'>{_rs(s['net_rs'])}</b> "
                 f"(2021–23 {_rs(ins)}, 2024–26 {_rs(outs)}), win {s['win_pct']}%, max drawdown {_rs(s['max_drawdown_rs'])}.</li>")
    s, w = g("w400_late")
    if s is not None and "late_entry" in w:
        late = w[w.late_entry]
        b.append(f"<li><b>Late entries work too:</b> with the 400 filter, {len(late)} weeks were entered after Day-1 09:20 "
                 f"(a later leader change widened the range) — net <b class='up'>{_rs(late.net_rs.sum())}</b>, win {(late.net_rs > 0).mean() * 100:.0f}%. "
                 f"Most came on Day 1 itself; entries only 3–4 days before expiry added little.</li>")
    if s0:
        b.append(f"<li>For comparison, trading <b>every</b> week (no filter) made {_rs(s0['net_rs'])} with a {_rs(s0['max_drawdown_rs'])} drawdown: "
                 f"the narrow weeks the filter skips were, on balance, losing weeks.</li>")
    b.append("<li><b>Caution:</b> the 350 / 400 thresholds come from looking at this same data, the worst week (−₹36,755) is unchanged "
             "because it was a wide-range week, and naked positions still carry crash risk this 5-year sample does not contain.</li>")
    return "".join(b)


def _late_table(weeks):
    rows = []
    for v in ("w400_late", "w350_late"):
        w = weeks.get(v)
        if w is None or "late_entry" not in w:
            continue
        w = w.copy()
        ed = pd.to_datetime(w.entry_time.str[:10])
        w["dayno"] = [len(pd.bdate_range(a, b)) for a, b in zip(pd.to_datetime(w.day1), ed)]
        w["when"] = ["Day 1 · 09:20" if not l else f"Day {d} · after a leader change" for l, d in zip(w.late_entry, w.dayno)]
        for k, x in w.groupby("when", sort=True):
            rows.append(f"<tr><td>{v.replace('_late', ' late').replace('w', '≥ ')}</td><td>{k}</td><td>{len(x)}</td>"
                        f"<td>{(x.net_rs > 0).mean() * 100:.0f}%</td>{_c(x.pnl_pts.sum(), lambda z: f'{z:,.0f}')}{_c(x.net_rs.sum(), b=True)}</tr>")
    return "".join(rows)


def build(groups=None, show=None, text=None, out_name="compare_report.html"):
    groups, show = groups or GROUPS, show or SHOW
    data, weeks = {}, {}
    for v in sorted(os.listdir(ROOT)):
        f = os.path.join(ROOT, v, "summary.json")
        if os.path.isfile(f):
            data[v] = json.load(open(f, encoding="utf-8"))
            w = pd.read_csv(os.path.join(ROOT, v, "weeks.csv"))
            w["year"] = w.expiry.str[:4]
            weeks[v] = w

    # comparison table
    rows = []
    for g, keys in groups:
        rows.append(f'<tr class="grp"><td colspan="12">{g}</td></tr>')
        for v in keys:
            if v not in data:
                continue
            s, w = data[v], weeks[v]
            ins, outs = w[w.expiry < "2024-01-01"].net_rs.sum(), w[w.expiry >= "2024-01-01"].net_rs.sum()
            best = v == "nohedge_nosl"
            rows.append(f"<tr{' class=best' if best else ''}><td>{s['config']['label']}</td><td>{s['weeks']}</td><td>{s['win_pct']}%</td>"
                        f"<td>{s['avg_premium']}</td>{_c(s['points'], lambda x: f'{x:,.0f}')}<td>{_rs(s['charges'])}</td>"
                        f"{_c(ins)}{_c(outs)}{_c(s['net_rs'], b=True)}<td class='down'>{_rs(s['max_drawdown_rs'])}</td>"
                        f"<td class='down'>{_rs(s['worst']['net_rs'])} <span class='note'>{s['worst']['expiry']}</span></td>"
                        f"<td>{int((w.net_rs < -20000).sum())}</td></tr>")
    cmp_t = "".join(rows)

    # by year
    years = sorted(set().union(*[set(w.year) for w in weeks.values()]))
    yr = ["<tr><th>Variant</th>" + "".join(f"<th>{y}</th>" for y in years) + "</tr>"]
    for v in show:
        if v in weeks:
            g = weeks[v].groupby("year").net_rs.sum()
            yr.append(f"<tr><td>{data[v]['config']['label']}</td>" + "".join(_c(float(g.get(y, 0))) for y in years) + "</tr>")
    yr_t = "".join(yr)

    # disaster-stop weeks vs the same weeks without a stop
    stop_t = ""
    if "nohedge_dstop3x" in weeks and "nohedge_nosl" in weeks:
        a = weeks["nohedge_nosl"].set_index("expiry")
        b = weeks["nohedge_dstop3x"].set_index("expiry")
        j = b[b.exit == "STOP LOSS"].join(a[["pnl_pts", "net_rs"]], rsuffix="_ns")
        stop_t = "".join(f"<tr><td>{e}</td>{_c(r.pnl_pts, lambda x: f'{x:,.0f}')}{_c(r.pnl_pts_ns, lambda x: f'{x:,.0f}')}"
                         f"{_c(r.pnl_pts - r.pnl_pts_ns, lambda x: f'{x:+,.0f}')}</tr>" for e, r in j.iterrows())
        stop_t += (f"<tr class='tot'><td>{len(j)} stopped weeks</td>{_c(j.pnl_pts.sum(), lambda x: f'{x:,.0f}')}"
                   f"{_c(j.pnl_pts_ns.sum(), lambda x: f'{x:,.0f}')}{_c((j.pnl_pts - j.pnl_pts_ns).sum(), lambda x: f'{x:+,.0f}')}</tr>")

    # hedge cost: points by leg
    leg_rows = []
    for v in ("nohedge_nosl", "fixhedge200_nosl", "fixhedge500_nosl", "hedge500_nosl"):
        f = os.path.join(ROOT, v, "trades.csv")
        if not os.path.isfile(f):
            continue
        t = pd.read_csv(f)
        if "side" not in t:
            t["side"] = "SELL"
        g = t.groupby(["side", "type"]).pnl_pts.sum()
        leg_rows.append(f"<tr><td>{data[v]['config']['label']}</td>" + "".join(
            _c(float(g.get(k, 0)), lambda x: f"{x:,.0f}") for k in (("SELL", "CE"), ("SELL", "PE"), ("BUY", "CE"), ("BUY", "PE")))
            + _c(float(g.sum()), lambda x: f"{x:,.0f}", b=True) + "</tr>")
    leg_t = "".join(leg_rows)

    series = []
    for i, v in enumerate(show):
        if v in weeks:
            w = weeks[v]
            series.append({"label": data[v]["config"]["label"], "color": COLORS[i],
                           "pts": [[e, round(c)] for e, c in zip(w.expiry, w.net_rs.cumsum())]})
    page = TEMPLATE
    if text:
        a, b = page.index('<div class="card"><h2>Common rules</h2>'), page.index('<div class="card"><h2>All variants</h2>')
        page = (page[:a] + '<div class="card"><h2>Common rules</h2><div class="rules">'
                '<b>Entry</b><span>Sell the 50-pt strikes nearest UPPER / LOWER of the volume-leader range (UPPER = CE leader + PE leader\'s first 5-min high, LOWER = PE leader − CE leader\'s first 5-min high).</span>'
                '<b>Range &amp; rolls</b><span>Range recalculated only on a leader change (5-min checks). Call rolls down to the strike nearest the new UPPER, put rolls up to the strike nearest the new LOWER; never away; at most 100 pts past a straddle.</span>'
                '<b>Exit</b><span>Target: net P&amp;L after charges ≥ 70% of the entry premium. Otherwise held to expiry and settled vs NIFTY\'s close.</span>'
                + text["rules"] + '</div></div><div class="card"><h2>What the test says</h2><ul>' + _auto_bullets(data, weeks) + (_tgt_bullets(weeks) if text.get("extra") == "{{TGT}}" else _nar_bullets(data, weeks) if text.get("extra") == "{{NAR}}" else _tgt2_bullets(data, weeks) if text.get("extra") == "{{TGT2}}" else _be_bullets(data, weeks) if text.get("extra") == "{{BE}}" else text.get("extra", "")) + '</ul></div>'
                + ('<div class="card"><h2>When the late-entry weeks were entered</h2><div class="scroll"><table><tr><th>Filter</th><th>Entered</th><th>Weeks</th><th>Win %</th><th>Points</th><th>Net ₹</th></tr>'
                   + _late_table(weeks) + '</table></div></div>')
                + page[b:])
        page = page.replace("Positional range method — fixed hedge &amp; disaster stop", text["title"]).replace(
            "<title>Positional Hedge & Stop Test</title>", "<title>Wide-Range Entry Test</title>")
    page = (page.replace("{{CMP}}", cmp_t).replace("{{YEAR}}", yr_t).replace("{{STOP}}", stop_t)
            .replace("{{LEG}}", leg_t).replace("/*SERIES*/", json.dumps(series)))
    out = os.path.join(ROOT, out_name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Positional Hedge & Stop Test</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--best:#e9f8f1;--grp:#eef0f6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--best:#16302a;--grp:#23262e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1360px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
tr.best td{background:var(--best)}tr.grp td{background:var(--grp);font-weight:700;font-size:12px;color:var(--muted)}tr.tot td{font-weight:700;border-top:2px solid var(--border)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:1000px){.grid2{grid-template-columns:1fr}}
.rules{display:grid;grid-template-columns:140px 1fr;gap:6px 14px;font-size:13.5px}.rules b{color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;margin-top:6px}.legend i{display:inline-block;width:14px;height:3px;margin-right:6px;vertical-align:middle}
svg text{fill:var(--muted);font-size:11px}ul{margin:0;padding-left:18px}li{margin:5px 0}
</style></head><body><div class="wrap">
<h1>Positional range method — fixed hedge &amp; disaster stop</h1>
<p class="note" style="margin-bottom:16px">~255 weekly expiries, Sep 2021 → Sep 2026 · 1 lot (lot size of that time) · 1-min data · 0.5 pt slippage per order · all charges.</p>
<div class="card"><h2>Common rules</h2><div class="rules">
<b>Entry</b><span>Day 1 (first day after the previous expiry) 09:20: sell the 50-pt strikes nearest UPPER / LOWER of the volume-leader range.</span>
<b>Range &amp; rolls</b><span>Range recalculated only on a leader change (5-min checks). Call rolls down to the strike nearest the new UPPER, put rolls up to the strike nearest the new LOWER; never away; at most 100 pts past a straddle.</span>
<b>Exit</b><span>Target: net P&amp;L after charges ≥ 70% of the entry premium (net credit when hedged). Otherwise held to expiry and settled vs NIFTY's close.</span>
<b>Fixed hedge</b><span>Buy a CE 200–500 pts above the sold call and a PE the same distance below the sold put at entry; never moved (only the sold legs roll). No stop.</span>
<b>Disaster stop</b><span>No hedge; close everything only when the loss (booked + open) reaches 2× or 3× the entry premium.</span>
</div></div>
<div class="card"><h2>What the test says</h2><ul>
<li><b>No hedge, no stop stays the best</b> in both periods. Every stop or hedge tested here lowered the 5-year result.</li>
<li><b>A fixed hedge is much cheaper than a moving one</b> (charges ≈ ₹93k vs ₹1.45 lakh) but still loses: the wings cost 24 pts a week at 500 pts (≈ 8 CE + 16 PE — puts are dear) and 200-pt wings eat half the premium. Over ~250 weeks that is more than the method earns, and big breakouts that the wings pay for were rare.</li>
<li><b>The fixed 200-pt hedge does cut the tail:</b> 4 weeks worse than −₹20k instead of 11, and a smaller 5% worst-case — at the cost of ₹3.2 lakh of profit.</li>
<li><b>A disaster stop mostly sells the low:</b> at 3× it fired in 15 weeks; only 4 of them ended better than holding. It also did not stop the worst week (7 Apr 2025 gapped straight through it).</li>
<li><b>Caution:</b> the no-stop, no-hedge result leans on 2025; 2023 and 2024 were losing years, and 5 years contain no Covid-style crash. Naked short options can lose far more than the worst week seen here.</li>
</ul></div>
<div class="card"><h2>All variants</h2><div class="scroll"><table>
<tr><th>Variant</th><th>Weeks</th><th>Win %</th><th>Premium</th><th>Points</th><th>Charges</th><th>2021–23</th><th>2024–26</th><th>Net ₹</th><th>Max DD</th><th>Worst week</th><th>Weeks &lt; −₹20k</th></tr>
{{CMP}}</table></div></div>
<div class="card"><h2>Cumulative net ₹</h2><div id="eq"></div><div class="legend" id="lg"></div></div>
<div class="card"><h2>Net ₹ by year</h2><div class="scroll"><table>{{YEAR}}</table></div></div>
<div class="grid2">
<div class="card"><h2>Where the points came from (per leg, whole period)</h2><div class="scroll"><table>
<tr><th>Variant</th><th>Sold CE</th><th>Sold PE</th><th>Hedge CE</th><th>Hedge PE</th><th>Total</th></tr>{{LEG}}</table></div>
<p class="note">Hedge columns are what the bought wings made or cost in total.</p></div>
<div class="card"><h2>Disaster stop 3×: stopped weeks vs holding</h2><div class="scroll"><table>
<tr><th>Expiry</th><th>With stop (pts)</th><th>Held (pts)</th><th>Stop helped</th></tr>{{STOP}}</table></div></div>
</div>
</div><script>
const SERIES = /*SERIES*/;
(function () {
  const W = 1300, H = 340, pl = 80, pr = 12, pt = 12, pb = 26;
  const all = SERIES.flatMap(s => s.pts.map(p => p[0])).sort();
  const t0 = new Date(all[0]).getTime(), t1 = new Date(all[all.length - 1]).getTime();
  const vals = SERIES.flatMap(s => s.pts.map(p => p[1])).concat([0]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = d => pl + (new Date(d).getTime() - t0) / (t1 - t0) * (W - pl - pr);
  const y = v => pt + (hi - v) / (hi - lo || 1) * (H - pt - pb);
  const rs = v => (v < 0 ? "-" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
  const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 4 || 1))) * 2;
  const ticks = []; for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  const years = []; for (let yv = new Date(all[0]).getFullYear() + 1; yv <= new Date(all[all.length - 1]).getFullYear(); yv++) years.push(yv);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Cumulative net">`;
  svg += ticks.map(v => `<line x1="${pl}" x2="${W - pr}" y1="${y(v)}" y2="${y(v)}" stroke="var(--border)"${v === 0 ? ' stroke-width="2"' : ""}/><text x="${pl - 6}" y="${y(v) + 4}" text-anchor="end">${rs(v)}</text>`).join("");
  svg += years.map(yv => { const xx = x(yv + "-01-01"); return `<line x1="${xx}" x2="${xx}" y1="${pt}" y2="${H - pb}" stroke="var(--border)" stroke-dasharray="3,3"/><text x="${xx + 4}" y="${H - 8}">${yv}</text>`; }).join("");
  svg += SERIES.map(s => `<path d="${s.pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("")}" fill="none" stroke="${s.color}" stroke-width="1.8"/>`).join("");
  document.getElementById("eq").innerHTML = svg + "</svg>";
  document.getElementById("lg").innerHTML = SERIES.map(s => `<span><i style="background:${s.color}"></i>${s.label} (${rs(s.pts[s.pts.length - 1][1])})</span>`).join("");
})();
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print(build())
    print(build(WIDTH_GROUPS, WIDTH_SHOW, WIDTH_TEXT, "width_report.html"))
    print(build(TGT_GROUPS, TGT_SHOW, TGT_TEXT, "target_report.html"))
    print(build(NAR_GROUPS, NAR_SHOW, NAR_TEXT, "narrow_exit_report.html"))
    print(build(TGT2_GROUPS, TGT2_SHOW, TGT2_TEXT, "target_levels_report.html"))
    print(build(BE_GROUPS, BE_SHOW, BE_TEXT, "breakeven_report.html"))
