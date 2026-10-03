"""Stage 2D Gate 3c (T12): language lint for claims and reports.

A forbidden term is a finding unless its sentence negates or limits it (e.g. "not evidence of a tradable edge"). Statuses: FLAG (needs rewording), NEGATED_OK (term present but negated/limited).
Structural ledger rules (`check_ledger`): ratio-estimand interval claims carry the T11 under-coverage warning; interval claims say synthetic coverage is not transferable; no Tier 1 without all pre-registered conditions.
"""
import os
import re
from typing import Dict, Iterable, List, Sequence

NEG = re.compile(r"\b(not|no|never|nor|cannot|without|neither|none|unassessable|unknown|forbidden|does not|did not|do not|n't)\b|NOT_|not assessable|n/a", re.I)
STAT = re.compile(r"statistic|holm|p\s*[=<≈]|p-value|bootstrap|interval|permutation|wald|sign test", re.I)
TECH_EDGE = re.compile(r"(bucket|interval|band|bound|boundary|closed|DTE|coverage|slot|segment)\s+edges?|edges?\s+(of|case)", re.I)
PRIOR = re.compile(r"previously viewed|untouched|reserved|prospective|never", re.I)

RULES = [
    ("edge_profit", re.compile(r"\b(edge|profitable|profitability|profits?|alpha|tradable|tradeable|arbitrage|money-making)\b", re.I), re.compile(NEG.pattern + "|" + TECH_EDGE.pattern, re.I), "tradability / profit vocabulary needs negation or limitation"),
    ("risk_premium", re.compile(r"\b(risk premium|variance risk premium|VRP|compensation for risk)\b", re.I), NEG, "premium claim needs negation or limitation"),
    ("proof", re.compile(r"\b(proves?|proven|proof|guarantee[sd]?|(confirms?|confirmed|validates?|validated)\s+(the\s+)?(premium|effect|edge|ratio|result|finding|hypothesis|strategy))\b", re.I), NEG, "overclaiming verb"),
    ("out_of_sample", re.compile(r"out[- ]of[- ]sample", re.I), re.compile(NEG.pattern + "|" + PRIOR.pattern, re.I), "holdout is previously viewed, never out-of-sample"),
    ("significance", re.compile(r"\bsignifican(t|tly|ce)\b", re.I), STAT, "'significant' without a statistical qualifier"),
    ("mispricing", re.compile(r"\b(overpriced|underpriced|expensive|cheap|mispriced)\b", re.I), NEG, "price-level judgement about options"),
    ("nominal_coverage", re.compile(r"\b(exact|guaranteed|true)\s+95\s*%", re.I), NEG, "95% is nominal, never guaranteed"),
]


def sentences(text: str) -> List[str]:
    out = []
    for block in re.split(r"\n+", text):
        out += [x.strip() for x in re.split(r"(?<=[.!?;])\s+", block) if x.strip()]
    return out


def lint_text(text: str, source: str = "") -> List[Dict]:
    res = []
    for i, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("**Forbidden wording.**"):
            continue                                              # the forbidden phrases are listed there on purpose
        for sent in sentences(line):
            for rid, pat, allow, msg in RULES:
                if pat.search(sent):
                    res.append(dict(source=source, line=i, rule=rid, status="NEGATED_OK" if allow.search(sent) else "FLAG", message=msg, sentence=sent[:300]))
    return res


def lint_files(paths: Iterable[str]) -> List[Dict]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out += lint_text(f.read(), p)
    return out


def check_ledger(claims: Sequence, tier1_ok: Dict[str, bool]) -> List[str]:
    """Structural rules; returns a list of violations (empty = clean)."""
    bad = []
    for c in claims:
        w = c.warnings.lower()
        if any(e in c.estimands for e in ("R1", "R1e", "R2", "R2e", "R3")) and c.interval_based and "under-cover" not in w and "covered the true value of the ratio" not in w:
            bad.append(f"{c.claim_id}: ratio-estimand interval claim lacks the T11 under-coverage warning")
        if c.interval_based and "not transferable" not in w and "not extrapolated" not in w:
            bad.append(f"{c.claim_id}: interval claim lacks 'synthetic coverage is not transferable'")
        if c.tier == "1" and not all(tier1_ok.values()):
            bad.append(f"{c.claim_id}: Tier 1 without all pre-registered conditions")
        if c.status in ("NOT_ASSESSABLE", "NOT_CLAIMED") and c.tier == "1":
            bad.append(f"{c.claim_id}: not-assessable claim cannot be Tier 1")
        for f in ("claim", "allowed"):
            for r in lint_text(getattr(c, f), c.claim_id):
                if r["status"] == "FLAG":
                    bad.append(f"{c.claim_id}.{f}: {r['rule']}: {r['sentence']}")
        for r in lint_text(c.warnings + " " + c.key_numbers, c.claim_id):
            if r["status"] == "FLAG":
                bad.append(f"{c.claim_id}.warnings/key_numbers: {r['rule']}: {r['sentence']}")
    return bad
