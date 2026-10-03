"""Mutation check for Gate 3c: applies each fault to a COPY of the repository and runs tests/test_stage2d_gate3c.py (usage: python mutation_check_gate3c.py WORKDIR from the repo root)."""
import os, shutil, subprocess, sys
SRC = os.getcwd()
W = sys.argv[1]
if os.path.exists(W): shutil.rmtree(W)
os.makedirs(W)
for d in ("optionsengine", "tests", "research_output"):
    shutil.copytree(os.path.join(SRC, d), os.path.join(W, d), ignore=shutil.ignore_patterns("__pycache__", "variants"))
S = "optionsengine/research/stage2d/"
M = [
 ("lint: negation word 'not' removed", S+"claim_lint.py", 'NEG = re.compile(r"\\b(not|no|', 'NEG = re.compile(r"\\b(no|'),
 ("lint: out-of-sample always allowed", S+"claim_lint.py", 're.compile(NEG.pattern + "|" + PRIOR.pattern, re.I), "holdout is previously viewed', 're.compile(".*"), "holdout is previously viewed'),
 ("lint: forbidden-wording lines linted", S+"claim_lint.py", 'if line.lstrip().startswith("**Forbidden wording.**"):', 'if False:'),
 ("lint: significance always allowed", S+"claim_lint.py", 'STAT = re.compile(r"statistic|', 'STAT = re.compile(r".*|statistic|'),
 ("rule: under-coverage check inverted", S+"claim_lint.py", 'and "under-cover" not in w and', 'and "under-cover" in w and'),
 ("rule: transferability check dropped", S+"claim_lint.py", 'if c.interval_based and "not transferable" not in w', 'if False and "not transferable" not in w'),
 ("rule: tier 1 without conditions allowed", S+"claim_lint.py", 'if c.tier == "1" and not all(tier1_ok.values()):', 'if False:'),
 ("tier1: interval condition or->and", S+"claims.py", '(row.ci_lo > null or row.ci_hi < null)', '(row.ci_lo > null and row.ci_hi < null)'),
 ("tier1: dev/holdout sign test inverted", S+"claims.py", '(t5.dev_estimate - null) * (t5.holdout_estimate - null) > 0', '(t5.dev_estimate - null) * (t5.holdout_estimate - null) < 0'),
 ("tier1: p threshold flipped", S+"claims.py", 'od.p_permutation_two_sided > 0.05', 'od.p_permutation_two_sided < 0.05'),
 ("t11: ratio rows pick spreads", S+"claims.py", 'rat = b0[b0.estimand.str.startswith("R")].coverage', 'rat = b0[b0.estimand.str.startswith("S")].coverage'),
 ("C05 width ratio inverted", S+"claims.py", 'nar = float(s.val(G1, "width", estimand=sc, method="expiry_moving_block_b5")) / float(s.val(G1, "width", estimand=sc, method="row_iid_NAIVE_reference"))', 'nar = float(s.val(G1, "width", estimand=sc, method="row_iid_NAIVE_reference")) / float(s.val(G1, "width", estimand=sc, method="expiry_moving_block_b5"))'),
 ("C10 Manski bounds swapped", S+"claims.py", '[{float(mc.manski_lower):.3f}, {float(mc.manski_upper):.3f}]', '[{float(mc.manski_upper):.3f}, {float(mc.manski_lower):.3f}]'),
 ("C04 status upgraded", S+"claims.py", 'C("C04", "Ratio of summed variances (R3)", "NOT_SUPPORTED"', 'C("C04", "Ratio of summed variances (R3)", "SUPPORTED_WITH_CAVEATS"'),
 ("C07 status upgraded", S+"claims.py", '"INCONCLUSIVE", "0",', '"SUPPORTED_WITH_CAVEATS", "0",'),
 ("builder: stage2d guard removed", S+"build_gate3c.py", 'if "stage2d" not in os.path.normpath(out).split(os.sep):', 'if False:'),
 ("builder: ratio warning column dropped", S+"build_gate3c.py", 'RATIO INTERVAL UNDER-COVERS', 'RATIO NOTE'),
]
res = []
for name, f, a, b in M:
    p = os.path.join(W, f)
    s = open(p).read()
    if a == b:
        res.append((name, "EQUIVALENT (not applied)")); continue
    if a not in s:
        res.append((name, "PATTERN NOT FOUND")); continue
    open(p, "w").write(s.replace(a, b, 1))
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_stage2d_gate3c.py", "-x", "-q", "-p", "no:cacheprovider"], cwd=W, capture_output=True, text=True)
    res.append((name, "KILLED" if r.returncode != 0 else "SURVIVED"))
    open(p, "w").write(s)
for n, r in res: print(f"{r:28s} {n}")
