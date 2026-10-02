import os, shutil, subprocess, sys
SRC = os.getcwd()
W = sys.argv[1]
if os.path.exists(W): shutil.rmtree(W)
os.makedirs(W)
for d in ("optionsengine", "tests"):
    shutil.copytree(os.path.join(SRC, d), os.path.join(W, d), ignore=shutil.ignore_patterns("__pycache__"))
S = "optionsengine/research/stage2d/"
M = [
 ("transition kernel half-weight", S+"synth_process.py", "M[i, j] = 0.5 * min(1.0, pi[j] / pi[i])", "M[i, j] = 0.5 * min(1.0, pi[i] / pi[j])"),
 ("kernel eps mixing", S+"synth_process.py", "+ dgp.eps * np.outer(np.ones(K), pi)", "+ dgp.eps * np.outer(np.ones(K), pi) * 0.5"),
 ("mean_after slot0", S+"synth_process.py", "0: sum(dgp.seg_mean[1:])", "0: sum(dgp.seg_mean[2:])"),
 ("to_aligned iv", S+"synth_process.py", "iv = 100.0 * np.sqrt(W / T_years)", "iv = 100.0 * np.sqrt(W * T_years)"),
 ("to_aligned rv_session", S+"synth_process.py", "np.sqrt(252.0 * V / d.session_equivalents", "np.sqrt(365.0 * V / d.session_equivalents"),
 ("to_aligned bucket", S+"synth_process.py", 'd.T_days <= 3, "1-3d"', 'd.T_days <= 4, "1-3d"'),
 ("wilson centre", S+"coverage_study.py", "c = p + z * z / (2 * n)", "c = p"),
 ("coverage inclusive", S+"coverage_study.py", "(lo[ok, i] <= t) & (t <= hi[ok, i])", "(lo[ok, i] < t) & (t < hi[ok, i])"),
 ("coverage band flag", S+"coverage_study.py", "0.93 <= k / nn <= 0.97", "0.90 <= k / nn <= 0.97"),
 ("undercovers flag", S+"coverage_study.py", "undercovers=bool(w[1] < 0.93)", "undercovers=bool(w[0] < 0.93)"),
 ("truth_difference sign", S+"coverage_study.py", "a1 - a0", "a0 - a1"),
 ("pipeline gcum offset", S+"synth_pipeline.py", "gcum[e + 1] - gcum[i + 1]", "gcum[e + 1] - gcum[i]"),
 ("recovery ratio truth", S+"synth_pipeline.py", "ratio_true=a.W_true / a.V_true", "ratio_true=a.W_true / a.V_true * 1.0 + 0.5"),
 ("truth_aligned rv", S+"build_gate3b.py", "rs = 100 * np.sqrt(252.0 * truth.V_true / truth.session_equivalents)", "rs = 100 * np.sqrt(365.0 * truth.V_true / truth.session_equivalents)"),
 ("stage2d refusal", S+"build_gate3b.py", 'if "stage2d" not in', 'if False and "stage2d" not in'),
]
res = []
for name, f, a, b in M:
    p = os.path.join(W, f)
    s = open(p).read()
    if a == b:
        continue
    if a not in s:
        res.append((name, "PATTERN NOT FOUND")); continue
    open(p, "w").write(s.replace(a, b, 1))
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_stage2d_gate3b.py", "-x", "-q", "-p", "no:cacheprovider"], cwd=W, capture_output=True, text=True)
    res.append((name, "KILLED" if r.returncode != 0 else "SURVIVED"))
    open(p, "w").write(s)
for n, r in res: print(f"{r:10s} {n}")
