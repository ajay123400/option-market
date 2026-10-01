"""Confirms that Stage 2D Gate 1 touched nothing that existed at the baseline commit.

Compares the working tree with BASE (default: the approved baseline HEAD) for every tracked path, and lists everything new. Pass criteria:
no tracked file modified/deleted/renamed, and every untracked/new path lies under the Stage 2D locations.
Usage: python research_output/stage2d/gate1/validation/immutability_check.py [BASE]
"""
import subprocess
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "3b2a8218e5aeb8b394705e3f1c3b7b9290211403"
ALLOWED_PREFIXES = ("optionsengine/research/stage2d/", "research_output/stage2d/", "tests/test_stage2d_")
PROTECTED = ["app.py", "templates", "optionsengine/greeks.py", "iv_surface_research.py", "optionsengine/surface.py", "optionsengine/research/loaders.py",
             "optionsengine/research/build_surface.py", "optionsengine/research/report.py", "optionsengine/research/session_calendar.py",
             "optionsengine/research/spot_quality.py", "optionsengine/research/realized_vol.py", "optionsengine/research/build_rv.py",
             "optionsengine/research/iv_rv.py", "optionsengine/research/iv_rv_stats.py", "optionsengine/research/build_iv_rv.py",
             "optionsengine/research/annualization_audit.py", "optionsengine/research/build_annualization_audit.py",
             "research_output/stage2a", "research_output/stage2b", "research_output/stage2c", "research_output/.gitignore"]


def run(*a):
    return subprocess.run(a, capture_output=True, text=True, check=True).stdout


head = run("git", "rev-parse", "HEAD").strip()
print(f"baseline {BASE}\nHEAD     {head}  ({'same' if head == BASE else 'DIFFERENT'})")
changed = run("git", "diff", "--name-status", BASE).splitlines()                    # tracked files, working tree vs baseline
print(f"tracked files changed vs baseline: {len(changed)}")
for c in changed:
    print("   ", c)
status = [l for l in run("git", "status", "--porcelain", "--untracked-files=all").splitlines()]
new = [l[3:] for l in status if l.startswith("??")]
outside = [p for p in new if not p.startswith(ALLOWED_PREFIXES)]
print(f"untracked new files: {len(new)}  (outside the Stage 2D paths: {len(outside)})")
for p in outside:
    print("    OUTSIDE:", p)
prot = run("git", "diff", "--stat", BASE, "--", *PROTECTED).strip()
print("protected / Stage 2A-2B-2C paths diff vs baseline:", prot or "none")
ok = head == BASE and not changed and not outside and not prot
print("\nRESULT:", "IMMUTABLE (only new Stage 2D files exist)" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)
