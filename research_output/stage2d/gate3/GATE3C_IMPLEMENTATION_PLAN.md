# Stage 2D — Gate 3c (T12): implementation plan (written before any Gate 3c code)

Baseline: HEAD `f4ff9995f5330b5dccf86cd69d1c3ba5dafee909` (T11), over Gate 3a `b2a7584`, Gate 2 `b29dc2c`, Gate 1 `f2a9a2f`; only Stage 2C is pushed.
Scope: T12 only — claim ledger, consolidated final report, language lint, prospective-test protocol. **No new statistics, no strategy, no P&L, no signals, no sizing, no Fyers link.**
Gates 1, 2, 3a and 3b files are read, never edited. All outputs are new files under Stage 2D paths. Nothing is committed or pushed.

## Deliverables
1. **Claim ledger** (`claim_ledger.csv`, rendered `claim_ledger.md`): every claim Stage 2D may or may not make, with status, evidence tier, the numbers behind it (read from the committed
   CSVs by code, never typed), sources, required warnings, allowed wording and forbidden wording. Evidence tiers are the Gate 2 pre-registration tiers (0 descriptive; 1 statistical evidence; 2 tradable edge: not assessable).
2. **T11 warning rule (structural, enforced by lint and tests):** every claim whose evidence is an interval/p-value for a ratio estimand (R1, R1e, R2, R2e, R3) must carry an explicit under-coverage warning quoting the synthetic coverage (88.7–91.7% in B0, b = 5; materially worse under strong dependence);
   every interval-based claim states that synthetic coverage is not extrapolated to the real data; every claim relying on row-level inference is marked invalid.
3. **Language lint** (`claim_lint.py`, `lint_report.csv`): forbidden vocabulary (edge, profit, tradable, alpha, arbitrage, risk premium, proves, guarantee, out-of-sample for the holdout, "significant" without a statistical qualifier, 95% described as guaranteed) is flagged unless the sentence negates or limits it.
   Run over the ledger, final report, protocol and all committed Stage 2D reports. Findings in committed gate reports are reported, not edited (those files are frozen).
4. **Consolidated final report** (`FINAL_REPORT.md`), generated from the ledger and the source CSVs so numbers cannot drift.
5. **Prospective-test protocol** (`PROSPECTIVE_PROTOCOL.md`): the frozen specification for expiries after 2026-09-30, SHA-256 recorded; **no prospective result is computed, simulated or fabricated.**
6. Tests (`tests/test_stage2d_gate3c.py`), mutation checks, an independent recomputation of every ledger number, determinism rerun, immutability check, full pytest.

## Rules for the ledger
* A claim's status is one of SUPPORTED_WITH_CAVEATS, SUPPORTED_METHOD (a statement about the statistical procedures), INCONCLUSIVE, NOT_SUPPORTED, NOT_ASSESSABLE, NOT_CLAIMED.
* Tier 1 is assigned only if: interval excludes the null under the expiry-block method, sign same in dev and holdout, in every expiry year, and in every specification; and T9 shows no detectable outcome-dependent selection. A claim failing any condition is Tier 0 or lower, whatever its p-value.
* "Statistically supported" never becomes "tradable": all tradability, premium and edge claims are NOT_ASSESSABLE or NOT_CLAIMED with the missing information named (bid/ask, costs, margin, execution, prospective data).
