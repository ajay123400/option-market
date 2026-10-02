# Stage 2D — Gate 2 (T5–T8) implementation plan

Baseline: HEAD `f2a9a2f` (Gate 1 committed locally, not pushed); parent `3b2a821` is the pushed Stage 2C. Scope: T5 dev-vs-holdout inference and forking-paths ledger, T6 strata with heterogeneity tests,
T7 convention sensitivity, T8 filter sensitivity. Not in scope: T9–T12, any strategy/P&L/signal/Fyers work. The frozen design is in `PREREGISTRATION.md`.

## New code (all new files; nothing existing is edited)
* `optionsengine/research/stage2d/contrasts.py` — split contrasts (independent expiry-block bootstraps per split), post-stratified contrast, Holm adjustment.
* `optionsengine/research/stage2d/strata.py` — stratum estimates, joint expiry-block bootstrap Wald heterogeneity test, Mincer–Zarnowitz slope.
* `optionsengine/research/stage2d/alt_rv.py` — RV recomputation for 5/15-minute sampling and the intraday-variance-profile partial-session weight, reading the 1-min spot through the existing Stage 2B session index (read-only).
* `optionsengine/research/stage2d/spec_curve.py` — runs the Stage 2C/audit builders on variant inputs (rate, forward gate, stale age, DTE cap) into `research_output/stage2d/gate2/variants/<name>/` and assembles the specification curve.
* `optionsengine/research/stage2d/build_gate2.py` — orchestrator + CLI; asserts baseline reproduction (Gate 1 numbers) and records the pre-registration SHA-256.

## Compute
Stage 2A re-runs for r = 5.5% and 7.5% (≈ 3 min each, 4 cores); Stage 2C + audit builders per variant (≈ 2 min); about 8 variant pipelines in total (2 rates, 2 extra stale ages reuse existing smiles, forward-gate variants reuse existing smiles, DTE ≤ 21). Total ≈ 25–30 min; disk use ≈ 1–2 GB under a gitignored variants directory.

## Tests (deterministic)
Hand-computed contrasts and post-stratification weights; Holm adjustment known values; Wald test calibration on synthetic strata (equal strata ⇒ rejection rate ≈ α; shifted stratum ⇒ high power; correlated strata); Mincer–Zarnowitz recovery of a known slope; 5/15-minute RV equal to hand values on synthetic sessions and equal to the 1-min value at step 1; partial-session weight reduces to uniform for a flat profile;
no-look-ahead (every variant uses only information at or before the snapshot for features and only later bars for targets; alternate RV windows start strictly after the observation point); variant builders never write outside `stage2d`; baseline specification in the curve reproduces Gate 1 and Stage 2C numbers; small-cluster guard in every stratum; seeds reproducible; immutability of Stage 2A/2B/2C files.

## Independent validation
Pure-Python recomputation of the T5 contrasts, one stratum estimate set, the 5-min RV for sampled expiries from raw bars, and the rate-variant IV change for sampled snapshots (Black-76 with r changed) — compared with the Gate 2 tables.

## Deliverables
`research_output/stage2d/gate2/REPORT.md` (methods, results, deviations, limitations), tables, validation outputs; no commit or push without explicit approval; stop after Gate 2.
