# Supervisor Review — 2026-04-30 17:30 UTC (Initial)

## Status: OFF COURSE

## Summary
The system has spent the last 48 hours almost entirely on infrastructure (wiring dead modules, expanding tests, building new components) while VISION.md's P1 priority is "Run Live Mandates to Verify Improvements." Only 4 out of 182 winners have meaningful rolling walk-forward data. The orchestrator is doing useful work but NOT the right work — it's building a better car instead of driving it.

## Priority Directives (ordered)

1. **Run 3 real mandates this cycle** — Pick from `mandates/` (momentum_spy, breakout_tsla, volume_aapl — any 3 different types). Run full 7-turn refinement loops. Do NOT wire any more modules until you have at least 3 new strategies with rolling WF results. This is VISION.md P1.

2. **Commit and push existing uncommitted work NOW** — There are debug files (debug_test.py, debug_test2.py, debug_strategy.py, debug_structure.py) and modified files (scripts/refinement_loop.py, test files, results/) sitting in git status. Commit the useful stuff, discard the debug artifacts, push. Do not start mandates until the working tree is clean.

3. **Fix the 1 test failure in test_param_optimizer.py::TestRunParamBacktest::test_basic_backtest** — The pre-commit hook will block commits on this. Either fix the test or skip it with a clear reason. Don't leave broken tests in the tree.

4. **Stop expanding tests** — 5000+ tests is enough. The "Continuous Improvement" section in overnight-tasks.md encourages test expansion as default work. This is diminishing returns. Remove or deprioritize that section.

5. **Update VISION.md metrics after mandates run** — The "Current Reality" section reports 6.8% per-turn success rate and 33% mandate convergence. These are from old data. After running new mandates, update these numbers with real rolling-WF-validated results.

## Warnings
- **Pre-commit hook was pointing to WRONG venv** (`~/development/QuantFactory/.venv` instead of `~/development/CrabQuant/.venv`). Fixed 2026-04-30. Any commits made before this fix were tested against the wrong Python environment.
- **Slippage sensitivity testing is broken** — Strategy reports show 0 trades at all slippage levels. The test harness is not actually running. Non-blocking but means registry reports are incomplete.
- **173/182 winners have NO walk-forward data** — They were promoted on raw backtest Sharpe alone. The "182 winners" metric is misleading. Only 4 have meaningful rolling WF results, and only 1 (roc_ema_volume_googl, 5/6 windows, avg Sharpe 1.175) is genuinely robust.
- **Last 24h real mandate success rate: 0/23 (0%)** — Despite ~70% per-turn success rate in run_history, the actual strategies aren't surviving walk-forward validation.

## Stale Items
- **overnight-tasks.md Task 0 (Registry Audit)** — This was completed in Cycle 19. Remove or mark as DONE with a clear note.
- **overnight-tasks.md "Continuous Improvement" section** — Encourages test expansion as default work. This should be removed or replaced with "Run mandates and analyze results."
- **VISION.md Success Metrics table** — "Per-turn success rate: 6.8%" is from old data before infrastructure fixes. Needs updating.
- **VISION.md "Current Reality" table** — "99 ROBUST (audited)" implies these strategies passed rolling WF. They passed OLD single-split WF. Needs honest update.

## Metric Reality Check

| Metric | VISION.md Says | Actual Value | Honest Assessment |
|--------|---------------|--------------|-------------------|
| Strategies in registry | 99 ROBUST | 99 ROBUST, but only 4 have rolling WF data | ⚠️ Inflated — promoted on old validation |
| Winners | 178 (119 promoted) | 182 total, 9 with any WF data, 4 with positive WF | ⚠️ Inflated — most are backtest-only |
| Per-turn success rate | 6.8% | ~70% (post-fix) but measures backtest Sharpe, not WF | ⚠️ Misleading — low bar |
| Mandate convergence | 33% (7/21) | Unknown with rolling WF — likely much lower | 🔴 Needs real measurement |
| Test coverage | 4430+ tests | 5000+ tests | ⚠️ Diminishing returns — stop expanding |
| Validation pass rate | 84% | 4/182 winners pass rolling WF = 2.2% | 🔴 Reality: 2.2%, not 84% |

## Previous Directives: N/A (first review)
