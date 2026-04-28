# CrabQuant Overnight Build — Task Queue

**Created:** 2026-04-28
**Project:** ~/development/CrabQuant/
**Venv:** source ~/development/CrabQuant/.venv/bin/activate
**Context Docs:** ROADMAP.md, VISION.md, docs/INDICATOR_API.md

## Global Rules

1. **Venv**: Always `source ~/development/CrabQuant/.venv/bin/activate` before any Python command
2. **Test after every task**: `cd ~/development/CrabQuant && python -m pytest tests/ -q --tb=short 2>&1 | tail -20`
   - Known: 4 pre-existing test errors in standalone debug scripts in root — ALWAYS IGNORE THESE
   - Only fail if NEW test failures appear
3. **Design principles** (from ROADMAP.md): Direct Python execution, JSON on disk, deterministic before intelligent, subprocess isolation for parallelism
4. **Strategy file format**: Every strategy MUST have `generate_signals(df, params)`, `DEFAULT_PARAMS`, `DESCRIPTION`
5. **Commit after each completed task**: `git add -A && git commit -m "phase5.6: <task description>"`
6. **Read before write**: Always read existing files before modifying them. Understand the codebase conventions.
7. **Don't break existing tests**: If a change breaks tests, fix them before marking the task complete.
8. **No LLM API calls needed**: These are code implementation tasks, not pipeline runs.

## Tasks

### Phase 5.6: Invention Accelerators

- [x] 1. **Implement Parallel Strategy Spawning (5.6.2)**
  - File: `scripts/refinement_loop.py`
  - On turn 1, when `config.parallel_invention == True`:
    - Spawn N strategies (from `config.parallel_invention_count`, default 3) via N separate LLM calls
    - Each gets a slightly different prompt variant (different indicator focus, different entry logic style)
    - Add prompt variant generation to `crabquant/refinement/prompts.py` — `get_parallel_prompt_variants(base_prompt, count)` returning list of N prompts
    - Backtest all N strategies, rank by composite score (sharpe * sqrt(trades/20) * (1 - abs(max_dd)))
    - Best strategy proceeds to refinement loop as normal
    - Log all parallel results to the run directory for analysis
  - Fallback: if `parallel_invention_count == 1`, behave as normal sequential
  - Add tests: `tests/refinement/test_parallel_invention.py` (5+ tests)
  - Depends on: nothing

- [ ] 2. **Implement Soft-Promote Tier (5.6.3)**
  - Create `results/candidates/` directory
  - File: `crabquant/refinement/promotion.py`
  - After full validation check:
    - If `validation.passed == True` → full promote to STRATEGY_REGISTRY (existing behavior)
    - If `validation.passed == False` BUT:
      - `avg_test_sharpe >= config.soft_promote_sharpe` (default 0.5) AND
      - `windows_passed >= config.soft_promote_min_windows` (default 2)
      → soft-promote: save to `results/candidates/{strategy_name}_{timestamp}.json` with `needs_ongoing_validation: true`
    - Regime-specific strategies get lower threshold: `avg_test_sharpe >= 0.3`
  - Candidate file format: `{name, timestamp, avg_test_sharpe, windows_passed, regime_tags, strategy_code, backtest_results}`
  - Wire into `refinement_loop.py` — after validation fails, check soft promote before giving up
  - Add tests: `tests/refinement/test_soft_promote.py` (5+ tests)
  - Depends on: nothing (parallel with task 1)

- [ ] 3. **Mode System Integration**
  - File: `scripts/refinement_loop.py`, `crabquant/refinement/config.py`
  - Verify all 4 mode presets work correctly:
    - `conservative`: cross_run=False, parallel=False, soft_promote=False
    - `explorer`: cross_run=True, parallel=True, soft_promote=True
    - `fast`: cross_run=True, parallel=False, soft_promote=False
    - `balanced` (default): cross_run=True, parallel=True, soft_promote=False
  - Individual toggles in mandate JSON should override mode presets
  - Ensure `refinement_loop.py` reads mode/toggles from mandate and applies to config
  - Add integration test: `tests/refinement/test_mode_system.py` (4+ tests — one per preset)
  - Depends on: tasks 1, 2

- [ ] 4. **Phase 5.6 E2E Validation**
  - Run full test suite, verify no regressions
  - Run a short real LLM E2E test with `explorer` mode: `python scripts/refinement_loop.py --mandate <test_mandate> --max-turns 3 --sharpe-target 1.0`
  - Verify parallel spawning produces 3 strategies on turn 1 (check logs)
  - Verify soft-promote creates candidate file when applicable
  - Commit: `git add -A && git commit -m "phase5.6: invention accelerators complete"`
  - Depends on: task 3

### Phase 5.5 Remaining

- [ ] 5. **Implement Regime-Aware Scanner (5.5.6)**
  - File: `crabquant/production/regime_scanner.py` (exists, may need enhancement)
  - Modify scanner to filter strategies by detected current regime
  - Only run strategies whose `preferred_regimes` include the current market regime
  - Fall back to regime-agnostic strategies when regime is unclear
  - Use `crabquant/refinement/regime_tagger.py` for regime detection
  - Add tests: `tests/production/test_regime_scanner.py` (5+ tests)
  - Depends on: nothing (can run parallel with Phase 5.6 tasks)

- [ ] 6. **Implement Portfolio Regime Router (5.5.7)**
  - File: `crabquant/production/regime_router.py` (exists, may need enhancement)
  - Read current regime from regime detection
  - Select top-N strategies from STRATEGY_REGISTRY that match current regime
  - Weight allocation by regime Sharpe scores
  - Fallback: equal weight when regime is unclear
  - Add tests: `tests/production/test_regime_router.py` (5+ tests)
  - Depends on: nothing (can run parallel with Phase 5.6 tasks)

### Phase 6 Prep (if time permits)

- [ ] 7. **API Budget Tracker**
  - File: `crabquant/refinement/api_budget.py`
  - Track prompt count per day/week, throttle to glm-4.7 at 80% budget, alert at 90%
  - `ApiBudgetTracker` class with `record_prompt()`, `should_throttle()`, `get_recommended_model()`, persistence
  - Add tests: `tests/refinement/test_api_budget.py` (10+ tests)
  - Depends on: nothing

- [ ] 8. **Resource Limiter**
  - File: `crabquant/refinement/resource_limiter.py`
  - Monitor CPU/RAM/disk, adjust parallel count dynamically, pause if RAM < 2GB
  - `ResourceLimiter` class with `check_resources()`, `get_recommended_parallel()`, `should_pause()`
  - Use psutil
  - Add tests: `tests/refinement/test_resource_limiter.py` (10+ tests)
  - Depends on: nothing

- [ ] 9. **Wire Phase 6 Components**
  - Wire api_budget into `crabquant/refinement/llm_api.py`
  - Wire resource_limiter into `scripts/run_pipeline.py`
  - Add fields to state.py
  - Run full test suite
  - Depends on: tasks 7, 8

## Decision Log

- [2026-04-28 09:17] Starting overnight build. Phase 5.6 is priority — parallel invention + soft promote. Then 5.5 remaining (scanner/router). Phase 6 prep if time permits.
- [2026-04-28 ~10:00] Phase 5.6.2 Parallel Strategy Spawning COMPLETE. Key decisions:
  - `PARALLEL_VARIANT_FOCI` list with 8 named indicator focus variants (trend_following, mean_reversion, momentum_breakout, volatility_regime, multi_timeframe, statistical_arb, volume_confirmation, multi_signal)
  - `get_parallel_prompt_variants(base_prompt, count)` — batch API returning list of N modified prompts
  - `get_variant_bias_text(variant_index, variant_count)` — single-variant API for use inside `call_llm_inventor`
  - Variant bias injected into user message in `llm_api.py` by checking for `parallel_variant_index`/`parallel_variant_count` keys in context
  - `_run_parallel_invention()` in `refinement_loop.py` — spawns N variants, validates gates, loads modules, backtests, ranks by composite score, returns best
  - Parallel invention only runs on turn 1; 2 retries per variant instead of 3; falls back to sequential if all variants fail
  - "balanced" mode added to `RefinementConfig.apply_mode()`: cross_run=True, parallel=True, soft_promote=False
  - `modification = {}` for parallel path — downstream code defaults to "novel" action, which is correct for turn 1
  - Module reloaded from file after parallel path (line 707) — consistent with sequential path, ensures clean state
  - 45 new tests in `test_parallel_invention.py` — all pass. Full suite: 1016 passed, 0 failures.

## Errors / Blockers

(none yet)
