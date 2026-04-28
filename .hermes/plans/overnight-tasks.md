# CrabQuant Overnight Build — Task Queue

**Created:** 2026-04-28
**Project:** ~/development/CrabQuant/
**Venv:** source ~/development/CrabQuant/.venv/bin/activate
**Branch:** `phase5.6-overnight` (create from main, PR when done)
**Context Docs:** ROADMAP.md, VISION.md, docs/INDICATOR_API.md

---

## Global Rules

1. **Venv**: ALWAYS `source ~/development/CrabQuant/.venv/bin/activate` before any Python command
2. **Branch**: All work goes on `phase5.6-overnight`. Create it from main before starting. Commit after each task. Never push to main directly.
3. **Test after every change**: `cd ~/development/CrabQuant && python -m pytest tests/ -q --tb=short 2>&1 | tail -20`
   - Known: 4 pre-existing test errors in standalone debug scripts (`debug_test.py`, `test_invented_*.py`, `test_new_strategy.py`) in root — ALWAYS IGNORE THESE
   - Only fail if NEW test failures appear
4. **Design principles**: Direct Python execution, JSON on disk, deterministic before intelligent, subprocess isolation for parallelism
5. **Strategy file format**: Every strategy MUST have `generate_signals(df, params)`, `DEFAULT_PARAMS`, `DESCRIPTION`
6. **Read before write**: Always read existing files before modifying them. Understand the codebase conventions.
7. **Don't break existing tests**: If a change breaks tests, fix them before marking the task complete.
8. **Skill file**: Read `~/.hermes/skills/software-development/crabquant-development/SKILL.md` for architecture details and mock path rules for tests. Update it if you change architecture.

---

## Guardrails (Hard Limits)

### Test Time Budget
- **Unit tests**: Always run full suite (`pytest tests/ -q --tb=short`) — completes in ~50s. No time limit needed.
- **E2E tests** (tasks that run the actual LLM pipeline): **MAX 3 minutes wall time**. Use `--timeout 180`.
  - Only run 1 mandate, max 3 turns, sharpe_target 1.5 (higher target = fewer validation runs = faster)
  - Never run the daemon (`--wave-only` only, no continuous loop)
  - If E2E test is just verifying structure (not LLM output), mock the LLM call instead of making real API calls

### Commit Discipline
- Commit after EVERY completed task: `git add -A && git commit -m "phase5.6: <brief description>"`
- If a task partially completes but is blocked, commit what you have with a WIP message
- Never leave uncommitted work at the end of a run

### Blocker Protocol
- If stuck on a task for >10 min: log it in Decision Log, move to next task
- If ALL tasks are blocked: log blockers, commit, stop
- Common blockers and fixes:
  - Import errors: check `__init__.py` files, verify mock paths per SKILL.md
  - Test failures: read the failure message, fix the test or the code, re-run
  - Circular imports: move imports inside function bodies
  - Missing files: check if another task was supposed to create them

---

## Autonomy

You are an autonomous builder. Within these rules, you have full discretion:
- **Task ordering**: Follow priority order but skip blocked tasks. If you finish all tasks, look for improvements mentioned in ROADMAP.md Phase 6 prep.
- **Self-correction**: If you notice a bug or improvement opportunity while working on a task, fix it. Log it in Decision Log.
- **Test additions**: Add tests for edge cases you discover. More coverage is better.
- **Documentation**: Update SKILL.md if you change architecture, mock paths, or known bugs.
- **No permission needed**: Implement, test, commit. Don't wait for human input.

---

## Tasks

### Phase 5.6: Invention Accelerators

- [x] 1. **Cross-Run Learning (5.6.1)** ✅ DONE
  - `get_winner_examples()` in context_builder.py, winner_examples in prompts, formatting fix in llm_api.py
  - 12 tests in `tests/refinement/test_cross_run_learning.py`
  - Config toggle `cross_run_learning` + mode presets

- [x] 2. **Parallel Strategy Spawning (5.6.2)** ✅ DONE (committed by previous agent)
  - Verify it works by reading the implementation and running existing tests

- [ ] 3. **Soft-Promote Tier (5.6.3)**
  - **Priority: HIGH** — stops throwing away near-miss strategies
  - Create `results/candidates/` directory (add `.gitkeep`)
  - File: `crabquant/refinement/promotion.py`
  - After `run_full_validation_check()`:
    - If `validation.passed == True` → full promote to STRATEGY_REGISTRY (existing behavior, no change)
    - If `validation.passed == False` BUT:
      - `avg_test_sharpe >= config.soft_promote_sharpe` (default 0.5) AND
      - `windows_passed >= config.soft_promote_min_windows` (default 2)
      → soft-promote: save to `results/candidates/{strategy_name}_{timestamp}.json`
    - Regime-specific strategies get lower threshold: `avg_test_sharpe >= 0.3`
  - Candidate file format: `{name, timestamp, avg_test_sharpe, windows_passed, total_windows, regime_tags, source_code, backtest_metrics}`
  - Wire into `refinement_loop.py` — after validation fails, check soft promote before marking turn as failed
  - Print clear log message when soft-promoting: `"📝 Soft-promoted to candidates (avg_test_sharpe={:.3f}, {}/{} windows passed)"`
  - Config: `soft_promote: bool = True` already exists in `config.py` — verify it's wired up
  - Tests: `tests/refinement/test_soft_promote.py` (6+ tests):
    - Test soft promote triggers when thresholds met
    - Test soft promote skips when thresholds not met
    - Test regime-specific lower threshold
    - Test candidate file written correctly
    - Test full promote takes priority over soft promote
    - Test soft_promote=False disables the path
  - **E2E verification**: NO real LLM calls needed. Mock the validation result to return `passed=False` with `avg_test_sharpe=0.6, windows_passed=3`. Verify candidate file created.

- [ ] 4. **Mode System Integration**
  - **Priority: HIGH** — makes the toggles user-accessible
  - File: `scripts/refinement_loop.py`, `crabquant/refinement/config.py`
  - Verify `apply_mode()` method works for all 4 presets:
    - `conservative`: cross_run=False, parallel=False, soft_promote=False
    - `explorer`: cross_run=True, parallel=True, soft_promote=True
    - `fast`: cross_run=True, parallel=False, soft_promote=False
    - `balanced` (default): cross_run=True, parallel=True, soft_promote=False
  - Wire mandate JSON `mode` field into config: if mandate has `"mode": "explorer"`, call `config.apply_mode("explorer")`
  - Individual toggles in mandate JSON should override mode presets (e.g., `mode: "fast"` + `soft_promote: true` → soft promote enabled)
  - Wire `mode` and `toggles` from mandate JSON through `crabquant_cron.py` → `refinement_loop.py` → `RefinementConfig`
  - Tests: `tests/refinement/test_mode_system.py` (5+ tests):
    - Test each preset sets correct toggle values
    - Test individual toggle overrides mode preset
    - Test mandate JSON mode field parsed correctly
    - Test missing mode falls back to balanced defaults
  - **E2E verification**: NO real LLM calls. Just verify config values are set correctly.

- [ ] 5. **Phase 5.6 E2E Validation**
  - **Priority: MEDIUM** — integration check
  - Run full test suite, verify no regressions (should be 980+ tests)
  - Verify all 3 accelerators can be enabled simultaneously via `explorer` mode
  - Verify all 3 can be disabled via `conservative` mode
  - Run a short E2E test with `explorer` mode: `python scripts/refinement_loop.py --mandate <test_mandate> --max-turns 2 --sharpe-target 2.0 --timeout 180`
    - High sharpe target (2.0) to avoid triggering validation (saves time)
    - Just verify it starts, runs 2 turns, logs parallel spawning, exits cleanly
  - Commit: `git add -A && git commit -m "phase5.6: invention accelerators complete"`
  - **E2E time limit: 3 minutes max**

### Phase 5.5 Remaining

- [ ] 6. **Regime-Aware Scanner Enhancement (5.5.6)**
  - **Priority: MEDIUM** — production feature, not blocking research
  - File: `crabquant/production/regime_scanner.py` (exists, needs enhancement)
  - Modify scanner to filter strategies by detected current regime using `regime_tagger.py`
  - Only run strategies whose `preferred_regimes` include the current market regime
  - Fall back to regime-agnostic strategies (empty regime_tags) when regime is unclear
  - Tests: `tests/production/test_regime_scanner.py` (5+ tests)

- [ ] 7. **Portfolio Regime Router Enhancement (5.5.7)**
  - **Priority: MEDIUM** — production feature, not blocking research
  - File: `crabquant/production/regime_router.py` (exists, needs enhancement)
  - Read current regime from `regime.py` detection
  - Select top-N strategies from STRATEGY_REGISTRY that match current regime
  - Weight allocation by regime Sharpe scores
  - Fallback: equal weight when regime is unclear
  - Tests: `tests/production/test_regime_router.py` (5+ tests)

### Phase 6 Prep (if time permits)

- [ ] 8. **API Budget Tracker**
  - **Priority: LOW** — needed for production, not for research quality
  - File: `crabquant/refinement/api_budget.py`
  - `ApiBudgetTracker` class: `record_prompt()`, `should_throttle()`, `get_recommended_model()`
  - Persist state to `results/api_budget.json`
  - Throttle at 80% daily budget, alert at 90%
  - Tests: `tests/refinement/test_api_budget.py` (10+ tests)
  - No E2E needed — pure unit tests

- [ ] 9. **Resource Limiter**
  - **Priority: LOW** — needed for production parallelism
  - File: `crabquant/refinement/resource_limiter.py`
  - `ResourceLimiter` class: `check_resources()`, `get_recommended_parallel()`, `should_pause()`
  - Use `psutil` (check if installed, add to requirements if not)
  - Monitor CPU%, available RAM, disk space
  - Tests: `tests/refinement/test_resource_limiter.py` (10+ tests)
  - No E2E needed — pure unit tests

- [ ] 10. **Wire Phase 6 Components**
  - **Priority: LOW** — integration task
  - Wire api_budget into `crabquant/refinement/llm_api.py` — check `should_throttle()` before each LLM call
  - Wire resource_limiter into `scripts/run_pipeline.py` — check before spawning parallel tasks
  - Add fields to state.py for tracking
  - Run full test suite
  - Depends on: tasks 8, 9

### Improvements (autonomous — do these if you finish all above)

- [ ] 11. **Prompt Engineering: Indicator Selection Guidance**
  - **Why**: LLM still picks suboptimal indicators. It should know which indicator families work best for which archetypes.
  - Add archetype-specific indicator guidance to turn 1 prompt:
    - Momentum: ROC, MACD, ADX, rate of change
    - Mean reversion: RSI, Bollinger Bands, z-score, CCI
    - Breakout: ATR, Keltner Channels, Donchian, volatility ratio
    - Volume: OBV, volume ROC, VWAP, accumulation/distribution
  - Keep it concise — 1-2 lines per archetype
  - Test by running a quick E2E (2 turns, 3 min max) and checking if indicator selection matches archetype

- [ ] 12. **Stagnation Recovery Improvements**
  - **Why**: When Sharpe plateaus, the LLM gets generic feedback ("try different approach")
  - Add specific recovery strategies to stagnation detection:
    - If stuck at Sharpe 0.0-0.3 for 2+ turns: suggest indicator family change
    - If stuck at Sharpe 0.5-0.8 for 2+ turns: suggest parameter tuning
    - If stuck at Sharpe 0.8-1.0 for 2+ turns: suggest adding a filter/confirmation layer
  - Wire into `refinement_loop.py` stagnation logic

- [ ] 13. **Composite Score in Refinement Loop**
  - **Why**: Currently the loop only checks Sharpe >= target. A strategy with Sharpe 1.5 and 5 trades gets promoted over Sharpe 1.4 with 80 trades.
  - Use composite score (sharpe * sqrt(trades/20) * (1 - abs(max_dd))) for best_strategy tracking
  - Log both Sharpe and composite score per turn
  - Only trigger validation if BOTH Sharpe >= target AND num_trades >= 20 (already done) AND composite_score >= threshold
  - This prevents the "high Sharpe, low trades" trap at a deeper level

---

## Decision Log

- [2026-04-28 09:44] Phase 5.6.1 committed to main (cross-run learning, min trade gate, examples formatting)
- [2026-04-28 09:44] Overnight build started. Tasks restructured with guardrails, autonomy rules, ETE time limits.
- [2026-04-28 09:44] Branch strategy: all work on `phase5.6-overnight`, PR when complete.

## Errors / Blockers

(none yet)
