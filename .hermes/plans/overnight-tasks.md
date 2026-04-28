# CrabQuant Overnight Build — Task Queue

**Created:** 2026-04-28
**Updated:** 2026-04-28 (restructured for invention-speed focus)
**Project:** ~/development/CrabQuant/
**Venv:** source ~/development/CrabQuant/.venv/bin/activate
**Branch:** `phase5.6-overnight` (create from main, PR when done)
**Context Docs:** ROADMAP.md, VISION.md, docs/INDICATOR_API.md
**Test baseline:** 1033 passed, 4 pre-existing errors (always ignore)

---

## Global Rules

1. **Venv**: ALWAYS `source ~/development/CrabQuant/.venv/bin/activate` before any Python command
2. **Branch**: Create `phase5.6-overnight` from main. ALL commits go here. NEVER touch main.
3. **Test after every change**: `cd ~/development/CrabQuant && python -m pytest tests/ -q --tb=short 2>&1 | tail -20`
   - 4 pre-existing errors in root debug scripts — ALWAYS IGNORE
   - Only fail on NEW test failures
4. **Strategy format**: `generate_signals(df, params)`, `DEFAULT_PARAMS`, `DESCRIPTION`
5. **Read before write**: Always read existing files before modifying
6. **Skill file**: Read `~/.hermes/skills/software-development/crabquant-development/SKILL.md` for architecture + mock paths
7. **Design principles**: Direct Python, JSON on disk, deterministic before intelligent

---

## How This Works

This runs as a repeating cron job (every 45 min). Each run:
1. Check out branch `phase5.6-overnight` (create from main if doesn't exist)
2. Read this file for progress
3. Pick next incomplete task
4. Implement it + tests + commit
5. Update this file (mark done, log decisions)
6. Stop after 1-2 tasks (next cron tick continues)

**If ALL tasks are done:** See the "Continuous Improvement" section at the bottom — there's always more to do.

---

## E2E Test Policy

**3 minutes max wall time for any E2E test.** Here's how:
- Feature validation: mock the LLM, don't make real API calls
- Pipeline smoke test: use `--sharpe-target 5.0` (impossibly high) + `--max-turns 2` — exits in ~2 min without triggering validation
- Actual strategy invention runs: ONLY in "Continuous Improvement" section, one at a time, 8 min max per run
- NEVER run the daemon (`--wave-only` only, no continuous loop)

---

## Blocker Protocol

- Stuck >10 min → log in Decision Log, skip to next task
- ALL tasks blocked → log blockers, commit, stop
- Common fixes: import errors → check `__init__.py`; circular imports → move imports into functions; test failures → read the error, fix code or test

---

## Tasks — Priority Order

### Tier 1: Directly Improves Invention Speed

- [x] 1. **Cross-Run Learning (5.6.1)** ✅ DONE
  - Winners feed into LLM context. On main.

- [x] 2. **Parallel Strategy Spawning (5.6.2)** ✅ DONE
  - 8 variant foci, composite ranking, wired into turn 1. On main.

- [x] 3. **Prompt Engineering: Anti-Overfitting** ✅ DONE
  - Added BAD vs GOOD examples to turn 1 prompt, failure_guidance per mode, inline notes in history
  - Commit: `cb4b48d`
  - **Also partially completed Task 4**: `build_failure_guidance()` function provides per-failure-mode actionable advice, and `format_previous_attempts_section()` adds inline ⚠️ notes for `too_few_trades_for_validation` and `validation_failed` failures
  - Remaining for Task 4: rolling WF window breakdown showing which windows passed/failed

- [x] 4. **Negative Example Feedback Loop** ✅ DONE
  - Added `_format_window_breakdown()` in `prompts.py` — per-window rolling WF table with train/test Sharpe, degradation, pass/fail, and actionable summary
  - Enhanced `format_previous_attempts_section()` with per-failure-mode inline guidance:
    - `too_few_trades_for_validation`: warns about restrictive conditions
    - `validation_failed`: shows per-window breakdown with train/test Sharpe
    - `low_sharpe` with < 10 trades: CURVE-FITTING RISK warning
    - `regime_fragility`: explains regime dependency, suggests detection or more robust indicators
  - Enhanced `build_failure_guidance()` to accept `validation` dict and include window breakdown for `validation_failed`
  - Added curve-fitting warning tiers: < 10 trades = CRITICAL, < 15 trades = unreliable
  - **Critical wiring**: Changed `llm_api.py` to use `format_previous_attempts_section()` instead of raw JSON dump for previous attempts
  - Wired `build_refinement_prompt()` to pass validation data to `build_failure_guidance()`
  - 25 unit tests in `tests/refinement/test_negative_feedback.py`
  - Commit: phase5.6-negative-feedback

- [x] 5. **Strategy Archetype Templates** ✅ DONE
  - **File**: `crabquant/refinement/archetypes.py` — 4 archetypes with skeleton code, default params, anti-patterns, regime affinity
  - **Critical fix**: `build_turn1_prompt()` computed `archetype_text` but never injected it into the `TURN1_PROMPT` template. Added `{archetype_section}` placeholder and wired it.
  - `get_archetype(name)` — case-insensitive lookup, returns `None` for unknown
  - `format_archetype_for_prompt()` — formats skeleton code, description, anti-patterns for LLM consumption
  - `list_archetypes()` — returns all archetype names
  - Archetypes: mean_reversion (RSI/BB, ranging), momentum (EMA/ROC, trending), breakout (ATR/range expansion, volatile), volatility (ATR ratio/BB bandwidth, volatile)
  - **91 unit tests** in `tests/refinement/test_archetypes.py`
  - Commit: `b556511`

- [ ] 6. **Run 3+ Full Mandates and Analyze**
  - **Priority: HIGH** — we need real data on what the LLM does with the new features
  - Run 3 mandates: `python scripts/refinement_loop.py --mandate <mandate> --timeout 480`
  - Use the default SPY mandate but with `mode: "explorer"` to enable all accelerators
  - After each run, read `results/refinement_runs/<run_dir>/state.json` and extract:
    - Turns used, best Sharpe, best trade count, failure modes per turn
    - Did cross-run learning fire? Did parallel spawning fire?
    - What indicators did the LLM pick?
  - Summarize patterns in Decision Log: what works, what doesn't, specific prompt improvements needed
  - **Time budget**: 8 min per mandate max, 25 min total for this task
  - **This is the ONLY task that should run real LLM calls**

### Tier 2: Infrastructure That Enables Better Invention

- [x] 7. **Soft-Promote Tier (5.6.3)** ✅ DONE
  - Already implemented — `soft_promote()` in `promotion.py` with threshold enforcement
  - Wired into refinement loop at line 939 (after validation_failed)
  - `results/candidates/` with `.gitkeep`
  - Regime-specific lower threshold (0.3 vs 0.5)
  - 16 existing tests passing
  - Commit: verified + `.gitkeep` added

- [x] 8. **Mode System Integration** ✅ DONE
  - Already implemented — `apply_mode()` supports conservative/fast/explorer/balanced/custom
  - Mandate `mode` field wired in refinement_loop.py line 502-504
  - Individual toggles override mode presets (lines 506-520)
  - 16 new tests in `tests/refinement/test_mode_system.py`
  - Commit: `26bbd7d`

- [x] 9. **Composite Score for Best-Strategy Tracking** ✅ DONE
  - Added `best_composite_score` field to `RunState` in `schemas.py`
  - Changed all 4 best-strategy tracking points in `refinement_loop.py` to use `compute_composite_score()` instead of raw Sharpe
  - Formula: `sharpe * sqrt(trades/20) * (1 - abs(max_drawdown))` — penalizes few trades and high drawdown
  - Validation gate still uses raw Sharpe (unchanged) — only best-strategy tracking uses composite
  - Added `composite_score` to all history entries (3 branches: too_few_trades, validation success, normal tracking)
  - Updated context_builder to pass `best_composite_score` to LLM context
  - Updated max_turns_exhausted log to show both Sharpe and composite
  - 15 new unit tests in `tests/refinement/test_composite_score.py`
  - Commit: phase5.6-composite-score
  - Total tests: 1179 passing

### Tier 3: Continuous Improvement (do after all tasks done)

If you complete all tasks above, keep going. Here's the priority order:

1. **Analyze mandate run results** (from task 6) and identify the top 3 prompt improvements. Implement them.
2. **Add more archetypes** — volatility breakout, statistical arbitrage, pair trading templates
3. **Stagnation recovery** — when Sharpe plateaus, give the LLM specific recovery strategies based on WHERE it's stuck (0.0-0.3 → change indicator family, 0.5-0.8 → tune params, 0.8-1.0 → add filter)
4. **Multi-ticker support** — run strategy on SPY+QQQ+IWM simultaneously, require pass on 2/3
5. **Feature importance feedback** — after backtest, tell the LLM which indicators contributed most to Sharpe
6. **Update SKILL.md** with any architecture changes you made
7. **Update ROADMAP.md** — mark completed items, add new items you discovered
8. **Run more mandates** to validate improvements (8 min max each)
9. **Create the PR** — `gh pr create --title "Phase 5.6: Invention Accelerators" --body "..."` from `phase5.6-overnight` to `master`

**IMPORTANT: Never stop just because the task list is done. The whole point is that you keep going.**

---

## Decision Log

- [2026-04-28 09:44] Phase 5.6.1 committed to main (cross-run learning, min trade gate, examples formatting)
- [2026-04-28 09:47] Phase 5.6.2 committed to main by overnight agent (parallel spawning, 8 variant foci)
- [2026-04-28 10:01] Restructured task list: removed low-priority infrastructure (API budget, resource limiter), added prompt engineering + archetype system + negative feedback loop. Focus on invention speed.
- [2026-04-28 10:01] 1033 tests passing, 4 pre-existing errors (ignore)
- [2026-04-28 10:52] Task 4 (Negative Example Feedback Loop) completed. Key insight: `call_llm_inventor` was dumping previous_attempts as raw JSON — changed to use `format_previous_attempts_section()` for readable, guidance-rich output. Added per-window breakdown for validation_failed, curve-fitting warnings for low_sharpe + few trades, regime dependency warnings for regime_fragility. 25 new tests. Total: 1057 passing.
- [2026-04-28 11:46] Task 5 (Strategy Archetype Templates) completed. Found critical wiring bug: `build_turn1_prompt()` computed `archetype_text` but never injected it into `TURN1_PROMPT` template. Added `{archetype_section}` placeholder + wired it. 4 archetypes with skeleton code, anti-patterns, regime affinity. 91 new tests. Total: 1148 passing.
- [2026-04-28 12:39] Task 9 (Composite Score for Best-Strategy Tracking) completed. `compute_composite_score()` was already defined in prompts.py and imported in refinement_loop.py but never used for tracking. Added `best_composite_score` field to RunState, changed all 4 tracking points (too_few_trades, validation success, main tracking, post-loop) to use composite score instead of raw Sharpe. Validation gate unchanged (still raw Sharpe). Key scenario verified: Sharpe 4.0 with 5 trades (composite=1.59) loses to Sharpe 1.8 with 60 trades (composite=3.50). 15 new tests. Total: 1179 passing.

## Errors / Blockers

(none yet)
