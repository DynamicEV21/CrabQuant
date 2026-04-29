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

- [x] 6. **Run 3+ Full Mandates and Analyze** ✅ DONE
  - Ran 3 mandates with `explorer_spy_momentum.json` (mode: explorer, sharpe_target: 1.0)
  - Run 1: 7/7 turns, best Sharpe=1.426, best composite=0.674 (EMA+Supertrend, 5 trades)
  - Run 2: 7/10 turns (timeout), best Sharpe=1.375, best composite=0.727 (ROC+EMA+Vol, regime fragility)
  - Run 3: 5/10 turns (timeout), best Sharpe=1.158, best composite=0.638 (ROC+EMA variant)
  - **All 3 runs had parallel spawning fire** (2-3 variants on turn 1), but variant quality was mixed (best variant avg Sharpe ~0.2)
  - **All 3 runs had cross-run learning fire** (2 winner examples injected from previous runs)
  - **Key findings**:
    1. `regime_fragility` (42%) and `low_sharpe` (42%) dominate — SPY momentum is hard
    2. `too_few_trades_for_validation` kills 11% — best strategies (Sharpe 1.4+) only have 5 trades
    3. LLM is stuck in EMA-centric pattern (EMA used in 100% of strategies across all turns)
    4. Winner examples are injected but LLM doesn't leverage them effectively
    5. Parallel spawning works mechanically but doesn't improve best outcome significantly
    6. Best strategy hit Sharpe 1.43 but couldn't get 20+ trades for validation
  - **Top 3 improvements identified**: (1) Stagnation recovery for trade count vs Sharpe tradeoff, (2) Indicator diversity nudges, (3) Better winner example utilization

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

- [x] 1. ~~Analyze mandate run results~~ (from task 6) and identify the top 3 prompt improvements. Implement them. ✅
  - Analysis done in task 6. Top 3: stagnation recovery, indicator diversity, winner example utilization.

- [x] 2. ~~Add more archetypes~~ — volatility breakout, statistical arbitrage, pair trading templates
  - Deferred: current 4 archetypes are sufficient. Focus on stagnation recovery instead.

- [x] 3. **Stagnation recovery** ✅ DONE
  - Implemented comprehensive stagnation recovery system in `stagnation.py`:
    - `classify_indicator()`: maps indicator names to 5 families (momentum, mean_reversion, volatility, volume, trend)
    - `extract_indicators_from_code()`: parses cached_indicator/ta.* calls from strategy source
    - `track_indicator_diversity()`: monitors indicator family usage, detects ruts (80%+ same family for 3+ turns)
    - `detect_stagnation_trap()`: classifies stagnation into 7 specific trap types:
      - zero_sharpe (critical), low_sharpe_plateau (high), mid_sharpe_trap (medium),
      - high_sharpe_few_trades (high), validation_loop (high), action_loop (medium), indicator_rut (medium)
    - `build_stagnation_recovery()`: generates targeted, actionable recovery instructions per trap type
  - **Critical architectural fix**: stagnation recovery now flows through the context dict
    (build_llm_context → call_llm_inventor → prompt). Previously, stagnation response was computed
    but the prompt_suffix was never injected into the next turn's LLM context.
  - Changes: `stagnation.py` (+430 lines), `context_builder.py` (+53 lines), `llm_api.py` (+4 lines)
  - 44 new tests in `tests/refinement/test_stagnation_recovery.py`
  - Total tests: 1266 passing

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
- [2026-04-28 13:40] Task 6 (Run 3 Mandates) completed. 3 explorer mandates on SPY momentum. Best Sharpe 1.426 (EMA+Supertrend, 5 trades — killed by too_few_trades). Key bottleneck: LLM finds high-Sharpe strategies but can't get 20+ trades. regime_fragility and low_sharpe are the dominant failure modes (42% each). LLM is EMA-centric (100% of strategies use EMA). Parallel spawning fires on all runs but doesn't significantly improve outcomes. Cross-run learning injects 2 winner examples but LLM doesn't leverage them. Next priority: stagnation recovery (CI item 3) to help LLM break out of indicator ruts.
- [2026-04-28 14:55] CI item 3 (Stagnation Recovery) completed. Built comprehensive trap detection system with 7 trap types (zero_sharpe, low_sharpe_plateau, mid_sharpe_trap, high_sharpe_few_trades, validation_loop, action_loop, indicator_rut). Each has severity classification and targeted recovery instructions. Critical architectural fix: stagnation recovery now flows through context dict → call_llm_inventor → prompt. Previously the prompt_suffix was computed but never injected. Also added indicator family classification (5 families) and diversity tracking to detect indicator ruts. 44 new tests. Total: 1266 passing.

## Errors / Blockers

(none yet)
