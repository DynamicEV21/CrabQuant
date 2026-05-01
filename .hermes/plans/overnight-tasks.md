# CrabQuant Overnight Build — Task Queue

**Created:** 2026-04-28 | **Last Updated:** 2026-05-01 ~01:00 UTC
**Project:** ~/development/CrabQuant/
**Venv:** `source ~/development/CrabQuant/.venv/bin/activate`
**Branch:** `phase5.6-overnight`
**Test baseline:** 5187 tests pass (0 failures)

---

## Governance

**SUPERVISOR OVERRIDES THIS FILE.** The supervisor cron runs every 2h and writes directives to `.hermes/plans/supervisor-review.md`. The orchestrator reads that FIRST. This file is secondary — it provides task context but does NOT set priorities.

---

## Global Rules

1. **Venv**: ALWAYS `source ~/development/CrabQuant/.venv/bin/activate`
2. **Branch**: All commits go to current branch. Push after each commit.
3. **Test after changes**: `python -m pytest tests/ -x -q --timeout=60 -k "not slow" 2>&1 | tail -20`
4. **Strategy format**: `generate_signals(df, params)`, `DEFAULT_PARAMS`, `DESCRIPTION`
5. **Read before write**: Always read existing files before modifying
6. **Commit discipline**: Commit ALL staged changes promptly. Push after every commit. Use `orch:` prefix.

---

## Active Tasks (ordered by priority)

### P0: Run 4 new mandates to test pipeline improvements (Directive 11)
- [x] Run mandate: breakout_spy (7 turns) | priority: HIGH | cycles: 1 | directive: #11
  - ✅ Completed: Best Sharpe 1.14 at turn 3, 7/7 turns exhausted, dominant failure: regime_fragility + low_sharpe
  - Did NOT converge (target 1.5). Turn 7 code generation failed (zero trades).
  - 3 auto-reverts (turns 4-6 all regressed from turn 3 best)
- [ ] Run mandate: mean_reversion_aapl (7 turns) | priority: HIGH | cycles: 1 | directive: #11
  - Mean reversion archetype on AAPL — different ticker from previous xom success
  - Full 7-turn refinement loop
- [ ] Run mandate: volume_btc (7 turns) | priority: HIGH | cycles: 1 | directive: #11
  - Volume archetype on BTC — high volatility, should produce frequent signals
  - Full 7-turn refinement loop
- [ ] Run mandate: trend_tsla (7 turns) | priority: HIGH | cycles: 1 | directive: #11
  - Trend following on TSLA — complements existing strategies
  - Full 7-turn refinement loop

### P1: Use param_optimizer aggressively (Directive 12)
- [ ] Use param_optimizer before spending LLM turns on parameter tweaks | priority: MEDIUM | cycles: 0 | directive: #12
  - Per VISION.md: param_optimizer rescued a mandate to Sharpe 1.54 (saved 3 turns)
  - Use it EARLY in the refinement loop (turn 2-3), not as last resort
  - This applies to ALL mandates above

---

## Completed Tasks (archive)

- ✅ Fix too_few_trades bottleneck — threshold lowered 20→10 (commit 3a93513)
- ✅ mean_reversion_xom mandate — SUCCESS Sharpe 1.73, 21 trades, turn 2, auto-promoted
- ✅ volume_nvda mandate — Best Sharpe 1.21, 7 turns, excessive_drawdown dominant
- ✅ momentum_msft mandate — Best Sharpe 1.05, 7 turns, low_sharpe dominant
- ✅ Orchestrator status file — now updated with meaningful data
- ✅ winners_with_wf KPI bug — FIXED (0 → 137)
- ✅ KPI prev/current rotation — VERIFIED working (timestamps differ)
- ✅ 12 staged files committed (commits 689c3fe, 3376c9c)
- ✅ All Tier 1-3 tasks (cross-run learning, parallel spawning, anti-overfitting, etc.)
- ✅ Phase 5.6 PR merged (#1, 28 files, +4,761 lines)
- ✅ Registry integrity audit (Cycle 19) — 99 ROBUST, 19 DEMOTED
- ✅ Full wiring audit — zero new bugs
- ✅ 14 pipeline enhancements implemented (DE optimizer, deflated Sharpe, complexity scoring, etc.)

---

## R&D Recommendations
None this cycle. Pipeline enhancements are built. Need mandate execution to validate.

---

## Blocked Issues
None. All blockers resolved.
