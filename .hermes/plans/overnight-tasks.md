# CrabQuant Overnight Build — Task Queue

**Created:** 2026-04-28 | **Last Updated:** 2026-05-01 ~03:15 UTC
**Project:** ~/development/CrabQuant/
**Venv:** `source ~/development/CrabQuant/.venv/bin/activate`
**Branch:** `phase5.6-overnight`
**Test baseline:** 5180 tests pass (0 failures)

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

### P0: Restart daemon + run 3 pending mandates (Directive 14)
- [ ] RESTART daemon: `python -m crabquant.daemon` OR run mandates directly via `python scripts/refinement_loop.py`
- [ ] Run mandate: mean_reversion_aapl (7 turns) | priority: HIGH | cycles: 1 | directive: #14
  - Mean reversion archetype on AAPL — mandate file already created
  - Full 7-turn refinement loop — use param_optimizer early (Directive 12)
  - **Previous status: BLOCKED — UNBLOCK NOW via direct script execution**
- [ ] Run mandate: volume_btc (7 turns) | priority: HIGH | cycles: 1 | directive: #14
  - Volume archetype on BTC — create mandate file if needed
  - Full 7-turn refinement loop
- [ ] Run mandate: trend_tsla (7 turns) | priority: HIGH | cycles: 1 | directive: #14
  - Trend following on TSLA — mandate file exists
  - Full 7-turn refinement loop

### P1: Investigate daemon down (Directive 15)
- [ ] Why did daemon stop after 27 mandates? | priority: MEDIUM | cycles: 1
  - Check PID file, logs, crash traces
  - Determine if graceful shutdown or crash
  - Recommend auto-restart mechanism

### P1: Fix Operations KPI staleness (Directive 16)
- [ ] ops-kpis.json 86 min stale — investigate Operations cron | priority: LOW | cycles: 0
  - Check if health-check script is running
  - Verify cron schedule

### P2: Remaining dead-code wiring (lower priority)
- [ ] Wire explainer agent into refinement_loop.py | priority: LOW | cycles: 0 | enhancement: #4
  - Module exists (`explainer.py`) but not called from pipeline
  - Designed for human review — lower value in autonomous mode
- [ ] Add crypto indicators to strategy_helpers.py | priority: LOW | cycles: 0 | enhancement: #13
  - LOW priority per VISION.md

---

## Completed Tasks (archive)

- ✅ breakout_spy mandate — Best Sharpe 1.14, 7 turns, regime_fragility + low_sharpe (not converged)
- ✅ Wire DE optimizer into optimize_parameters (commit 44d59bd)
- ✅ Wire AST sanitizer into validation gate 1 (commit 496a7cd)
- ✅ Enhancement audit — 12/14 verified working
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
None this cycle. Pipeline enhancements are built and being wired. Need mandate execution to validate.

---

## Blocked Issues
- 🔴 Daemon NOT running — mandates cannot execute without daemon or direct script invocation
- ⚠️ Operations KPIs 86 min stale — possible Operations cron gap
- ⚠️ Mandate execution stalled 3+ hours — Directive 14 must unblock this cycle
