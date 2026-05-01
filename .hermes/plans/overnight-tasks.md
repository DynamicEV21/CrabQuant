# CrabQuant Overnight Build — Task Queue

**Created:** 2026-04-28 | **Last Updated:** 2026-05-01 ~10:29 UTC (Director Review #7)
**Project:** ~/development/CrabQuant/
**Venv:** `source ~/development/CrabQuant/.venv/bin/activate`
**Branch:** `phase5.6-overnight`
**Test baseline:** 5182 tests pass (0 failures), 81% coverage

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

### P0: Fix Cron Infrastructure (Directive 20 — CEO ACTION REQUIRED)
- [ ] Verify crontab schedules: `crontab -l` should show Ops (5m), Tech Lead (30m), Director (2h)
- [ ] Create or fix `crabquant/daemon/` module — currently doesn't exist (cannot import)
- [ ] Check agent execution logs for crash/exit patterns
- [ ] If crons can't be fixed: run mandates manually via `python scripts/refinement_loop.py`
- **Status: BLOCKED on CEO manual intervention**

### P1: Continue Mandate Execution (Directive 21)
- [ ] Run mandate: volatility_amd (7 turns) | priority: HIGH | cycles: 1 | directive: #21
  - Volatility archetype on AMD — new ticker, diverse coverage
  - Use param_optimizer early (Directive 12)
- [ ] Run mandate: multi_indicator_goog (7 turns) | priority: HIGH | cycles: 1 | directive: #21
  - Multi-indicator archetype on GOOG — new ticker
- [ ] Run mandate: mean_reversion_meta (7 turns) | priority: HIGH | cycles: 1 | directive: #21
  - Mean reversion on META — new ticker
- **Note: Pipeline validated at 66.7% convergence (2/3 last batch). Keep volume up.**

### P2: Clean Up Dead Code (Directive 22)
- [ ] Archive/delete `validation_probe.py` (337 lines, 0% coverage)
- [ ] Archive/delete `verify_validation.py` (46 lines, 0% coverage)
- [ ] Fix 322 pandas FutureWarnings in `signal_analysis.py` (fillna downcasting)

---

## Completed Tasks (archive)

- ✅ mean_reversion_aapl mandate — Sharpe 2.15, 7 turns, AUTO-PROMOTED (Ops execution)
- ✅ trend_tsla mandate — Sharpe 1.92, 7 turns, AUTO-PROMOTED (param_optimizer rescue, Ops execution)
- ✅ volume_btc mandate — Sharpe 0.51, 7 turns, not converged (Ops execution)
- ✅ All Directive 14 mandates executed (Ops intervention)
- ✅ Stale state committed (Directive 19, commit 1688295)
- ✅ breakout_spy mandate — Best Sharpe 1.14, 7 turns (not converged)
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

## Blocked Issues
- 🔴 No crontab configured — agent crons not running as scheduled
- 🔴 crabquant.daemon module doesn't exist — cannot start daemon
- 🔴 Tech Lead unresponsive 6.7h — cron infrastructure failure, not agent failure
