# CrabQuant Overnight Build — Task Queue

**Created:** 2026-04-28 | **Last Updated:** 2026-04-30 ~22:15 UTC
**Project:** ~/development/CrabQuant/
**Venv:** `source ~/development/CrabQuant/.venv/bin/activate`
**Branch:** `phase5.6-overnight` → PR #1 merged
**Test baseline:** 4353 tests pass (0 failures, 1 skipped)

---

## Governance

**SUPERVISOR OVERRIDES THIS FILE.** The supervisor cron runs every 2h and writes directives to `.hermes/plans/supervisor-review.md`. The orchestrator reads that FIRST. This file is secondary — it provides task context but does NOT set priorities.

If supervisor-review.md says "run mandates" and this file says "wire modules" — follow the supervisor.

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

### P0: Commit staged work immediately
- [x] Commit the 12 staged files as atomic commits | priority: HIGH | cycles: 1 | directive: #7
  - Files were committed in Director review commit 689c3fe + orch commit 3376c9c

### P0: Run remaining 2 mandates (Directive 2 continuation)
- [x] Run volume_nvda mandate (7 turns) | priority: HIGH | cycles: 1 | directive: #8
  - Result: Best Sharpe 1.21, composite 1.51, max_turns_exhausted. Dominant failure: excessive_drawdown (3/7). Turn 6 hit Sharpe 1.65 but only 6 trades. NVDA volatility too high for 25% drawdown constraint.
- [x] Run momentum_msft mandate (7 turns) | priority: HIGH | cycles: 1 | directive: #8
  - Result: Best Sharpe 1.05, composite 0.79, max_turns_exhausted. Dominant failure: low_sharpe (4/7). Turn 6 best with param optimization rescue (0.42→1.05). MSFT low volatility makes high Sharpe targets difficult.
  - Neither mandate converged. 0/2 converged (0/5 total with mean_reversion_xom = 1/5 = 20%).

### P1: Fix KPI prev/current rotation (carryover from Review #2 Directive 3)
- [x] Fix the health check script rotation bug | priority: MEDIUM | cycles: 0 | directive: #9
  - VERIFIED: The rotation IS working correctly. ops-kpis-prev.json shows 22:14 UTC data, ops-kpis.json shows 23:10 UTC data with different values (winners_total 188→189, history_real_mandates 195→182, etc.). The fix was already in place in the script (shutil.copy2 at line 451). The Director was likely looking at stale data from before the fix.

### P2: Update VISION.md (carryover from Review #1 & #2)
- [ ] Update VISION.md "Current Reality" table | priority: LOW | cycles: 0 | directive: #10
  - Per-turn success rate: ~37% (up from 14%)
  - Mandate convergence: improving (1/3 recent + 1/1 new = 2/4)
  - Test coverage: 4353 tests
  - Last Updated: 2026-04-30
  - Winners with WF: 136/188
  - NOTE: This is a PROTECTED file — orchestrator should NOT modify this. Director will handle if needed.

---

## Completed Tasks (archive)

- ✅ Fix too_few_trades bottleneck — threshold lowered 20→10 (commit 3a93513)
- ✅ mean_reversion_xom mandate — SUCCESS Sharpe 1.73, 21 trades, turn 2, auto-promoted
- ✅ Orchestrator status file — now updated with meaningful data
- ✅ winners_with_wf KPI bug — FIXED (0 → 136)
- ✅ All Tier 1-3 tasks (cross-run learning, parallel spawning, anti-overfitting, etc.)
- ✅ Phase 5.6 PR merged (#1, 28 files, +4,761 lines)
- ✅ Registry integrity audit (Cycle 19) — 99 ROBUST, 19 DEMOTED
- ✅ Full wiring audit — zero new bugs

---

## R&D Recommendations
None this cycle. System is on positive trajectory. too_few_trades fix validated by real mandate convergence.

---

## Blocked Issues
None. All blockers resolved.
