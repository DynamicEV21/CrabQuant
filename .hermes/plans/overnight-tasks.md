# CrabQuant Overnight Build — Task Queue

**Created:** 2026-04-28 | **Last Updated:** 2026-04-30
**Project:** ~/development/CrabQuant/
**Venv:** `source ~/development/CrabQuant/.venv/bin/activate`
**Branch:** `phase5.6-overnight` → PR #1 merged
**Test baseline:** 349+ tests pass (9 pre-existing failures in vectorbt/param_optimizer — ignore)
**Context Docs:** ROADMAP.md, VISION.md, docs/INDICATOR_API.md

---

## Governance

**SUPERVISOR OVERRIDES THIS FILE.** The supervisor cron runs every 2h and writes directives to `.hermes/plans/supervisor-review.md`. The orchestrator reads that FIRST. This file is secondary — it provides task context but does NOT set priorities.

If supervisor-review.md says "run mandates" and this file says "wire modules" — follow the supervisor.

---

## Global Rules

1. **Venv**: ALWAYS `source ~/development/CrabQuant/.venv/bin/activate`
2. **Branch**: All commits go to current branch. Push after each commit.
3. **Test after changes**: `python -m pytest tests/refinement/test_schemas.py tests/refinement/test_config.py -q` (< 10s)
   - Full suite has 9 pre-existing failures (vectorbt API change + param_optimizer fixture) — ignore
4. **Strategy format**: `generate_signals(df, params)`, `DEFAULT_PARAMS`, `DESCRIPTION`
5. **Read before write**: Always read existing files before modifying
6. **Skill file**: `~/.hermes/skills/software-development/crabquant-development/SKILL.md`
7. **Pre-commit hook**: 7s import check + critical tests. Fix broken imports before committing.

---

## Completed Tasks (archive)

All Tier 1-3 tasks completed. Summary:
- ✅ Cross-run learning, parallel spawning, anti-overfitting prompts
- ✅ Negative feedback loop, archetype templates, stagnation recovery (7 trap types)
- ✅ Composite score, soft-promote, mode system, multi-ticker
- ✅ Feature importance, regime diagnosis, sharpe diagnosis
- ✅ Action validator, cosmetic guard upgrade, wiring audit
- ✅ Validation funnel fix (0% → 100% pass rate on rolling WF)
- ✅ First promotions (4 strategies), batch promote (118 registry entries)
- ✅ Phase 5.6 PR merged (#1, 28 files, +4,761 lines)
- ✅ Test suite: 1033 → 4567+ tests

Full decision history: `.hermes/plans/decision-log-archive.md`

---

## Blocker Protocol

- Stuck >10 min → log in decision-log.md, skip to next task
- ALL tasks blocked → log blockers, commit, stop
- Pre-existing test failures: vectorbt Portfolio API change + param_optimizer fixture — ignore these

---

## Active Work

**Supervisor directives in `.hermes/plans/supervisor-review.md` take priority over anything below.**

### Known Bugs (fix when supervisor requests)
1. `vectorbt` API change — `Portfolio` attribute missing (9 test failures in validation_gates)
2. `param_optimizer.py` test fixture — sample data produces no actionable trades (1 failure)
3. Slippage sensitivity tests — returns 0 trades at all levels (report generation bug)

### Registry Status
- 118 strategies in registry (99 ROBUST, 19 DEMOTED)
- 182 winners, only 9 have walk-forward data (4 pass rolling WF with 4+/6 windows)
- 173 winners have NO walk-forward validation — unproven backtest results only
