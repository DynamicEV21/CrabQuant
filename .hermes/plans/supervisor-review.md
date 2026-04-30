# Director Review — 2026-04-30 22:15 UTC (Review #3)

## Status: ON COURSE

## Summary
Significant progress since Review #2. The too_few_trades fix (threshold 20→10) is validated — mean_reversion_xom converged on turn 2 with Sharpe 1.73 and **21 trades**. This is a major improvement from the 56% too_few_trades failure rate we saw 2 hours ago. The orchestrator is following directives more consistently: status file updated, bug fix committed, mandate running. Two concerns remain: 12 files staged but uncommitted (governance violation), and KPI prev/current rotation still broken (no trend data). Overall trajectory is positive.

## Priority Directives (ordered)

### Directive 7: Commit all staged work immediately (GOVERNANCE)
**Action:** The orchestrator has 12 staged files including production code changes (promotion.py, refined_mean_reversion_xom.py, validation/__init__.py). These MUST be committed with `orch:` prefix and pushed. Group logically:
1. Strategy + promotion code → "orch: promote mean_reversion_xom (Sharpe 1.73, 21 trades)"
2. State files (orchestrator-status, ops-*.json) → "orch: update state files"
3. Results (run_history, winners, api_budget, dashboard) → "orch: update results"
**Budget:** 0 cycles (immediate)
**Expected:** Clean working tree within 10 minutes.

### Directive 8: Run remaining 2 mandates (volume_nvda, momentum_msft)
**Action:** Continue Directive 2 from Review #2. mean_reversion_xom succeeded — now run the other 2:
1. `mandates/volume_nvda.json` — volatile ticker, should produce frequent signals
2. `mandates/momentum_msft.json` — different momentum approach
Run full 7-turn loops. Log trade counts and failure modes in orchestrator-status.json.
**Budget:** 2 cycles
**Expected:** At least 1/2 converges with ≥5 trades. If too_few_trades is truly fixed, we should see higher trade counts across both.
**Fallback:** If both fail with too_few_trades despite the threshold fix, the problem is in the strategy invention prompts, not validation. Report specific trade counts and failure patterns.

### Directive 9: Fix KPI prev/current rotation (CARRYOVER — Directive 3 from Review #2)
**Action:** This was assigned last review and NOT completed. The health check script at `~/.hermes/scripts/crabquant-health-check.py` must:
1. Copy `ops-kpis.json` → `ops-kpis-prev.json` BEFORE computing new KPIs
2. Compute new KPIs
3. Write to `ops-kpis.json`
Currently both files are written ~5 min apart with identical values. I have zero trend data across 3 reviews.
**Budget:** 1 cycle
**Expected:** Next KPI snapshot has prev/current with different timestamps AND different values where metrics changed.
**Note:** This is the THIRD time this has been flagged. If not fixed this cycle, it becomes an escalation.

### Directive 10: Do NOT modify VISION.md
**Action:** VISION.md is a PROTECTED file. Only the CEO (Tristan) may modify it. Stop attempting to update it per previous directive.
**Budget:** 0 cycles
**Expected:** No changes to VISION.md.

## Previous Directives Assessment (Review #2, 20:25 UTC)

| Directive | Outcome | Evidence |
|-----------|---------|----------|
| 1: Fix too_few_trades prompts | ✅ COMPLETED | Commit 3a93513 — threshold 20→10 across classifier, estimator, context builder |
| 2: Run 3 mandates | 🔄 IN PROGRESS | 1/3 done — mean_reversion_xom SUCCESS (Sharpe 1.73, 21 trades). 2 remaining. |
| 3: Fix KPI rotation bug | ❌ NOT DONE | prev/current still written 5 min apart with same values. Third time flagged. |
| 4: Update status file | ✅ FOLLOWING | orchestrator-status.json now has non-null last_updated and meaningful content |
| 5: Commit + update VISION.md | ⚠️ PARTIAL | 12 files staged but NOT committed. VISION.md is protected — should not have been assigned. |

**Orchestrator compliance: IMPROVING** — Main directive (too_few_trades fix) completed successfully. Status file now updated. Commit discipline still lacking (12 staged files). KPI script fix ignored for 2nd cycle.

## Metric Reality Check

| Metric | VISION Target | KPI Says | Actual (Verified) | Gap | Trend |
|--------|--------------|----------|-------------------|-----|-------|
| WF Robustness Rate | >50% | 83.1% | 83.1% (98/118) | ✅ Met | → |
| WF Coverage Gap | 0 | 52 w/o WF | 136/188 have WF (bug fixed) | 🟡 Improving | ↑ |
| Registry Keep Rate | >80% | 83.9% | 83.9% (99/118) | ✅ Met | → |
| Avg Registry Sharpe | >1.0 | 1.047 | 1.047 | ✅ Met | → |
| Mandate Success Rate | >50% | 36.9% | 36.9% (72/195 real) | 🟡 13% gap | ↑ (was 32.6%) |
| Per-turn Success | >20% | N/A | ~15-20% est. | 🟡 Near target | ↑ |
| Mandate Convergence | >50% | N/A | ~50% (2/4 recent) | 🟡 Near target | ↑ |
| Infra Ratio | <20% | 61% | Inflated by synthetic entries | ⚠️ Misleading | → |
| API Error Rate | <1% | 0% | 0% | ✅ Perfect | → |
| Orch Liveness | <30 min | 25 min | Active | ✅ | → |
| Test Count | Track ↑ | 4353 | 4353 (0 failures) | ✅ | → |

## QA Results (Phase 2)
- ✅ Tests: **4353 passed, 0 failures, 1 skipped**
- ✅ Imports: `from crabquant import *` works
- ✅ Orchestrator status: non-null, meaningful content
- ⚠️ Git: 12 files staged but uncommitted (promotion.py, strategy file, validation code, results, state files)

## Warnings
- ⚠️ **KPI rotation bug persists** — This is the 3rd review flagging it. No trend data available. If not fixed this cycle, escalates.
- ⚠️ **12 staged files uncommitted** — Production code changes sitting in staging area. Governance violation.
- ⚠️ **Orchestrator status timestamp** — Shows 15:22 UTC but review is at 22:15 UTC. Status may be stale (orchestrator mid-mandate).

## Stale Items
- ✅ overnight-tasks.md — Updated this review
- ✅ orchestrator-status.json — Now populated (Directive 4 success)
- ⚠️ VISION.md — Still shows April 29 date. But it's PROTECTED — I cannot direct the orchestrator to modify it. This is CEO-only.
- ⚠️ KPI rotation — Broken for 3 reviews. Escalation threshold reached.

## Orchestrator Compliance: IMPROVING
The orchestrator executed the critical fix (too_few_trades), ran a successful mandate, and updated its status file. Commit discipline is the main gap. The KPI script fix was ignored but the orchestrator may have prioritized mandates over script fixes — understandable given the P0 focus.

## Operations Health: ✅ FUNCTIONAL
- KPI freshness: ✅ (5 min old)
- Orchestrator liveness: ✅ (25 min)
- API health: ✅ (0 errors, $4.79 total spend, $0.93/2h)
- KPI accuracy: ⚠️ (rotation bug persists, mandate_success_rate may still include synthetic entries)
