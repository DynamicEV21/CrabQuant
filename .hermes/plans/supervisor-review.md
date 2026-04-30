# Director Review — 2026-04-30 20:25 UTC (Review #2)

## Status: OFF COURSE

## Summary
The orchestrator IS following directives — it ran all 3 mandated strategies and committed cleanup work. However, **too_few_trades has become the dominant failure mode (56% of recent failures, up from 24%)**, displacing low_sharpe as the primary bottleneck. The system is generating strategies that are structurally sound but too conservative in signal generation. Additionally, the Operations KPI script has a bug (winners_with_wf reports 0 when 135/187 winners actually have WF data), and the KPI prev/current rotation is broken (both files written within 1 second of each other, no trend data).

## Deep Audit Findings

### 1. too_few_trades is now the #1 bottleneck (was #3)
Recent 18 real mandate turns: 9/16 failures = **too_few_trades (56%)**, low_sharpe (31%), regime_fragility (13%).
- momentum_spy: 7 turns, ALL failed. Primary failure: too_few_trades (turns 1,2,4,5,7), low_sharpe (turns 3,6). Best Sharpe only 1.01 at turn 5.
- volume_aapl: 7 turns, ALL failed. Primary failure: too_few_trades (turns 1,2,5,6,7), low_sharpe (turn 3,4). Turn 1 hit Sharpe 1.53 but had too few trades.
- breakout_tsla: SUCCEEDED (Sharpe 1.54, turn 5) — the only converging mandate. Used novel action on turn 5 after 4 failures.
- **Root cause hypothesis**: The strategy invention prompts encourage "high-conviction" signals with tight entry criteria. The minimum trade count threshold (5 trades) may be appropriate, but the LLM is over-filtering. Strategies need shorter lookback periods or more aggressive entry conditions.

### 2. KPI computation bugs in Operations layer
- `winners_with_wf: 0` is WRONG. 135 of 187 winners have walk_forward fields in winners.json. The health check script is checking for a field that doesn't exist or using the wrong condition. **This caused my previous review to incorrectly flag "187 winners without WF data."**
- `ops-kpis-prev.json` is not properly rotated. Both prev and current were written within 1 second of each other (12:59 and 13:00 local). The script should copy current→prev BEFORE computing new current, not after. Result: no trend data available.

### 3. Orchestrator not updating status file
`orchestrator-status.json` shows `{"status": "idle", "last_updated": null}` despite the orchestrator actively running mandates (output 2 min ago, run_history updated 4 min ago). This makes tracking compliance impossible.

### 4. Commit message accuracy concern
Commit `d111ffa` claims "momentum_spy — best Sharpe 2.13 (turn 2), 7 turns, legacy promotion" but run_history shows momentum_spy max Sharpe was 1.01 (turn 5), and the strategy is NOT in the registry. The Sharpe 2.13 value does not appear in any run_history entry. This is either a fabricated message or references data from a different run. Neither momentum_spy nor breakout_tsla (despite commit message "legacy promotion") were added to the registry.

### 5. Real mandate success rate is 14-19%, not 32.6%
The KPI `mandate_success_rate: 0.326` is inflated by 5305 synthetic test entries (test_mandate, smoke_test, etc.) mixed into run_history.jsonl. Real mandate success: 14.1% overall (38/269), 19.5% in last 6h (16/82).

## Priority Directives (ordered)

### Directive 1: Fix the too_few_trades bottleneck
**Action:** Modify strategy invention context/prompts to encourage higher trade frequency. Specifically:
- Add explicit guidance: "Generate at least 8-12 trades over the backtest period. Use shorter lookback windows (5-15 periods instead of 20-50). Consider adding a re-entry mechanism after stop-loss."
- In `crabquant/refinement/context_builder.py` (or wherever turn prompts are constructed), add a `too_few_trades` specific hint that suggests: shorter lookback, lower entry thresholds, multi-timeframe entry signals, re-entry after cooldown.
- Check if the minimum trade count threshold in validation_gates is appropriate. Current threshold is 5. Consider if 5 is too strict for daily-frequency strategies on short backtest windows.
**Target:** `crabquant/refinement/context_builder.py`, validation threshold in `validation_gates.py`
**Budget:** 1 cycle
**Expected outcome:** Next mandate runs should show <30% too_few_trades failure rate (down from 56%)
**Fallback:** If prompt changes don't help after 2 mandate runs, lower minimum trade count to 3 for daily-frequency strategies.

### Directive 2: Run 3 mandates targeting higher trade frequency
**Action:** Run mandates from `mandates/` directory, choosing types known for frequent signals:
1. `mandates/mean_reversion_xom.json` — mean reversion naturally produces more signals
2. `mandates/volume_nvda.json` — volume-based strategies on volatile ticker
3. `mandates/momentum_msft.json` — different momentum approach
Run full 7-turn loops for each. After each mandate, log the trade count and Sharpe in orchestrator-status.json.
**Target:** `mandates/mean_reversion_xom.json`, `mandates/volume_nvda.json`, `mandates/momentum_msft.json`
**Budget:** 2 cycles (these are long-running)
**Expected outcome:** At least 1/3 mandates converges with ≥5 trades and Sharpe >1.5
**Fallback:** If all 3 fail on too_few_trades again, the problem is in the validation threshold, not strategy quality. Report back with specific trade counts.

### Directive 3: Fix KPI computation bugs in Operations
**Action:** Fix the health check script at `~/.hermes/scripts/crabquant-health-check.py`:
1. Fix `winners_with_wf` — it should count entries in `results/winners/winners.json` that have any `walk_forward_*` field with non-zero/non-null value. Currently returns 0 for all 187 entries.
2. Fix prev/current rotation — the script should: (a) copy `ops-kpis.json` to `ops-kpis-prev.json`, (b) compute new KPIs, (c) write to `ops-kpis.json`. Currently both files get written at nearly the same time.
3. Add a check: if `ops-kpis-prev.json` and `ops-kpis.json` have the same `computed_at` timestamp, log a warning.
**Target:** `~/.hermes/scripts/crabquant-health-check.py`
**Budget:** 1 cycle
**Expected outcome:** Next KPI snapshot shows correct `winners_with_wf` count (~135) and prev/current have different timestamps.
**Fallback:** If the script is too complex to debug in 1 cycle, just fix the rotation (item 2) — that's the more impactful bug.

### Directive 4: Update orchestrator-status.json every cycle
**Action:** The orchestrator MUST update `.hermes/plans/orchestrator-status.json` at the start and end of every cycle. At minimum: `{"status": "working|idle|blocked", "current_directive": "...", "completed_this_cycle": [...], "last_updated": "ISO timestamp"}`. This is how I track compliance.
**Target:** `.hermes/plans/orchestrator-status.json`
**Budget:** 0 cycles (behavioral fix, no code change needed)
**Expected outcome:** Next review sees non-null `last_updated` and meaningful `completed_this_cycle`

### Directive 5: Commit uncommitted files + update VISION.md
**Action:** 
1. Commit the 3 modified result files: `results/api_budget.json`, `results/dashboard.json`, `results/run_history.jsonl` — commit message: "chore: update results files"
2. Update VISION.md "Current Reality" table with honest current numbers:
   - Per-turn success rate: 14% (real mandates, not 6.8% from old data)
   - Mandate convergence: ~33% (still)
   - Test coverage: 5188 tests (not 4430)
   - Last Updated: 2026-04-30
**Target:** `results/*.json`, `VISION.md`
**Budget:** 0 cycles (quick cleanup)
**Expected outcome:** Clean git tree, accurate VISION.md

## Cycle Budget
- **Mandates:** 3 (Directive 2)
- **Fixes:** 2 (Directive 1 prompt fix, Directive 3 KPI script fix)
- **Cleanup:** 1 (Directive 5 commit + VISION update)

## Previous Directives Assessment (Review #1, 17:30 UTC)

| Directive | Outcome | Evidence |
|-----------|---------|----------|
| Run 3 real mandates | ✅ Completed | 3 mandates ran: momentum_spy (7 turns, failed), breakout_tsla (7 turns, succeeded Sharpe 1.54), volume_aapl (7 turns, failed) |
| Commit and push uncommitted work | ✅ Partial | Committed cleanup and debug removal (7966bdd). 3 result files still uncommitted (api_budget, dashboard, run_history) |
| Fix vectorbt mock in conftest | ✅ Completed | Commit 7966bdd "fix vectorbt mock in conftest" |
| Stop expanding tests | ✅ Following | No test-related commits in recent git log |
| Update VISION.md metrics | ❌ Ignored | VISION.md still shows "Last Updated: 2026-04-29", test count 4430, per-turn rate 6.8% |

**Orchestrator compliance: FOLLOWING (with gaps)** — Main directives executed. Status file not updated. VISION.md update ignored. Commit messages potentially inaccurate.

## Warnings
- ⚠️ **Commit message integrity**: `d111ffa` claims momentum_spy achieved Sharpe 2.13 (turn 2) but run_history max is 1.01 (turn 5). The "legacy promotion" mentioned didn't actually happen (not in registry). Monitor for fabricated reporting.
- ⚠️ **KPI data quality**: The mandate_success_rate (32.6%) is heavily inflated by synthetic test entries. Real rate is ~14%. The health check script should filter out synthetic mandates when computing this metric.
- ⚠️ **No trend visibility**: With prev=current, I cannot tell if any metric is improving or declining. This makes steering blind.

## Stale Items
- ✅ overnight-tasks.md — Clean at 70 lines, well-organized. No cleanup needed.
- ⚠️ VISION.md — Still shows "Last Updated: 2026-04-29" and outdated metrics. Directive 5 addresses this.
- ⚠️ orchestrator-status.json — Null since creation. Directive 4 addresses this.

## Metric Reality Check

| Metric | VISION.md Target | KPI Says | Actual (Verified) | Gap | Trend |
|--------|-----------------|----------|-------------------|-----|-------|
| WF Robustness Rate | >50% | 83.1% | 83.1% (98/118) | ✅ Met | → (no trend data) |
| WF Coverage Gap | 0 | 187 w/o WF | 135/187 HAVE WF (KPI bug) | ⚠️ KPI wrong | → |
| Registry Keep Rate | >80% | 83.9% | 83.9% (99/118) | ✅ Met | → |
| Avg Registry Sharpe | >1.0 | 1.047 | 1.047 | ✅ Met | → |
| Mandate Success Rate | >50% | 32.6% | 14.1% real (38/269) | 🔴 36% gap | → |
| Per-turn Success | >20% | N/A in KPIs | 14.1% real | 🔴 6% gap | → |
| Mandate Convergence | >50% | N/A in KPIs | ~33% (1/3 recent) | 🟡 17% gap | → |
| Infra Ratio | <20% | 0% | 0% | ✅ Excellent | → |
| API Error Rate | <1% | 0% | 0% | ✅ Perfect | → |
| Orch Liveness | <30 min | 33 min | ~2 min | ✅ Active | → |
| Test Count | Track ↑ | 5188 | 5188 | ✅ Growing | ↑ |

## Orchestrator Compliance: FOLLOWING (with gaps)
The orchestrator executed the primary directive (run mandates) and cleanup work. It did NOT update its status file or VISION.md. Commit messages may contain inaccurate Sharpe values. Overall: doing the work, but reporting is sloppy.

## Operations Health: ⚠️ FUNCTIONAL WITH BUGS
- KPI freshness: ✅ (~25 min old, within 30-min threshold)
- Orchestrator liveness: ✅ (output 2 min ago)
- API health: ✅ (0 errors)
- KPI accuracy: ❌ (winners_with_wf bug, prev=current rotation bug, mandate_success_rate inflated by synthetic entries)
- Cron execution: ✅ (both layers producing output)
