# Promotion Gap Analysis

**Date:** 2026-04-27  
**Scope:** Why 0 strategies were promoted despite 2 mandates appearing successful

## Executive Summary

**One strategy WAS promoted.** The mandate `e2e_test_momentum_20260426_232049` (Sharpe 1.537) successfully passed full validation and was auto-promoted to both `winners.json` and `STRATEGY_REGISTRY`. It appears at the end of `results/winners/winners.json` as `"strategy": "refined_e2e_test_momentum"`.

The second mandate (`e2e_test_momentum_20260426_190716`, Sharpe 1.873) was **never promoted because its status was `max_turns_exhausted`, not `success`**. Despite having the best Sharpe (1.87) well above its target (0.5), it never triggered the promotion code path because the success check (`result.sharpe >= sharpe_target and result.passed`) was only evaluated **during** the loop, and the best Sharpe was achieved on turn 2 which continued to turn 3 (turn 3 then failed code generation).

---

## Detailed Analysis

### Mandate 1: e2e_test_momentum_20260426_232049 ✅ PROMOTED
- **Status:** `success`
- **Sharpe target:** 1.5 | **Best Sharpe:** 1.537 (turn 1)
- **Tickers:** `["AAPL"]`
- **Validation:** `passed: true` (walk-forward robust: true, cross-ticker robust: true)
- **Walk-forward:** Train Sharpe 1.18 → Test Sharpe 1.13 (4.3% degradation)
- **Cross-ticker:** null (only 1 ticker in mandate, so skipped — passes by default)
- **Outcome:** Successfully auto-promoted to `results/winners/winners.json` as `"refined_e2e_test_momentum"`

### Mandate 2: e2e_test_momentum_20260426_190716 ❌ NOT PROMOTED
- **Status:** `max_turns_exhausted`
- **Sharpe target:** 0.5 | **Best Sharpe:** 1.873 (turn 2)
- **Tickers:** `["AAPL"]`
- **Turn 1:** Sharpe 1.16 → continued (failure_mode: "low_sharpe" but above target of 0.5... **BUG**)
- **Turn 2:** Sharpe 1.87 → continued (failure_mode: "too_few_trades")
- **Turn 3:** `code_generation_failed` — "Overtrading: 150 entry signals on 251 bars"
- **Outcome:** Loop exhausted. Status set to `max_turns_exhausted`. **No promotion code was ever reached.**

---

## Code Path Trace: Orchestrator Success → Promotion

### The success check (scripts/refinement_loop.py, ~line 327):
```python
if result.sharpe >= sharpe_target and result.passed:
    # ... run_full_validation_check ...
    # ... auto_promote if validation passed ...
    return state
```

This check is **only inside the main for-loop**, executed at the end of each turn's backtest. The promotion path is:

1. **Sharpe check:** `result.sharpe >= sharpe_target` AND `result.passed`
2. **Full validation:** `run_full_validation_check(strategy_fn, params, discovery_ticker, validation_tickers)`
   - Walk-forward test (min OOS Sharpe ≥ 0.5, must be `robust`)
   - Cross-ticker test (min avg Sharpe ≥ 0.5, must be `robust`) — skipped if only 1 ticker
3. **Auto-promote:** If `validation["passed"]` → calls `auto_promote()` which:
   - Checks `is_already_registered()` (prevents duplicates)
   - Calls `register_strategy()` (writes .py file, inserts into STRATEGY_REGISTRY)
   - Calls `promote_to_winner()` (writes to winners.json)
4. **Fallback:** If validation NOT passed → calls `promote_to_winner()` (legacy path, writes to winners only)

### What happens on max_turns_exhausted (line ~375):
```python
state.status = "max_turns_exhausted"
save_state(run_dir, state)
```
**No promotion logic.** The loop simply exits with no post-loop promotion step.

---

## Bug: Mandate 2 Should Have Been Promoted

### Why it wasn't caught on turn 1:
- Turn 1 Sharpe: 1.16 vs target 0.5 → `result.sharpe >= sharpe_target` is True
- BUT the code also requires `result.passed` to be True
- The classifier assigned `failure_mode: "low_sharpe"` even though 1.16 > 0.5
- This suggests `result.passed` may have been False (possibly due to trade count or other guardrail failures)

### Why it wasn't caught on turn 2:
- Turn 2 Sharpe: 1.87 vs target 0.5 → `result.sharpe >= sharpe_target` is True
- BUT `failure_mode: "too_few_trades"` was assigned
- Again, `result.passed` was likely False due to too few trades

### The classifier seems misaligned with the sharpe_target:
The `classify_failure()` function assigns failure modes based on its own internal thresholds (e.g., minimum trade count), which may be independent of the mandate's sharpe_target. A strategy with excellent Sharpe (1.87) but few trades gets `passed=False` from the backtest engine, which blocks the promotion path.

---

## Root Causes

### 1. No Post-Loop Promotion (Critical)
When a mandate exhausts all turns, the best strategy (stored in `state.best_sharpe`, `state.best_code_path`) is **never evaluated for promotion**. The promotion code only runs inside the loop when a turn immediately succeeds. If the best turn had `passed=False` (due to secondary constraints like trade count), it never gets promoted even if its Sharpe far exceeds the target.

### 2. `result.passed` Blocks Promotion Despite High Sharpe
The success check requires BOTH `result.sharpe >= sharpe_target` AND `result.passed`. The `passed` flag comes from the backtest engine and may be False due to non-Sharpe constraints (trade count, drawdown, etc.). This means a strategy with excellent Sharpe can be blocked from promotion by unrelated guardrails.

### 3. Mandate 2's sharpe_target Was 0.5 — Should Have Passed Easily
With a target of 0.5 and best Sharpe of 1.87, this mandate should have succeeded on turn 1. The fact that it didn't suggests the `result.passed` flag is the blocker, not the Sharpe check.

---

## Recommendations

1. **Add post-loop promotion for `max_turns_exhausted`:** After the loop exits due to max turns, check if `state.best_sharpe >= sharpe_target` and if so, attempt promotion on the best strategy even though it wasn't an immediate success.

2. **Decouple `result.passed` from the success check:** The promotion path should use Sharpe as the primary gate. If `passed=False` due to trade count, still promote but log a warning. Or use a softer "promotable" flag that only requires Sharpe and basic viability.

3. **Fix classifier/sharpe_target alignment:** The `classify_failure()` function should be aware of the mandate's `sharpe_target`. If Sharpe exceeds the target, it should not classify as "low_sharpe" regardless of other failure modes.

4. **Add "best effort" promotion on abandonment:** Even abandoned mandates that achieved a valid backtest (Sharpe > 0) should be evaluated — they might still produce useful strategies even if below the target Sharpe.
