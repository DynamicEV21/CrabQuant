# E2E Promotion Test Results

**Date:** 2026-04-27 03:20–03:40 PDT  
**Model:** glm-5-turbo (default, GLM-4.7 not available as CLI flag)  
**Sharpe Target:** 0.5 (low, to maximize promotion chance)  
**Max Turns:** 5  

---

## Summary

| Mandate | Status | Best Sharpe | Turns Used | Promoted? | Notes |
|---------|--------|-------------|------------|-----------|-------|
| momentum_aapl | `max_turns_exhausted` | 0.28 | 5/5 | ❌ No | Below 0.5 target |
| breakout_spy | `running` (killed) | -0.11 | 2/5 | ❌ No | SIGKILL'd (LLM timeout cascade) |
| e2e_test_momentum | `success` | **1.55** | 2/5 | ✅ Yes (legacy) | In winners.json; NOT in STRATEGY_REGISTRY |

---

## Mandate 1: momentum_aapl

**Run directory:** `refinement_runs/momentum_aapl_20260427_032056`

### Per-Turn Breakdown

| Turn | Sharpe | Status | Notes |
|------|--------|--------|-------|
| 1 | N/A | `code_generation_failed` | 3 validation attempts failed (atr() API errors, SyntaxError) |
| 2 | 0.27 | backtested | First successful backtest |
| 3 | -0.97 | backtested | Negative Sharpe, regression |
| 4 | 0.10 | backtested | Slight improvement |
| 5 | 0.28 | backtested | Best but still below 0.5 target |

### Result
- **Final status:** `max_turns_exhausted`
- **Best Sharpe:** 0.278 (turn 5)
- **Promotion:** Not attempted (Sharpe < 0.5 target)

### Full stdout
```
Starting refinement loop for: mandates/momentum_aapl.json

============================================================
Turn 1/5 | Best Sharpe: -999.00
  Calling LLM...
  LLM responded in 26.5s
  Gate failed (attempt 1): ["Runtime error in generate_signals: atr() missing 2 required positional arguments: 'low' and 'close'"]
  Calling LLM...
  LLM responded in 20.5s
  Gate failed (attempt 2): ["Runtime error in generate_signals: atr() missing 2 required positional arguments: 'low' and 'close'"]
  Calling LLM...
  LLM responded in 22.3s
  Gate failed (attempt 3): ['SyntaxError: line 61: unindent does not match any outer indentation level']
  All 3 validation attempts failed. Advancing turn.

============================================================
Turn 2/5 | Best Sharpe: -999.00
  Calling LLM...
  LLM responded in 30.3s
  Running backtest...
  Backtest completed in 4.9s
  Turn 2 completed in 35.3s

============================================================
Turn 3/5 | Best Sharpe: 0.27
  Calling LLM...
  LLM responded in 57.6s
  Gate failed (attempt 1): ["Runtime error in generate_signals: atr() missing 2 required positional arguments: 'low' and 'close'"]
  Calling LLM...
  LLM responded in 37.9s
  Running backtest...
  Backtest completed in 0.2s
  Turn 3 completed in 95.7s

============================================================
Turn 4/5 | Best Sharpe: 0.27
  Calling LLM...
  LLM responded in 74.6s
  Running backtest...
  Backtest completed in 0.1s
  Turn 4 completed in 74.7s

============================================================
Turn 5/5 | Best Sharpe: 0.27
  Calling LLM...
  LLM responded in 78.3s
  Running backtest...
  Backtest completed in 0.1s
  Turn 5 completed in 78.4s
  Max turns exhausted. Best Sharpe: 0.28 at turn 5

Final status: max_turns_exhausted
Best Sharpe: 0.28
```

---

## Mandate 2: breakout_spy

**Run directory:** `refinement_runs/breakout_spy_20260427_032702`

### Per-Turn Breakdown

| Turn | Sharpe | Status | Notes |
|------|--------|--------|-------|
| 1 | -0.61 | backtested | Negative Sharpe |
| 2 | -0.11 | backtested | Improving but still negative |
| 3 | N/A | `llm_failed` | All 3 LLM retries timed out → SIGKILL |

### Result
- **Final status:** `running` (interrupted by SIGKILL)
- **Best Sharpe:** -0.106 (turn 2)
- **Promotion:** Not attempted (killed before completion)
- **Error:** LLM API returned ReadTimeout on 3 consecutive retries (2s, 4s, 8s backoff), then the entire process was SIGKILL'd (likely OOM from accumulating retry state, or external kill)

### Full stdout
```
Starting refinement loop for: mandates/breakout_spy.json

============================================================
Turn 1/5 | Best Sharpe: -999.00
  Calling LLM...
  LLM responded in 64.4s
  Running backtest...
  Backtest completed in 4.9s
  Turn 1 completed in 69.3s

============================================================
Turn 2/5 | Best Sharpe: -0.61
  Calling LLM...
  LLM responded in 27.3s
  Running backtest...
  Backtest completed in 0.1s
  Turn 2 completed in 27.4s

============================================================
Turn 3/5 | Best Sharpe: -0.11
  Calling LLM...
  LLM call failed (ReadTimeout), retry 1/3 in 2s...
  Calling LLM...
  LLM call failed (ReadTimeout), retry 2/3 in 4s...
  Calling LLM...
  LLM call failed (ReadTimeout), retry 3/3 in 8s...
  Calling LLM...
LLM inventor call failed: The read operation timed out
  LLM call failed (attempt 1)
  Calling LLM...
[Process killed by SIGKILL]
```

---

## Mandate 3: e2e_test_momentum ✅ PROMOTED

**Run directory:** `refinement_runs/e2e_test_momentum_20260427_033745`

### Per-Turn Breakdown

| Turn | Sharpe | Status | Notes |
|------|--------|--------|-------|
| 1 | N/A | `code_generation_failed` | 3 validation attempts failed (atr() API errors, zero signals) |
| 2 | **1.55** | backtested | **🏆 SUCCESS** — Sharpe 1.55 >= 0.5 target |

### Result
- **Final status:** `success`
- **Best Sharpe:** 1.555 (turn 2)
- **Promotion:** ✅ YES — added to `results/winners/winners.json`
  - Strategy name: `refined_e2e_test_momentum`
  - Promoted at: `2026-04-27T10:39:20 UTC`
  - **NOT registered in STRATEGY_REGISTRY** (validation failed)

### Validation Details
- **Walk-forward test:** FAILED — severe regime shift
  - Train Sharpe: 1.37, Test Sharpe: 0.01
  - Degradation: 99.3%
  - Train regime: `mean_reversion` → Test regime: `trending_up`
- **Cross-ticker robustness:** PASSED (not null)
- **Overall validation.passed:** `false` (walk-forward failed)

The strategy was promoted via the **"legacy promotion"** path (which adds to winners.json regardless of validation status), but was **blocked from STRATEGY_REGISTRY** by `auto_promote()` which requires `validation["passed"] == True`.

### Full stdout
```
Starting refinement loop for: mandates/e2e_test_momentum.json

============================================================
Turn 1/5 | Best Sharpe: -999.00
  Calling LLM...
  LLM responded in 27.0s
  Gate failed (attempt 1): ["Runtime error in generate_signals: atr() missing 2 required positional arguments: 'low' and 'close'"]
  Calling LLM...
  LLM responded in 22.2s
  Gate failed (attempt 2): ['Zero entry signals generated on test data']
  Calling LLM...
  LLM responded in 25.4s
  Gate failed (attempt 3): ["Runtime error in generate_signals: atr() missing 2 required positional arguments: 'low' and 'close'"]
  All 3 validation attempts failed. Advancing turn.

============================================================
Turn 2/5 | Best Sharpe: -999.00
  Calling LLM...
  LLM responded in 14.9s
  Running backtest...
  Backtest completed in 5.0s
  🏆 SUCCESS! Sharpe 1.55 >= 0.5
    - Low trade count: 7 (marginal statistical significance)
  📋 Validation not passed — using legacy promotion...
  Turn 2 completed in 20.1s

Final status: success
Best Sharpe: 1.55
```

---

## Recurring Issues

1. **`atr()` API confusion:** The LLM consistently generates `atr()` calls with wrong arguments (`atr(close)` instead of `atr(high, low, close)`). This causes ~40% of validation failures across all mandates. Consider adding a more explicit example or helper wrapper.

2. **LLM timeouts:** The z.ai API can time out (ReadTimeout) especially for longer prompts. The retry mechanism (3 retries with 2/4/8s backoff) works but sometimes all retries fail, leading to process death.

3. **Walk-forward regime shift:** Even strategies with high Sharpe on training data fail walk-forward validation due to regime shifts. This is expected but means the `auto_promote` → STRATEGY_REGISTRY path is very strict.

4. **Low trade count:** The promoted strategy only had 7 trades, flagged as "marginal statistical significance." Consider enforcing a higher `min_trades` constraint.

5. **Legacy promotion vs. auto_promote gap:** There's a disconnect where `promote_to_winner()` (legacy) adds to winners.json unconditionally, but `auto_promote()` requires validation to pass for STRATEGY_REGISTRY. This means winners.json can accumulate strategies that aren't actually robust.

---

## Pipeline Health Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| LLM API connectivity | ⚠️ Partial | Works but intermittent timeouts |
| Code generation | ⚠️ Partial | atr() API errors common; needs better prompting |
| Validation gates | ✅ Working | Correctly catches runtime errors and zero signals |
| Backtesting | ✅ Working | Fast and reliable |
| Sharpe evaluation | ✅ Working | Correct threshold comparison |
| Walk-forward validation | ✅ Working | Correctly detects regime shifts |
| Legacy promotion | ✅ Working | Adds to winners.json on success |
| auto_promote → STRATEGY_REGISTRY | ✅ Working (correctly strict) | Blocks unvalidated strategies |
| Error recovery | ✅ Working | Retries on validation failures; advances turn on exhaustion |
| SIGKILL resilience | ❌ No | Process dies on cascading LLM timeouts |
