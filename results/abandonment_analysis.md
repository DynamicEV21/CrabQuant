# Abandoned Mandates Analysis

**Date:** 2026-04-27  
**Scope:** 5 mandates abandoned at turn 2 with Sharpe -999 (never got a valid backtest)

## Executive Summary

All 5 abandoned mandates share the **exact same failure pattern**: the LLM generated code that failed validation gates on all 3 retry attempts in turn 1, causing the circuit breaker to open and immediately abort on turn 2. **No mandate ever reached the backtest phase.**

The circuit breaker's configuration (window=20, min_pass_rate=30%) is too aggressive for the 3-attempt retry loop — if all 3 attempts in turn 1 fail (0/3 = 0%), the breaker opens immediately with no recovery path.

---

## Per-Mandate Breakdown

### 1. momentum_aapl_20260426_233327
- **Turn 1:** `code_generation_failed` — `SyntaxError: line 55: unindent does not match any outer indentation level`
- **Turn 2:** `circuit_breaker_open` — Window: 0/3 (0.0%) Overall: 0/3
- **Root cause:** LLM produced Python code with invalid indentation. A syntax-level error that should be trivial to fix but the retry loop didn't repair it.

### 2. momentum_nvda_20260426_233537
- **Turn 1:** `code_generation_failed` — `Zero entry signals generated on test data`
- **Turn 2:** `circuit_breaker_open` — Window: 0/3 (0.0%) Overall: 0/3
- **Root cause:** LLM's entry conditions were too strict (MACD bullish + RSI dip/recovery + volume spike all required simultaneously). Over-constrained signal logic produced zero entries on test data. The strategy concept was sound but the RSI recovery threshold (45) combined with volume spike filter was too restrictive.

### 3. multi_rsi_googl_20260426_233645
- **Turn 1:** `code_generation_failed` — `Zero entry signals generated on test data`
- **Turn 2:** `circuit_breaker_open` — Window: 0/3 (0.0%) Overall: 0/3
- **Root cause:** Same as NVDA — over-constrained entry logic producing zero signals. LLM generated overly complex multi-RSI conditions that never triggered.

### 4. trend_amzn_20260426_234020
- **Turn 1:** `code_generation_failed` — `Runtime error in generate_signals: 'ATRr_14'`
- **Turn 2:** `circuit_breaker_open` — Window: 0/3 (0.0%) Overall: 0/3
- **Root cause:** LLM referenced an indicator column name `ATRr_14` that doesn't exist in the indicator cache output. The LLM hallucinated the column name instead of using the correct naming convention. This is a schema mismatch — the LLM doesn't know the exact column naming convention of the `cached_indicator("atr", ...)` output.

### 5. volume_msft_20260426_234247
- **Turn 1:** `code_generation_failed` — `Runtime error in generate_signals: unhashable type: 'Series'`
- **Turn 2:** `circuit_breaker_open` — Window: 0/3 (0.0%) Overall: 0/3
- **Root cause:** LLM used a pandas Series as a dict key or set element — a common pandas programming error. The LLM doesn't have good enough Python/pandas reasoning to avoid this anti-pattern.

---

## Failure Pattern Classification

| Error Category | Mandates | Count |
|---|---|---|
| Syntax error (indentation) | momentum_aapl | 1 |
| Zero entry signals (over-constrained) | momentum_nvda, multi_rsi_googl | 2 |
| Column name hallucination | trend_amzn | 1 |
| Pandas type error | volume_msft | 1 |

## Root Causes

### Primary: Circuit Breaker Too Aggressive
The circuit breaker opens when pass rate drops below 30% in a window of 20. The retry loop makes exactly 3 attempts per turn. If all 3 fail (0/3 = 0% < 30%), the breaker **immediately opens on the very next turn with no chance to recover**. This is the proximate cause of all 5 abandonments — the circuit breaker killed them, not the underlying errors.

### Secondary: LLM Code Quality Issues (4 distinct types)
1. **Syntax errors** — basic Python indentation failures
2. **Over-constrained signals** — entry conditions too strict, producing zero trades
3. **Schema hallucination** — wrong indicator column names (e.g., `ATRr_14` vs actual naming)
4. **Pandas misuse** — using Series as dict keys

### Tertiary: No Error Feedback Loop
The validation gates detect the errors but the 3 retry attempts all call the LLM with the same (or minimally different) context. There's no mechanism to feed the specific error back to the LLM within the retry loop (e.g., "your code had an indentation error on line 55, fix it"). Each retry is essentially a fresh generation attempt.

---

## Recommendations

1. **Increase circuit breaker window or lower threshold** — either require more attempts before opening (e.g., window=10 instead of 20 with the same 30%), or lower min_pass_rate to 15-20%, so that 0/3 failures don't immediately trigger it.
2. **Add error feedback to retry loop** — when a gate fails with a specific error, inject that error into the next retry's LLM context so the LLM can fix it.
3. **Add indicator schema to LLM context** — include the exact column naming conventions for `cached_indicator` outputs so the LLM doesn't hallucinate column names.
4. **Decouple circuit breaker from turn advancement** — allow turn 2 to proceed even if turn 1 had failures, giving the LLM a fresh attempt with updated context (the turn 1 failure would be in the history).
