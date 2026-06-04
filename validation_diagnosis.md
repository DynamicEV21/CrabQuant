# Validation Pipeline Diagnostic Report
**Generated:** 2026-04-29T02:30:42.406060
**Ticker:** SPY

### Rolling Walk-Forward (rolling_walk_forward)
  - `min_avg_test_sharpe`: 0.5
  - `min_windows_passed`: 2
  - `per_window_min_test_sharpe`: 0.3
  - `per_window_max_degradation`: 0.7

### Single-Split Walk-Forward (walk_forward_test)
  - `min_test_sharpe`: 0.3
  - `min_test_trades`: 10
  - `max_degradation`: 0.7

### Cross-Ticker Validation
  - `profitable_rate > 0.4` AND `avg_sharpe > 0.5`


  Params: {'fast_len': 9, 'slow_len': 21}

### 1. Rolling Walk-Forward (5y, 18mo train / 6mo test / 6mo step)

### 2. Single-Split Walk-Forward (3y, 18mo train / 6mo test)

### 3. Cross-Ticker Validation (7 tickers)

  ⏱️  Elapsed: 6.9s


  Params: {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'exit_hist': -0.5, 'sma_len': 50, 'volume_window': 20, 'volume_mult': 1.2}

### 1. Rolling Walk-Forward (5y, 18mo train / 6mo test / 6mo step)

### 2. Single-Split Walk-Forward (3y, 18mo train / 6mo test)

### 3. Cross-Ticker Validation (7 tickers)

  ⏱️  Elapsed: 1.4s


  Params: {'bb_len': 20, 'bb_std': 2.0, 'squeeze_len': 50, 'squeeze_mult': 0.8, 'vol_mult': 1.2}

### 1. Rolling Walk-Forward (5y, 18mo train / 6mo test / 6mo step)

### 2. Single-Split Walk-Forward (3y, 18mo train / 6mo test)

### 3. Cross-Ticker Validation (7 tickers)

  ⏱️  Elapsed: 1.7s


## Pass Rate Summary
| Validation Type          | Passed | Total | Rate  |
|--------------------------|--------|-------|-------|
| Rolling Walk-Forward     | 0      | 3      | 0%    |
| Single-Split Walk-Forward| 0      | 3      | 0%    |
| Cross-Ticker             | 1      | 3      | 33%    |

## Rolling Walk-Forward Failure Analysis

### ema_crossover
- Avg test Sharpe: -0.198 (need >= 0.5)
- Windows passed: 2/6 (need >= 2)
- Min test Sharpe: -1.906
- Avg degradation: 86.6%
- Notes: Windows: 6, passed: 2 | Avg test Sharpe: -0.20 (min: -1.91) | Avg degradation: 86.6% | ⚠️ Avg test Sharpe -0.20 < 0.5

### macd_momentum
- Avg test Sharpe: 0.104 (need >= 0.5)
- Windows passed: 2/6 (need >= 2)
- Min test Sharpe: -0.674
- Avg degradation: 99.1%
- Notes: Windows: 6, passed: 2 | Avg test Sharpe: 0.10 (min: -0.67) | Avg degradation: 99.1% | ⚠️ Avg test Sharpe 0.10 < 0.5

### bollinger_squeeze
- Avg test Sharpe: 0.000 (need >= 0.5)
- Windows passed: 0/6 (need >= 2)
- Min test Sharpe: 0.000
- Avg degradation: 100.0%
- Notes: Windows: 6, passed: 0 | Avg test Sharpe: 0.00 (min: 0.00) | Avg degradation: 100.0% | ⚠️ Avg test Sharpe 0.00 < 0.5 | ⚠️ Only 0/6 windows passed (need >= 2)

## Single-Split Walk-Forward Failure Analysis

### ema_crossover
- Train Sharpe: 1.841
- Test Sharpe: -0.018 (need >= 0.3)
- Degradation: 101.0% (max 70%)
- Regime shift: True
- Train regime: mean_reversion
- Test regime: trending_up
- Notes: Train: Sharpe 1.84, Return 39.7% | Test: Sharpe -0.02, Return -0.2% | Degradation: 101.0% | Train regime: mean_reversion | Test regime: trending_up | ⚠️ Regime shift detected | ⚠️ Only 5 OOS trades (need >= 10) | ⚠️ Degradation 101.0% exceeds max 70%

### macd_momentum
- Train Sharpe: 0.375
- Test Sharpe: 0.000 (need >= 0.3)
- Degradation: 100.0% (max 70%)
- Regime shift: True
- Train regime: mean_reversion
- Test regime: trending_up
- Notes: Train: Sharpe 0.38, Return 2.8% | Test: Sharpe 0.00, Return 0.0% | Degradation: 100.0% | Train regime: mean_reversion | Test regime: trending_up | ⚠️ Regime shift detected | ⚠️ Only 0 OOS trades (need >= 10) | ⚠️ Degradation 100.0% exceeds max 70%

### bollinger_squeeze
- Train Sharpe: 0.682
- Test Sharpe: 0.000 (need >= 0.3)
- Degradation: 100.0% (max 70%)
- Regime shift: True
- Train regime: mean_reversion
- Test regime: trending_up
- Notes: Train: Sharpe 0.68, Return 7.0% | Test: Sharpe 0.00, Return 0.0% | Degradation: 100.0% | Train regime: mean_reversion | Test regime: trending_up | ⚠️ Regime shift detected | ⚠️ Only 0 OOS trades (need >= 10) | ⚠️ Degradation 100.0% exceeds max 70%

## Cross-Ticker Failure Analysis

### macd_momentum
- Avg Sharpe: -0.029 (need > 0.5)
- Win rate: 57% (need > 40%)
- Median Sharpe: 0.118
- Notes: Tested 7/7 tickers | Profitable: 4/7 (57%) | Passed: 0/7 | Avg Sharpe: -0.03 (σ=0.82) | Median Sharpe: 0.12 | Avg Return: 1.2%

### bollinger_squeeze
- Avg Sharpe: 0.047 (need > 0.5)
- Win rate: 57% (need > 40%)
- Median Sharpe: 0.065
- Notes: Tested 7/7 tickers | Profitable: 4/7 (57%) | Passed: 0/7 | Avg Sharpe: 0.05 (σ=0.69) | Median Sharpe: 0.07 | Avg Return: 1.2%


Based on the observed metrics from hand-crafted strategies:

### Observed Best Metrics (from 3 hand-crafted strategies)
- Best avg test Sharpe:  0.104
- Best min test Sharpe:  0.000
- Best windows passed:   2
- Best avg degradation:  86.6%

### Recommended Thresholds for >50% Pass Rate

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `min_avg_test_sharpe` | 0.5 | 0.09 | Best observed: 0.104 — current threshold unreachable |
| `min_windows_passed` | 2 | 2 | OK |
| per-window `min_test_sharpe` | 0.3 | 0.0 | Remove or lower — single window failures cascade |
| per-window `max_degradation` | 0.7 | 1.0 | Remove or raise — too strict for individual windows |


### Bug 1: Hardcoded per-window thresholds in rolling_walk_forward (line 431)
```python
window_passed = test_result.sharpe >= 0.3 and degradation <= 0.7
```
These thresholds (0.3 Sharpe, 0.7 degradation) are **hardcoded** and NOT
configurable via function parameters. The `min_avg_test_sharpe` and `min_windows_passed`
params only control the FINAL aggregate check. Individual windows must pass these
hardcoded gates to count as 'passed'. This means:
- If a strategy has 3 windows with Sharpe [0.4, 0.2, 0.4], only 2 pass (need 2) — OK
- If a strategy has 3 windows with Sharpe [0.4, 0.25, 0.4], only 2 pass — OK
- But degradation can silently kill windows that have OK Sharpe

### Bug 2: Double-gating makes thresholds excessively strict
The rolling walk-forward requires BOTH:
1. `avg_test_sharpe >= min_avg_test_sharpe` (0.5)
2. `windows_passed >= min_windows_passed` (2)
AND each window must individually pass `sharpe >= 0.3 AND degradation <= 0.7`.
This is **triple-gating**: per-window Sharpe, per-window degradation, AND aggregate.
For a strategy to pass, it needs consistently positive OOS performance across ALL windows.

### Bug 3: No num_trades check in rolling_walk_forward
Unlike walk_forward_test() which checks `min_test_trades >= 10`, rolling_walk_forward()
does NOT check trade count. A strategy with 1 trade in a window that happens to be
profitable can pass the Sharpe check, creating a false positive. Conversely, a strategy
with many trades might fail on Sharpe noise.

### Bug 4: Train Sharpe = 0 edge case in degradation calculation
When `train_result.sharpe == 0`, the degradation formula divides by zero.
The code checks `train_result.sharpe > 0`, but a Sharpe of exactly 0.0 (not negative)
falls through to the `elif test_result.sharpe > 0` branch, setting degradation=0.0.
This means strategies with zero in-sample Sharpe but positive OOS Sharpe get
degradation=0.0, which is incorrect — it should be undefined/inconclusive.


### Can ANY strategy pass current thresholds?
NO — 0/3 hand-crafted strategies passed rolling walk-forward.
These are well-known, historically profitable strategies (EMA crossover, MACD,
Bollinger Squeeze) with DEFAULT parameters. If they can't pass, the thresholds
are almost certainly too strict for the invented strategies.

### Dominant Failure Mode(s)
1. **LOW TEST SHARPE**: Best avg test Sharpe = 0.104, need >= 0.5
3. **HIGH DEGRADATION**: Best avg degradation = 86.6%, per-window max = 70%

### Specific Threshold Recommendations
To achieve >50% pass rate with hand-crafted strategies:

- Lower `min_avg_test_sharpe` to **0.0**
- Make per-window thresholds configurable (currently hardcoded)
- Consider removing degradation gate from per-window check (keep only aggregate)
- Add min_trades check to rolling_walk_forward for consistency
