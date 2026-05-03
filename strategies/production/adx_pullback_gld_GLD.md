# adx_pullback_gld — GLD

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T13:47:50.260878+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 1.6480 |
| Return | 0.00% |
| Max Drawdown | 0.00% |
| Trades | 0 |

## Validation

| Check | Result |
|-------|--------|
| Walk-Forward Robust | ✅ |
| Cross-Ticker Robust | ✅ |

## Parameters

```json
{
  "adx_len": 14,
  "adx_threshold": 25,
  "atr_len": 14,
  "atr_mult": 2.0,
  "pullback_lookback": 5
}
```
