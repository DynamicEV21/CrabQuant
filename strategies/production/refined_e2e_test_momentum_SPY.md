# refined_e2e_test_momentum — SPY

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T12:47:16.265896+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 1.8700 |
| Return | 30.00% |
| Max Drawdown | -12.00% |
| Trades | 15 |

## Validation

| Check | Result |
|-------|--------|
| Walk-Forward Robust | ✅ |
| Cross-Ticker Robust | ✅ |

## Parameters

```json
{
  "roc_len": 14,
  "ema_len": 21,
  "vol_len": 20,
  "vol_mult": 1.1,
  "exit_roc": -1.0
}
```
