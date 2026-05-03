# refined_roc_ema_volume_googl — SPY

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T12:47:20.385835+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 1.5000 |
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
  "roc_len": 10,
  "ema_len": 20,
  "vol_sma_len": 20,
  "atr_len": 14,
  "atr_mult": 2.0,
  "trailing_len": 20
}
```
