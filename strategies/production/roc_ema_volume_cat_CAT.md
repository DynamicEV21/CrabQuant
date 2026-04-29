# roc_ema_volume_cat — CAT

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T13:47:50.260919+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 1.6260 |
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
  "atr_len": 14,
  "atr_mult": 2.0,
  "ema_len": 20,
  "roc_len": 10,
  "trailing_len": 20,
  "vol_sma_len": 20
}
```
