# roc_ema_volume_jpm — JPM

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T15:38:15.709760+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 0.8150 |
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
  "roc_len": 10,
  "ema_len": 20,
  "vol_sma_len": 20,
  "atr_len": 14,
  "atr_mult": 2.0,
  "trailing_len": 20
}
```
