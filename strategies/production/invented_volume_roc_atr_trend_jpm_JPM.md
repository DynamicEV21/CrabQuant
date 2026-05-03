# invented_volume_roc_atr_trend_jpm — JPM

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T13:47:50.260915+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 2.3390 |
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
  "ema_len": 20,
  "roc_len": 10,
  "vol_sma_len": 20
}
```
