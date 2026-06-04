# invented_volume_roc_atr_trend_googl — GOOGL

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T15:38:15.699611+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 1.3030 |
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
  "roc_len": 21,
  "ema_len": 20,
  "vol_sma_len": 20,
  "volume_mult": 1.5,
  "atr_len": 14,
  "atr_mult": 2.5,
  "rsi_len": 14,
  "rsi_overbought": 70
}
```
