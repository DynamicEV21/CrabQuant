# invented_volume_adx_ema_nvda — NVDA

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T13:47:50.260910+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 0.8360 |
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
  "ema_len": 20,
  "volume_sma": 20
}
```
