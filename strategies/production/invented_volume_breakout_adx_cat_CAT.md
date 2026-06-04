# invented_volume_breakout_adx_cat — CAT

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T13:47:50.260912+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 1.3630 |
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
  "adx_threshold": 25,
  "atr_len": 14,
  "atr_mult": 2.0,
  "volume_mult": 1.5
}
```
