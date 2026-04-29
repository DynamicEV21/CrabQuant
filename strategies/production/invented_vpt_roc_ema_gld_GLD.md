# invented_vpt_roc_ema_gld — GLD

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T13:47:50.260917+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 1.3390 |
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
  "ema_len": 20,
  "roc_len": 10,
  "vpt_len": 20
}
```
