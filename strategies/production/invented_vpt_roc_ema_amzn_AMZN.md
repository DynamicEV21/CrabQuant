# invented_vpt_roc_ema_amzn — AMZN

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T15:38:15.701512+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 0.4710 |
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
  "vpt_len": 14,
  "roc_len": 21,
  "ema_len": 20,
  "roc_threshold": 1.0
}
```
