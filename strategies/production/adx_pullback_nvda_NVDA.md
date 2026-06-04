# adx_pullback_nvda — NVDA

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T15:38:15.707147+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 0.9160 |
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
  "adx_threshold": 25,
  "ema_len": 20,
  "take_atr": 3
}
```
