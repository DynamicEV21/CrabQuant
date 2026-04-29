# injected_momentum_atr_volume_gld — GLD

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T13:47:50.260902+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 2.1660 |
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
  "ema_fast": 9,
  "ema_slow": 21,
  "volume_threshold": 1.5
}
```
