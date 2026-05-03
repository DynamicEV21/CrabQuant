# informed_simple_adaptive_spy — SPY

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T13:47:50.260900+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 1.7560 |
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
  "rsi_len": 14,
  "rsi_oversold": 35,
  "rsi_overbought": 65,
  "volume_window": 20,
  "volume_mult": 1.3
}
```
