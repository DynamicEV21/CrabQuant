# invented_momentum_rsi_stoch_tsla — TSLA

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T15:38:15.703409+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 0.5620 |
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
  "rsi_len": 14,
  "rsi_oversold": 35,
  "volume_window": 20,
  "volume_mult": 1.2
}
```
