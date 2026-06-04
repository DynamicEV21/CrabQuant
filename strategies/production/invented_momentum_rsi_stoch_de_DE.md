# invented_momentum_rsi_stoch_de — DE

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T13:47:50.260908+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 1.3900 |
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
  "momentum_len": 10,
  "rsi_len": 14,
  "stoch_k": 14,
  "stoch_d": 3
}
```
