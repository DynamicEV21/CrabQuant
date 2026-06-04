# invented_volume_breakout_adx_meta — META

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T15:38:15.702890+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 1.0480 |
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
  "vol_sma_len": 20,
  "vol_mult": 2.0,
  "adx_len": 14,
  "adx_threshold": 25,
  "rsi_len": 14,
  "atr_len": 14,
  "atr_mult": 2.5,
  "sma_fast": 10,
  "sma_slow": 30
}
```
