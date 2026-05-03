# invented_momentum_confluence_jpm — JPM

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T15:38:15.704826+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 1.7270 |
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
  "roc_len": 10,
  "rsi_len": 14,
  "adx_len": 14,
  "atr_len": 14,
  "vol_sma_len": 20,
  "rsi_oversold": 35,
  "rsi_overbought": 65,
  "adx_threshold": 25,
  "adx_weak_threshold": 20,
  "volume_mult": 1.5,
  "atr_mult": 2.0
}
```
