# injected_momentum_atr_volume_tsla — TSLA

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T15:38:15.705807+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 0.4280 |
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
  "roc_len": 5,
  "roc_threshold": 0.5,
  "vol_sma_len": 10,
  "vol_threshold": 1.2,
  "rsi_len": 10,
  "rsi_min_uptrend": 25,
  "rsi_max_downtrend": 75,
  "ema_short_len": 10,
  "ema_long_len": 20,
  "atr_len": 10,
  "atr_mult": 1.5
}
```
