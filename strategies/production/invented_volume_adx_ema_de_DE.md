# invented_volume_adx_ema_de — DE

**Source:** refinement_pipeline
**Promoted:** 2026-04-29T15:38:15.706576+00:00
**Verdict:** ROBUST

## Backtest Results

| Metric | Value |
|--------|-------|
| Sharpe | 0.3900 |
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
  "obv_fast": 10,
  "obv_slow": 20,
  "adx_len": 14,
  "adx_threshold": 25,
  "ema_len": 50,
  "rsi_len": 14,
  "rsi_oversold": 30,
  "rsi_overbought": 70,
  "atr_mult": 2.0
}
```
