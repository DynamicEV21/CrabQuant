# CAT / roc_ema_volume — PRODUCTION
**Promoted:** 2026-04-25
**Verdict:** ROBUST

## VectorBT Results
- Sharpe: 2.66 | Return: 160.0% | MaxDD: -12.3% | Trades: 15 | Win Rate: 53.3%
- Composite Score: 2.02

## Confirmation Results (backtesting.py)
- Sharpe: 1.62 | Return: 114.4% | MaxDD: -11.1% | Trades: 18 | Win Rate: 50.0%
- Realistic fill degradation: -28% return, -1.03 Sharpe

## Slippage Sensitivity
- 0.0% slippage: ✅ Sharpe 0.00
- 0.1% slippage: ✅ Sharpe 0.00
- 0.2% slippage: ✅ Sharpe 0.00

## Period Performance
- 2y: Sharpe 1.62, Return 114.4%

## Strategy Parameters
- roc_len: 8
- ema_len: 15
- vol_sma_len: 15
- atr_len: 14
- atr_mult: 1.5
- trailing_len: 10

<!-- METADATA
{"strategy_name": "roc_ema_volume", "ticker": "CAT", "params": {"roc_len": 8, "ema_len": 15, "vol_sma_len": 15, "atr_len": 14, "atr_mult": 1.5, "trailing_len": 10}, "date_promoted": "2026-04-25", "verdict": "ROBUST", "vbt_sharpe": 2.6576223219722452, "vbt_total_return": 1.5999390361426409, "vbt_max_drawdown": -0.12306584314143265, "vbt_num_trades": 15, "vbt_win_rate": 0.5333333333333333, "vbt_score": 2.0183239833233593, "confirm_sharpe": 1.6247, "confirm_total_return": 1.1442, "confirm_max_drawdown": -0.1105, "confirm_num_trades": 18, "confirm_win_rate": 0.5, "confirm_profit_factor": 5.5853, "confirm_expectancy": 635.68, "slippage_results": [{"slippage_pct": 0.0, "sharpe": 0, "total_return": 0, "max_drawdown": 0, "num_trades": 0, "win_rate": 0, "passed": true}, {"slippage_pct": 0.001, "sharpe": 0, "total_return": 0, "max_drawdown": 0, "num_trades": 0, "win_rate": 0, "passed": true}, {"slippage_pct": 0.002, "sharpe": 0, "total_return": 0, "max_drawdown": 0, "num_trades": 0, "win_rate": 0, "passed": true}], "period_results": [{"period": "2y", "sharpe": 1.6247, "total_return": 1.1442, "max_drawdown": -0.1105, "num_trades": 18, "win_rate": 0.5, "passed": true}], "regime_info": {"best_regime": "", "works_in": [], "avoid_in": []}, "key": "roc_ema_volume|CAT|de4a4b06ae51"}
-->