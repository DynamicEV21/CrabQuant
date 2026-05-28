# CrabQuant Mandates

Mandates define the target specifications for strategy discovery and refinement. Each mandate is a JSON file that tells the QuantFactory refinement pipeline what kind of strategy to build, what ticker to target, and what performance constraints to satisfy.

## Mandate Format

```json
{
  "name": "momentum_spy",
  "description": "Broad market momentum strategy using MACD crossovers and relative strength on SPY.",
  "strategy_archetype": "momentum",
  "tickers": ["SPY", "QQQ", "DIA"],
  "primary_ticker": "SPY",
  "period": "2y",
  "sharpe_target": 1.0,
  "max_turns": 7,
  "constraints": {
    "max_parameters": 8,
    "min_trades": 10,
    "max_drawdown_pct": 20
  }
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | ✅ | Unique identifier, typically `{archetype}_{ticker}` |
| `description` | ✅ | Human-readable description of the strategy intent |
| `strategy_archetype` | ✅ | One of: `momentum`, `mean_reversion`, `breakout`, `trend_following`, `volume`, `volatility`, `multi_indicator` |
| `tickers` | ✅ | Array of tickers to use in backtesting (primary + peers) |
| `primary_ticker` | ✅ | The main ticker this strategy targets |
| `period` | ✅ | Backtest lookback period (e.g. `"2y"`, `"1y"`) |
| `sharpe_target` | ✅ | Minimum Sharpe ratio the refined strategy must achieve |
| `max_turns` | ✅ | Maximum refinement iterations allowed |
| `constraints` | ✅ | Object with performance and complexity limits |

### Constraints

| Constraint | Default | Description |
|------------|---------|-------------|
| `max_parameters` | 8 | Maximum number of strategy parameters |
| `min_trades` | 10 | Minimum number of trades over the backtest period |
| `max_drawdown_pct` | 25 | Maximum acceptable drawdown percentage |

## Available Mandates

### Momentum
| File | Primary Ticker | Sharpe Target | Description |
|------|---------------|---------------|-------------|
| `momentum_aapl.json` | AAPL | 1.5 | MACD + RSI momentum on Apple |
| `momentum_msft.json` | MSFT | 1.0 | Large cap tech momentum on Microsoft |
| `momentum_nvda.json` | NVDA | 1.5 | MACD + RSI momentum on NVIDIA |
| `momentum_spy.json` | SPY | 1.0 | Broad market momentum on S&P 500 |
| `momentum_googl.json` | GOOGL | 1.0 | Large cap tech momentum on Alphabet |

### Mean Reversion
| File | Primary Ticker | Sharpe Target | Description |
|------|---------------|---------------|-------------|
| `mean_reversion_spy.json` | SPY | 1.5 | Bollinger Bands + RSI mean reversion on S&P 500 |
| `mean_reversion_nvda.json` | NVDA | 1.2 | Volatile mean reversion on NVIDIA |
| `mean_reversion_nflx.json` | NFLX | 1.2 | Volatile mean reversion on Netflix |

### Breakout
| File | Primary Ticker | Sharpe Target | Description |
|------|---------------|---------------|-------------|
| `breakout_aapl.json` | AAPL | 1.0 | ATR + Donchian breakout on Apple |
| `breakout_amd.json` | AMD | 1.2 | High beta breakout on AMD |
| `breakout_spy.json` | SPY | 1.0 | Broad market breakout on S&P 500 |
| `breakout_tsla.json` | TSLA | 1.5 | ATR + Donchian breakout on Tesla |

### Trend Following
| File | Primary Ticker | Sharpe Target | Description |
|------|---------------|---------------|-------------|
| `trend_amzn.json` | AMZN | 1.5 | Moving averages + ADX trend following on Amazon |
| `trend_msft.json` | MSFT | 1.0 | Large cap trend following on Microsoft |
| `trend_tsla.json` | TSLA | 1.2 | Volatile trend following on Tesla |

### Volume
| File | Primary Ticker | Sharpe Target | Description |
|------|---------------|---------------|-------------|
| `volume_aapl.json` | AAPL | 1.0 | OBV + VWAP volume signals on Apple |
| `volume_msft.json` | MSFT | 1.5 | OBV + volume profile on Microsoft |
| `volume_nvda.json` | NVDA | 1.2 | OBV divergence + volume profile on NVIDIA |

### Volatility
| File | Primary Ticker | Sharpe Target | Description |
|------|---------------|---------------|-------------|
| `volatility_spy.json` | SPY | 1.0 | Volatility regime strategy on S&P 500 |
| `volatility_tsla.json` | TSLA | 1.2 | High volatility strategy on Tesla |

### Multi-Indicator
| File | Primary Ticker | Sharpe Target | Description |
|------|---------------|---------------|-------------|
| `multi_rsi_googl.json` | GOOGL | 1.5 | Multi-indicator RSI on Alphabet |

### Testing / E2E
| File | Primary Ticker | Sharpe Target | Description |
|------|---------------|---------------|-------------|
| `e2e_test_momentum.json` | AAPL | 0.5 | E2E integration test (momentum) |
| `e2e_stress_test.json` | SPY | 1.5 | E2E stress test (momentum + volume) |

## Creating New Mandates

1. Copy an existing mandate as a template
2. Update `name` to `{archetype}_{ticker}`
3. Set `strategy_archetype` and `primary_ticker`
4. Choose related tickers for the `tickers` array (helps with relative strength and correlation analysis)
5. Set `sharpe_target`: use **1.0** for stable/large-cap combos, **1.2** for volatile/high-beta combos, **1.5** for aggressive targets
6. Adjust `constraints` based on volatility profile of the ticker
7. Validate JSON with `python -m json.tool mandates/{name}.json`
