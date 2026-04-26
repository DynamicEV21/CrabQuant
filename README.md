# CrabQuant — CodeCrab's Autonomous Strategy Engine

Clean, fast, direct. No LangGraph. No agent overhead. Just Python + VectorBT + judgment.

## What It Does

Autonomously discovers, backtests, and validates trading strategies:
1. **Discovery** — Runs 9 strategy archetypes across 30+ tickers with iterative parameter tuning
2. **Validation** — Walk-forward testing + cross-ticker validation to separate real edges from curve-fitting
3. **Scoring** — Composite score that penalizes overfit (low trades, high drawdown)

## Quick Start

```bash
# Install deps (or use QuantFactory's venv)
pip install -r requirements.txt

# Full discovery sweep
python -m crabquant.run

# Validate winners
python -m crabquant.run --validate

# Single strategy deep dive
python -m crabquant.run --strategy macd_momentum

# Single ticker
python -m crabquant.run --ticker AAPL,MSFT,GOOGL
```

## Strategy Library

| Strategy | Best Ticker | Sharpe | Description |
|----------|-------------|--------|-------------|
| macd_momentum | AMD | 2.15 | MACD histogram shift + 200 SMA trend filter |
| adx_pullback | NFLX | 2.09 | ADX trend + pullback to EMA |
| rsi_crossover | GOOGL | 1.68 | Fast/slow RSI crossover + regime filter |
| atr_channel_breakout | ORCL | 1.59 | Keltner channel breakout + volume |
| volume_breakout | NFLX | 1.52 | Donchian channel + volume spike |
| multi_rsi_confluence | — | — | Triple RSI oversold confluence |
| ema_ribbon_reversal | — | — | EMA alignment + RSI dip |
| bollinger_squeeze | — | — | BB squeeze + breakout |
| ichimoku_trend | — | — | Simplified Ichimoku cloud |

## Architecture

```
crabquant/
├── engine/          # VectorBT backtest engine + metrics
├── strategies/      # Strategy library (modular, testable)
├── data/            # Data loader (yfinance with caching)
├── validation/      # Walk-forward + cross-ticker validation
└── run.py           # Main CLI runner
```

## Validation Philosophy

A strategy that works on one ticker in one time period is **not** a strategy.
We require:
- **Walk-forward**: Train on 18mo, test on 6mo — does Sharpe hold?
- **Cross-ticker**: Test on 15+ other tickers — is it generalizable?
- **Both must pass** for a strategy to be marked "robust"

## Why Not LangGraph?

QuantFactory used LangGraph to orchestrate strategy creation via LLM agents.
This was slower, had 50%+ timeout rates, and produced similar-or-worse results.

CrabQuant uses direct Python execution. I (CodeCrab) write the strategies, run the
backtests, analyze the results, and iterate — all in one session. No graph overhead,
no prompt chains, no wasted tokens.

## Project Context

This is a fork-evolution from [QuantFactory](https://github.com/DynamicEV21/QuantFactory).
QF had good ideas (strategy archetypes, iterative improvement) but the LangGraph
architecture was the wrong abstraction layer. CrabQuant keeps what worked and
ditches what didn't.

## License

MIT
