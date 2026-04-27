# CrabQuant — CodeCrab's Autonomous Strategy Engine

Clean, fast, direct. No LangGraph. No agent overhead. Just Python + VectorBT + LLM judgment.

## What It Does

Autonomously discovers, backtests, validates, and **refines** trading strategies:
1. **Discovery** — Runs 9 strategy archetypes across 30+ tickers with iterative parameter tuning
2. **Validation** — Walk-forward testing + cross-ticker validation to separate real edges from curve-fitting
3. **Refinement** — LLM-driven iterative improvement: invent new strategies, diagnose failures, refine until they converge
4. **Scoring** — Composite score that penalizes overfit (low trades, high drawdown)
5. **Always-On Daemon** — Runs 24/7 as a persistent process with state persistence, health checks, and supervisor monitoring

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

## Always-On Daemon

CrabQuant runs as a persistent daemon (`scripts/run_pipeline.py`) that continuously:
- Generates mandates from strategy archetypes and market conditions
- Runs parallel refinement waves (up to 5 concurrent LLM-driven strategy improvements)
- Promotes winning strategies to the registry
- Persists state across restarts (picks up where it left off)
- Reports health via heartbeat file and JSON endpoint

A supervisor cron checks every 5 minutes and restarts the daemon if it dies.

```bash
# Start the daemon
python scripts/run_pipeline.py --daemon

# Check status
python scripts/run_pipeline.py --status

# Stop gracefully
python scripts/run_pipeline.py --stop

# Run health check
cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m crabquant.production.health
```

## Architecture

```
crabquant/
├── engine/          # VectorBT backtest engine + metrics
├── strategies/      # Strategy library (28+ strategies, modular, testable)
├── data/            # Data loader (yfinance with caching)
├── validation/      # Walk-forward + cross-ticker validation
├── refinement/      # LLM-driven strategy refinement pipeline (31 components)
├── production/      # Strategy promotion, health checks, scanner
├── confirm/         # Slippage/commission confirmation (bar-by-bar)
└── run.py           # Main CLI runner

scripts/
├── run_pipeline.py  # Always-on daemon (start/stop/status)
├── refinement_loop.py  # Per-mandate refinement orchestrator
└── wave_runner.py   # Parallel wave execution CLI
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
