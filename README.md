# CrabQuant

Autonomous quantitative strategy research engine. Discovers, backtests, validates, and refines trading strategies using walk-forward optimization — no human intervention required.

## What It Does

1. **Discovery** — Runs strategy archetypes across 30+ tickers with iterative parameter tuning
2. **Validation** — Walk-forward testing + cross-ticker validation to separate real edges from curve-fitting
3. **Refinement** — LLM-driven iterative improvement: invent strategies, diagnose failures, refine until convergence
4. **Scoring** — Composite score penalizing overfit (low trades, high drawdown), with Sortino ratio and expected value
5. **Promotion** — Robust strategies enter the production registry (118 entries, audited)

## Quick Start

```bash
# Install deps
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

## Architecture

```
crabquant/
├── data/            # Hybrid data loader (yfinance + massive.com parquet overlay)
├── engine/          # VectorBT backtest engine + composite metrics
├── strategies/      # Strategy library (150 files, 9 archetypes + invented variants)
├── validation/      # Walk-forward + cross-ticker validation
├── refinement/      # LLM-driven iterative refinement pipeline (31 components)
├── production/      # Strategy promotion, health checks, regime routing, decay monitoring
├── confirm/         # Slippage/commission confirmation (bar-by-bar)
├── analysis/        # Portfolio correlation analysis
├── brief/           # Discovery brief generation
├── run.py           # Main CLI runner
└── strategy_adapter.py  # Porting adapter for legacy/new strategies

loops/               # Autonomous optimization loops
├── diversity-explorer/  # Quality-Diversity portfolio coverage optimization
├── sharpe-optimizer/    # Single-metric Sharpe optimization for near-miss strategies
├── meta-analyzer/       # Cross-run learning and failure pattern analysis
└── run.py               # Loop runner CLI

scripts/             # Operational scripts
├── run_pipeline.py      # Always-on daemon (start/stop/status)
├── refinement_loop.py   # Per-mandate refinement orchestrator
└── wave_runner.py       # Parallel wave execution CLI

strategies/production/   # Production registry (118 validated strategies)
results/                 # Backtest results, winners, sweep data
tests/                   # Test suite (unit + integration + E2E)
```

## Data

Hybrid loading strategy — best quality available for each date range:

| Tier | Source | Quality | Coverage |
|------|--------|---------|----------|
| **TIER 1** | massive.com parquet | Real VWAP, real n_trades | ~1 year, 33 tickers |
| **TIER 2** | yfinance API | Approx VWAP, no n_trades | Deep history (configurable) |

```python
from crabquant.data import load_data
df = load_data("AAPL", period="2y")  # massive.com takes priority on overlap
```

Tickers: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, NFLX, AMD, ORCL, SPY, QQQ, JPM, JNJ, UNH, CAT, DE, GLD, and more.

## Strategy Library

9 core archetypes with LLM-invented variants:

| Archetype | Best Result | Description |
|-----------|-------------|-------------|
| macd_momentum | AMD Sharpe 2.15 | MACD histogram shift + 200 SMA trend filter |
| adx_pullback | NFLX Sharpe 2.09 | ADX trend + pullback to EMA |
| rsi_crossover | GOOGL Sharpe 1.68 | Fast/slow RSI crossover + regime filter |
| atr_channel_breakout | ORCL Sharpe 1.59 | Keltner channel breakout + volume |
| volume_breakout | NFLX Sharpe 1.52 | Donchian channel + volume spike |
| multi_rsi_confluence | — | Triple RSI oversold confluence |
| bollinger_squeeze | — | BB squeeze + breakout |
| roc_ema_volume | — | Rate-of-change + EMA + volume (highest variant count) |
| invented_* | — | LLM-discovered strategies (momentum, volume, volatility families) |

150 strategy files total. Each includes `generate_signals()`, `generate_signals_matrix()`, `DEFAULT_PARAMS`, and `PARAM_GRID`.

## Validation Philosophy

A strategy that works on one ticker in one time period is not a strategy. Requirements:

- **Walk-forward**: Rolling windows (6 windows, configurable train/test split) — does Sharpe hold?
- **Cross-ticker**: Test on 15+ other tickers — is it generalizable?
- **Overfit detection**: Composite scoring penalizes low trade counts, high drawdown, and regime-dependent performance
- **Both must pass** for a strategy to be marked ROBUST and enter the production registry

## Refinement Pipeline

31-component LLM-driven iterative improvement loop:

- Up to 7 refinement turns per mandate
- Circuit breaker (window=20, min 30% pass rate)
- Stagnation detection with 4 recovery strategies (abandon, nuclear, pivot, broaden)
- Cross-run learning feeds proven winners into invention context
- Phase 5.6 invention accelerators: parallel invention, semantic validation, positive feedback damping
- AST sanitization on all generated code
- Adaptive prompts that shrink/expand based on turn progress

## Production Registry

118 validated strategies in `strategies/production/registry.json`:

- Each entry: strategy name, ticker, params, verdict (ROBUST/backtest_only), promotion timestamp, report file
- Regime-aware routing via `crabquant/production/regime_router.py`
- Strategy decay monitoring via `crabquant/production/strategy_decay.py`
- Integrity audits demote strategies that fail re-validation

## Autonomous Loops

The `loops/` system runs persistent optimization programs:

- **diversity-explorer** — Quality-Diversity optimization for portfolio coverage
- **sharpe-optimizer** — Single-metric Sharpe ratio optimization for near-miss strategies
- **meta-analyzer** — Cross-run pattern analysis and failure mode learning

```bash
python loops/run.py list                    # List available loops
python loops/run.py diversity-explorer      # Print program spec
python loops/run.py diversity-explorer --dry-run  # Validate config
```

## Always-On Daemon

```bash
# Start
python scripts/run_pipeline.py --daemon

# Check status
python scripts/run_pipeline.py --status

# Stop gracefully
python scripts/run_pipeline.py --stop
```

The daemon continuously generates mandates, runs parallel refinement waves (up to 5 concurrent), promotes winners, and persists state across restarts.

## Project Status

**Phase 5.6 (Invention Accelerators) — COMPLETE**

| Phase | Status |
|-------|--------|
| 0-3 | MVP + Refinement Pipeline |
| 4-4.5 | Integration + Convergence Tuning |
| 5A | Daemon Core |
| 5 | Fix the Funnel |
| 5.5 | Regime-Aware Registry |
| **5.6** | **Invention Accelerators** |
| **6** | **Production Validation** (next) |
| 7 | Intelligence Layer |
| 8 | Deployment Readiness |
| 9 | Live Trading |

**Infrastructure:**
- Gas City (gc) + Beads (bd) orchestration active
- 22 validated strategies in production registry
- Hermes-native pipeline with vectorbt + pandas_ta
- Autonomous build factory via self-driving-gc

See `VISION.md` for the full project vision, `ROADMAP.md` for detailed phase breakdowns.

## Testing

```bash
source .venv/bin/activate
python -m pytest tests/ -x -q
```

## Tech Stack

- Python 3.12
- VectorBT (backtesting)
- yfinance + massive.com parquet (data)
- pandas_ta (technical indicators)
- httpx (LLM API calls)
- pytest (testing)

## License

MIT
