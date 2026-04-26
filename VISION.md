"""
CrabQuant Vision & Roadmap

## What CrabQuant Is Becoming

An autonomous quantitative strategy research system that continuously discovers,
validates, and improves trading strategies without human intervention.

## The Loop

1. **DISCOVER** — Cron grinds through strategy/ticker/parameter combos
2. **ANALYZE** — Self-improvement agent examines winners, finds patterns
3. **IMPROVE** — Writes new strategies, refines params, fixes bugs
4. **VALIDATE** — Walk-forward + cross-ticker on promising strategies
5. **REPEAT** — Each cycle makes the system smarter

## Principles

- **Direct Python execution** — no LangGraph, no agent overhead
- **Composite scoring** — penalizes overfit, rewards consistency
- **Validation before promotion** — walk-forward AND cross-ticker must pass
- **Self-aware** — tracks what works, adapts strategies accordingly
- **Transparent** — every decision logged, reproducible

## The Pipeline

```
INVENT (LLM agent, every 3h)
  ↓ Analyzes market data, winner patterns
  ↓ Writes new strategy Python code
  ↓ Tests it produces signals
  ↓ Registers in strategy library
  ↓
OPTIMIZE (cron, every 10m)
  ↓ Tests each strategy across 56 tickers
  ↓ Directed parameter search (explore → narrow → fine-tune)
  ↓ Saves winners to winners.json
  ↓ Runs analysis every 10 cycles
  ↓
VALIDATE (improve agent, every 3h)
  ↓ Walk-forward: train 18mo / test 6mo
  ↓ Cross-ticker: test on 15+ other tickers
  ↓ Promotes robust strategies
  ↓
EVOLVE (improve agent, every 3h)
  ↓ Reads insights.json patterns
  ↓ Expands winning param grids
  ↓ Fixes dead combos
  ↓ Combines winning signals into composites
  ↓
REPEAT (continuous)
```

## Current Phase: v0.2 — Invention + Optimization

### Done
- [x] 9 strategy archetypes with proven results
- [x] VectorBT backtest engine with composite scoring
- [x] Walk-forward + cross-ticker validation
- [x] Autonomous cron v2 (directed search, smart ordering, dead combo tracking)
- [x] Self-improvement analysis (strategy rates, ticker hotspots, param patterns)
- [x] Strategy invention module (market analysis + pattern detection)
- [x] Inventor cron (every 3h, writes + tests + registers new strategies)
- [x] Full test suite (25/25 passing)

### In Progress
- [ ] First invented strategy (sub-agent running)
- [ ] Verify inventor produces working strategies
- [ ] Validate initial winners from sweep

### Next (v0.3)
- [ ] Composite strategy aggregation (combine signals from multiple strategies)
- [ ] Regime detection layer (trending vs mean-reverting vs volatile)
- [ ] Strategy decay detection (Sharpe dropping = stop using it)
- [ ] Position sizing (Kelly criterion / risk parity)

### Future (v0.4+)
- [ ] Multi-timeframe analysis
- [ ] Sector rotation signals
- [ ] Paper trading interface
- [ ] Portfolio-level optimization
- [ ] Live broker integration (Interactive Brokers / Alpaca)

## Success Metrics

- **Sharpe ≥ 1.5** on validated (walk-forward + cross-ticker) strategies
- **Robust rate ≥ 30%** — at least 30% of winners survive validation
- **Generalization** — strategies work across sectors, not just one ticker
- **No overfit** — composite score penalizes low-trade, high-drawdown results
- **Self-sustaining** — system improves itself without human input for weeks
- **Invention rate** — inventor produces at least 1 working strategy per cycle

## The Dream

You give it a vision. It runs. It finds edges. It invents new approaches. It validates them. It improves itself.
You check in, review the results, and decide what to deploy."""

VISION = """
CrabQuant should become a self-improving autonomous research engine.

The core loop:
1. Cron discovers strategy/ticker combos that meet criteria
2. Self-improvement agent (me, in heartbeat cycles) analyzes what's working
3. Based on patterns, I write new strategies, refine param grids, fix issues
4. Those improvements get committed, cron picks them up next cycle
5. Validation catches curve-fits before they waste time

What makes this different from QuantFactory:
- No LLM agent overhead — direct Python execution
- Composite scoring that penalizes overfit
- Validation is mandatory, not optional
- I (CodeCrab) am the intelligence layer, not a LangGraph graph
- The system gets smarter over time because I learn from the data
"""
