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

## Current Phase: v0.1 — Foundation

### Done
- [x] 9 strategy archetypes with proven results
- [x] VectorBT backtest engine with composite scoring
- [x] Walk-forward + cross-ticker validation
- [x] Autonomous cron (10min cycles, 504 combos)
- [x] Full test suite (25/25 passing)

### In Progress
- [ ] Directed parameter search (replace random mutation)
- [ ] Self-improvement agent (analyze → suggest → implement → validate)
- [ ] Zero-trade early exit with smart strategy skipping
- [ ] Pattern detection across winners

### Next (v0.2)
- [ ] Regime detection layer (trending vs mean-reverting vs volatile)
- [ ] Composite strategy aggregation (combine signals from multiple strategies)
- [ ] Position sizing (Kelly criterion / risk parity)
- [ ] Strategy decay detection (Sharpe dropping over time = stop using it)

### Future (v0.3+)
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

## The Dream

You give it a vision. It runs. It finds edges. It validates them. It improves itself.
You check in, review the results, and decide what to deploy.
"""

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
