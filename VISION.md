# CrabQuant — Vision

**Last Updated:** 2026-04-28
**Current Phase:** Phase 5.6 — Invention Accelerators

---

## What CrabQuant Is

CrabQuant is an **autonomous, always-on quantitative strategy research system** that continuously invents, tests, validates, refines, and promotes trading strategies without human intervention.

It's not a backtesting tool. It's not a strategy library. It's a **research engine** — a system that does what a quant researcher does: come up with ideas, test them, figure out why they fail, iterate, and keep the winners.

The end state: you give it compute and an API budget. It runs. It finds edges. It invents new approaches. It validates them. It promotes the survivors. You check in, review results, and decide what to deploy.

---

## Core Philosophy

### Always-On, Not Periodic

The system should run **continuously** — not as a set of cron jobs that fire every few hours, but as an always-on engine that's constantly churning through strategies.

### LLMs as Consultants, Not Conductors

Python owns the loop. JSON on disk is state. LLMs are called when intelligence is needed (inventing strategies, diagnosing failures, proposing fixes) but they don't orchestrate anything.

### Realistic Realism

No shortcuts that sacrifice accuracy. Real OHLCV data from Yahoo Finance. Realistic fill simulation. Walk-forward validation. Cross-ticker validation. Composite scoring that penalizes overfit.

If a strategy looks too good, it probably is. The system's job is to catch that.

### Self-Improving

Every cycle makes the system smarter. The meta-analyzer studies what indicator families work on which tickers, which failure modes are most common, which actions lead to breakthroughs. That knowledge feeds back into invention prompts.

### Fast Iteration

Minutes between strategy discoveries, not hours. The refinement pipeline should produce a complete strategy lifecycle (invent → backtest → diagnose → refine → validate) in under 15 minutes per mandate.

---

## What's Built

### Research Engine (Refinement Pipeline)
The core. An LLM-driven iterative refinement loop that takes a strategy mandate and loops up to 7 turns, using LLM intelligence to improve strategies until they hit a Sharpe target.

31 components: orchestrator, LLM API, validation gates, backtest engine, circuit breaker, stagnation detection, action analytics, auto-promotion, wave manager, regime tagger, rolling walk-forward, cross-run learning, and more. All tested (972 tests passing).

### Backtest Engine
VectorBT-based with composite scoring, parallel execution across tickers, real OHLCV from Yahoo Finance with pickle caching (20hr TTL).

### Strategy Library
25 strategies across archetypes (momentum, mean reversion, breakout, trend, volume). Mix of hand-crafted and LLM-invented.

### Validation
Walk-forward validation (configurable train/test split), cross-ticker validation across 15+ tickers, overfitting detection, guardrails.

### Daemon
Persistent process with PID management, state persistence, graceful shutdown, health check endpoint. Has run 11 waves and 23 mandates.

---

## Current Reality (April 2026)

The pipeline runs end-to-end. Strategies get invented and backtested. Some hit Sharpe >2.0 in 2 turns. But the system has a fundamental funnel problem:

| Metric | Value | Problem? |
|--------|-------|----------|
| Total mandates run | 80 | — |
| Backtest successes (Sharpe ≥ target) | ~10% | Low but workable |
| Code gen failure rate | 54% | 🔴 Too high |
| Strategies registered in registry | **0** (from invention) | 🔴 Critical |
| winners.json entries | 58 (50 are sweep debris) | 🔴 Mostly noise |
| Validation pass rate | 0% (0/8 refined) | 🔴 Wall |

**The one thing that matters:** Zero strategies from the invention pipeline have ever been registered in STRATEGY_REGISTRY. The walk-forward validation gate is a wall — it always detects regime shift and rejects. The `auto_promote()` path is effectively dead code.

Everything else works. The LLM can write good strategies. The backtest engine is fast and accurate. The daemon runs reliably. But nothing survives validation, so nothing gets promoted.

**Phase 5 fixes the funnel. Phase 5.5 adds regime awareness. Phase 5.6 accelerates invention.** See ROADMAP.md.

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Convergence rate | >20% | ~10% |
| Code gen failure rate | <30% | 54% |
| Validation pass rate | >50% | 0% |
| Strategies in registry (from invention) | 10+ | 0 |
| Unattended runtime | 7+ days | ~1 day |
| Test coverage | 100% of new code | 972 tests |

---

## The Dream

> You give it a vision. It runs. It finds edges. It invents new approaches. It validates them. It improves itself. You check in, review the results, and decide what to deploy.

This is the north star. Every architectural decision should move us closer to this.
