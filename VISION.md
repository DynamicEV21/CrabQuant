# CrabQuant — Vision

**Last Updated:** 2026-04-29
**Current Phase:** Phase 6 — Production Validation (prep)

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

31 components: orchestrator, LLM API, validation gates, backtest engine, circuit breaker, stagnation detection, action analytics, auto-promotion, wave manager, regime tagger, rolling walk-forward, cross-run learning, feature importance, stagnation recovery, semantic action validation, and more. All tested (4430+ tests passing).

### Backtest Engine
VectorBT-based with composite scoring, parallel execution across tickers, real OHLCV from Yahoo Finance with pickle caching (20hr TTL).

### Strategy Library
25+ strategies across archetypes (momentum, mean reversion, breakout, trend, volume). Mix of hand-crafted and LLM-invented.

### Validation
Walk-forward validation (configurable train/test split), cross-ticker validation across 15+ tickers, overfitting detection, guardrails.

### Daemon
Persistent process with PID management, state persistence, graceful shutdown, health check endpoint. Has run 11 waves and 23 mandates.

---

## Current Reality (April 2026)

The pipeline runs end-to-end. Strategies get invented and backtested. Some hit Sharpe >2.0 in 2 turns. The promotion pipeline gap has been fixed — 118 strategies now in the production registry.

| Metric | Value | Problem? |
|--------|-------|----------|
| Total mandates run | 30 unique (147 turns) | — |
| Backtest successes (real mandates) | 6.8% per-turn | 🔴 Low |
| Mandate convergence (≥1 success) | 33% (7/21 real) | 🟡 Needs work |
| Code gen failure rate (real mandates) | **0%** | ✅ Fixed |
| Strategies in production registry | **118** (ROBUST) | ✅ Exceeded target |
| winners.json entries | 178 (119 promoted) | ✅ Strong pipeline |
| Validation pass rate | 67% (119/178 winners) | ✅ Exceeded target |

**The funnel is open.** After fixing threshold bugs (Cycle 10-12) and the promotion pipeline gap (Cycle 13), 118 validated strategies are now in the production registry.

**Key insight (Cycle 15):** The "54% code gen failure rate" was a phantom metric inflated by smoke_test and test_mandate entries. Real production mandates have **0% code gen failures**. The real bottleneck is **strategy quality** — 83% of real mandate turns fail on performance metrics:
- low_sharpe: 35% — strategies run but underperform
- regime_fragility: 25% — strategies work in some years but not others
- too_few_trades: 24% — strategies too selective (< 5 trades)
- excessive_drawdown: 10% — strategies lose too much

**Fixes applied (Cycle 15):**
- Sharpe Root Cause Analyzer: 12 diagnosis patterns for low_sharpe with specific actionable fixes
- Regime Diagnosis System: 9 regime patterns with per-year breakdown and targeted fixes

**Phase 5 fixes the funnel. Phase 5.5 adds regime awareness. Phase 5.6 accelerates invention.** See ROADMAP.md.

---

## Success Metrics

| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| Validation pass rate | >50% | 67% (119/178) | ✅ Met |
| Strategies in registry (from invention) | 10+ | 118 promoted | ✅ Exceeded |
| Mandate convergence rate | >50% | 33% (7/21) | 🟡 17% |
| Per-turn success rate | >20% | 6.8% (10/147) | 🔴 13% |
| Code gen failure rate | <30% | 0% (real) | ✅ Fixed |
| Test coverage | 100% of new code | 4430+ tests | ✅ Surpassed |

---

## Continuous Improvement Engine

This section tells the orchestrator what to do when all planned tasks are complete. It does NOT need human direction — it reads this section, identifies priorities, and works autonomously.

### How the Priority System Works

1. **Read the Success Metrics table above** — the Gap column shows relative urgency
2. **The largest gap = current priority** — work on whatever moves the biggest gap metric
3. **Blocking chains matter** — some metrics block others (see below)
4. **Check diminishing returns** — if a metric has improved >50% toward target, deprioritize it
5. **One priority at a time** — focus workers on the current P0, don't spread thin

### Priority Queue

The orchestrator should recalculate priorities every cycle by looking at actual metric values (not just what's written here — check real files like `results/winners/`, `STRATEGY_REGISTRY`, test counts, etc.):

**🔴 P0 — Improve Per-Turn Success Rate (6.8% → >20%)**
- The biggest gap. 83% of real mandate turns fail on performance, not code quality.
- **low_sharpe (35%)** — ✅ Sharpe Root Cause Analyzer added (Cycle 15). Verified wired correctly (Cycle 17 wiring audit).
- **regime_fragility (25%)** — ✅ Regime Diagnosis System added (Cycle 15). Verified wired correctly.
- **too_few_trades (24%)** — ✅ Guidance template added (Cycle 15 orchestrator fix). Verified wired correctly.
- **excessive_drawdown (10%)** — Minor. Existing guidance is adequate.
- **Wiring bugs fixed (Cycle 17)**: `detect_regime` imported from wrong module, `analyze_failure_patterns` called with wrong arg type, 2 silent `except-pass` blocks now log warnings. All 3 bugs would have caused dead-code paths in the Turn 1 and Turn 2+ prompt builders.
- **Full wiring audit (Cycle 17)**: Zero new bugs found across all 17 modules. All function signatures match their call sites.
- Action: Run live mandates to verify all diagnosis systems work end-to-end (now P1).

**🟡 P1 — Run Live Mandates to Verify Improvements**
- The diagnosis systems (sharpe, regime) were built based on historical data analysis
- Need to run actual mandates to verify they improve the per-turn success rate
- Run explorer mode mandates on diverse tickers (not just SPY)
- Verify the feedback loop actually changes LLM behavior

**🟢 P2 — Polish (only after P0/P1 are significantly improved)**
- Mandate convergence rate (33% → >50%)
- ROADMAP Phase 6 prep items — daemon config, budget tracking, Telegram briefs

**⚪ P3 — Diminished Returns (avoid unless directly tied to P0/P1 work)**
- Test expansion — we have 4430+ tests. Only write tests for code you're actively modifying for a higher-priority goal.

### Dynamic Priority Rules

These rules override the static queue above when conditions change:

| Condition | Action |
|-----------|--------|
| A metric has improved >50% toward target | Deprioritize — shift workers to next biggest gap |
| P0 is partially resolved (e.g., validation rate hits 20%) | Keep P0 but split workers: 2 on P0, 1 on P1 |
| Same task has failed 3+ times across cycles | Stop retrying — escalate to orchestrator self-rescue or skip |
| A worker discovers a bug blocking P0 | Immediately redirect 1 worker to fix it |
| All P0 tasks are in-progress or blocked | Move to P1 — don't idle |
| Test count for a module exceeds 50 | STOP adding tests to that module — diminishing returns |

### Work Categories

| Category | Examples | When to Use |
|----------|----------|-------------|
| **Fix blockers** | Debug validation failures, fix code gen errors, resolve import bugs | Always — highest value |
| **Build features** | Implement ROADMAP items, add modules from current phase plan | When planned tasks exist |
| **Investigate & diagnose** | Run diagnostic mandates, analyze failure patterns, profile bottlenecks | When root cause is unclear |
| **Improve quality** | Better error messages, type hints, logging, docstrings | Only alongside active feature/fix work |
| **Expand tests** | Add tests for new/modified modules | Only for code being changed for P0/P1 goals |

### Discovery Budget

Each cycle, the orchestrator MAY allocate **at most 1 worker slot** to "discovered" work — something the agent finds on its own that wasn't in the task queue.

**Allowed discovery:**
- Finding and fixing bugs that block success metrics
- Improving prompts or config that would move P0/P1 metrics
- Adding small utility functions that reduce code duplication
- Implementing ROADMAP items that weren't in the task queue but align with current phase

**NOT allowed discovery:**
- Expanding test files for well-tested modules
- Refactoring working code that isn't blocking anything
- Adding features not in VISION.md or ROADMAP.md
- Changing business logic or algorithm behavior without investigation

Discovery workers should commit with prefix `discovery:` so their work is distinguishable from planned work.

### Anti-Patterns (Do NOT Do These)

- ❌ Expanding test files for modules with >50 tests (diminishing returns)
- ❌ Adding tests that don't correspond to any feature/fix/bug work
- ❌ Working on P3 priorities when P0/P1 gaps exist
- ❌ Refactoring code that's already working and not blocking anything
- ❌ Repeating a task that failed 3+ times without changing approach
- ❌ Writing code without understanding why previous attempts failed

---

## The Dream

> You give it a vision. It runs. It finds edges. It invents new approaches. It validates them. It improves itself. You check in, review the results, and decide what to deploy.

This is the north star. Every architectural decision should move us closer to this.
