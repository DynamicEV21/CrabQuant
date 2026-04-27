# CrabQuant — Vision & Roadmap

**Last Updated:** 2026-04-26  
**Status:** Phase 1-3 refinement pipeline built (31/31 components passing), moving to integration

---

## 1. What CrabQuant Is

CrabQuant is an **autonomous, always-on quantitative strategy research system** that continuously invents, tests, validates, refines, and promotes trading strategies without human intervention.

It's not a backtesting tool. It's not a strategy library. It's a **research engine** — a system that does what a quant researcher does: come up with ideas, test them, figure out why they fail, iterate, and keep the winners.

The end state: you give it compute and an API budget. It runs. It finds edges. It invents new approaches. It validates them. It promotes the survivors. You check in, review results, and decide what to deploy.

---

## 2. Core Philosophy

### Always-On, Not Periodic

The system should run **continuously** — not as a set of cron jobs that fire every few hours, but as an always-on engine that's constantly churning through strategies. Every few minutes, a new strategy should be entering the pipeline.

The current cron-based architecture (4 agents firing at different intervals) was a necessary MVP, but the vision is a **persistent loop** that never stops. Cron agents should become lightweight health checkers and status reporters, not the core research engine.

### LLMs as Consultants, Not Conductors

Python owns the loop. JSON on disk is state. LLMs are called when intelligence is needed (inventing strategies, diagnosing failures, proposing fixes) but they don't orchestrate anything. The deterministic code handles flow control, validation, and resource management.

### Realistic Realism

No shortcuts that sacrifice accuracy. Real OHLCV data from Yahoo Finance. Realistic fill simulation. Walk-forward validation (train 18mo / test 6mo). Cross-ticker validation across 15+ tickers. Composite scoring that penalizes overfit (`sharpe * sqrt(trades/20) * (1 - abs(max_dd))`).

If a strategy looks too good, it probably is. The system's job is to catch that.

### Self-Improving

Every cycle makes the system smarter. The meta-analyzer studies what indicator families work on which tickers, which failure modes are most common, which actions lead to breakthroughs. That knowledge feeds back into invention prompts, making future strategies better.

### Fast Iteration

Minutes between strategy discoveries, not hours. The vectorized sweep benchmark is ~10 seconds for 8,748 parameter combos on a single ticker. The refinement pipeline should produce a complete strategy lifecycle (invent → backtest → diagnose → refine → validate) in under 15 minutes per mandate.

---

## 3. Key Components

### 3.1 The Research Engine (Refinement Pipeline) — NEW, PHASE 1-3 BUILT

The core of the vision. An LLM-driven iterative refinement loop that takes a strategy mandate and loops up to 7 turns, using LLM intelligence to improve strategies until they hit a Sharpe target.

**Built components (31/31 passing tests):**

| Component | Purpose | Status |
|-----------|---------|--------|
| `schemas.py` | RunState, BacktestReport, StrategyModification dataclasses | ✅ |
| `config.py` | RefinementConfig with all defaults, mandate loading | ✅ |
| `module_loader.py` | Temp file + importlib strategy loading | ✅ |
| `classifier.py` | 6 deterministic failure modes (too_few_trades, flat_signal, excessive_drawdown, regime_fragility, overtrading, low_sharpe) | ✅ |
| `validation_gates.py` | 3-gate validation (syntax+import, signal sanity, smoke backtest) | ✅ |
| `diagnostics.py` | run_backtest_safely, compute_sharpe_by_year, compute_strategy_hash | ✅ |
| `llm_api.py` | z.ai OpenAI-compatible API wrapper with JSON extraction | ✅ |
| `context_builder.py` | build_llm_context, get_strategy_examples, compute_delta | ✅ |
| `prompts.py` | Turn 1 (invention) and Turn 2+ (refinement) prompt templates | ✅ |
| `orchestrator.py` | Main refinement loop (LLM → validate → backtest → classify → repeat) | ✅ |
| `wave_manager.py` | Parallel mandate execution via subprocess isolation | ✅ |
| `stagnation.py` | Stagnation scoring + response protocol (pivot, nuclear, abandon) | ✅ |
| `gate3_smoke.py` | Smoke backtest validation gate | ✅ |
| `guardrails_integration.py` | Existing guardrails supplement failure classification | ✅ |
| `hypothesis_enforcement.py` | Validate LLM output has non-trivial hypothesis | ✅ |
| `cosmetic_guard.py` | Force structural intervention after 3 consecutive `modify_params` | ✅ |
| `circuit_breaker.py` | Halt if LLM validation pass rate drops below 30% | ✅ |
| `tier1_diagnostics.py` | sharpe_by_year, previous_attempts with params and deltas | ✅ |
| `prompt_refinement.py` | Refined LLM prompts based on Phase 1 observations | ✅ |
| `wave_scaling.py` | Increase parallel limit, add wave status | ✅ |
| `per_wave_metrics.py` | Track convergence rate per wave | ✅ |
| `tier2_diagnostics.py` | Regime decomposition, top drawdowns, portfolio correlation | ✅ |
| `regime_sharpe.py` | Sharpe broken down by market regime | ✅ |
| `portfolio_correlation.py` | Equity curve correlation to existing winners | ✅ |
| `action_analytics.py` | Track which action types succeed/fail across all runs | ✅ |
| `cron_integration.py` | Continuous autonomous wave execution via cron | ✅ |
| `mandate_generator.py` | Auto-generate mandates from strategy catalog + market analysis | ✅ |
| `full_validation_promotion.py` | Walk-forward + cross-ticker before winner promotion | ✅ |
| `wave_dashboard.py` | Real-time view of all running mandates | ✅ |
| `auto_promotion.py` | Strategies that pass validation auto-register in STRATEGY_REGISTRY | ✅ |
| `e2e_smoke` / `e2e_phase3` | End-to-end integration tests | ✅ |

### 3.2 Backtest Engine — EXISTING, PROVEN

- `BacktestEngine`: VectorBT-based backtesting with composite scoring
- `parallel_backtest`: ProcessPoolExecutor across tickers
- Real OHLCV data from Yahoo Finance with pickle caching (20hr TTL)
- 28 strategies registered (9 base + 2 invented + 5 QF-proven + 2 injected + 10 more invented)
- Vectorized sweep: ~8,748 combos in ~10 seconds per ticker

### 3.3 Validation — EXISTING, PROVEN

- `walk_forward_test()`: 18mo train / 6mo test split with regime detection
- `cross_ticker_validation()`: Test strategy across 15+ tickers
- `full_validation()`: Combined walk-forward + cross-ticker
- `GuardrailConfig` + `check_guardrails()`: Configurable threshold checking
- `OverfittingDetector`: Curve-fitting detection across param combos

### 3.4 Strategy Library — EXISTING, GROWING

28 strategies across archetypes:
- **Momentum**: macd_momentum, ema_crossover, roc_ema_volume, invented_momentum_*
- **Mean Reversion**: rsi_crossover, bollinger_squeeze, bb_stoch_macd, multi_rsi_confluence
- **Breakout**: atr_channel_breakout, volume_breakout, invented_volatility_rsi_breakout
- **Trend**: ichimoku_trend, ema_ribbon_reversal, adx_pullback, informed_*
- **Volume**: vpt_crossover, invented_volume_*

### 3.5 Meta-Analysis — EXISTING, EVOLVING

- Market regime detection (5 regimes via SPY+VIX)
- Strategy-to-regime affinity mapping
- Winner pattern analysis (which tickers, which indicator families)
- API budget optimization (GLM-4.7 for mechanical tasks, GLM-5-Turbo for invention)

### 3.6 Cron Agents — EXISTING, BEING PHASED OUT

The original 4-agent cron architecture that got us here:
- **crabquant-wave** (every 15min): Parameter sweep across strategy/ticker combos
- **crabquant-invent** (every 2h): LLM-driven strategy invention
- **crabquant-validate** (every 2h): Walk-forward + cross-ticker validation
- **crabquant-meta** (every 3h): Meta-learning, grid expansion, retirement

**These are the MVP. The refinement pipeline replaces them as the primary research engine.**

---

## 4. Architecture Goals

### 4.1 Always-On Continuous Loop

The refinement pipeline runs as a persistent process (not periodic cron). It:
- Picks the next mandate from a queue
- Spawns parallel subprocesses (up to 5 concurrent)
- Each subprocess runs the full 7-turn refinement loop
- On completion, promotes winners and queues new mandates
- Never stops — when mandates run out, generates new ones from market analysis

Cron agents transition to **lightweight supervisors**:
- Health check: verify the loop is running, restart if crashed
- Status reporting: periodic summary to Telegram
- Budget monitoring: track API usage, throttle if needed
- Data freshness: verify price data cache isn't stale

### 4.2 Parallel Execution

Multiple strategies researched simultaneously via subprocess isolation:
- Each subprocess has its own `STRATEGY_REGISTRY`, `indicator_cache`, `sys.modules`
- No shared mutable state between parallel runs
- Resource-aware: max 5 concurrent based on 12-thread CPU and ~17GB free RAM
- ~200MB per process (pandas DataFrames) → 5 parallel = ~1GB

### 4.3 Self-Improving

The system learns from its own experience:
- **Action analytics**: Which LLM actions (replace_indicator, add_filter, change_entry_logic, etc.) lead to breakthroughs? Feed winning patterns back into prompts.
- **Regime awareness**: Which indicator families work in which market conditions? Adapt invention prompts accordingly.
- **Failure pattern analysis**: If a certain failure mode keeps occurring, the system adjusts its validation thresholds or invention constraints.
- **Strategy deduplication**: Hash-based detection prevents the LLM from producing identical strategies across runs.

### 4.4 Realistic Backtesting

No shortcuts:
- Real OHLCV from Yahoo Finance (501 bars, 2 years)
- Realistic fill simulation via VectorBT's `Portfolio.from_signals()`
- Walk-forward validation before any promotion
- Cross-ticker validation across 15+ tickers
- Composite score penalizes low-trade-count, high-drawdown results
- OverfittingDetector catches curve-fitting across param combos
- Slippage modeling (via confirm module) for final validation

### 4.5 Fast Iteration

Target cadence:
- **Strategy invention → first backtest**: <2 minutes
- **Full 7-turn refinement loop**: <15 minutes per mandate
- **Parallel wave of 5 mandates**: <15 minutes (subprocess parallelism)
- **Time between new strategy discoveries**: every few minutes
- **Full sweep (all strategies × all tickers)**: <1 hour

---

## 5. Current State vs Vision

### What's Working (MVP, Deployed)

- [x] 28 strategies with proven backtest results
- [x] VectorBT engine with vectorized sweep (~600x faster than sequential)
- [x] Walk-forward + cross-ticker validation
- [x] 4-agent cron pipeline running autonomously
- [x] Strategy invention module (LLM writes new strategy code)
- [x] Meta-analysis with regime detection
- [x] Guardrails + overfitting detection
- [x] API budget optimization (GLM-4.7 for mechanical, GLM-5-Turbo for creative)
- [x] Pushed to GitHub: [DynamicEV21/CrabQuant](https://github.com/DynamicEV21/CrabQuant)
- [x] 96+ tests passing for core engine
- [x] 31/31 refinement pipeline components built with unit tests

### What's Built But Not Yet Integrated

The refinement pipeline components are individually tested but not yet wired into a working end-to-end loop:
- [ ] End-to-end integration test with real LLM calls
- [ ] Real mandate execution (invent → backtest → refine → promote)
- [ ] Cron integration for continuous autonomous operation
- [ ] Wave dashboard wired to actual running mandates

### What Needs to Be Built (Production Gaps)

- [ ] **Always-on daemon**: The refinement pipeline currently runs as a script. Needs to become a persistent process with supervisor.
- [ ] **Portfolio-level optimization**: Currently optimizes individual strategies. Needs portfolio construction (risk parity, correlation-aware allocation).
- [ ] **Strategy decay detection**: Monitor live-ish performance of promoted strategies. Retire strategies whose edge fades.
- [ ] **Paper trading interface**: Feed validated strategies into a simulated portfolio to track real-time performance.
- [ ] **Multi-timeframe analysis**: Strategies that combine signals across daily, weekly, hourly timeframes.
- [ ] **Slippage + commission modeling**: The confirm module exists but isn't integrated into the refinement loop.
- [ ] **Broker integration** (far future): Interactive Brokers or Alpaca for live trading.

### What's Been Learned

From the QuantFactory → CrabQuant migration and 2 days of intensive development:
1. **No LangGraph** — agent orchestration frameworks add overhead without value for deterministic loops
2. **Python owns the loop** — JSON on disk, subprocesses for isolation, LLMs as consultants
3. **Deterministic diagnosis before LLM reasoning** — classify failures by code, then ask the LLM to fix them
4. **Three-gate validation catches 90% of bad LLM code** before expensive backtesting
5. **Stagnation detection is critical** — the LLM will happily tweak parameters forever without converging
6. **Subprocess isolation is non-negotiable** for parallel execution (shared module state will corrupt runs)
7. **GLM-5-turbo wraps JSON in code blocks** — always use extraction, never raw `json.loads()`
8. **API budget is prompts, not tokens** — GLM Coding Plan limits by prompt count, so minimize LLM calls
9. **Vectorized sweep is 600x faster** — always use `generate_signals_matrix()` + `Portfolio.from_signals()` for param grids
10. **GPU doesn't help** — VectorBT free is CPU-only (Pro-only), CuPy can't accelerate VectorBT internals

---

## 6. Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Convergence rate | >15% of mandates hit Sharpe target | Not yet measured (E2E not run) |
| Average turns to converge | <5 | Not yet measured |
| Validation pass rate | >60% (gates) | Not yet measured |
| Best Sharpe quality | ≥1.5 on walk-forward | Best raw: 2.40 (CAT/macd_momentum) |
| Robust rate | ≥30% survive walk-forward | Not yet measured |
| Unattended runtime | 48+ hours without intervention | Cron runs ~hours, not tested 48h |
| Strategies discovered | Growing weekly | 28 registered, 2 LLM-invented |
| Test coverage | 100% of new code | 31/31 refinement components tested |

---

## 7. Roadmap

### Phase 0: MVP ✅ COMPLETE
- Core engine, 9 base strategies, cron sweep, validation, 4-agent pipeline
- Replaced QuantFactory entirely

### Phase 1-3: Refinement Pipeline ✅ BUILT (not integrated)
- 31 components with unit tests
- Full PRD (2107 lines, 24 sections)
- Parallel wave execution designed
- Ready for E2E integration

### Phase 4: Integration (NEXT)
- Wire the refinement pipeline into a working end-to-end loop
- Run real mandates with actual LLM calls
- Fix integration bugs (4 test collection errors already known)
- Measure convergence rate, adjust thresholds

### Phase 5: Always-On Production
- Convert from cron-triggered to persistent daemon
- Supervisor process for health monitoring
- Auto-mandate generation from market analysis
- Continuous wave execution

### Phase 6: Intelligence Layer
- Portfolio-level optimization
- Strategy decay detection
- Action analytics feedback loop
- Adaptive invention prompts based on historical success patterns

### Phase 7: Deployment Readiness
- Paper trading interface
- Slippage + commission modeling in refinement loop
- Multi-timeframe strategies
- Real-time performance monitoring dashboard

### Phase 8: Live Trading (Far Future)
- Broker integration (Interactive Brokers / Alpaca)
- Risk management layer
- Position sizing (Kelly criterion / risk parity)
- Automated deployment pipeline

---

## 8. Design Principles (Non-Negotiable)

1. **Direct Python execution** — no LangGraph, no agent framework overhead
2. **JSON on disk is state** — no databases, no message queues, just files
3. **Deterministic before intelligent** — classify failures by code, then ask LLM
4. **Subprocess isolation for parallelism** — shared mutable state is the enemy
5. **Validation before promotion** — walk-forward AND cross-ticker, no exceptions
6. **Composite scoring penalizes overfit** — sharpe * sqrt(trades/20) * (1 - abs(max_dd))
7. **Transparent and reproducible** — every decision logged, every strategy traceable
8. **Budget-aware** — GLM-4.7 for mechanical tasks, GLM-5-Turbo only for invention
9. **Free/open-source only** — no VectorBT Pro, no paid APIs beyond z.ai
10. **Real data, realistic conditions** — no shortcuts that sacrifice accuracy for speed

---

## 9. The Dream

> You give it a vision. It runs. It finds edges. It invents new approaches. It validates them. It improves itself. You check in, review the results, and decide what to deploy.

This is the north star. Every architectural decision should move us closer to this.
