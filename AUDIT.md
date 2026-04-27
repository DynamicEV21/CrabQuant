# CrabQuant Codebase Audit

**Date:** 2026-04-26
**Auditor:** CodeCrab 🦀
**Total Python files:** 185 (8,299 lines tests, 4,830 lines refinement, 2,794 lines core engine)

---

## 1. Project Structure

```
CrabQuant/
├── crabquant/                          # Core library (2,794 LOC)
│   ├── engine/                         # Backtest engine (540 LOC)
│   │   ├── backtest.py                 # VectorBT wrapper, composite scoring
│   │   └── parallel.py                 # ProcessPoolExecutor sweep
│   ├── validation/                     # Walk-forward + cross-ticker (322 LOC)
│   ├── data/                           # Yahoo Finance OHLCV (114 LOC)
│   ├── production/                     # Strategy promotion (900 LOC)
│   │   ├── promoter.py                 # Winner registration
│   │   ├── report.py                   # HTML reports
│   │   └── scanner.py                  # Live scanner
│   ├── strategies/                     # 28 strategy files
│   ├── refinement/                     # Iterative refinement pipeline (4,830 LOC)
│   │   └── [26 modules]
│   ├── brief/                          # Market briefings
│   ├── confirm/                        # Confirmation backtesting
│   ├── guardrails.py                   # Guardrail checking
│   ├── invention.py                    # LLM strategy invention
│   ├── regime.py                       # Market regime detection
│   └── run.py                          # CLI entry point (344 LOC)
├── scripts/                            # Execution scripts (3,177 LOC)
│   ├── refinement_loop.py              # Main refinement entry point (546 LOC)
│   ├── cron_task.py                    # Cron agent dispatcher (795 LOC)
│   ├── crabquant_cron.py               # Cron orchestration (391 LOC)
│   ├── sweep_task.py                   # Parameter sweep (249 LOC)
│   ├── validate_task.py                # Validation runner (323 LOC)
│   ├── improve_task.py                 # Strategy improvement (322 LOC)
│   ├── meta_task.py                    # Meta-analysis (227 LOC)
│   └── [others]
├── tests/refinement/                   # Refinement tests (8,299 LOC, 636 tests)
├── results/                            # Run outputs, state
├── refinement_runs/                    # E2E run artifacts
├── mandates/                           # Strategy mandates (JSON)
└── refinement/BUILD_STATUS.json        # Build tracking
```

---

## 2. Component-by-Component Analysis

### 2.1 Core Engine (PROVEN, STABLE)

| Module | LOC | Quality | Notes |
|--------|-----|---------|-------|
| `engine/backtest.py` | 396 | ✅ Good | VectorBT wrapper with composite scoring. Battle-tested. |
| `engine/parallel.py` | 137 | ✅ Good | ProcessPoolExecutor across tickers. Works. |
| `validation/__init__.py` | 322 | ✅ Good | Walk-forward + cross-ticker. Core feature. |
| `data/__init__.py` | 114 | ✅ Good | Yahoo Finance with pickle cache. Simple, works. |
| `guardrails.py` | 200 | ✅ Good | Configurable threshold checking. |

### 2.2 Strategy Library (GROWING)

28 strategies registered. Mix of hand-written and LLM-invented. Quality varies — some invented strategies are essentially copies with renamed variables. No deduplication mechanism.

### 2.3 Production Module (WORKING, BASIC)

| Module | LOC | Quality | Notes |
|--------|-----|---------|-------|
| `production/promoter.py` | 329 | ⚠️ Fair | Registers winners but no portfolio-level logic |
| `production/report.py` | 218 | ⚠️ Fair | HTML reports, not integrated with refinement pipeline |
| `production/scanner.py` | 148 | ⚠️ Fair | Live scanner, not wired to refinement |

### 2.4 Refinement Pipeline (BUILT, NEEDS INTEGRATION)

| Module | LOC | Quality | Issues |
|--------|-----|---------|--------|
| `llm_api.py` | 302 | ⚠️ Fair | Works but ~50s per call. No async, no streaming. No batching. |
| `prompts.py` | 394 | ⚠️ Fair | Good templates but no A/B testing mechanism. Hardcoded. |
| `context_builder.py` | 242 | ⚠️ Fair | Builds context from strategy examples. Stripping logic fragile. |
| `diagnostics.py` | 352 | ✅ Good | Safe backtesting with timing. Sharpe-by-year. |
| `validation_gates.py` | 200 | ✅ Good | 3-gate system. Works well after PARAM_GRID fix. |
| `module_loader.py` | 90 | ✅ Good | Temp file + importlib. Clean. |
| `classifier.py` | 82 | ⚠️ Fair | Only 6 failure modes. Deterministic only, no nuance. |
| `config.py` | 106 | ✅ Good | RefinementConfig with defaults. Clean. |
| `schemas.py` | 220 | ✅ Good | RunState, BacktestReport dataclasses. |
| `stagnation.py` | 128 | ✅ Good | Stagnation scoring + pivot/nuclear/abandon. |
| `wave_manager.py` | 250 | ⚠️ Fair | Subprocess isolation. Max 5 parallel. |
| `mandate_generator.py` | 220 | ⚠️ Fair | Auto-generates mandates but no market data integration. |
| `promotion.py` | 309 | ⚠️ Fair | Walk-forward + cross-ticker before promotion. Imports from validation. |
| `action_analytics.py` | 195 | ✅ Good | Tracks which actions succeed. JSON persistence. |
| `per_wave_metrics.py` | 187 | ✅ Good | Per-wave convergence tracking. |
| `wave_dashboard.py` | 203 | ⚠️ Fair | Real-time dashboard. Not wired to actual process. |
| `regime_sharpe.py` | 172 | ✅ Good | Sharpe by market regime. |
| `portfolio_correlation.py` | 164 | ✅ Good | Equity curve correlation to existing winners. |
| `circuit_breaker.py` | 185 | ✅ Good | Halt if LLM pass rate drops below 30%. |
| `cosmetic_guard.py` | 125 | ✅ Good | Force structural intervention after 3x modify_params. |
| `hypothesis_enforcement.py` | 120 | ✅ Good | Reject generic hypotheses. |
| `gate3_smoke.py` | 126 | ✅ Good | Smoke backtest gate. |
| `guardrails_integration.py` | 117 | ✅ Good | Existing guardrails supplement classification. |
| `tier1_diagnostics.py` | 156 | ✅ Good | Sharpe-by-year, previous attempts. |
| `wave_scaling.py` | 185 | ✅ Good | Parallel limit, wave status tracking. |

### 2.5 Scripts (FUNCTIONAL, FRAGMENTED)

| Script | LOC | Quality | Issues |
|--------|-----|---------|--------|
| `refinement_loop.py` | 546 | ⚠️ Fair | Main entry point. Works but monolithic. Should be thinner. |
| `cron_task.py` | 795 | ⚠️ Fair | Largest script. Dispatches to sweep/improve/validate/meta. |
| `crabquant_cron.py` | 391 | ⚠️ Fair | Cron orchestration. Single-threaded loop. |
| `sweep_task.py` | 249 | ⚠️ Fair | Parameter sweep. Works but isolated from refinement. |
| `validate_task.py` | 323 | ⚠️ Fair | Validation runner. Separate from refinement promotion. |
| `improve_task.py` | 322 | ⚠️ Fair | Strategy improvement. Overlaps with refinement. |
| `meta_task.py` | 227 | ⚠️ Fair | Meta-analysis. Runs every 3h, not continuous. |

---

## 3. Architecture Diagram

```
                    ┌─────────────────────┐
                    │   Cron Agents (5)    │
                    │  (supervisors only)  │
                    └──────────┬──────────┘
                               │ health check
                               ▼
┌──────────┐    ┌─────────────────────────────┐    ┌──────────┐
│ Mandate  │───▶│   Refinement Pipeline       │───▶│ Promoted │
│ Generator│    │  (should be always-on)       │    │ Winners  │
└──────────┘    │                             │    └────┬─────┘
                │  ┌───────┐  ┌────────────┐  │         │
                │  │ LLM   │─▶│ 3-Gate     │  │         │
                │  │ API   │  │ Validation  │  │         │
                │  └───────┘  └─────┬──────┘  │         │
                │                   ▼         │         │
                │  ┌──────────────────────┐   │         │
                │  │   Backtest Engine    │   │         │
                │  │   (VectorBT)         │   │         │
                │  └──────────┬───────────┘   │         │
                │             ▼               │         │
                │  ┌──────────────────────┐   │         │
                │  │  Failure Classifier  │   │         │
                │  │  + Stagnation Detect │   │         │
                │  └──────────┬───────────┘   │         │
                │             ▼               │         │
                │  ┌──────────────────────┐   │         │
                │  │  Refine or Pivot     │◀──┘         │
                │  │  (back to LLM)       │             │
                │  └──────────────────────┘             │
                └──────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Meta-Analysis      │
                    │  (learns patterns)  │
                    └─────────────────────┘
```

**Current problem:** The pipeline above exists as 26 separate modules but there's NO main loop that runs it continuously. `refinement_loop.py` runs ONE mandate and exits. The cron agents (sweep/improve/validate/meta) are SEPARATE scripts that don't use the refinement pipeline at all.

---

## 4. Data Flow Map

```
mandates/*.json ──▶ refinement_loop.py ──▶ refinement_runs/<name>/
                        │                        │
                        ▼                        ▼
                   LLM API (z.ai)         strategy_v*.py
                        │                        │
                        ▼                        ▼
                   validation_gates          backtest
                        │                        │
                        ▼                        ▼
                   (pass/fail)             BacktestReport
                        │                        │
                        ▼                        ▼
                   classifier.py           diagnostics.py
                        │                        │
                        └──────────┬─────────────┘
                                   ▼
                            context_builder.py
                                   │
                                   ▼
                            LLM (refine)
                                   │
                                   └──▶ repeat (up to 7 turns)

WINNERS ──▶ promotion.py ──▶ STRATEGY_REGISTRY
                                       │
                                       ▼
                              results/insights.json
                                       │
                                       ▼
                              meta_task.py (every 3h)
```

**Gap:** Winners from the refinement pipeline aren't flowing to the cron agents. The cron agents use the old sweep/improve/validate scripts which don't know about refinement runs.

---

## 5. Top 10 Problems (by severity)

### 🔴 CRITICAL

**1. No continuous execution loop**
The refinement pipeline runs ONE mandate and exits. There's no daemon that picks up mandates, runs them, promotes winners, and queues new ones. This is the #1 blocker for Tristan's vision.

**2. Cron agents don't use the refinement pipeline**
`crabquant-wave` runs parameter sweeps. `crabquant-improve` runs old-style improvements. `crabquant-validate` runs walk-forward. None of them use the 31-component refinement system. The refinement pipeline is completely disconnected from the running system.

**3. Two separate strategy worlds**
The refinement pipeline generates strategies in `refinement_runs/` as temp files. The cron agents work with `crabquant/strategies/` registered strategies. There's no bridge — refinement winners don't get discovered by the sweep engine.

### 🟠 HIGH

**4. LLM API is synchronous and slow**
Every LLM call takes 30-60 seconds. No async, no streaming, no batching. In a 7-turn refinement loop with 3 retries per gate, that's 7 × 2 × 45s = 10.5 minutes of just waiting for the LLM. The system can't run "every few minutes" with this bottleneck.

**5. Mandate generator has no market data integration**
`mandate_generator.py` creates mandates from the strategy catalog, but it doesn't look at current market conditions, regime, or what tickers are performing well. It generates generic mandates.

**6. No resource management**
The wave manager limits parallel processes to 5, but there's no CPU/RAM monitoring. No API budget tracking. No throttling when rate-limited (we saw rate limits kill the wave cron).

### 🟡 MEDIUM

**7. Scripts are monolithic and overlapping**
`cron_task.py` (795 LOC), `refinement_loop.py` (546 LOC), `crabquant_cron.py` (391 LOC) all have overlapping concerns. `improve_task.py` does strategy improvement, which is what the refinement pipeline does. This should be consolidated.

**8. No strategy deduplication**
The LLM can generate essentially identical strategies across runs. There's a hash-based check in diagnostics.py but it's not used in the refinement loop to prevent duplicate research.

**9. Classifier is too simple**
Only 6 deterministic failure modes. No nuance — "too_few_trades" doesn't distinguish between "0 trades" (broken) and "8 trades" (might just need parameter tuning). The LLM gets blunt failure labels.

**10. No observability**
No metrics, no dashboards, no alerts. `wave_dashboard.py` exists but isn't wired to anything. You can't see "3 strategies being researched, 1 stuck on overtrading gate, 2 pending" without manually checking files.

---

## 6. Top 10 Improvements (by impact)

### 1. **Build the continuous execution loop** (CRITICAL)
Create `scripts/research_daemon.py` — a persistent process that:
- Maintains a mandate queue (JSON file or directory)
- Spawns wave_manager for each batch of mandates
- Promotes winners automatically
- Generates new mandates when queue runs low
- Handles graceful shutdown (SIGTERM)
- Logs state to `results/daemon_state.json`

**Impact:** This is THE missing piece. Without it, the refinement pipeline is just a library.

### 2. **Unify the two systems** (CRITICAL)
Bridge the refinement pipeline to the existing cron infrastructure:
- Refinement winners → auto-register in STRATEGY_REGISTRY
- Cron sweep → includes refinement-invented strategies
- Meta-analysis → reads refinement run history
- Remove redundant scripts (improve_task.py replaced by refinement)

### 3. **Async LLM calls with concurrency** (HIGH)
Replace synchronous `urllib.request` with `asyncio` + `aiohttp`:
- Parallel gate retries (3 retries can run simultaneously)
- Parallel mandate processing (send multiple LLM calls at once)
- Streaming responses for faster time-to-first-token
- Target: cut effective LLM latency by 3-5x

### 4. **Smart mandate generation** (HIGH)
Feed market data into mandate creation:
- Current regime (bull/bear/volatile) → suggest appropriate archetypes
- Recent winner patterns → generate similar mandates
- Ticker performance ranking → focus on high-opportunity tickers
- Gap analysis → identify uncovered ticker/archetype combos

### 5. **Resource-aware scheduling** (MEDIUM)
- Track API budget (prompt count, not tokens)
- Monitor CPU/RAM before spawning new waves
- Rate limit detection → exponential backoff
- Priority queue: promising strategies get more turns

### 6. **Consolidate scripts** (MEDIUM)
- `research_daemon.py` replaces `cron_task.py` + `crabquant_cron.py` + `refinement_loop.py`
- `sweep_task.py` becomes a daemon sub-command
- `improve_task.py` deleted (replaced by refinement pipeline)
- Total: 3,177 LOC → ~1,000 LOC

### 7. **Strategy deduplication in loop** (MEDIUM)
- Before running backtest, hash the strategy code
- Check against `results/strategy_hashes.json`
- If duplicate, skip backtest and tell LLM "this is identical to strategy X"
- Saves API budget and compute

### 8. **Richer failure classification** (MEDIUM)
- Expand from 6 to 12+ failure modes
- Add severity levels (broken vs suboptimal)
- Include quantitative hints (e.g., "max drawdown 45% at bar 87")
- Feed these into context for better LLM diagnosis

### 9. **Observability layer** (LOW-MEDIUM)
- Prometheus-style metrics export (or simple JSON)
- Running count: mandates queued, in-progress, completed, winners
- Per-component timing: LLM calls, backtests, gates
- Telegram push for notable events (new winner, stagnation detected)
- `wave_dashboard.py` wired to real state

### 10. **Prompt optimization** (LOW)
- A/B test prompt variants
- Track which prompt patterns lead to faster convergence
- Extract winning prompt techniques from successful runs
- Feed back into `prompts.py`

---

## 7. Integration Gaps

| Built Component | Integrated With | Status |
|----------------|-----------------|--------|
| refinement/orchestrator.py | scripts/refinement_loop.py | ✅ Wired |
| refinement/promotion.py | refinement_loop.py | ❌ Not called |
| refinement/mandate_generator.py | Nothing | ❌ Standalone |
| refinement/wave_dashboard.py | Nothing | ❌ Standalone |
| refinement/action_analytics.py | refinement_loop.py | ❌ Not called |
| refinement/auto_promotion (in wave_manager) | Nothing | ❌ Not called |
| refinement/cron_integration.py | Nothing | ❌ Standalone |
| production/promoter.py | promote_task.py | ✅ Wired (old system) |
| production/scanner.py | Nothing | ❌ Standalone |

**Summary:** 26 refinement modules built, 3 wired into the main loop, 23 standalone. The integration gap is massive.

---

## 8. Test Coverage

| Area | Tests | Status |
|------|-------|--------|
| refinement/ (unit) | 636 | ✅ All passing |
| refinement/ (E2E) | 36 | ✅ All passing |
| core engine | ~96 | ✅ Passing |
| scripts/ | 0 | ❌ No tests |
| integration | 0 | ❌ No tests |

The refinement pipeline is well-tested in isolation but there are zero tests for the scripts that actually run the system (`cron_task.py`, `refinement_loop.py`, etc.). This is why integration bugs (like the PARAM_GRID mismatch) only show up in real E2E runs.

---

## 9. Summary

CrabQuant has excellent building blocks — a fast backtest engine, a sophisticated refinement pipeline with 31 tested components, and proven validation. But it's currently a collection of parts, not a system. The critical missing piece is the **continuous execution loop** that ties everything together into an always-on research engine.

**Priority order:**
1. Build `research_daemon.py` (the always-on loop)
2. Wire existing components into the loop
3. Async LLM calls for speed
4. Bridge refinement winners to strategy registry
5. Everything else
