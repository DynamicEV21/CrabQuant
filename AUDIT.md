# CrabQuant Codebase Audit

**Date:** 2026-04-26  
**Auditor:** CodeCrab 🦀 (automated code review)  
**Scope:** Full codebase — 185 Python files, ~55K lines (library + scripts + tests)

---

## 1. Project Structure Overview

```
~/development/CrabQuant/
├── crabquant/                      # Core library (13,814 LOC)
│   ├── __init__.py                 # 11 LOC
│   ├── run.py                      # 344 LOC — main CLI runner
│   ├── invention.py                # 180 LOC — strategy invention (STUB)
│   ├── regime.py                   # 294 LOC — market regime detection
│   ├── guardrails.py               # 200 LOC — backtest guardrails
│   ├── indicator_cache.py          # 110 LOC — module-level indicator cache
│   ├── strategies/                 # 22 strategy files (3,100 LOC)
│   │   ├── __init__.py             # 226 LOC — STRATEGY_REGISTRY + imports
│   │   └── *.py                    # 22 strategy implementations
│   ├── engine/                     # Backtest engine (533 LOC)
│   │   ├── backtest.py             # 396 LOC — VectorBT BacktestEngine
│   │   └── parallel.py             # 137 LOC — ProcessPoolExecutor
│   ├── data/                       # Data loading (114 LOC)
│   │   └── __init__.py             # yfinance + pickle cache
│   ├── confirm/                    # Confirmation (1,573 LOC)
│   │   ├── __init__.py             # 93 LOC — ConfirmationResult
│   │   ├── runner.py               # 220 LOC — bar-by-bar backtest
│   │   ├── batch.py                # 154 LOC — batch confirmation
│   │   └── strategy_converter.py   # 1,106 LOC — VBT→backtesting.py converters
│   ├── validation/                 # Walk-forward (322 LOC)
│   │   └── __init__.py             # walk_forward + cross_ticker
│   ├── production/                 # Registry (800 LOC)
│   │   ├── __init__.py, promoter.py, scanner.py, report.py
│   ├── brief/                      # Daily brief (581 LOC)
│   │   ├── __init__.py, models.py, formatter.py, market.py, discoveries.py
│   └── refinement/                 # LLM refinement (~5,200 LOC, 28 files)
│       ├── prompts.py, context_builder.py, diagnostics.py, llm_api.py
│       ├── wave_manager.py, schemas.py, mandate_generator.py, validation_gates.py
│       ├── promotion.py, classifier.py, config.py, module_loader.py
│       ├── stagnation.py, circuit_breaker.py, cosmetic_guard.py, gate3_smoke.py
│       ├── action_analytics.py, guardrails_integration.py, hypothesis_enforcement.py
│       ├── per_wave_metrics.py, portfolio_correlation.py, regime_sharpe.py
│       ├── tier1_diagnostics.py, wave_dashboard.py, wave_scaling.py
├── scripts/                        # Cron entry points (3,177 LOC, 11 files)
│   ├── cron_task.py                # 795 LOC — main sweep + backtest
│   ├── refinement_loop.py          # 546 LOC — LLM refinement orchestrator
│   ├── crabquant_cron.py           # 391 LOC — cron lifecycle
│   ├── validate_task.py            # 323 LOC
│   ├── improve_task.py             # 322 LOC
│   ├── sweep_task.py               # 249 LOC
│   ├── meta_task.py                # 227 LOC
│   ├── confirm_task.py             # 198 LOC
│   ├── wave_runner.py, promote_task.py, brief_task.py
├── tests/                          # Test suite (10,751 LOC, 30+ files)
├── results/                        # Runtime data
│   ├── cron_state.json             # ~55 completed combos
│   ├── winners/, confirmed/, logs/, plots/
│   └── insights.json, meta_report.json
└── debug_test.py, debug_strategy.py
```

**File counts:** 185 `.py` files  
**Lines of code:** ~55K total (library: 13.8K, scripts: 3.2K, tests: 10.8K, strategies: 3.1K)  
**Test-to-code ratio:** ~1:1.5 (thorough)

---

## 2. Component-by-Component Analysis

### 2.1 Strategies (`crabquant/strategies/`) — 22 files, 3,100 LOC

**What:** 22 trading strategies. Each exports `generate_signals(df, params) -> (entries, exits)`, `DEFAULT_PARAMS`, `PARAM_GRID`, `DESCRIPTION`. `__init__.py` builds `STRATEGY_REGISTRY`.

**Quality:** Good. Clean pattern, consistent interface, pandas_ta indicators.

**Problems:**
- `__init__.py` (226 LOC): Verbose per-strategy imports — should use `__all__` + lazy loading
- `informed_adaptive_trend_reversion.py` and `informed_simple_adaptive.py` reference `generate_signals_matrix` — missing from most strategies
- `ichimoku_trend.py` (54 LOC): Unusually short — likely incomplete
- 22 strategies but only 17 have confirmation converters (see 2.4)

### 2.2 Engine (`crabquant/engine/`) — 2 files, 533 LOC

**What:** VectorBT `BacktestEngine` with param grid sweeps. `parallel.py` uses ProcessPoolExecutor for multi-ticker.

**Quality:** Solid, well-structured.

**Problems:**
- `parallel.py` line ~20: `from crabquant.strategies STRATEGY_REGISTRY` — syntax error (missing `import`). Only works if monkey-patched at runtime
- `backtest.py` line ~10: Default `sharpe_target=1.5` is aggressive — many profitable strategies have Sharpe 0.8–1.2
- `parallel.py`: Each worker loads data independently — no shared memory, wastes RAM

### 2.3 Data (`crabquant/data/__init__.py`) — 114 LOC

**What:** yfinance OHLCV loader with pickle caching to `~/.cache/crabquant/`, 20-hour TTL.

**Quality:** Simple, functional.

**Problems:**
- Cache validity uses `time.time()` — doesn't account for market hours (caches stale overnight data)
- No validation of returned DataFrame (could silently return empty data)
- Hardcoded cache path — not configurable

### 2.4 Confirmation (`crabquant/confirm/`) — 4 files, 1,573 LOC

**What:** VectorBT fast screen → backtesting.py realistic bar-by-bar confirmation. `strategy_converter.py` (1,106 LOC) converts VBT strategies to `backtesting.py` Strategy subclasses.

**Quality:** Well-structured registry pattern, but the converter is a maintenance burden.

**Problems:**
- **`strategy_converter.py` (1,106 LOC):** The single largest file. Contains 17 hand-written converters + ~15 indicator helpers. Every new strategy requires a new converter.
- Lines 30–150: Pure-numpy re-implementations of rolling functions (`_rolling_max`, `_rolling_min`, `_rsi`, `_atr`, `_adx`, `_macd`, `_stoch`, `_bbands`) — O(n×window) instead of O(n), significantly slower than pandas
- `_CONVERTERS` dict has 17 entries but `STRATEGY_REGISTRY` has 22 strategies — **5 strategies cannot be confirmed** (the `informed_*` and `invented_volume_*` variants)
- `runner.py` line ~79: `lambda size, price: _slippage_commission(size, price, slip)` — closure captures variable
- `batch.py`: Circular import risk with `confirm/__init__.py`

### 2.5 Validation (`crabquant/validation/__init__.py`) — 322 LOC

**What:** Walk-forward (60/40 split) and cross-ticker validation with regime detection.

**Quality:** Good concept.

**Problems:**
- Hardcoded 60/40 split — not configurable
- Single split, not rolling/expanding window
- Cross-ticker assumes equal applicability across tickers

### 2.6 Refinement (`crabquant/refinement/`) — 28 files, ~5,200 LOC

**What:** LLM-driven strategy improvement. Includes prompt engineering, validation gates, circuit breakers, stagnation detection, cosmetic guards, action analytics, wave-based parallel execution.

**Quality:** Most sophisticated package. Good separation of concerns.

**Problems:**
- **`llm_api.py` (302 LOC):** Uses raw `urllib.request` — no retry logic, no connect timeout, no connection pooling, no streaming. Can hang indefinitely.
- **`wave_manager.py` (250 LOC):** Uses `subprocess.run()` per mandate — no resource limits, no timeout, no output capture limits
- **`validation_gates.py`:** Gate 3 smoke backtest uses VectorBT engine, not the confirmation runner — inconsistent validation
- **`mandate_generator.py`:** Doesn't check `cron_state.json` for already-completed combos — duplicates work
- **`classifier.py` (82 LOC):** Only 5 failure modes; new types silently become "unknown"
- **`context_builder.py` (242 LOC):** No token limit awareness — could exceed LLM context window
- **`stagnation.py`:** Weights (0.4 sharpe_trend, 0.2 variance, 0.4 repetition) are hardcoded — not tunable per strategy
- **`circuit_breaker.py`:** Default `min_pass_rate=0.3` over `window=20` — once opened, there's no automatic recovery mechanism documented

### 2.7 Production (`crabquant/production/`) — 4 files, 800 LOC

**What:** Promotes ROBUST strategies to production registry with markdown reports + embedded JSON metadata.

**Quality:** Clean dedup logic, good design.

**Problems:**
- `promoter.py` line ~150: `_extract_slippage_results()` parses notes by string matching — extremely fragile
- `promoter.py` line ~200: ROBUST verdicts with no parseable notes get dummy SlippageResults with all zeros — **hides missing data**
- **No retirement mechanism** — promoted strategies stay forever
- `scanner.py` and `promoter.py` both compute params hash independently — should share utility

### 2.8 Brief (`crabquant/brief/`) — 5 files, 581 LOC

**What:** Daily Telegram-friendly market brief with regime, production status, cron health.

**Quality:** Clean, well-structured.

**Problems:**
- `discoveries.py` line ~120: `get_cron_status()` shells out to `openclaw cron list` and **parses text output** — extremely fragile
- `market.py` line ~30: `get_market_regime()` loads SPY data every call — no caching
- `formatter.py` line ~65: Hard 800-char limit with naive line truncation

### 2.9 Regime (`crabquant/regime.py`) — 294 LOC

**What:** Score-based market regime classifier (5 regimes) using SPY SMA slopes, ROC, BB width, realized vol, VIX.

**Quality:** Sound multi-signal approach.

**Problems:**
- `REGIME_STRATEGY_AFFINITY` (210 LOC): **Hardcoded scores** for 5 regimes × 22 strategies — not data-driven, never updated from backtest results
- Line ~40: `sma50_slope` from 6-bar lookback — too short for 50-day SMA
- Confidence formula `(best - second) / (best + 0.001)` — epsilon makes low scores appear confident

### 2.10 Guardrails (`crabquant/guardrails.py`) — 200 LOC

**What:** Configurable guardrails with conservative/moderate/aggressive presets.

**Quality:** Clean, well-documented.

**Problems:**
- Overfitting detection only checks trade count — no parameter stability, return distribution, or correlation checks
- Conservative `min_trades=30` too strict for 6-month backtests
- No per-regime adjustment

### 2.11 Scripts (`scripts/`) — 11 files, 3,177 LOC

**What:** Cron entry points. Self-contained with arg parsing, logging, error handling.

**Quality:** Functional but duplicated patterns.

**Problems:**
- `cron_task.py` (795 LOC): **Monolith** — sweep logic + backtesting + winner detection + state management in one file
- `refinement_loop.py` (546 LOC): De facto orchestrator, but lives as a script
- Every script has `sys.path.insert(0, ...)` boilerplate
- `improve_task.py` depends on `invention.py` which has hardcoded mock data

### 2.12 Invention (`crabquant/invention.py`) — 180 LOC

**What:** Strategy invention system.

**Quality:** **Stub. Not production-ready.**

**Problems:**
- Line 15: `analyze_market_data()` returns **hardcoded mock data**
- Line 70: `test_strategy_code()` checks for `generate_signals_matrix` — missing from most strategies
- Line 90: `importlib.reload()` in running process — risky

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OpenClaw Cron Agents (4)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │  Wave    │ │ Validate │ │ Confirm  │ │ Promote  │              │
│  │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │              │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘              │
│       └────────────┴────────────┴────────────┘                      │
│                         │                                          │
│  ┌──────────────────────▼──────────────────────────────────────┐   │
│  │                  scripts/ layer (3,177 LOC)                  │   │
│  │  cron_task.py │ validate_task.py │ confirm_task.py          │   │
│  │  sweep_task.py │ promote_task.py │ brief_task.py            │   │
│  │  improve_task.py │ meta_task.py │ refinement_loop.py        │   │
│  │  wave_runner.py │ crabquant_cron.py                         │   │
│  └──────────────────────┬──────────────────────────────────────┘   │
│                         │                                          │
├─────────────────────────▼──────────────────────────────────────────┤
│                     crabquant/ library                              │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐      │
│  │  engine/    │  │   data/      │  │    strategies/        │      │
│  │  backtest   │◄─┤  yfinance    │◄─┤  22 strategies        │      │
│  │  parallel   │  │  pickle cache│  │  STRATEGY_REGISTRY    │      │
│  └──────┬──────┘  └──────────────┘  └──────────────────────┘      │
│         │                                                           │
│  ┌──────▼──────────────────────────────────────────────────────┐   │
│  │                    Pipeline Stages                           │   │
│  │                                                              │   │
│  │  1. SWEEP     engine.BacktestEngine.run_vectorized()        │   │
│  │       ↓                                                      │   │
│  │  2. SCREEN    guardrails.check_guardrails()                  │   │
│  │       ↓                                                      │   │
│  │  3. VALIDATE  validation.walk_forward_test()                 │   │
│  │       ↓                                                      │   │
│  │  4. CONFIRM   confirm.run_confirmation() (backtesting.py)    │   │
│  │       ↓                                                      │   │
│  │  5. PROMOTE   production.promote_strategy()                  │   │
│  │       ↓                                                      │   │
│  │  6. REFINE    refinement/ (LLM-driven improvement)           │   │
│  │       ↓                                                      │   │
│  │  7. REPORT    brief.generate_brief() → Telegram              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              refinement/ (LLM pipeline, ~5,200 LOC)          │   │
│  │                                                              │   │
│  │  llm_api ──► context_builder ──► prompts ──► wave_manager    │   │
│  │                                    │                         │   │
│  │  validation_gates ◄────────────────┘                         │   │
│  │       ↓                                                      │   │
│  │  circuit_breaker ──► cosmetic_guard ──► stagnation           │   │
│  │       ↓                                                      │   │
│  │  action_analytics ──► diagnostics ──► classifier             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐      │
│  │ regime.py   │  │ guardrails.py│  │   production/         │      │
│  │ 5 regimes   │  │ 3 presets    │  │  registry + reports   │      │
│  └─────────────┘  └──────────────┘  └──────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘

  External:    yfinance (data)    Zhipu AI GLM (LLM)    Telegram (briefs)
```

---

## 4. Data Flow Map

```
                    ┌─────────────┐
                    │  yfinance   │
                    │  28 tickers │
                    └──────┬──────┘
                           │ OHLCV DataFrames
                           ▼
                ┌──────────────────────┐
                │  data/__init__.py     │
                │  load_data() + cache  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  strategies/          │
                │  generate_signals()   │
                │  → entries, exits     │
                └──────────┬───────────┘
                           │
            ┌──────────────▼──────────────┐
            │  SWEEP: engine/backtest.py   │
            │  VectorBT param grid sweep   │
            │  (sharpe, return, dd, etc)  │
            └──────────────┬──────────────┘
                           │ BacktestResult
                           ▼
                ┌──────────────────────┐
                │  SCREEN: guardrails  │
                │  Sharpe ≥ 1.5        │
                │  DD ≤ 25%, Return ≥10%│
                │  Trades ≥ 5          │
                └──────────┬───────────┘
                           │ passed →
                           ▼
                ┌──────────────────────┐
                │  winners.json        │
                │  ~55 combos tracked  │
                └──────────┬───────────┘
                           │
              ┌────────────▼────────────┐
              │  VALIDATE: validation/  │
              │  walk-forward (60/40)   │
              │  cross-ticker check     │
              └────────────┬────────────┘
                           │ validated →
                           ▼
                ┌──────────────────────┐
                │  CONFIRM: confirm/   │
                │  backtesting.py      │
                │  3 periods × 3 slips │
                └──────────┬───────────┘
                           │ ROBUST / FRAGILE / FAILED
                           ▼
                ┌──────────────────────┐
                │  confirmed.json      │
                └──────────┬───────────┘
                           │ ROBUST only →
                           ▼
                ┌──────────────────────┐
                │  PROMOTE: production/│
                │  registry.json + .md │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  BRIEF → Telegram    │
                └──────────────────────┘


   ─── REFINEMENT BRANCH (LLM-driven) ───

   Winners that fail validation →
        │
        ▼
   ┌──────────────────────┐
   │  refinement/         │
   │  mandate_generator   │
   │  (strategy, ticker,  │
   │   failure_mode)      │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │  wave_manager        │
   │  subprocess per      │
   │  mandate             │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │  LLM API (GLM)       │
   │  + context_builder   │
   │  + prompts.py        │
   └──────────┬───────────┘
              │ new strategy code
              ▼
   ┌──────────────────────┐
   │  validation_gates    │
   │  G1: syntax/import   │
   │  G2: generate_signals│
   │  G3: smoke backtest  │
   └──────────┬───────────┘
              │ passed →
              ▼
   ┌──────────────────────┐
   │  Backtest + evaluate │
   │  + circuit_breaker   │
   │  + stagnation check  │
   │  + cosmetic_guard    │
   └──────────┬───────────┘
              │ improved →
              ▼
   ┌──────────────────────┐
   │  Save strategy code  │
   │  → strategies dir    │
   │  → back in pipeline  │
   └──────────────────────┘
```

---

## 5. Top 10 Problems (Ranked by Severity)

### 🔴 P1: `invention.py` is entirely stubbed with mock data
**File:** `crabquant/invention.py` lines 15–30  
**Impact:** The "improve" cron agent (`scripts/improve_task.py`) depends on this for strategy invention. It returns hardcoded ticker lists and regime classifications. Any "invented" strategies are based on fake analysis.  
**Severity:** HIGH — undermines the entire autonomous improvement pipeline.

### 🔴 P2: `parallel.py` has a syntax error in import
**File:** `crabquant/engine/parallel.py` line ~20  
**Impact:** `from crabquant.strategies STRATEGY_REGISTRY` is invalid Python. This file cannot be imported directly. Works only if the import is monkey-patched at runtime by scripts.  
**Severity:** HIGH — latent bug, breaks if imported outside the specific script context.

### 🔴 P3: 5 of 22 strategies cannot be confirmed
**File:** `crabquant/confirm/strategy_converter.py` `_CONVERTERS` dict  
**Impact:** `informed_adaptive_trend_reversion`, `informed_simple_adaptive`, `invented_volume_breakout_adx`, `invented_volume_adx_ema`, and `invented_volume_momentum_trend` have no backtesting.py converters. If any of these win the VectorBT sweep, they **cannot proceed through confirmation**.  
**Severity:** HIGH — silent pipeline failure.

### 🟡 P4: `llm_api.py` uses raw `urllib.request` with no reliability features
**File:** `crabquant/refinement/llm_api.py` lines 1–302  
**Impact:** No retry on transient failures, no connect timeout (can hang indefinitely), no connection pooling, no streaming support. Every LLM call is a cold connection.  
**Severity:** MEDIUM-HIGH — the refinement pipeline's LLM dependency is fragile.

### 🟡 P5: Strategy converter re-implements rolling functions in pure Python
**File:** `crabquant/confirm/strategy_converter.py` lines 30–150  
**Impact:** `_rolling_max`, `_rolling_min`, `_rsi`, `_atr`, `_adx`, `_macd`, `_stoch`, `_bbands` are O(n×window) pure-numpy loops. For a 2-year daily series (500 bars) with typical windows, this is ~10× slower than pandas native. The batch confirm runs 9 backtests per strategy (3 periods × 3 slippage levels) — this compounds.  
**Severity:** MEDIUM — significant performance bottleneck.

### 🟡 P6: No strategy retirement mechanism
**File:** `crabquant/production/` (all files)  
**Impact:** Once promoted to `registry.json`, a strategy stays forever. No re-validation, no expiry, no removal for degraded performance. Over time, the production registry accumulates stale strategies.  
**Severity:** MEDIUM — data integrity risk grows over time.

### 🟡 P7: `discoveries.py` parses `openclaw cron list` text output
**File:** `crabquant/brief/discoveries.py` lines 120–160  
**Impact:** Shells out to `openclaw cron list` and parses stdout with string matching. Any change in OpenClaw's output format breaks the brief. The fallback "assume 4 crons" is misleading.  
**Severity:** MEDIUM — fragile integration.

### 🟡 P8: `promoter.py` hides missing slippage data with dummy values
**File:** `crabquant/production/promoter.py` lines ~195–210  
**Impact:** When `_extract_slippage_results()` can't parse notes, it creates `SlippageResult(passed=True, sharpe=0, return=0, ...)` for ROBUST verdicts. The markdown report shows "✅ 0.0% slippage: Sharpe 0.00" — looks like real data but is fabricated.  
**Severity:** MEDIUM — misleading production reports.

### 🟠 P9: Hardcoded regime-strategy affinity scores
**File:** `crabquant/regime.py` lines 80–290 (210 LOC)  
**Impact:** 110 hardcoded affinity scores (5 regimes × 22 strategies) are never validated against actual backtest results. The brief recommends strategies based on these assumed affinities, not evidence.  
**Severity:** MEDIUM-LOW — could mislead trading decisions.

### 🟠 P10: `mandate_generator.py` doesn't deduplicate against completed work
**File:** `crabquant/refinement/mandate_generator.py`  
**Impact:** Can generate mandates for strategy/ticker combos already in `cron_state.json` completed list. Wastes LLM tokens and compute on already-tested combos.  
**Severity:** MEDIUM-LOW — inefficiency.

---

## 6. Top 10 Improvements (Ranked by Impact)

### 💡 I1: Replace `invention.py` mock data with real analysis
**Impact:** HIGH — unlocks the autonomous improvement pipeline  
**What:** Replace `analyze_market_data()` with actual regime detection using `crabquant/regime.py` + historical performance analysis from `results/winners/`. The `meta_task.py` already does some of this — consolidate.  
**Effort:** 2–3 hours

### 💡 I2: Fix `parallel.py` import + add missing strategy converters
**Impact:** HIGH — fixes latent bug + unblocks 5 strategies  
**What:** Change `from crabquant.strategies STRATEGY_REGISTRY` to `from crabquant.strategies import STRATEGY_REGISTRY`. Add converters for the 5 missing strategies (or remove them from the registry).  
**Effort:** 1–2 hours per converter

### 💡 I3: Replace `urllib` with `httpx` in `llm_api.py`
**Impact:** HIGH — reliable LLM calls with retries, timeouts, pooling  
**What:** Replace `urllib.request` with `httpx.Client(retries=3, timeout=30)`. Add exponential backoff, connection pooling, and proper error handling.  
**Effort:** 1–2 hours

### 💡 I4: Add strategy retirement to production pipeline
**Impact:** MEDIUM-HIGH — keeps production registry clean  
**What:** Add a `retire_strategy()` function that marks strategies as inactive when re-validation fails. Add a weekly re-validation cron. Show active/retired counts in the brief.  
**Effort:** 3–4 hours

### 💡 I5: Use `backtesting.lib` instead of hand-rolled indicators in strategy_converter
**Impact:** MEDIUM — 10× faster confirmation backtests  
**What:** `backtesting.py` includes `backtesting.lib` with optimized `RSI`, `SMA`, `EMA`, `ATR`, etc. Replace the 15 custom indicator functions. If missing functions, use `talib` (already a dependency of pandas_ta).  
**Effort:** 2–3 hours

### 💡 I6: Split `cron_task.py` monolith into library + thin script
**Impact:** MEDIUM — better testability, reusability  
**What:** Extract sweep logic, winner detection, and state management into `crabquant/pipeline/sweep.py` (or similar). The script becomes a 20-line entry point.  
**Effort:** 2–3 hours

### 💡 I7: Make regime-strategy affinity data-driven
**Impact:** MEDIUM — evidence-based brief recommendations  
**What:** After each confirmation batch, compute actual Sharpe-by-regime from backtest results. Update affinity scores from data, not hardcoded assumptions. Store in `results/regime_affinity.json`.  
**Effort:** 3–4 hours

### 💡 I8: Replace `openclaw cron list` text parsing with proper API
**Impact:** MEDIUM — robust cron health monitoring  
**What:** Either use `openclaw cron list --json` (if available) or read cron state from `results/cron_state.json` directly instead of shelling out.  
**Effort:** 1 hour

### 💡 I9: Add token limit awareness to `context_builder.py`
**Impact:** MEDIUM-LOW — prevents LLM context overflow  
**What:** Estimate token count of built prompts. If approaching limit, truncate earlier turns or summarize. Use a simple character-based heuristic (1 token ≈ 4 chars for GLM).  
**Effort:** 1–2 hours

### 💡 I10: Proper Python packaging (pyproject.toml + `pip install -e .`)
**Impact:** MEDIUM-LOW — eliminates `sys.path` hacks  
**What:** Create `pyproject.toml` with project metadata and dependencies. Run `pip install -e .` once. Remove all `sys.path.insert(0, ...)` from scripts.  
**Effort:** 1 hour

---

## 7. Integration Gaps

### G1: Refinement pipeline ↔ Production pipeline
**What's built:** The refinement pipeline generates improved strategy code and saves it to the strategies directory.  
**What's missing:** Improved strategies are not automatically re-entered into the sweep/validation/confirmation pipeline. A human must manually trigger a new sweep.  
**Fix:** Add a post-refinement hook that queues the improved strategy for immediate sweep+confirm.

### G2: Meta-learner ↔ Strategy registry
**What's built:** `meta_task.py` analyzes win rates and suggests grid expansions.  
**What's missing:** Grid expansion suggestions are logged to `meta_report.json` but not automatically applied to strategy files. Requires manual editing.  
**Fix:** Add auto-apply mode to `meta_task.py` that writes expanded `PARAM_GRID` directly to strategy files.

### G3: Regime detection ↔ Backtest thresholds
**What's built:** `regime.py` detects 5 market regimes. `guardrails.py` has 3 preset configurations.  
**What's missing:** No automatic adjustment of backtest thresholds based on detected regime. A strategy discovered during HIGH_VOLATILITY is evaluated with the same thresholds as one discovered during LOW_VOLATILITY.  
**Fix:** Add `regime_adjusted_config()` that loosens/tightens thresholds per regime.

### G4: Confirmation results ↔ LLM context
**What's built:** `action_analytics.py` tracks LLM action success rates.  
**What's missing:** Confirmation results (ROBUST/FRAGILE/FAILED with degradation metrics) are not fed back to the LLM. The LLM doesn't learn that certain modifications tend to produce strategies that fail realistic-fill testing.  
**Fix:** Extend `context_builder.py` to include confirmation degradation data for previous attempts of the same strategy.

### G5: Cross-strategy correlation monitoring
**What's built:** `portfolio_correlation.py` exists in refinement/.  
**What's missing:** Not integrated into the production pipeline. No check whether newly promoted strategies are correlated with existing ones. Could end up with 5 strategies that all trigger on the same RSI signal.  
**Fix:** Add a correlation check in `promoter.py` before promotion. Reject strategies with >0.7 correlation to existing production strategies on the same ticker.

### G6: Agent workspaces are empty shells
**What's built:** 4 agent directories in `~/development/crabquant-agents/` (invent, meta, validate, wave) with SOUL.md, TOOLS.md, AGENTS.md, etc.  
**What's missing:** The `agent/` subdirectories (where agent workspace files would live) are **empty** — no prompt.md, no workspace files. The agents have identity/persona files but no actual task definitions or workspace context. They're bootstrapped but not configured.  
**Fix:** Create agent workspace files with task prompts, reference paths, and workspace context for each agent.

### G7: Batch confirm ↔ Validation
**What's built:** `confirm/batch.py` runs 3 periods × 3 slippage levels = 9 backtests per strategy.  
**What's missing:** The walk-forward validation in `validation/__init__.py` is a separate step that doesn't use the batch confirm results. The two systems produce independent verdicts with no reconciliation.  
**Fix:** Unify validation and confirmation into a single pipeline stage, or at minimum cross-reference their verdicts.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Python files | 185 |
| Total LOC | ~55,000 |
| Library code | 13,814 LOC |
| Script code | 3,177 LOC |
| Test code | 10,751 LOC |
| Strategy code | ~3,100 LOC |
| Test-to-code ratio | ~1:1.5 |
| Strategies defined | 22 |
| Strategies with converters | 17 (5 missing) |
| Completed combos | ~55 |
| Cron agents | 4 (wave, validate, confirm, promote) |
| Refinement modules | 28 files |
| Critical bugs | 3 (mock invention, syntax error, missing converters) |
| Integration gaps | 7 |
| Hardcoded data | 110 affinity scores + invention mock data |
