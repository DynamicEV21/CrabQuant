# CrabQuant — Roadmap

**Last Updated:** 2026-04-28
**Status:** Phase 5.5 (Regime-Aware Registry) — CURRENT

---

## Phase History

| Phase | Name | Status |
|-------|------|--------|
| 0 | MVP | ✅ Complete |
| 1-3 | Refinement Pipeline | ✅ Built |
| 4 | Integration | ✅ Done |
| 4.5 | Convergence Tuning | ✅ Done |
| 5A | Daemon Core | ✅ Done |
| 5 | Fix the Funnel | ✅ Mostly Done |
| **5.5** | **Regime-Aware Registry** | **✅ Done** |
| **5.6** | **Invention Accelerators** | **🔴 CURRENT** |
| 6 | Production Validation | Planned |
| 7 | Intelligence Layer | Planned |
| 8 | Deployment Readiness | Planned |
| 9 | Live Trading | Far Future |

---

## Phase 5: Fix the Funnel 🔴 CURRENT

**Why this phase exists:** The pipeline runs end-to-end, strategies get invented, backtests produce Sharpe ratios >2.0, but the system is fundamentally broken at the promotion layer. After 80 mandates and 58 winner entries, **zero strategies have been registered in STRATEGY_REGISTRY** through the invention pipeline. The funnel leaks at three points.

### The Three Leaks

**Leak 1: Walk-forward validation kills everything (8/8 refined winners rejected)**
Walk-forward splits data 75/25 (train/test). The most recent 25% is always a trending-up regime while the training period is mean-reversion. This is a data artifact, not a real regime shift — but the validation sees `regime_shift=true` and degradation >70%, then fails the strategy. The `auto_promote()` path (which registers in STRATEGY_REGISTRY) requires `validation.passed == True`, so this gate is a wall.

**Leak 2: Code generation fails on 54% of runs**
Not syntax errors — runtime errors. Top causes:
- Zero entry signals (18%) — conditions too restrictive
- Wrong indicator API signatures (10%) — `atr(close)` instead of `atr(high, low, close)`
- Pandas type confusion (13%) — returns DataFrame instead of `pd.Series[bool]`
The circuit breaker (20% pass rate, grace=2 turns) fires too early, killing mandates that could have recovered.

**Leak 3: winners.json is 86% noise**
50 of 58 entries are parameter sweeps of 3 pre-existing strategies (`roc_ema_volume:36, informed_simple_adaptive:9, rsi_crossover:5`). No way to distinguish LLM-invented from sweep-found. The 8 actual refined entries all got `backtest_only` status. Makes it impossible to measure research progress.

### Deliverables

#### 5.1 Fix Walk-Forward Validation
- **Problem:** Fixed 75/25 split creates systematic regime shift artifact
- **Fix:** Use rolling/expanding walk-forward windows instead of single split. Or adjust degradation threshold to account for expected regime differences. Or add a "soft pass" tier that promotes to registry with a `needs_ongoing_validation` flag instead of full validation
- **Success:** At least 1 refined strategy reaches `validation.passed == True` and gets registered in STRATEGY_REGISTRY
- **Files:** `crabquant/refinement/full_validation_promotion.py`, `crabquant/refinement/config.py` (WalkForwardConfig)

#### 5.2 Fix Code Generation Reliability
- **Problem:** 54% of runs hit code gen failures, mostly from wrong indicator usage and overly restrictive conditions
- **Fix:**
  - Add explicit `pd.Series[bool]` return type examples to system prompt
  - Add "zero signals" recovery hint to retry feedback ("try relaxing your entry conditions")
  - Tune circuit breaker: increase grace_turns from 2→3, lower min_pass_rate from 20%→15%
  - Trim prompt bloat — remove duplicate indicator reference (~2000 wasted tokens)
- **Success:** Code gen failure rate drops below 30%
- **Files:** `crabquant/refinement/prompts.py`, `crabquant/refinement/context_builder.py`, `crabquant/refinement/circuit_breaker.py`

#### 5.3 Clean Winners Pipeline
- **Problem:** winners.json mixes sweep results with invention results, no distinction
- **Fix:**
  - Add `source` field: `"sweep"` vs `"invention"` vs `"refinement"`
  - Separate `invention_winners.json` from sweep results
  - Fix `measure_convergence.py` bug: `promoted_codes` param never passed, promotion rate always 0%
  - Track: unique strategies invented, convergence rate over time, per-archetype success rates
- **Success:** Can clearly see how many strategies the LLM invented vs how many are sweeps
- **Files:** `results/winners/winners.json`, `crabquant/refinement/promotion.py`, `scripts/measure_convergence.py`

#### 5.4 Prompt Optimization
- **Problem:** ~6000-10000 tokens of context per call, with double inclusion of indicator reference
- **Fix:**
  - Remove duplicate indicator_quick_ref from Turn 1 prompt (already in system prompt via INDICATOR_API.md injection)
  - Trim strategy examples from 2 full source files to 1 + key snippets
  - Add explicit return type contract: "entries and exits MUST be pd.Series[bool]"
- **Success:** Prompt tokens reduced by ~30%, no quality loss
- **Files:** `crabquant/refinement/prompts.py`, `crabquant/refinement/context_builder.py`

### Phase 5 Success Criteria
- [x] Rolling walk-forward replaces single-split (fixes Leak 1)
- [ ] At least 3 refined strategies registered in STRATEGY_REGISTRY
- [ ] Code gen failure rate < 30%
- [x] winners.json clearly separates invention from sweep (source field added)
- [ ] Convergence rate > 20% (from current ~10%)
- [x] measure_convergence.py reports accurate promotion rate

---

## Phase 5.5: Regime-Aware Registry 🔴 CURRENT

**Why this phase exists:** CrabQuant's strategy catalog has 23 strategies but zero regime metadata. QuantFactory (the inspiration repo) baked regime detection into every layer — strategies knew which market conditions they thrived in, the portfolio selector picked strategies based on the current regime, and walk-forward analysis explicitly tested regime robustness. Without this, the system treats all strategies as regime-agnostic, which they aren't. This phase brings regime awareness into the registry and promotion pipeline.

### What QuantFactory Did Right (That We're Adopting)

1. **Regime tags on every strategy** — each strategy had `preferred_regimes`, `acceptable_regimes`, `weak_regimes` computed from per-regime Sharpe analysis
2. **Rolling walk-forward** — multiple windows instead of a single train/test split, catching strategies that only work in specific regimes
3. **Regime-aware promotion** — strategies flagged as regime-specific get different validation thresholds (they can be great in their regime without being universal)
4. **Dict-based registry entries** — extensible metadata vs rigid tuples

### What We Left Behind

1. **Full mode system (day trading / swing trading)** — too complex, added modes we'd never use. CrabQuant's mandate system is simpler and more flexible
2. **Web UI** — premature. CLI + Telegram first, UI when we have something worth dashboarding
3. **Strategy factory patterns** — QuantFactory's abstract base classes were over-engineered for 23 strategies. Flat function-based strategies stay
4. **Full portfolio optimizer** — too early. We need more winners before optimizing allocation between them

### Deliverables

#### 5.5.1 Regime Tagger Module ✅ Done
- **New file:** `crabquant/refinement/regime_tagger.py`
- `compute_strategy_regime_tags(strategy_fn, params, tickers)` → `{preferred_regimes, acceptable_regimes, weak_regimes, regime_sharpes, is_regime_specific}`
- `get_regime_strategies(regime, registry, min_sharpe)` → filter registry by regime compatibility
- Graceful fallback: returns empty tags on data errors

#### 5.5.2 Rolling Walk-Forward in Promotion ✅ Done
- **Modified:** `crabquant/refinement/promotion.py`
- `run_full_validation_check` now calls `rolling_walk_forward` (4 windows, 60/40 split) instead of single-split `walk_forward_test`
- Return dict includes `rolling_windows`, `avg_test_sharpe`, `worst_window_sharpe`, `regime_shift_windows`
- Detects regime-specific strategies: if a strategy passes in its preferred regime but fails in others, still promotes with a warning

#### 5.5.3 Registry Format Migration ✅ Done
- **New file:** `crabquant/strategies/_registry_compat.py` — backward-compat shim
- `register_strategy()` now stores dict entries with regime tags
- `_registry_compat` helpers (`get_fn`, `get_defaults`, `get_regime_tags`, etc.) handle both tuple and dict formats
- Existing 23 tuple entries continue to work; newly promoted entries are dicts
- **Modified:** `crabquant/refinement/context_builder.py` — updated to use compat helpers

#### 5.5.4 VALIDATION_CONFIG Update ✅ Done
- **Modified:** `crabquant/refinement/config.py`
- Rolling WF defaults: `n_windows=4`, `train_pct=0.6`, `min_test_sharpe=0.5`
- Soft promotion threshold: `soft_promote_test_sharpe=0.3` (registers with `needs_ongoing_validation` flag)
- Per-regime thresholds: regime-specific strategies get lower cross-ticker bar

#### 5.5.5 Test Coverage ✅ Done
- **New:** `tests/refinement/test_regime_tagger.py` (5 tests)
- **New:** `tests/refinement/test_registry_compat.py` (13 tests)
- **Updated:** `tests/refinement/test_promotion.py` (rolling WF mocks, regime tag mocks)
- **Updated:** `tests/refinement/test_e2e_phase3.py` (rolling WF mocks)
- **Total:** 932 tests pass, 0 regressions

### Remaining Work (Phase 5.5)

#### 5.5.6 Regime-Aware Scanner (Planned)
- Modify `production/scanner.py` to filter strategies by detected current regime
- Only run strategies whose `preferred_regimes` include the current market regime
- Fall back to regime-agnostic strategies when regime is unclear

#### 5.5.7 Portfolio Regime Router (Planned)
- New module: `production/regime_router.py`
- Reads current regime from `regime.py` detection
- Selects top-N strategies from STRATEGY_REGISTRY that match current regime
- Weight allocation by regime Sharpe scores

### Phase 5.5 Success Criteria
- [x] Regime tags computed on promotion
- [x] Registry entries carry regime metadata
- [x] Rolling walk-forward replaces single split
- [x] Backward compatibility maintained (tuple + dict)
- [ ] Scanner filters by regime
- [ ] Portfolio router selects by current regime
- [ ] At least 1 strategy promoted with regime tags (requires daemon run)

---

## Phase 5.6: Invention Accelerators 🔴 CURRENT

**Why this phase exists:** Live runs revealed the core bottleneck — the LLM can hit Sharpe 1.0-1.7 in-sample but collapses to ~0 out-of-sample. The refinement loop wastes 2-3 turns on garbage strategies before finding something viable, and each successful strategy fails rolling walk-forward validation. Three acceleration features address this: cross-run learning (smarter first turns), parallel invention (explore more strategy space), and soft promotion (stop throwing away good-enough strategies).

### Architecture: Modes + Toggles

Users pick a **mode preset** or configure individual **toggles** in mandates/config:

**Mode Presets:**
| Mode | cross_run | parallel | soft_promote | Description |
|------|-----------|----------|-------------|-------------|
| `conservative` | off | off | off | Current behavior — strict sequential, strict promotion |
| `explorer` | on | on | on | Maximum discovery — learn from history, spawn variants, keep candidates |
| `fast` | on | off | off | Quick wins — smarter prompts, sequential, strict promotion |
| `balanced` (default) | on | on | off | Smart + parallel but still strict on promotion |

**Individual Toggles (override mode):**
```json
{
  "mode": "balanced",
  "toggles": {
    "cross_run_learning": true,
    "parallel_invention": true,
    "parallel_count": 3,
    "soft_promote": false,
    "soft_promote_min_sharpe": 0.5,
    "soft_promote_min_windows": 2
  }
}
```

### Deliverables

#### 5.6.1 Cross-Run Learning ✅ Doing First
- **Problem:** Each mandate starts blind — the LLM invents from scratch, wasting 2-3 turns on basic patterns
- **Fix:** Feed top winners from `results/winners/winners.json` as example strategies into the LLM context
  - On turn 1: inject 2-3 winning strategies with similar archetype/ticker as code examples
  - On refinement turns: include "what worked for similar mandates" section
  - Rank winners by Sharpe × sqrt(trades) to favor robust strategies over curve-fits
  - Deduplicate by strategy_hash to avoid showing near-identical variants
- **Files:** `crabquant/refinement/context_builder.py`, `crabquant/refinement/prompts.py`
- **Success:** Turn 1 average Sharpe improves by 50%+ (from ~0.0 to ~0.5+)

#### 5.6.2 Parallel Strategy Spawning
- **Problem:** Sequential invention explores one path at a time; 2-3 turns wasted on dead ends
- **Fix:** On turn 1, spawn N strategies in parallel (default 3), backtest all, keep the best
  - Each parallel strategy gets a slightly different prompt variant (different indicator focus, different entry logic style)
  - Best strategy by composite score proceeds to refinement loop
  - Falls back to sequential if `parallel_count: 1`
- **Files:** `scripts/refinement_loop.py`, `crabquant/refinement/prompts.py` (prompt variants)
- **Success:** Discovery phase cut from 4-6 min to ~2 min, higher best-Sharpe on turn 1

#### 5.6.3 Soft-Promote Tier
- **Problem:** Binary pass/fail promotion throws away strategies that are good but not perfect
- **Fix:** Add "candidate" promotion tier for near-misses
  - Full promote: rolling WF passes, registered in STRATEGY_REGISTRY (current behavior)
  - Soft promote: avg_test_sharpe >= 0.5 AND 2+ windows passed → goes to `results/candidates/` with `needs_ongoing_validation` flag
  - Candidates can be promoted to full status after paper trading or additional validation windows
  - Regime-specific strategies get lower soft-promote threshold (avg_test_sharpe >= 0.3)
- **Files:** `crabquant/refinement/promotion.py`, `crabquant/refinement/config.py`, `results/candidates/`
- **Success:** 5+ candidate strategies accumulated from 50 mandates (vs 0 full promotes)

### Phase 5.6 Success Criteria
- [ ] Cross-run learning: turn 1 average Sharpe > 0.5 (from ~0.0 baseline)
- [ ] Parallel invention: discovery time < 3 min for 3 parallel strategies
- [ ] Soft promote: candidate pool has 5+ entries from 50 mandates
- [ ] Mode system: all 4 presets work, individual toggles override correctly
- [ ] Tests: all 960+ pass, new test coverage for all 3 features

---

## Phase 6: Production Validation (Planned)

**Prerequisite:** Phase 5 complete — the funnel actually works.

Run the daemon for real and validate the system works at scale.

### Deliverables
- Run daemon continuously for 7+ days
- API budget tracking (now matters — we're burning API calls)
- Resource-aware parallelism (CPU/RAM throttling)
- Telegram status reporting (daily briefs)
- Real convergence baseline from 200+ mandates

### Success Criteria
- 7+ days unattended runtime
- Convergence rate > 20% sustained
- 10+ strategies in STRATEGY_REGISTRY from invention
- No API rate limit failures
- Telegram daily briefs arriving reliably

### Notes
This absorbs the useful parts of old Phase 5B (API budget, resource limiter, Telegram) but only AFTER the research quality is proven. No point optimizing a factory that doesn't produce.

---

## Phase 7: Intelligence Layer (Planned)

**Prerequisite:** Phase 6 complete — we have real production data to learn from.

The system learns from its own experience.

### Deliverables
- **Adaptive prompts:** Analyze which LLM actions lead to breakthroughs per archetype/ticker, inject winning patterns into prompts
- **Strategy decay detection:** Monitor promoted strategies over time, flag when performance degrades
- **Portfolio correlation gate:** When promoting, check correlation with existing winners — avoid promoting strategies that are just variants of existing ones
- **Mandate prioritization:** Focus research on underexplored archetype×ticker combinations
- **Failure pattern analysis:** If a failure mode spikes, auto-adjust validation thresholds or prompt constraints

### Notes
This absorbs the useful parts of old Phase 6. The intelligence layer needs real production data to be meaningful — that's why it comes after Phase 6, not before.

---

## Phase 8: Deployment Readiness (Planned)

**Prerequisite:** Phase 7 complete — we have a robust, self-improving research engine.

Bridge from research to deployment.

### Deliverables
- **Slippage integration:** Wire `confirm/` module into refinement loop (code exists, not connected)
- **Paper trading:** Forward-test promoted strategies with live data
- **Multi-timeframe support:** Allow strategies to use daily + intraday data
- **Walk-forward in refinement loop:** Real-time regime detection during strategy invention
- **Telegram dashboard:** On-demand status, strategy cards, portfolio overview

### Notes
This absorbs the useful parts of old Phase 7. Slippage is the most important item — the rest is nice-to-have.

---

## Phase 9: Live Trading (Far Future)

- Broker integration (Interactive Brokers / Alpaca)
- Risk management layer
- Position sizing (Kelly criterion / risk parity)
- Automated deployment pipeline

---

## Old PRDs (Reference Only)

These documents have been superseded or absorbed into the restructured phases. Kept for reference.

| PRD | Status | What happened |
|-----|--------|---------------|
| `PHASE45_PRD.md` | ✅ Complete | All deliverables implemented. Indicator API injection, walk-forward config, diversity scoring, promotion tracking. |
| `PHASE5B_PRD.md` | 📦 Absorbed | API budget, resource limiter, Telegram alerts → moved to Phase 6 (after funnel is fixed) |
| `PHASE6_PRD.md` | 📦 Absorbed | Intelligence layer → moved to Phase 7 (needs production data first) |
| `PHASE7_PRD.md` | 📦 Absorbed | Deployment readiness → moved to Phase 8 (slippage most important item) |

---

## Design Principles (Non-Negotiable)

1. **Direct Python execution** — no LangGraph, no agent framework overhead
2. **JSON on disk is state** — no databases, no message queues, just files
3. **Deterministic before intelligent** — classify failures by code, then ask LLM
4. **Subprocess isolation for parallelism** — shared mutable state is the enemy
5. **Validation before promotion** — walk-forward AND cross-ticker, no exceptions
6. **Composite scoring penalizes overfit** — sharpe * sqrt(trades/20) * (1 - abs(max_dd))
7. **Transparent and reproducible** — every decision logged, every strategy traceable
8. **Budget-aware** — minimize LLM calls, use cheaper models for mechanical tasks
9. **Free/open-source only** — no VectorBT Pro, no paid APIs beyond z.ai
10. **Real data, realistic conditions** — no shortcuts that sacrifice accuracy for speed
