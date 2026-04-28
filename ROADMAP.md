# CrabQuant — Roadmap

**Last Updated:** 2026-04-28
**Status:** Phase 5 (Fix the Funnel) — CURRENT

---

## Phase History

| Phase | Name | Status |
|-------|------|--------|
| 0 | MVP | ✅ Complete |
| 1-3 | Refinement Pipeline | ✅ Built |
| 4 | Integration | ✅ Done |
| 4.5 | Convergence Tuning | ✅ Done |
| 5A | Daemon Core | ✅ Done |
| **5** | **Fix the Funnel** | **🔴 CURRENT** |
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
- [ ] At least 3 refined strategies registered in STRATEGY_REGISTRY
- [ ] Code gen failure rate < 30%
- [ ] winners.json clearly separates invention from sweep
- [ ] Convergence rate > 20% (from current ~10%)
- [ ] measure_convergence.py reports accurate promotion rate

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
