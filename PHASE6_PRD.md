# Phase 6 PRD — Intelligence Layer

**Goal:** Make the system learn from its own experience — not just run mandates, but get smarter about which mandates to run, what prompts work, when strategies are decaying, and how to build a diversified portfolio.

**Scope:** Action analytics feedback loop, adaptive invention prompts, strategy decay detection, portfolio correlation in promotion, mandate prioritization, failure pattern analysis.
**Out of scope:** Slippage integration, paper trading, multi-timeframe, live dashboard (Phase 7).

**Dependencies:** Phase 5B complete (48h+ of daemon run data, API budget tracking, status reporting).

**Prerequisites:** At least 100 mandates completed so action analytics has enough data to be meaningful.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Intelligence Layer                         │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────┐                │
│  │  Action         │───►│  Adaptive        │                │
│  │  Analytics      │    │  Prompts         │                │
│  │  (feedback)     │    │  (invention)     │                │
│  └────────┬────────┘    └──────────────────┘                │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────┐    ┌──────────────────┐                │
│  │  Failure        │───►│  Mandate         │                │
│  │  Pattern        │    │  Prioritizer     │                │
│  │  Analysis       │    │  (scoring)       │                │
│  └─────────────────┘    └────────┬─────────┘                │
│                                  │                           │
│  ┌─────────────────┐             │                           │
│  │  Portfolio      │◄────────────┘                           │
│  │  Correlation    │                                         │
│  │  + Scoring      │                                         │
│  └────────┬────────┘                                         │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────┐                                         │
│  │  Strategy       │                                         │
│  │  Decay          │                                         │
│  │  Detection      │                                         │
│  └─────────────────┘                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component 1: Action Analytics Feedback Loop

**File:** `~/development/CrabQuant/crabquant/refinement/action_analytics.py` (enhance existing)

### 2.1 Current State

`action_analytics.py` already exists with:
- `load_run_history()` — reads `results/run_history.jsonl`
- `track_action_result()` — records action outcomes
- `get_action_stats()` — computes success rates per action type

### 2.2 Enhancement Requirements

Wire action analytics into the LLM context so the system learns from its own experience:

1. **Context injection**: When `context_builder.py` builds the Turn 2+ context, include a section: "Actions that historically improved Sharpe for this failure mode" and "Actions that historically failed for this failure mode"
2. **Failure-mode-specific stats**: Not just overall action success rates, but success rates broken down by failure mode (e.g., `replace_indicator` works 40% of the time for `too_few_trades` but only 10% for `excessive_drawdown`)
3. **Recent bias**: Weight recent results (last 50 runs) more heavily than historical data to adapt to changing market conditions
4. **Persistence**: Ensure run history is written to JSONL after every mandate (not just on daemon shutdown)

### 2.3 New/Enhanced Interface

```python
def get_failure_mode_action_stats(
    history: list[dict],
    failure_mode: str,
    recent_window: int = 50,
) -> dict[str, dict]:
    """Get action success rates broken down by failure mode.
    
    Returns: {
        "replace_indicator": {"success": 0.45, "count": 20, "avg_delta": 0.3},
        "add_filter": {"success": 0.20, "count": 15, "avg_delta": 0.1},
        ...
    }
    Sorted by success rate descending.
    """

def format_action_feedback_for_context(
    failure_mode: str,
    action_stats: dict[str, dict],
    max_actions: int = 5,
) -> str:
    """Format action analytics as a context block for the LLM.
    
    Output example:
    "Based on 50 recent runs with 'too_few_trades':
     - ✅ replace_indicator (45% success, avg +0.3 Sharpe)
     - ✅ change_entry_logic (35% success, avg +0.2 Sharpe)
     - ❌ modify_params (10% success, avg +0.05 Sharpe)
     Consider trying: replace_indicator or change_entry_logic"
    """
```

### 2.4 Integration Points

- `crabquant/refinement/context_builder.py` — call `get_failure_mode_action_stats()` and `format_action_feedback_for_context()` when building Turn 2+ context. Append to the LLM prompt.
- `scripts/refinement_loop.py` — ensure `track_action_result()` is called after every refinement turn with the action taken and outcome.

### 2.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_action_analytics_enhanced.py` (8+ tests)

- `test_failure_mode_specific_stats`
- `test_recent_window_bias`
- `test_empty_history_graceful`
- `test_format_context_block`
- `test_max_actions_limit`
- `test_unknown_failure_mode`
- `test_persistence_after_each_turn`
- `test_avg_delta_computation`

### 2.6 Effort Estimate
**S (1 session)** — existing module, add filtering + formatting.

---

## 3. Component 2: Adaptive Invention Prompts

**File:** `~/development/CrabQuant/crabquant/refinement/prompts.py` (enhance existing)

### 3.1 Current State

`prompts.py` has static Turn 1 (invention) and Turn 2+ (refinement) templates.

### 3.2 Enhancement Requirements

Make Turn 1 prompts dynamic based on:
1. **Action analytics**: Emphasize indicator families that have high convergence rates for the current market regime
2. **Regime awareness**: If current regime is HIGH_VOLATILITY, suggest volatility-based indicators (ATR, BB, VIX). If LOW_VOLATILITY, suggest momentum/trend indicators.
3. **Portfolio gaps**: If no breakout strategies promoted, add a nudge toward breakout-style entries
4. **Control group**: 20% of mandates use base prompts without adaptation (to measure if adaptation actually helps)

### 3.3 New Interface

```python
def build_adaptive_invention_prompt(
    base_prompt: str,
    regime: str,
    portfolio_gaps: dict[str, float],
    action_stats: dict[str, dict] | None = None,
    adaptation_rate: float = 0.80,
) -> str:
    """Build an invention prompt adapted to current conditions.
    
    With probability (1 - adaptation_rate), returns base_prompt unchanged (control group).
    Otherwise, appends regime-specific hints, portfolio gap nudges, and 
    historically successful indicator suggestions.
    
    Args:
        base_prompt: The static invention prompt template.
        regime: Current market regime string.
        portfolio_gaps: Archetype coverage scores from mandate_generator.
        action_stats: Action analytics from action_analytics.
        adaptation_rate: Fraction of mandates that get adaptive prompts (default 0.80).
    
    Returns:
        Adapted prompt string (or base prompt for control group).
    """
```

### 3.4 Adaptive Additions (capped at 500 tokens)

```
[Regime Context]
Current market regime: HIGH_VOLATILITY
Consider volatility-based indicators: ATR, Bollinger Bands, VIX

[Portfolio Gaps]
Underrepresented archetypes: breakout (0 promoted), mean_reversion (1 promoted)
Consider: channel breakout or range compression strategies

[Historical Success]
Indicators with high convergence in current regime:
- ATR + RSI combination (42% convergence rate)
- Bollinger Squeeze (38% convergence rate)
```

### 3.5 Integration Points

- `scripts/refinement_loop.py` — for Turn 1, call `build_adaptive_invention_prompt()` instead of using the static template directly.
- `crabquant/refinement/context_builder.py` — provide regime and portfolio gap data.

### 3.6 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_adaptive_prompts.py` (8+ tests)

- `test_adaptive_prompt_includes_regime`
- `test_adaptive_prompt_includes_portfolio_gaps`
- `test_control_group_returns_base_prompt`
- `test_adaptation_rate_respected`
- `test_token_limit_not_exceeded`
- `test_empty_stats_graceful`
- `test_high_volatility_suggests_volatility_indicators`
- `test_low_volatility_suggests_trend_indicators`

### 3.7 Effort Estimate
**S (1 session)** — prompt string manipulation + regime mapping.

---

## 4. Component 3: Strategy Decay Detection

**File:** `~/development/CrabQuant/crabquant/production/scanner.py` (enhance existing)

### 4.1 Current State

`scanner.py` exists in production/ but currently does basic scanning of promoted strategies.

### 4.2 Enhancement Requirements

Add periodic re-validation of promoted strategies:

1. **Daily backtest on recent data**: For each promoted strategy, run a backtest on the most recent 6 months of data
2. **Compare to promotion Sharpe**: If current Sharpe drops >30% from the value at promotion time, flag for retirement
3. **Consecutive check requirement**: Require 3 consecutive below-threshold checks before retiring (to avoid false positives from temporary drawdowns)
4. **Regime-aware**: Only compare against same-regime performance. If the regime changed, the strategy might still be fine — just not applicable to current conditions
5. **Retirement**: Move strategy to "retired" status in the registry (don't delete — keep for analysis)

### 4.3 New Interface

```python
@dataclass
class DecayCheckResult:
    strategy_name: str
    promotion_sharpe: float
    current_sharpe: float
    sharpe_decline_pct: float
    current_regime: str
    is_decayed: bool
    consecutive_decayed_checks: int
    should_retire: bool

def check_strategy_decay(
    strategy_name: str,
    promotion_sharpe: float,
    decay_threshold: float = 0.30,
    consecutive_required: int = 3,
) -> DecayCheckResult:
    """Run a backtest on recent data and compare to promotion-time Sharpe.
    
    Returns DecayCheckResult with is_decayed and should_retire flags.
    """

def check_all_strategies_decay(
    strategies_dir: Path | str,
    decay_state_file: str = "results/decay_state.json",
) -> list[DecayCheckResult]:
    """Check all promoted strategies for decay.
    
    Reads promotion metadata (original Sharpe) from strategy registry.
    Updates consecutive check count in decay_state.json.
    Returns list of results, sorted by severity.
    """
```

### 4.4 Integration Points

- Supervisor cron — run `check_all_strategies_decay()` once per day as part of health check.
- Status reporter — include decay check results in daily report.
- `crabquant/production/promoter.py` — add retirement function that moves strategy to inactive status.

### 4.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_strategy_decay.py` (8+ tests)

- `test_no_decay_good_performance`
- `test_decay_detected_sharpe_drop`
- `test_consecutive_required_before_retire`
- `test_regime_change_not_flagged`
- `test_decay_state_persistence`
- `test_all_strategies_check`
- `test_empty_registry_graceful`
- `test_retirement_marks_inactive`

### 4.6 Effort Estimate
**S-M (1 session)** — re-use existing backtest engine, add comparison logic.

---

## 5. Component 4: Portfolio Correlation in Promotion

**File:** `~/development/CrabQuant/crabquant/refinement/portfolio_correlation.py` (enhance existing)

### 5.1 Current State

`portfolio_correlation.py` exists with:
- `load_winners_equity_curves()` — loads equity data (currently returns empty since equity data not embedded)
- `compute_correlation_matrix()` — computes pairwise correlation
- `find_diversifying_pairs()` — identifies low-correlation pairs

### 5.2 Enhancement Requirements

Wire portfolio correlation into the promotion pipeline:

1. **Generate equity curves**: During backtest, save equity curve data (portfolio value over time) to the run directory
2. **Correlation check at promotion**: Before promoting a strategy, compute its equity curve correlation against all already-promoted strategies. Reject if correlation > 0.8 to any existing strategy (unless Sharpe is significantly higher — >50% above the correlated strategy's Sharpe)
3. **Rolling window**: Use 90-day rolling correlation, not full-history, to avoid stale correlation data
4. **Portfolio diversification score**: Compute an overall portfolio diversification score based on average pairwise correlation. Include in status reports.

### 5.3 New/Enhanced Interface

```python
def compute_strategy_correlation(
    new_equity_curve: pd.Series,
    existing_equity_curves: dict[str, pd.Series],
    window: int = 90,
) -> dict[str, float]:
    """Compute rolling correlation between new strategy and all existing.
    
    Returns: {"strategy_a": 0.85, "strategy_b": 0.32, ...}
    """

def should_promote_with_correlation(
    new_sharpe: float,
    new_equity_curve: pd.Series,
    existing_strategies: dict[str, dict],  # name -> {sharpe, equity_curve}
    max_correlation: float = 0.80,
    sharpe_improvement_required: float = 0.50,
) -> tuple[bool, str]:
    """Decide if a strategy should be promoted considering portfolio correlation.
    
    Returns: (should_promote, reason_string)
    
    Rejects if:
    - Correlation > max_correlation to any existing strategy AND
    - New Sharpe is NOT > sharpe_improvement_required * existing_sharpe
    """

def compute_portfolio_diversification_score(
    equity_curves: dict[str, pd.Series],
) -> float:
    """Compute 0-1 score where 1 = perfectly diversified (all uncorrelated).
    
    Based on average pairwise correlation: score = 1 - mean(abs(corr_matrix - I))
    """
```

### 5.4 Integration Points

- `crabquant/refinement/promotion.py` — call `should_promote_with_correlation()` before registering in STRATEGY_REGISTRY.
- `crabquant/refinement/diagnostics.py` — save equity curve data during backtest.
- Status reporter — include portfolio diversification score.

### 5.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_portfolio_correlation_enhanced.py` (8+ tests)

- `test_low_correlation_allows_promotion`
- `test_high_correlation_blocks_promotion`
- `test_high_sharpe_overrides_correlation`
- `test_rolling_window_computation`
- `test_empty_portfolio_allows_promotion`
- `test_diversification_score_perfect`
- `test_diversification_score_identical`
- `test_equity_curve_saving`

### 5.6 Effort Estimate
**S-M (1 session)** — existing math, add promotion gate.

---

## 5. Component 5: Mandate Prioritization

**File:** `~/development/CrabQuant/crabquant/refinement/mandate_generator.py` (enhance)

### 5.1 Requirements

Score and prioritize potential mandates by expected value. Instead of running mandates in random order or FIFO, prioritize based on:

1. **Convergence probability**: Historical convergence rate for this archetype×ticker combination
2. **Expected Sharpe**: Average Sharpe achieved by similar mandates
3. **Portfolio gap fill**: Bonus for archetypes with few promoted strategies
4. **Regime match**: Bonus for strategies suited to current market regime
5. **API budget**: Lower-priority mandates get skipped when budget is tight

### 5.2 New Interface

```python
def score_mandate(
    mandate: dict,
    convergence_history: dict[str, float],  # archetype -> convergence_rate
    portfolio_gaps: dict[str, float],        # archetype -> coverage_score
    current_regime: str,
    regime_affinity: dict[str, dict[str, float]],  # archetype -> regime -> affinity
) -> float:
    """Score a mandate on 0-1 scale.
    
    Formula:
    score = 0.3 * convergence_prob + 0.3 * (1 - portfolio_coverage) 
          + 0.2 * regime_match + 0.2 * expected_sharpe_normalized
    """

def prioritize_mandates(
    mandates: list[dict],
    convergence_history: dict[str, float],
    portfolio_gaps: dict[str, float],
    current_regime: str,
    regime_affinity: dict[str, dict[str, float]],
    api_budget_remaining: float = 1.0,
    top_n: int | None = None,
) -> list[dict]:
    """Sort mandates by score. If budget is tight, only return top-N.
    
    Returns sorted list of (mandate, score) tuples.
    """
```

### 5.3 Integration Points

- `scripts/run_pipeline.py` — call `prioritize_mandates()` when picking next mandates for a wave.
- Status reporter — show top-priority mandates and their scores.

### 5.4 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_mandate_prioritization.py` (8+ tests)

- `test_score_high_convergence`
- `test_score_portfolio_gap_bonus`
- `test_score_regime_match`
- `test_budget_truncation`
- `test_sort_order`
- `test_empty_history_defaults`
- `test_tiebreaking`
- `test_top_n_limit`

### 5.5 Effort Estimate
**S (1 session)** — scoring formula + sorting.

---

## 6. Component 6: Failure Pattern Analysis

**File:** `~/development/CrabQuant/crabquant/refinement/classifier.py` (enhance existing)

### 6.1 Current State

`classifier.py` classifies failures into 6 modes: `too_few_trades`, `flat_signal`, `excessive_drawdown`, `regime_fragility`, `overtrading`, `low_sharpe`.

### 6.2 Enhancement Requirements

1. **Aggregate statistics**: Track failure mode distribution across all runs. If one mode dominates (>60% of failures), flag it.
2. **Auto-adjustment**: When a failure mode is dominant, automatically adjust validation thresholds or add constraints to invention prompts. For example, if `too_few_trades` is 60% of failures, lower the minimum trade count threshold or add entry frequency hints to prompts.
3. **New failure mode**: Add `correlation_reject` for strategies rejected due to portfolio correlation (from Component 4).
4. **Report**: Include failure distribution in status reports.

### 6.3 New Interface

```python
def analyze_failure_patterns(
    history: list[dict],
    window: int = 100,
) -> dict[str, Any]:
    """Analyze failure mode distribution and detect patterns.
    
    Returns: {
        "distribution": {"too_few_trades": 0.35, "low_sharpe": 0.25, ...},
        "dominant_mode": "too_few_trades",
        "dominant_pct": 0.35,
        "recommendations": [
            "too_few_trades is 35% of failures. Consider lowering min_trades threshold."
        ]
    }
    """

def get_auto_adjustments(
    pattern_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Generate automatic adjustments based on failure patterns.
    
    Returns: {
        "min_trades_adjustment": -5,  # Lower min trades by 5
        "prompt_hints": ["Try more aggressive entry signals", ...],
        "sharpe_target_adjustment": -0.1,
    }
    """
```

### 6.4 Integration Points

- Status reporter — include failure distribution in daily report.
- Adaptive prompts — use auto-adjustments when building invention prompts.
- `crabquant/refinement/prompts.py` — incorporate prompt hints from failure analysis.

### 6.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_failure_patterns.py` (8+ tests)

- `test_distribution_computation`
- `test_dominant_mode_detection`
- `test_recommendations_generated`
- `test_auto_adjustment_trades`
- `test_auto_adjustment_sharpe`
- `test_no_adjustment_balanced_failures`
- `test_window_filtering`
- `test_empty_history`

### 6.6 Effort Estimate
**S (1 session)** — aggregation + threshold logic.

---

## 7. Dependencies Between Components

```
Action Analytics ──────────────► Adaptive Prompts (uses action stats)
Action Analytics ──────────────► Failure Patterns (uses action + failure data)
Failure Patterns ──────────────► Adaptive Prompts (uses auto-adjustments)
Portfolio Correlation ────────► Promotion Gate (blocks correlated strategies)
Mandate Prioritization ◄────── Portfolio Correlation (uses gaps + regime)
Strategy Decay ◄────────────── Promotion (needs promoted strategies to check)
```

**Build order:**
1. **Action Analytics Enhancement** (no deps — foundation for feedback)
2. **Failure Pattern Analysis** (depends on analytics)
3. **Adaptive Prompts** (depends on analytics + failure patterns)
4. **Portfolio Correlation Enhancement** (no deps — parallel with 1-3)
5. **Mandate Prioritization** (depends on portfolio correlation for gap data)
6. **Strategy Decay Detection** (depends on promotion pipeline)

---

## 8. Success Criteria

- [ ] Action analytics data flows into LLM context (verify by inspecting generated context for Turn 2+)
- [ ] Adaptive prompts include regime context and portfolio gaps (verify for Turn 1)
- [ ] Control group (20% base prompts) still runs for comparison
- [ ] Convergence rate improves by ≥5 percentage points vs Phase 5B baseline (measured over 100 mandates)
- [ ] Strategy decay detector flags at least 1 decaying strategy (if any promoted strategies exist)
- [ ] Portfolio correlation blocks promotion of highly correlated strategies
- [ ] Mandate prioritization produces different orderings than random/FIFO
- [ ] Failure pattern analysis generates actionable recommendations
- [ ] All unit tests pass (target: 48+ new tests)
- [ ] PHASE_CHECKLIST.md completed

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Feedback loop causes prompt drift | Prompts become too specialized | 20% control group; periodic prompt reset to base template |
| Portfolio correlation overfits to historical | Bad generalization | Use 90-day rolling correlation; don't block all correlated strategies |
| Decay detection false positives | Premature strategy retirement | Require 3 consecutive below-threshold checks |
| Adaptive prompts too complex for LLM | LLM confused by long context | Cap adaptive additions at 500 tokens; keep base prompt clean |
| Convergence rate doesn't improve | Intelligence layer adds no value | Measure control group vs adaptive; if no improvement after 100 mandates, simplify |

---

## 10. Effort Summary

| Component | Effort | Tests | Dependencies |
|-----------|--------|-------|-------------|
| Action Analytics Enhancement | S (1 session) | 8+ | None |
| Failure Pattern Analysis | S (1 session) | 8+ | Analytics |
| Adaptive Prompts | S (1 session) | 8+ | Analytics + Failures |
| Portfolio Correlation Enhancement | S-M (1 session) | 8+ | None |
| Mandate Prioritization | S (1 session) | 8+ | Correlation |
| Strategy Decay Detection | S-M (1 session) | 8+ | Promotion |
| Integration + Wiring | M (1-2 sessions) | — | All components |
| **Total** | **L (5-7 sessions)** | **48+** | — |

---

## 11. File Structure

```
~/development/CrabQuant/
├── crabquant/refinement/
│   ├── action_analytics.py      # ENHANCED — failure-mode-specific stats, context formatting
│   ├── prompts.py               # ENHANCED — adaptive invention prompts
│   ├── classifier.py            # ENHANCED — failure pattern analysis, auto-adjustments
│   ├── portfolio_correlation.py # ENHANCED — equity curves, promotion gate, diversification score
│   ├── mandate_generator.py     # ENHANCED — mandate scoring + prioritization
│   ├── context_builder.py       # ENHANCED — inject action feedback into Turn 2+ context
│   ├── promotion.py             # ENHANCED — correlation check before promotion
│   └── diagnostics.py           # ENHANCED — save equity curves during backtest
├── crabquant/production/
│   └── scanner.py               # ENHANCED — strategy decay detection
├── results/
│   └── decay_state.json         # NEW — decay check persistence
└── tests/refinement/
    ├── test_action_analytics_enhanced.py   # NEW
    ├── test_adaptive_prompts.py            # NEW
    ├── test_strategy_decay.py              # NEW
    ├── test_portfolio_correlation_enhanced.py  # NEW
    ├── test_mandate_prioritization.py      # NEW
    └── test_failure_patterns.py            # NEW
```
