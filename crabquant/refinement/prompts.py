"""
Refined LLM Prompts — Phase 2 prompt templates based on Phase 1 observations.

Improvements over Phase 1:
- Stricter hypothesis enforcement with examples
- Explicit action type list and guidance
- Sharpe-by-year in every refinement prompt
- Previous attempts with params and deltas
- Stagnation suffix injection
- Tier 2 diagnostics when available
- Strategy examples for reference

Two prompt modes:
1. Turn 1 (invention): Full context, no backtest report
2. Turns 2+ (refinement): Full backtest report + current code + history
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crabquant.refinement.sharpe_diagnosis import diagnose_low_sharpe
from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
from crabquant.refinement.positive_feedback import (
    analyze_positive_feedback,
    format_positive_feedback_for_prompt,
)

# ── Action Types ─────────────────────────────────────────────────────────────

VALID_ACTIONS = [
    "replace_indicator",
    "add_filter",
    "modify_params",
    "change_entry_logic",
    "change_exit_logic",
    "add_regime_filter",
    "full_rewrite",
    "novel",
]

# ── Recommended Actions per Failure Mode ────────────────────────────────────
# Maps failure_mode → (primary_action, reason)
# For low_sharpe, a dynamic lookup is used based on diagnosis metrics.
RECOMMENDED_ACTIONS: dict[str, tuple[str, str]] = {
    "too_few_trades_for_validation": ("change_entry_logic", "widen thresholds to increase trade frequency"),
    "validation_failed": ("full_rewrite", "simpler strategy that generalizes better"),
    "regime_fragility": ("add_regime_filter", "detect and adapt to market regime changes"),
    "low_sharpe": ("replace_indicator", "diagnosis-dependent — see low_sharpe overrides"),
    "too_few_trades": ("change_entry_logic", "loosen entry conditions for more signals"),
    "excessive_drawdown": ("add_filter", "add risk management (stop loss / volatility filter)"),
    "flat_signal": ("change_entry_logic", "fix signal logic so trades actually fire"),
    "overtrading": ("add_filter", "add cooldown between trades to reduce frequency"),
    "backtest_crash": ("novel", "rewrite strategy to fix the runtime bug"),
    "module_load_failed": ("novel", "rewrite strategy to fix the import/syntax error"),
}

# ── System Prompt (shared) ──────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the CrabQuant Strategy Inventor. You generate and refine trading strategies \
as complete Python files.

OUTPUT FORMAT — you MUST produce a JSON object with these exact keys:
{
  "hypothesis": "<causal hypothesis for why this change will improve Sharpe>",
  "action": "<one of: replace_indicator, add_filter, modify_params, change_entry_logic, \
change_exit_logic, add_regime_filter, full_rewrite, novel>",
  "addresses_failure": "<the failure_mode from the backtest report, or 'new_strategy' for turn 1>",
  "reasoning": "<why this modification targets the diagnosed problem>",
  "expected_impact": "<minor, moderate, or major>",
  "new_strategy_code": "<complete Python file content>"
}

STRATEGY FILE REQUIREMENTS:
- Must define: generate_signals(df, params) -> (entries, exits)
- Must define: DEFAULT_PARAMS, DESCRIPTION
- Do NOT include generate_signals_matrix or PARAM_GRID
- Use pandas_ta for indicators (via crabquant.indicator_cache.cached_indicator)
- df columns: open, high, low, close, volume (lowercase)
- entries and exits must be pd.Series[bool], same length as df, no NaN
- Param grid values should be lists with 3-5 values each

CRITICAL RULES:
1. You MUST state a causal hypothesis. "Adjust parameters" is NOT a hypothesis.
   Good: "The strategy loses money during high-volatility regimes because the signal \
threshold is too tight, generating whipsaws. Adding a volatility filter should \
reduce false signals by 30-50%."
   Bad: "I'll try different parameters to improve Sharpe."
2. Your modification must address the diagnosed failure_mode.
3. Do NOT introduce indicators that aren't in pandas_ta.
4. Do NOT use future data (no lookahead bias).
5. Generate COMPLETE file content, not diffs or patches.
6. Look at the EXAMPLE STRATEGIES below to understand the exact API patterns. \
Your code MUST follow the same structure.
7. TRADE FREQUENCY: Aim for 20-80 trades over the backtest period. Too few trades \
(< 10) means your conditions are too restrictive — simplify. Too many trades (> 200) \
means your signal is noise — add filters. Start SIMPLE on turn 1.
8. ANTI-OVERFITTING: A strategy with 3 trades at 100% win rate is WORSE than useless — \
it is curve-fit garbage that will fail out-of-sample. Prefer 40+ trades at 50-65% win rate. \
NEVER stack 4+ conditions — each additional condition narrows your signal and increases \
overfitting risk. If your entry fires fewer than once per month on average, OPEN IT UP.
9. REGULAR SIGNALS: Your entry conditions should fire REGULARLY (at least 2x per month). \
Avoid strategies that only trigger in rare market conditions (e.g., \"RSI < 15 AND price > SMA200 \
AND volume > 3x average AND MACD histogram > 0\" — this might fire once per year). Simpler \
conditions that fire often are more robust out-of-sample.

TOP 3 INDICATOR MISTAKES — YOUR CODE WILL CRASH IF YOU DO THESE:
1. atr(close, length=14) → WRONG. ATR requires: atr(high, low, close, length=14)
2. stoch(close, k=14, d=3) → WRONG. Stochastic requires: stoch(high, low, close, k=14, d=3)
3. adx(close, length=14) → WRONG. ADX requires: adx(high, low, close, length=14)
Multi-output indicators (macd, bbands, stoch, adx) return DataFrames — \
use .iloc[:, N] for column access, NOT named strings.
{stagnation_suffix}

## Indicator API Reference
{indicator_reference}
"""

# ── Turn 1 Prompt (Pure Invention) ──────────────────────────────────────────

TURN1_PROMPT = """\
## Task: Invent a New Strategy

### Mandate
- Name: {mandate_name}
- Archetype: {strategy_archetype}
- Target Sharpe: {sharpe_target}
- Tickers: {tickers}
- Period: {period}
{seed_section}
{archetype_section}
### Strategy Catalog (all available strategies for inspiration)
{strategy_catalog}

### Example Strategies (follow this exact pattern)
{strategy_examples}

{winner_examples_section}
### Your Task
Invent a complete trading strategy matching the mandate.
Use the example strategies as a TEMPLATE for code structure, function signatures, \
and indicator usage patterns. Your code MUST follow the same conventions.

START SIMPLE. For turn 1, prefer a single clear signal (e.g., EMA crossover, \
RSI oversold, MACD histogram flip) with 1-2 basic filters. Do NOT combine \
5+ indicators on turn 1 — you can add complexity in later turns if needed. \
Simple strategies with 30-80 trades per year are far more likely to succeed \
than complex multi-condition systems with 5 trades per year.

### Anti-Overfitting Examples

**BAD (overfit — will fail out-of-sample):**
```python
# Too many conditions — rarely fires, curve-fit to handful of data points
entries = (
    (rsi < 20)
    & (close > sma_200)
    & (volume > 2 * volume_sma)
    & (macd_hist > 0)
    & (atr < atr_sma)  # 5 conditions! Maybe 3 trades per year
)
```

**GOOD (robust — fires regularly, likely to generalize):**
```python
# Simple, clear signal — fires often, catches a real pattern
entries = (rsi < 30) & (rsi > rsi.shift(1))  # RSI oversold + turning up
exits = (rsi > 70) | (hold_periods > 10)     # Overbought or time stop
```

**GOOD (momentum):**
```python
entries = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))  # Crossover
exits = (ema_fast < ema_slow) | (hold_periods > 20)
```

{trade_count_guidance}

### Indicator Quick Reference — USE THESE SIGNATURES EXACTLY
{indicator_quick_ref}

Output ONLY the JSON object with the required fields.
For turn 1, use:
  "addresses_failure": "new_strategy"
  "action": "novel" (or any action that fits)
  "hypothesis": "<why this strategy should achieve the target Sharpe>"
"""

# ── Refinement Prompt (Turns 2+) ────────────────────────────────────────────

REFINEMENT_PROMPT = """\
## Backtest Report (Turn {current_turn}/{max_turns})

### Current Strategy Code
Below is the FULL strategy code that produced these results. You are MODIFYING this code.
Read it carefully before proposing changes.

```python
{current_strategy_code}
```

Current params: {current_params}

### Metrics
- Sharpe: {sharpe_ratio:.2f} (target: {sharpe_target})
- Return: {total_return_pct:.1%}
- Max Drawdown: {max_drawdown_pct:.1%}
- Trades: {total_trades}
- Win Rate: {win_rate:.1%}
- Profit Factor: {profit_factor:.2f}
- Calmar Ratio: {calmar_ratio:.2f}
- Sortino Ratio: {sortino_ratio:.2f}

### Failure Diagnosis (deterministic — Python computed this, not a guess)
- Mode: {failure_mode}
- Details: {failure_details}

{failure_guidance}
{positive_feedback_section}
{action_effectiveness_section}
{failure_pattern_section}
### Sharpe by Year
{sharpe_by_year}
{tier2_section}
{feature_importance_section}
### Previous Attempts
{previous_attempts_section}
### Stagnation
- Score: {stagnation_score:.2f} (0=progress, 1=stuck)
- Trend: {stagnation_trend}
- Best so far: Sharpe {best_sharpe:.2f} at turn {best_turn}
{stagnation_suffix}
### Example Strategies (for reference — use if you need to change indicator patterns)
{strategy_examples}
{winner_examples_section}

## Your Task
{targeted_task_guidance}

{trade_count_guidance}

### Indicator Quick Reference — USE THESE SIGNATURES EXACTLY
{indicator_quick_ref}

Output ONLY the JSON object with the required fields.
"""


# ── Parallel Prompt Variants (Phase 5.6.2) ────────────────────────────────────

# Different focus areas for parallel prompt variants.
# Each variant biases the LLM toward a different indicator family / entry style.
PARALLEL_VARIANT_FOCI = [
    {
        "name": "momentum",
        "bias": (
            "## Parallel Variant Focus: MOMENTUM\n"
            "For this variant, prioritize MOMENTUM-based indicators: MACD, ROC, EMA crossovers, "
            "and rate-of-change signals. Entry logic should detect accelerating price trends.\n"
            "Prefer: macd, roc, ema, wma. Avoid: rsi, stoch, bbands."
        ),
    },
    {
        "name": "mean_reversion",
        "bias": (
            "## Parallel Variant Focus: MEAN REVERSION\n"
            "For this variant, prioritize MEAN-REVERSION indicators: RSI, Bollinger Bands, "
            "Stochastic, and CCI. Entry logic should detect overbought/oversold conditions.\n"
            "Prefer: rsi, bbands, stoch, cci. Avoid: macd, roc, ema crossovers."
        ),
    },
    {
        "name": "volatility_breakout",
        "bias": (
            "## Parallel Variant Focus: VOLATILITY / BREAKOUT\n"
            "For this variant, prioritize VOLATILITY and BREAKOUT indicators: ATR, ADX, "
            "Supertrend, and Bollinger Band squeezes. Entry logic should detect expanding volatility.\n"
            "Prefer: atr, adx, supertrend, bbands (squeeze). Avoid: rsi, roc."
        ),
    },
    {
        "name": "volume_confirmation",
        "bias": (
            "## Parallel Variant Focus: VOLUME CONFIRMATION\n"
            "For this variant, prioritize VOLUME-BASED signals: OBV, VWAP, volume SMAs, "
            "and volume breakouts. Entry logic should require volume confirmation.\n"
            "Prefer: obv, vwap, sma(volume). Use volume > vol_avg as a filter."
        ),
    },
    {
        "name": "multi_signal",
        "bias": (
            "## Parallel Variant Focus: MULTI-SIGNAL CONFLUENCE\n"
            "For this variant, combine 2-3 DIFFERENT indicator families for confluence. "
            "E.g., MACD + RSI, or EMA crossover + volume confirmation + ATR filter. "
            "Each signal component should be simple; the combination provides edge."
        ),
    },
]


def get_parallel_prompt_variants(base_prompt: str, count: int) -> list[str]:
    """Generate N prompt variants for parallel strategy invention.

    Each variant appends a different indicator focus bias to the base prompt,
    encouraging the LLM to explore different regions of strategy space.

    Args:
        base_prompt: The base Turn 1 prompt (before variant injection).
        count: Number of variants to generate (1-5). If count > available foci,
            foci cycle with a slight rewording.

    Returns:
        List of prompt strings, one per variant. If count == 1, returns
        [base_prompt] unchanged (sequential fallback).
    """
    if count <= 1:
        return [base_prompt]

    count = min(count, len(PARALLEL_VARIANT_FOCI))
    variants = []
    for i in range(count):
        focus = PARALLEL_VARIANT_FOCI[i]
        # Inject variant bias at the end of the prompt, before any closing instruction
        variant_prompt = base_prompt.rstrip() + "\n\n" + focus["bias"] + "\n"
        variants.append(variant_prompt)

    return variants


def get_variant_bias_text(variant_index: int, variant_count: int) -> str:
    """Return the bias instruction text for a specific parallel variant.

    This is a convenience wrapper for use inside call_llm_inventor,
    where we already know which variant we're generating.

    Args:
        variant_index: 0-based index of the variant.
        variant_count: Total number of variants being generated.

    Returns:
        Bias instruction string, or empty string if out of range.
    """
    if variant_count <= 1 or variant_index < 0:
        return ""
    safe_index = variant_index % len(PARALLEL_VARIANT_FOCI)
    return PARALLEL_VARIANT_FOCI[safe_index]["bias"]


def compute_composite_score(
    sharpe: float,
    trades: int,
    max_drawdown: float,
    sortino: float = 0.0,
    expected_value: float = 0.0,
) -> float:
    """Compute composite score for ranking parallel strategies.

    Formula: (sortino_weighted + ev_weighted) * robustness_factor

    Where:
        sortino_weighted = min(max(sortino, 0) / 3.0, 1.0)
        ev_weighted = sign(EV) * min(abs(EV) / 100, 1.0)
        robustness_factor = sqrt(min(trades, 100) / 20) * (1 - min(abs(max_dd), 1.0))

    This rewards downside-risk-adjusted returns and positive expected value,
    while penalizing overfit (few trades) and excessive drawdown.

    Args:
        sharpe: Sharpe ratio of the strategy (kept for backward compat, not used in new formula).
        trades: Total number of trades.
        max_drawdown: Maximum drawdown as a fraction (e.g., 0.15 for 15%).
        sortino: Sortino ratio (default 0.0).
        expected_value: Expected value per trade in dollar terms (default 0.0).

    Returns:
        Composite score (higher is better). Returns 0.0 on invalid inputs.
    """
    import numpy as np

    if trades < 1:
        return 0.0

    trade_factor = (min(trades, 100) / 20) ** 0.5
    dd_penalty = 1.0 - min(abs(max_drawdown), 1.0)
    robustness_factor = trade_factor * dd_penalty

    sortino_safe = max(sortino, 0.0)
    if np.isinf(sortino_safe) or np.isnan(sortino_safe):
        sortino_safe = 0.0
    sortino_weighted = min(sortino_safe / 3.0, 1.0)

    ev_weighted = float(np.sign(expected_value)) * min(abs(expected_value) / 100.0, 1.0)

    # Fallback: if no sortino or EV provided, use sharpe for backward compat
    if sortino_weighted == 0.0 and ev_weighted == 0.0:
        sharpe_safe = max(sharpe, 0.0)
        if np.isinf(sharpe_safe) or np.isnan(sharpe_safe):
            sharpe_safe = 0.0
        return sharpe_safe * robustness_factor

    return (sortino_weighted + ev_weighted) * robustness_factor


# ── Helper Functions ─────────────────────────────────────────────────────────


def _find_project_root(start: Path | None = None) -> Path:
    """Find the project root by walking up from *start* until VISION.md or pyproject.toml.

    Args:
        start: Directory to start searching from. Defaults to this file's directory.

    Returns:
        Path to the project root.

    Raises:
        FileNotFoundError: If no root marker is found.
    """
    current = start or Path(__file__).resolve().parent
    for _ in range(20):  # safety bound
        if (current / "VISION.md").is_file() or (current / "pyproject.toml").is_file():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        "Cannot find project root (no VISION.md or pyproject.toml found)"
    )


def load_indicator_reference() -> str:
    """Load the full indicator API reference from docs/INDICATOR_API.md.

    Returns:
        Full text of the indicator reference, or a fallback string if the
        file is missing.
    """
    try:
        root = _find_project_root()
        ref_path = root / "docs" / "INDICATOR_API.md"
        return ref_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "# Indicator API Reference\n"
            "(docs/INDICATOR_API.md not found — indicator reference unavailable)\n"
        )


def extract_quick_reference(full_reference: str) -> str:
    """Extract the Quick Reference Card (section 7) from the full reference.

    Falls back to the full reference if section 7 cannot be found.

    Args:
        full_reference: Full text of INDICATOR_API.md.

    Returns:
        Just the Quick Reference Card section as a string.
    """
    # Try to find section 7 by its heading
    marker = "## 7. Quick Reference Card"
    idx = full_reference.find(marker)
    if idx >= 0:
        return full_reference[idx:].strip()
    # Fallback: return the whole thing (better than nothing)
    return full_reference


def format_stagnation_suffix(
    constraint: str | None,
    prompt_suffix: str | None,
) -> str:
    """Format stagnation response for injection into system prompt.

    Args:
        constraint: Stagnation constraint string (normal, pivot, nuclear, abandon).
        prompt_suffix: Prompt suffix from get_stagnation_response().

    Returns:
        Formatted string for prompt injection, or empty string if normal.
    """
    if not constraint or constraint == "normal":
        return ""
    if not prompt_suffix:
        return ""
    return f"\n{prompt_suffix}"


def format_tier2_section(tier1_report: dict) -> str:
    """Format Tier 2 diagnostics for refinement prompt.

    Only includes sections that have data (non-None values).

    Args:
        tier1_report: Tier 1 report dict that may contain Tier 2 fields.

    Returns:
        Formatted string for prompt injection, or empty string if no Tier 2 data.
    """
    sections = []

    regime_sharpe = tier1_report.get("regime_sharpe")
    if regime_sharpe:
        parts = [f"  {regime}: {sharpe:.2f}" for regime, sharpe in regime_sharpe.items()]
        sections.append("### Regime Decomposition\n" + "\n".join(parts))

    top_drawdowns = tier1_report.get("top_drawdowns")
    if top_drawdowns:
        dd_lines = []
        for dd in top_drawdowns:
            depth = dd.get("depth_pct", 0)
            duration = dd.get("duration_bars", 0)
            dd_lines.append(f"  - Depth: {depth:.1%}, Duration: {duration} bars")
        sections.append("### Top Drawdowns\n" + "\n".join(dd_lines))

    benchmark = tier1_report.get("benchmark_return_pct")
    if benchmark is not None:
        sections.append(f"### Benchmark (Buy & Hold)\n  Return: {benchmark:.1%}")

    return "\n\n".join(sections) if sections else ""


def _format_window_breakdown(validation: dict) -> str:
    """Format per-window rolling walk-forward results as a compact table.

    Shows train/test Sharpe, degradation, and pass/fail for each window.
    This tells the LLM WHICH specific windows failed and by how much.

    Args:
        validation: Validation dict with 'window_results' list.

    Returns:
        Formatted string with per-window breakdown, or empty string if no data.
    """
    window_results = validation.get("window_results", [])
    if not window_results:
        return ""

    lines = ["  Rolling Walk-Forward Window Breakdown:"]
    lines.append("  Window | Train Sharpe | Test Sharpe | Degradation | Result")
    lines.append("  -------|-------------|-------------|-------------|-------")
    for w in window_results:
        w_num = w.get("window", "?")
        train_s = w.get("train_sharpe", 0)
        test_s = w.get("test_sharpe", 0)
        deg = w.get("degradation", 0)
        passed = w.get("passed", False)
        result = "✅ PASS" if passed else "❌ FAIL"
        if w.get("error"):
            result = f"💥 ERROR: {w['error'][:40]}"
        lines.append(
            f"  {w_num:>6} | {train_s:>11.2f} | {test_s:>11.2f} | {deg:>10.0%} | {result}"
        )

    # Add actionable summary based on failure pattern
    passed_count = sum(1 for w in window_results if w.get("passed", False))
    total = len(window_results)
    if passed_count == 0:
        lines.append("\n  ❌ ALL windows failed. Strategy is completely overfit — SIMPLIFY drastically or start over.")
    elif passed_count <= 1:
        lines.append(f"\n  ❌ Only {passed_count}/{total} windows passed. Strategy is heavily overfit to one period.")
    elif passed_count < total:
        lines.append(f"\n  ⚠️ {passed_count}/{total} windows passed. Close but not robust enough — reduce parameters.")
    else:
        lines.append(f"\n  ✅ {passed_count}/{total} windows passed.")

    return "\n".join(lines)


def format_previous_attempts_section(previous_attempts: list[dict]) -> str:
    """Format previous attempt history for the refinement prompt.

    For each failed turn, adds specific inline guidance based on the failure mode:
    - too_few_trades_for_validation: warns about restrictive conditions
    - validation_failed: shows per-window breakdown with train/test Sharpe
    - low_sharpe with < 10 trades: warns about curve-fitting risk
    - regime_fragility: explains regime dependency
    """
    if not previous_attempts:
        return "  (no previous attempts — this is your first refinement)"

    lines = []
    for entry in previous_attempts:
        turn = entry.get("turn", "?")
        sharpe = entry.get("sharpe", 0.0)
        failure = entry.get("failure_mode", "unknown")
        action = entry.get("action", "unknown")
        hypothesis = entry.get("hypothesis", "N/A")
        params = entry.get("params_used", {})
        delta = entry.get("delta_from_prev", "N/A")

        note = ""
        if failure == "too_few_trades_for_validation":
            note = (
                f"\n  ⚠️ NOTE: Only {entry.get('num_trades', '?')} trades — "
                "OPEN UP conditions, remove filters, widen thresholds"
            )
        elif failure == "validation_failed":
            validation = entry.get("validation", {})
            note = "\n  ⚠️ NOTE: Passed in-sample but FAILED out-of-sample validation — REDUCE complexity, simplify conditions"
            if validation:
                avg_ts = validation.get("avg_test_sharpe", "?")
                wp = validation.get("windows_passed", "?")
                nw = validation.get("num_windows", "?")
                note += f"\n  📊 Validation: avg test Sharpe={avg_ts}, {wp}/{nw} windows passed"
                # Add per-window breakdown
                window_detail = _format_window_breakdown(validation)
                if window_detail:
                    note += "\n" + window_detail
        elif failure == "low_sharpe":
            num_trades = entry.get("num_trades", 0)
            if num_trades < 10:
                note = (
                    f"\n  ⚠️ CURVE-FITTING RISK: Only {num_trades} trades with low Sharpe. "
                    "This strategy is likely fitting noise, not a real pattern. "
                    "Start with a SIMPLER, well-known signal (single indicator crossover) "
                    "that fires 30+ times per year."
                )
        elif failure == "regime_fragility":
            # Use sharpe_by_year if available in the history entry for a specific note
            sby_entry = entry.get("sharpe_by_year", {})
            if sby_entry and len(sby_entry) >= 2:
                neg_years = [y for y, s in sby_entry.items() if s < 0]
                if neg_years:
                    note = (
                        f"\n  ⚠️ REGIME WARNING: Lost money in years: {', '.join(neg_years)}. "
                        "See the Regime Fragility Diagnosis below for specific fixes."
                    )
                else:
                    low_years = [f"{y} ({s:.1f})" for y, s in sorted(sby_entry.items(), key=lambda x: x[1])[:2]]
                    note = (
                        f"\n  ⚠️ REGIME WARNING: Weakest years: {', '.join(low_years)}. "
                        "See the Regime Fragility Diagnosis below for specific fixes."
                    )
            else:
                note = (
                    "\n  ⚠️ REGIME WARNING: This strategy only works in specific market conditions "
                    "(e.g., trending or ranging markets). It fails when the regime changes. "
                    "Either add regime detection (only trade when conditions match) or switch "
                    "to a more robust indicator that works across regimes."
                )

        elif failure == "too_few_trades":
            num_trades = entry.get("num_trades", 0)
            note = (
                f"\n  ⚠️ TOO SELECTIVE: Only {num_trades} trades. "
                "OPEN UP your entry conditions — widen thresholds, remove a filter, "
                "or shorten indicator periods. Target: 20-100 trades per year."
            )

        lines.append(
            f"Turn {turn}: Sharpe {sharpe:.2f}\n"
            f"  Failure: {failure} | Action: {action}\n"
            f"  Hypothesis: \"{hypothesis}\"\n"
            f"  Params: {params}\n"
            f"  Delta: {delta}"
            f"{note}"
        )

    return "\n".join(lines)


def build_targeted_task_guidance(
    failure_mode: str,
    *,
    sharpe_ratio: float = 0.0,
    sharpe_target: float = 1.5,
    total_trades: int = 0,
    turn_num: int = 2,
    max_turns: int = 7,
) -> str:
    """Generate failure-mode-specific task instructions for the refinement prompt.

    Instead of generic "read the code and fix it", this tells the LLM exactly
    what to focus on based on the failure mode. This is the "how to think about
    this problem" instruction, complementing build_failure_guidance which is the
    "what to do about it" instruction.

    Args:
        failure_mode: The classified failure mode.
        sharpe_ratio: Current Sharpe ratio.
        sharpe_target: Target Sharpe ratio.
        total_trades: Number of trades.
        turn_num: Current turn number (for urgency messaging).
        max_turns: Maximum turns (for urgency messaging).

    Returns:
        Formatted task guidance string.
    """
    # Base instructions (always present)
    base = (
        "1. Read the CURRENT STRATEGY CODE above — understand what it's doing\n"
        "2. Read the FAILURE DIAGNOSIS — understand what's wrong\n"
        "3. Read PREVIOUS ATTEMPTS — don't repeat what already failed\n"
    )

    # Failure-mode-specific step 4
    mode_step = ""
    if failure_mode == "low_sharpe":
        gap = sharpe_target - sharpe_ratio
        if gap < 0.5:
            mode_step = (
                f"4. You're CLOSE to target (gap: {gap:.2f}). Make ONE small, "
                f"targeted change — don't rewrite the whole strategy. "
                f"A small parameter tweak or one additional filter may close the gap.\n"
                f"5. Output the COMPLETE modified strategy file"
            )
        elif gap < 1.5:
            mode_step = (
                f"4. MODERATE gap to target (Sharpe {sharpe_ratio:.2f} vs {sharpe_target:.2f}). "
                f"Focus on ONE specific weakness identified in the diagnosis above. "
                f"Check the ✅ What's Working section — preserve those elements.\n"
                f"5. Output the COMPLETE modified strategy file"
            )
        else:
            mode_step = (
                f"4. LARGE gap to target (Sharpe {sharpe_ratio:.2f} vs {sharpe_target:.2f}). "
                f"The core signal may need replacement. Consider a different indicator family "
                f"or entry logic, but keep the overall structure simple.\n"
                f"5. Output the COMPLETE modified strategy file"
            )
    elif failure_mode == "too_few_trades" or failure_mode == "too_few_trades_for_validation":
        mode_step = (
            f"4. Your ONLY job this turn: increase trade frequency from {total_trades} to 20+.\n"
            f"   - Widen ONE threshold (e.g., RSI 20→35, or EMA period 50→20)\n"
            f"   - OR remove ONE condition from your entry logic\n"
            f"   - OR shorten your indicator lookback period\n"
            f"   Do NOT add new indicators. Do NOT change the exit logic. SIMPLIFY.\n"
            f"5. Output the COMPLETE modified strategy file"
        )
    elif failure_mode == "regime_fragility":
        mode_step = (
            f"4. Add a REGIME FILTER on top of your existing signal — don't replace the signal.\n"
            f"   - Use ADX > 25 to detect trending markets (only trade when trending)\n"
            f"   - OR use ATR ratio to detect volatility regime\n"
            f"   - OR use SMA(200) slope to detect market direction\n"
            f"   Your existing signal is good in some conditions — just skip the bad conditions.\n"
            f"5. Output the COMPLETE modified strategy file"
        )
    elif failure_mode == "excessive_drawdown":
        mode_step = (
            f"4. Add RISK MANAGEMENT — don't change the entry signal.\n"
            f"   - Add ATR-based stop loss: exit when loss exceeds 2× ATR(14)\n"
            f"   - OR add volatility filter: skip entries when ATR is above its SMA\n"
            f"   - OR add time-based stop: exit after N bars if not profitable\n"
            f"   Your entry signal generates returns — just cap the losses.\n"
            f"5. Output the COMPLETE modified strategy file"
        )
    elif failure_mode == "validation_failed":
        mode_step = (
            f"4. Your strategy works in-sample but fails out-of-sample — it's OVERFIT.\n"
            f"   - REMOVE the most recently added condition/indicator\n"
            f"   - Widen ALL thresholds by 20-30%\n"
            f"   - Replace multi-condition entries with a SINGLE simpler condition\n"
            f"   - Fewer parameters = less overfitting\n"
            f"5. Output the COMPLETE modified strategy file"
        )
    elif failure_mode == "flat_signal":
        mode_step = (
            f"4. Your strategy produces ZERO signals — the entry condition is never True.\n"
            f"   DEBUG: Check if you're using `and`/`or` instead of `&`/`|` for Series comparisons\n"
            f"   DEBUG: Check if your thresholds are impossible (e.g., RSI > 90 AND RSI < 10)\n"
            f"   FIX: Start with the SIMPLEST possible entry: `entries = (ema_fast > ema_slow)`\n"
            f"5. Output the COMPLETE modified strategy file"
        )
    elif failure_mode == "overtrading":
        mode_step = (
            f"4. Your strategy fires too many signals — transaction costs eat profits.\n"
            f"   - Add a COOLDOWN: minimum N bars between trades\n"
            f"   - OR use LONGER indicator periods\n"
            f"   - OR add a CONFIRMATION requirement (2+ indicators must agree)\n"
            f"5. Output the COMPLETE modified strategy file"
        )
    elif failure_mode in ("backtest_crash", "module_load_failed"):
        mode_step = (
            f"4. Your code has a bug — read the error message carefully and fix it.\n"
            f"   - Check the INDICATOR QUICK REFERENCE below for correct function signatures\n"
            f"   - Common: atr needs (high, low, close), stoch needs (high, low, close)\n"
            f"   - Check that entries/exits are pd.Series[bool], not scalars\n"
            f"5. Output the COMPLETE modified strategy file"
        )
    else:
        mode_step = (
            f"4. Propose a targeted modification with a causal hypothesis\n"
            f"5. Output the COMPLETE modified strategy file"
        )

    # Add urgency for later turns
    urgency = ""
    remaining = max_turns - turn_num
    if remaining <= 2:
        urgency = (
            f"\n⚠️ Only {remaining} turn(s) remaining. If this strategy can't hit the target "
            f"with one more modification, consider `full_rewrite` or `novel` action "
            f"to try a fundamentally different approach."
        )

    return base + mode_step + urgency


def build_failure_guidance(
    failure_mode: str,
    total_trades: int = 0,
    validation: dict | None = None,
    *,
    sharpe_ratio: float = 0.0,
    sharpe_target: float = 1.5,
    total_return_pct: float = 0.0,
    max_drawdown_pct: float = 0.0,
    win_rate: float = 0.0,
    profit_factor: float = 0.0,
    sortino_ratio: float = 0.0,
    calmar_ratio: float = 0.0,
    avg_holding_bars: float | None = None,
    sharpe_by_year: dict | None = None,
) -> str:
    """Generate actionable guidance based on failure mode.

    This is the KEY feedback mechanism -- instead of just showing the failure
    label, we tell the LLM exactly WHAT to do about it.

    Args:
        failure_mode: The failure mode string from the backtest report.
        total_trades: Number of trades in the backtest.
        validation: Optional validation results dict with 'window_results',
            'avg_test_sharpe', 'windows_passed', 'num_windows'. Used for
            validation_failed to include per-window breakdown.
        sharpe_ratio: Current Sharpe ratio (for low_sharpe diagnosis).
        sharpe_target: Target Sharpe ratio (for low_sharpe diagnosis).
        total_return_pct: Total return fraction (for low_sharpe diagnosis).
        max_drawdown_pct: Max drawdown fraction (for low_sharpe diagnosis).
        win_rate: Win rate fraction (for low_sharpe diagnosis).
        profit_factor: Profit factor (for low_sharpe diagnosis).
        sortino_ratio: Sortino ratio (for low_sharpe diagnosis).
        calmar_ratio: Calmar ratio (for low_sharpe diagnosis).
        avg_holding_bars: Average holding period (for low_sharpe diagnosis).
        sharpe_by_year: Per-year Sharpe (for low_sharpe diagnosis).

    Returns:
        Formatted guidance string for prompt injection.
    """
    # For validation_failed, include window breakdown if available
    window_breakdown = ""
    if failure_mode == "validation_failed" and validation:
        window_breakdown = _format_window_breakdown(validation)
        if window_breakdown:
            window_breakdown = "\n\n" + window_breakdown

    guidance_map = {
        "too_few_trades_for_validation": (
            "### ⚠️ ACTION REQUIRED: Increase Trade Frequency\n"
            "Your strategy only produced {trades} trades — FAR below the 20-trade minimum "
            "for validation. This means your entry conditions are too restrictive or "
            "curve-fit to a handful of data points.\n\n"
            "**What to do:**\n"
            "- REMOVE conditions — every additional filter reduces trade frequency\n"
            "- WIDEN thresholds — change RSI < 20 to RSI < 35, or SMA length from 200 to 50\n"
            "- Use a SIMPLER signal — single indicator crossover > multi-condition stack\n"
            "- Your goal: 30+ trades over the backtest period\n"
            "- A strategy with 40 trades and Sharpe 0.5 is far better than 3 trades and Sharpe 2.0"
        ),
        "validation_failed": (
            "### ⚠️ ACTION REQUIRED: Fix Out-of-Sample Failure\n"
            "Your strategy passed in-sample (Sharpe hit target) but FAILED out-of-sample "
            "in the rolling walk-forward validation. This is classic overfitting.\n\n"
            "**What to do:**\n"
            "- REDUCE complexity — fewer indicators, fewer conditions, fewer parameters\n"
            "- WIDEN your thresholds — tight thresholds overfit to specific price levels\n"
            "- Add a TIME STOP — exit after N bars regardless of signal (reduces curve-fitting)\n"
            "- Make sure your signal isn't just capturing one specific market regime\n"
            "- Simple strategies generalize better: prefer 2 conditions over 5"
            "{window_breakdown}"
        ),
        "regime_fragility": (
            "### ⚠️ Strategy is Regime-Dependent\n"
            "Your strategy performs well in one market regime but fails in others. "
            "The diagnosis below shows exactly WHICH years failed and WHY.\n\n"
            "{regime_diagnosis}"
        ),
        "low_sharpe": (
            "### Guidance: Improve Sharpe Ratio\n"
            "{curve_fit_warning}"
            "**What to do:**\n"
            "- Check your entry/exit logic — are you entering at good prices?\n"
            "- Try different indicator parameters or a different indicator family entirely\n"
            "- Consider adding a trend filter — only take signals in the direction of the trend\n"
            "{sharpe_diagnosis}"
        ),
        "too_few_trades": (
            "### ⚠️ Too Few Trades ({trades} trades, minimum 20 for validation)\n"
            "Your strategy's entry conditions are too restrictive, causing very few trades. "
            "With so few trades, the Sharpe ratio is unreliable (could be luck).\n\n"
            "**What to do (pick ONE):**\n"
            "- LOOSEN entry thresholds — widen RSI bands, reduce required indicator alignment\n"
            "- REMOVE a filter — if you require 3+ conditions to align, try 2 or even 1\n"
            "- SHORTEN indicator periods — shorter lookbacks fire more often (e.g., EMA 10 → EMA 5)\n"
            "- Try a MORE FREQUENT signal type — crossovers and threshold breaches fire more than extreme readings\n"
            "- Add SHORT side — if you only go long, adding short signals doubles your opportunities\n\n"
            "**Anti-patterns to avoid:**\n"
            "- Do NOT just add random filters — this makes the problem worse\n"
            "- Do NOT tighten exits — this doesn't create more entries\n"
            "- Do NOT use very long lookback periods (50+, 200+) as primary signals — they rarely cross\n"
            "**Target: 20-100 trades per year on daily data.**"
        ),
        "excessive_drawdown": (
            "### ⚠️ Excessive Drawdown (max drawdown exceeds 30%)\n"
            "Your strategy takes on too much risk. Large drawdowns wipe out gains and make "
            "the strategy unsuitable for real trading.\n\n"
            "**What to do (pick ONE):**\n"
            "- Add a STOP LOSS — exit when price drops X% from entry (e.g., ATR-based: 2× ATR)\n"
            "- Add a TREND FILTER — only take signals aligned with the major trend (200 EMA direction)\n"
            "- Use WIDER position sizing — fewer concurrent trades means less correlated risk\n"
            "- Add a VOLATILITY FILTER — don't trade when ATR or VIX is above normal (reduce position or skip)\n"
            "- Add a DRAWDOWN CIRCUIT BREAKER — stop trading after losing X% in N days\n\n"
            "**Anti-patterns to avoid:**\n"
            "- Do NOT just tighten take-profit — this doesn't limit downside risk\n"
            "- Do NOT add more indicators — complexity won't fix risk management\n"
            "- Do NOT remove the stop loss to 'let winners run' — you need defined risk first\n"
            "**Target: max drawdown < 25% for a robust strategy.**"
        ),
        "flat_signal": (
            "### ⚠️ Flat Signal (zero meaningful trades or returns)\n"
            "Your strategy is not generating any signals at all, or the signals cancel out. "
            "The entry/exit logic is either never triggered or entries and exits happen simultaneously.\n\n"
            "**What to do (pick ONE):**\n"
            "- Check your entry condition — is it possible for it to be True? Add debug prints or simplify\n"
            "- Use a SIMPLER signal — try a basic crossover (EMA fast > EMA slow) before adding complexity\n"
            "- Check for NaN handling — if indicators return NaN for the first N bars, all signals are blocked\n"
            "- Use .fillna(0) or .dropna() on indicator outputs before comparison\n"
            "- Ensure entries and exits don't fire on the SAME bar — add a delay or state check\n\n"
            "**Anti-patterns to avoid:**\n"
            "- Do NOT add more conditions — if nothing fires now, more conditions make it worse\n"
            "- Do NOT use extreme thresholds — RSI > 90 never triggers, RSI > 70 does\n"
            "**Target: at least 10 trades in the backtest period.**"
        ),
        "overtrading": (
            "### ⚠️ Overtrading (too many trades — transaction costs dominate)\n"
            "Your strategy fires signals too frequently. With high trade frequency, "
            "commission and slippage consume all profits, making the strategy net negative.\n\n"
            "**What to do (pick ONE):**\n"
            "- Add a COOLDOWN — minimum N bars between trades (e.g., wait 5 bars after exit)\n"
            "- Use LONGER indicator periods — shorter periods fire more often (EMA 5 → EMA 20)\n"
            "- Add a CONFIRMATION requirement — require 2+ indicators to agree before entry\n"
            "- WIDEN your thresholds — RSI crossing 50 fires often, RSI crossing 30/70 fires less\n"
            "- Use HIGHER timeframe signals — if trading daily, require weekly trend alignment\n\n"
            "**Anti-patterns to avoid:**\n"
            "- Do NOT remove the commission model — costs are real in production\n"
            "- Do NOT add a minimum holding period without also reducing signal frequency\n"
            "**Target: 20-100 trades per year on daily data (not 500+).**"
        ),
        "backtest_crash": (
            "### ⚠️ ACTION REQUIRED: Fix Code Crash\n"
            "Your strategy code crashed during backtesting. This is NOT a performance issue — "
            "your code has a bug that prevents it from running.\n\n"
            "**Common causes:**\n"
            "- Wrong column names: use lowercase 'open', 'high', 'low', 'close', 'volume'\n"
            "- Missing imports: import indicators from `crabquant.indicators`\n"
            "- Wrong function signature: `generate_signals(df, params)` returns `(entries, exits)`\n"
            "- NaN handling: use `.fillna()` or `.dropna()` on indicator outputs\n\n"
            "**Check the crash error details in the feedback section above and fix the exact error.**"
        ),
        "module_load_failed": (
            "### ⚠️ ACTION REQUIRED: Fix Import/Load Error\n"
            "Your strategy module failed to load. This means there is a syntax error, "
            "import error, or other issue preventing Python from loading your code.\n\n"
            "**Common causes:**\n"
            "- SyntaxError: check for missing colons, unmatched parentheses, bad indentation\n"
            "- ImportError: only use standard lib + pandas + numpy + crabquant modules\n"
            "- NameError at module level: all top-level code must be valid\n\n"
            "**Check the crash error details in the feedback section above and fix the exact error.**"
        ),
    }

    template = guidance_map.get(failure_mode, "")
    if not template:
        return ""

    # Build curve-fitting warning for low_sharpe with very few trades
    curve_fit_warning = ""
    sharpe_diagnosis = ""
    regime_diagnosis = ""
    if failure_mode == "low_sharpe":
        if total_trades < 10:
            curve_fit_warning = (
                f"⚠️ CRITICAL: Only {total_trades} trades — this strategy is almost certainly "
                "curve-fit to random noise. DO NOT try to optimize parameters further. "
                "Instead, REPLACE the entire strategy with a simpler one that fires 30+ times.\n\n"
            )
        elif total_trades < 15:
            curve_fit_warning = (
                f"⚠️ Very few trades ({total_trades}) — your Sharpe may be unreliable. "
                "Focus on increasing trade frequency FIRST.\n\n"
            )
        # Run the root cause analyzer
        sharpe_diagnosis = diagnose_low_sharpe(
            sharpe_ratio=sharpe_ratio,
            sharpe_target=sharpe_target,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            avg_holding_bars=avg_holding_bars,
            sharpe_by_year=sharpe_by_year,
        )
        if sharpe_diagnosis:
            sharpe_diagnosis = "\n\n" + sharpe_diagnosis

    elif failure_mode == "regime_fragility":
        # Run the regime diagnosis analyzer
        regime_diagnosis = diagnose_regime_fragility(sharpe_by_year or {})
        if not regime_diagnosis:
            # Fallback if no year data available
            regime_diagnosis = (
                "**What to do:**\n"
                "- Add regime detection — check if the market is trending or ranging before trading\n"
                "- Use adaptive parameters — different thresholds for different volatility levels\n"
                "- OR lean into it — make the strategy explicitly regime-specific (only trade when "
                "conditions match), but accept fewer trades"
            )

    # ── Determine recommended action ────────────────────────────────────────
    recommended_action_line = ""
    if failure_mode in RECOMMENDED_ACTIONS:
        action, reason = RECOMMENDED_ACTIONS[failure_mode]

        # Dynamic override for low_sharpe based on diagnosis metrics
        if failure_mode == "low_sharpe":
            sharpe_gap = sharpe_target - sharpe_ratio
            if win_rate < 0.35:
                action = "add_filter"
                reason = "add trend filter — low win rate suggests fighting the trend"
            elif profit_factor < 1.0:
                action = "change_exit_logic"
                reason = "add stop loss — losing more per trade than winning"
            elif sharpe_gap > 1.0:
                action = "full_rewrite"
                reason = "large Sharpe gap — current approach is far from viable"
            elif sharpe_gap < 0.3:
                action = "modify_params"
                reason = "close to target — small parameter tweaks may suffice"
            # else keep default: replace_indicator

        recommended_action_line = (
            f"\n\n**Recommended action:** `{action}` — {reason}"
        )

    formatted = template.format(
        trades=total_trades,
        window_breakdown=window_breakdown,
        curve_fit_warning=curve_fit_warning,
        sharpe_diagnosis=sharpe_diagnosis,
        regime_diagnosis=regime_diagnosis,
    )

    return formatted + recommended_action_line


def build_turn1_prompt(
    *,
    mandate: dict,
    current_turn: int,
    max_turns: int,
    seed_strategy_name: str | None = None,
    seed_code: str | None = None,
    seed_params: dict | None = None,
    strategy_examples: list[dict] | None = None,
    strategy_catalog: list[dict] | None = None,
    winner_examples: list[dict] | None = None,
    indicator_reference: str = "",
    indicator_quick_ref: str = "",
    archetype_section: str | None = None,
    effective_target: float | None = None,
    trade_count_guidance: str = "",
) -> str:
    """Build the complete Turn 1 (invention) prompt.

    Args:
        mandate: Mandate dict with name, archetype, tickers, etc.
        current_turn: Current turn number (should be 1).
        max_turns: Maximum turns in the run.
        seed_strategy_name: Optional name of seed strategy.
        seed_code: Optional seed strategy source code.
        seed_params: Optional seed strategy default params.
        strategy_examples: Optional list of example strategy dicts.
        strategy_catalog: Optional list of catalog entries.
        winner_examples: Optional list of proven winner strategy dicts (cross-run learning).
        indicator_reference: Full indicator API reference text.
        indicator_quick_ref: Quick reference card (section 7) for user message.
        archetype_section: Optional pre-formatted archetype template section.
        effective_target: Adaptive Sharpe target for this turn. If None, uses sharpe_target.
        trade_count_guidance: Pre-formatted trade count expectations text.

    Returns:
        Complete prompt string for the LLM.
    """
    seed_section = ""
    if seed_strategy_name and seed_code:
        seed_section = (
            f"\n### Seed Strategy: {seed_strategy_name}\n"
            f"The code below is your starting point. Modify it to improve performance.\n\n"
            f"```python\n{seed_code}\n```\n"
            f"Default params: {seed_params or {}}\n"
        )

    # Format strategy examples
    examples_text = ""
    if strategy_examples:
        parts = []
        for ex in strategy_examples:
            parts.append(
                f"#### {ex.get('name', 'unknown')}\n"
                f"Description: {ex.get('description', 'N/A')}\n"
                f"Default params: {ex.get('default_params', {})}\n"
                f"```python\n{ex.get('source_code', '# code unavailable')}\n```"
            )
        examples_text = "\n\n".join(parts)

    # Format strategy catalog
    catalog_text = ""
    if strategy_catalog:
        catalog_text = "\n".join(
            f"- **{entry['name']}**: {entry['description']}"
            for entry in strategy_catalog
        )

    # Format winner examples (cross-run learning)
    winner_section = ""
    if winner_examples:
        winner_parts = [
            "### Proven Strategies (from previous runs — these achieved high Sharpe with real trades)"
        ]
        for wex in winner_examples:
            ticker_str = f" ({wex['ticker']})" if wex.get("ticker") else ""
            winner_parts.append(
                f"#### ⭐ {wex['name']} — Sharpe {wex['sharpe']:.2f}, "
                f"{wex['trades']} trades{ticker_str}\n"
                f"```python\n{wex['source_code']}\n```"
            )
        winner_section = "\n\n".join(winner_parts) + "\n\n"

    # Format archetype template section
    if archetype_section is None:
        archetype_name = mandate.get("strategy_archetype", "any")
        if archetype_name != "any":
            from crabquant.refinement.archetypes import get_archetype, format_archetype_for_prompt
            archetype = get_archetype(archetype_name)
            if archetype:
                archetype_section = (
                    "## Strategy Archetype Template\n"
                    "Use this as your STARTING POINT. Customize the parameters and logic,\n"
                    "but keep the core structure. This template is proven to work.\n\n"
                    + format_archetype_for_prompt(archetype)
                )
    
    archetype_text = archetype_section or ""

    sharpe_target_val = mandate.get("sharpe_target", 1.5)
    eff_target = effective_target if effective_target is not None else sharpe_target_val
    
    # Build adaptive target display
    if effective_target is not None and abs(effective_target - sharpe_target_val) > 1e-9:
        target_display = f"{eff_target:.2f} (ramping toward final target: {sharpe_target_val:.2f})"
    else:
        target_display = f"{sharpe_target_val:.2f}"

    return TURN1_PROMPT.format(
        mandate_name=mandate.get("name", "unnamed"),
        strategy_archetype=mandate.get("strategy_archetype", "any"),
        sharpe_target=target_display,
        tickers=mandate.get("tickers", ["AAPL", "SPY"]),
        period=mandate.get("period", "2y"),
        seed_section=seed_section,
        archetype_section=archetype_text,
        strategy_catalog=catalog_text,
        strategy_examples=examples_text,
        winner_examples_section=winner_section,
        indicator_quick_ref=indicator_quick_ref,
        trade_count_guidance=trade_count_guidance,
    )


def build_refinement_prompt(
    *,
    tier1_report: dict,
    current_turn: int,
    max_turns: int,
    sharpe_target: float,
    effective_target: float | None = None,
    best_sharpe: float,
    best_turn: int,
    stagnation_suffix: str = "",
    strategy_examples: list[dict] | None = None,
    winner_examples: list[dict] | None = None,
    archetype_section: str | None = None,
    indicator_reference: str = "",
    indicator_quick_ref: str = "",
    action_effectiveness_section: str = "",
    failure_pattern_section: str = "",
    trade_count_guidance: str = "",
) -> str:
    """Build the complete refinement prompt for turns 2+.

    Args:
        tier1_report: Tier 1 diagnostic report dict.
        current_turn: Current turn number.
        max_turns: Maximum turns in the run.
        sharpe_target: Target Sharpe ratio (full/final target).
        effective_target: Adaptive Sharpe target for this turn. If None, uses sharpe_target.
        best_sharpe: Best Sharpe achieved so far.
        best_turn: Turn number of the best Sharpe.
        stagnation_suffix: Formatted stagnation response string.
        strategy_examples: Optional list of example strategy dicts.
        winner_examples: Optional list of proven winner strategy dicts (cross-run learning).
        indicator_reference: Full indicator API reference text.
        indicator_quick_ref: Quick reference card (section 7) for user message.
        action_effectiveness_section: Pre-formatted action effectiveness data for
            the current failure mode.
        trade_count_guidance: Pre-formatted trade count expectations text.

    Returns:
        Complete prompt string for the LLM.
    """
    # Resolve effective target for display
    eff = effective_target if effective_target is not None else sharpe_target
    if effective_target is not None and abs(effective_target - sharpe_target) > 1e-9:
        target_display = f"{eff:.2f} (ramping toward final target: {sharpe_target:.2f})"
    else:
        target_display = f"{sharpe_target:.2f}"
    # Format previous attempts
    prev_attempts = tier1_report.get("previous_attempts", [])
    prev_section = format_previous_attempts_section(prev_attempts)

    # Format Tier 2 section (only if data available)
    tier2_section = format_tier2_section(tier1_report)
    tier2_formatted = f"\n### Advanced Diagnostics (Tier 2)\n{tier2_section}" if tier2_section else ""

    # Format strategy examples
    examples_text = ""
    if strategy_examples:
        parts = []
        for ex in strategy_examples:
            parts.append(
                f"#### {ex.get('name', 'unknown')}\n"
                f"```python\n{ex.get('source_code', '# code unavailable')}\n```"
            )
        examples_text = "\n\n".join(parts)

    # Format winner examples (cross-run learning)
    winner_section = ""
    if winner_examples:
        winner_parts = [
            "### Proven Strategies (from previous runs — these achieved high Sharpe with real trades)"
        ]
        for wex in winner_examples:
            ticker_str = f" ({wex['ticker']})" if wex.get("ticker") else ""
            winner_parts.append(
                f"#### ⭐ {wex['name']} — Sharpe {wex['sharpe']:.2f}, "
                f"{wex['trades']} trades{ticker_str}\n"
                f"```python\n{wex['source_code']}\n```"
            )
        winner_section = "\n\n".join(winner_parts) + "\n\n"

    # Format failure guidance (specific actionable advice based on failure mode)
    failure_mode = tier1_report.get("failure_mode", "")
    total_trades = tier1_report.get("total_trades", 0)
    # Extract validation results from previous attempts if available
    validation_data = None
    prev_attempts = tier1_report.get("previous_attempts", [])
    for pa in prev_attempts:
        if pa.get("failure_mode") == "validation_failed" and pa.get("validation"):
            validation_data = pa["validation"]
            break
    failure_guidance = build_failure_guidance(
        failure_mode, total_trades, validation_data,
        sharpe_ratio=tier1_report.get("sharpe_ratio", 0.0),
        sharpe_target=tier1_report.get("sharpe_target", sharpe_target),
        total_return_pct=tier1_report.get("total_return_pct", 0.0),
        max_drawdown_pct=tier1_report.get("max_drawdown_pct", 0.0),
        win_rate=tier1_report.get("win_rate", 0.0),
        profit_factor=tier1_report.get("profit_factor", 0.0),
        sortino_ratio=tier1_report.get("sortino_ratio", 0.0),
        calmar_ratio=tier1_report.get("calmar_ratio", 0.0),
        avg_holding_bars=tier1_report.get("avg_holding_bars"),
        sharpe_by_year=tier1_report.get("sharpe_by_year"),
    )

    # Format sharpe_by_year
    sby = tier1_report.get("sharpe_by_year", {})
    if sby:
        sby_text = "\n".join(f"  {year}: {sharpe:.2f}" for year, sharpe in sby.items())
    else:
        sby_text = "  (not available)"

    # Format positive feedback (what's working — prevents regression)
    pos_feedback = analyze_positive_feedback(
        sharpe_ratio=tier1_report.get("sharpe_ratio", 0.0),
        sharpe_target=tier1_report.get("sharpe_target", sharpe_target),
        total_return_pct=tier1_report.get("total_return_pct", 0.0),
        max_drawdown_pct=tier1_report.get("max_drawdown_pct", 0.0),
        win_rate=tier1_report.get("win_rate", 0.0),
        profit_factor=tier1_report.get("profit_factor", 0.0),
        sortino_ratio=tier1_report.get("sortino_ratio", 0.0),
        calmar_ratio=tier1_report.get("calmar_ratio", 0.0),
        total_trades=tier1_report.get("total_trades", 0),
        avg_holding_bars=tier1_report.get("avg_holding_bars"),
        sharpe_by_year=tier1_report.get("sharpe_by_year"),
        failure_mode=failure_mode,
    )
    positive_feedback_section = format_positive_feedback_for_prompt(pos_feedback)

    # Build failure-mode-specific task guidance
    targeted_task_guidance = build_targeted_task_guidance(
        failure_mode,
        sharpe_ratio=tier1_report.get("sharpe_ratio", 0.0),
        sharpe_target=tier1_report.get("sharpe_target", sharpe_target),
        total_trades=total_trades,
        turn_num=current_turn,
        max_turns=max_turns,
    )

    return REFINEMENT_PROMPT.format(
        current_turn=current_turn,
        max_turns=max_turns,
        sharpe_target=target_display,
        current_strategy_code=tier1_report.get("current_strategy_code", "# code unavailable"),
        current_params=tier1_report.get("current_params", {}),
        sharpe_ratio=tier1_report.get("sharpe_ratio", 0.0),
        total_return_pct=tier1_report.get("total_return_pct", 0.0),
        max_drawdown_pct=tier1_report.get("max_drawdown_pct", 0.0),
        total_trades=tier1_report.get("total_trades", 0),
        win_rate=tier1_report.get("win_rate", 0.0),
        profit_factor=tier1_report.get("profit_factor", 0.0),
        calmar_ratio=tier1_report.get("calmar_ratio", 0.0),
        sortino_ratio=tier1_report.get("sortino_ratio", 0.0),
        failure_mode=tier1_report.get("failure_mode", "unknown"),
        failure_details=tier1_report.get("failure_details", "N/A"),
        failure_guidance=failure_guidance,
        positive_feedback_section=positive_feedback_section,
        sharpe_by_year=sby_text,
        feature_importance_section=tier1_report.get("feature_importance_section", ""),
        tier2_section=tier2_formatted,
        previous_attempts_section=prev_section,
        stagnation_score=tier1_report.get("stagnation_score", 0.0),
        stagnation_trend=tier1_report.get("stagnation_trend", "improving"),
        best_sharpe=best_sharpe,
        best_turn=best_turn,
        stagnation_suffix=stagnation_suffix,
        strategy_examples=examples_text,
        winner_examples_section=winner_section,
        indicator_quick_ref=indicator_quick_ref,
        action_effectiveness_section=action_effectiveness_section,
        failure_pattern_section=failure_pattern_section,
        trade_count_guidance=trade_count_guidance,
        targeted_task_guidance=targeted_task_guidance,
    )


# ── Phase 6: Crash error recovery hints ──────────────────────────────────────

_CRASH_HINTS: list[tuple[str, str, str]] = [
    # (error_type_pattern, message_pattern, hint)
    (
        "KeyError",
        "",
        "Check column names — use lowercase: 'open', 'high', 'low', 'close', 'volume'. "
        "DataFrame columns are lowercase in CrabQuant. Access via df['close'], not df.Close.",
    ),
    (
        "NameError",
        "not defined",
        "A variable or function is used without being defined or imported. "
        "If it's an indicator, import it from `crabquant.indicators` or use `cached_indicator('indicator_name', df)`.",
    ),
    (
        "TypeError",
        "generate_signals",
        "`generate_signals` must accept exactly two arguments: `(df, params)` and return "
        "a tuple of two Series `(entries, exits)` with boolean values. "
        "Do NOT add extra parameters.",
    ),
    (
        "TypeError",
        "missing",
        "A function is called with the wrong number of arguments. "
        "Check the function signature and ensure all required parameters are passed.",
    ),
    (
        "AttributeError",
        "",
        "An attribute or method does not exist on the object. "
        "For DataFrames, use bracket notation: df['close'] not df.close. "
        "Check that indicator functions return the expected type.",
    ),
    (
        "ImportError",
        "",
        "A module could not be imported. Only use standard library, pandas, pandas_ta, "
        "numpy, and crabquant modules. Do not import third-party libraries like talib, "
        "ta-lib, sklearn, or scipy unless explicitly available.",
    ),
    (
        "ModuleNotFoundError",
        "",
        "A module could not be found. Only use standard library, pandas, pandas_ta, "
        "numpy, and crabquant modules.",
    ),
    (
        "ValueError",
        "array",
        "An array operation failed — possibly due to NaN values or shape mismatch. "
        "Use `.dropna()`, `.fillna()`, or `np.nan_to_num()` to handle NaN values. "
        "Check that your indicators produce the same length as the input DataFrame.",
    ),
    (
        "ValueError",
        "Truth value",
        "You likely used a Series/DataFrame in a boolean context (e.g., `if df['signal']:`). "
        "Use `.any()`, `.all()`, or `.iloc[0]` to get a scalar boolean.",
    ),
    (
        "ZeroDivisionError",
        "",
        "Division by zero occurred — likely a denominator that can be zero. "
        "Add a small epsilon: `denom = series + 1e-10` or guard with `.replace(0, np.nan)`.",
    ),
    (
        "IndexError",
        "",
        "An index is out of bounds — likely accessing a list/array with an invalid index. "
        "Check array lengths before indexing, or use `.iloc[-1]` instead of `.iloc[N]`.",
    ),
    (
        "SyntaxError",
        "",
        "The generated code has invalid Python syntax. Check for missing colons, "
        "unmatched parentheses, or incorrect indentation. Make sure all string literals are properly closed.",
    ),
    (
        "IndentationError",
        "",
        "Python indentation is incorrect. Ensure consistent use of 4 spaces for indentation. "
        "Do not mix tabs and spaces.",
    ),
]


def get_crash_recovery_hints(error_type: str, error_message: str) -> str:
    """Return recovery hint text for a crash error.

    Matches the error_type and error_message against known patterns and returns
    the most specific hint available.

    Args:
        error_type: Exception class name (e.g., 'KeyError', 'NameError').
        error_message: Exception message string.

    Returns:
        A recovery hint string, or empty string if no hint matches.
    """
    hints: list[str] = []

    for pattern_type, pattern_msg, hint in _CRASH_HINTS:
        type_match = pattern_type.lower() in error_type.lower()
        msg_match = not pattern_msg or pattern_msg.lower() in error_message.lower()
        if type_match and msg_match:
            hints.append(hint)

    if hints:
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for h in hints:
            if h not in seen:
                seen.add(h)
                unique.append(h)
        return " ".join(unique)

    return ""


def build_crash_guidance(error_info: dict | None) -> str:
    """Build guidance text for crash errors (backtest_crash / module_load_failed).

    This is the Phase 6 equivalent of build_failure_guidance() but for code
    crashes rather than metric failures.

    Args:
        error_info: Dict with 'error_type', 'error_message', optionally 'error_traceback'.

    Returns:
        Formatted guidance string.
    """
    if not error_info:
        return ""

    error_type = error_info.get("error_type", "UnknownError")
    error_message = error_info.get("error_message", "No details")
    hints = get_crash_recovery_hints(error_type, error_message)

    lines = [
        f"Your strategy code crashed with: {error_type}: {error_message}",
    ]
    if hints:
        lines.append(f"Fix: {hints}")
    else:
        lines.append(
            "Fix: Review your code for bugs. Common issues: wrong column names, "
            "missing imports, incorrect function signatures, NaN handling."
        )

    return "\n".join(lines)
