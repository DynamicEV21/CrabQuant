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
1. Read the CURRENT STRATEGY CODE above — understand what it's doing
2. Read the FAILURE DIAGNOSIS — understand what's wrong
3. Read PREVIOUS ATTEMPTS — don't repeat what already failed
4. Propose a targeted modification with a causal hypothesis
5. Output the COMPLETE modified strategy file

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
) -> float:
    """Compute composite score for ranking parallel strategies.

    Formula: sharpe * sqrt(trades / 20) * (1 - abs(max_drawdown))

    This penalizes overfit (few trades) and excessive drawdown.

    Args:
        sharpe: Sharpe ratio of the strategy.
        trades: Total number of trades.
        max_drawdown: Maximum drawdown as a fraction (e.g., 0.15 for 15%).

    Returns:
        Composite score (higher is better). Returns 0.0 on invalid inputs.
    """
    if sharpe <= 0 or trades < 1:
        return 0.0

    trade_factor = (trades / 20) ** 0.5
    dd_penalty = 1.0 - min(abs(max_drawdown), 1.0)
    return sharpe * trade_factor * dd_penalty


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
            note = (
                "\n  ⚠️ REGIME WARNING: This strategy only works in specific market conditions "
                "(e.g., trending or ranging markets). It fails when the regime changes. "
                "Either add regime detection (only trade when conditions match) or switch "
                "to a more robust indicator that works across regimes."
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


def build_failure_guidance(
    failure_mode: str,
    total_trades: int = 0,
    validation: dict | None = None,
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
            "It may only work during trends, or only during ranges.\n\n"
            "**What to do:**\n"
            "- Add regime detection — check if the market is trending or ranging before trading\n"
            "- Use adaptive parameters — different thresholds for different volatility levels\n"
            "- OR lean into it — make the strategy explicitly regime-specific (only trade when "
            "conditions match), but accept fewer trades"
        ),
        "low_sharpe": (
            "### Guidance: Improve Sharpe Ratio\n"
            "{curve_fit_warning}"
            "**What to do:**\n"
            "- Check your entry/exit logic — are you entering at good prices?\n"
            "- Try different indicator parameters or a different indicator family entirely\n"
            "- Consider adding a trend filter — only take signals in the direction of the trend"
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

    return template.format(
        trades=total_trades,
        window_breakdown=window_breakdown,
        curve_fit_warning=curve_fit_warning,
    )


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

    return TURN1_PROMPT.format(
        mandate_name=mandate.get("name", "unnamed"),
        strategy_archetype=mandate.get("strategy_archetype", "any"),
        sharpe_target=mandate.get("sharpe_target", 1.5),
        tickers=mandate.get("tickers", ["AAPL", "SPY"]),
        period=mandate.get("period", "2y"),
        seed_section=seed_section,
        archetype_section=archetype_text,
        strategy_catalog=catalog_text,
        strategy_examples=examples_text,
        winner_examples_section=winner_section,
        indicator_quick_ref=indicator_quick_ref,
    )


def build_refinement_prompt(
    *,
    tier1_report: dict,
    current_turn: int,
    max_turns: int,
    sharpe_target: float,
    best_sharpe: float,
    best_turn: int,
    stagnation_suffix: str = "",
    strategy_examples: list[dict] | None = None,
    winner_examples: list[dict] | None = None,
    archetype_section: str | None = None,
    indicator_reference: str = "",
    indicator_quick_ref: str = "",
) -> str:
    """Build the complete refinement prompt for turns 2+.

    Args:
        tier1_report: Tier 1 diagnostic report dict.
        current_turn: Current turn number.
        max_turns: Maximum turns in the run.
        sharpe_target: Target Sharpe ratio.
        best_sharpe: Best Sharpe achieved so far.
        best_turn: Turn number of the best Sharpe.
        stagnation_suffix: Formatted stagnation response string.
        strategy_examples: Optional list of example strategy dicts.
        winner_examples: Optional list of proven winner strategy dicts (cross-run learning).
        indicator_reference: Full indicator API reference text.
        indicator_quick_ref: Quick reference card (section 7) for user message.

    Returns:
        Complete prompt string for the LLM.
    """
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
    failure_guidance = build_failure_guidance(failure_mode, total_trades, validation_data)

    # Format sharpe_by_year
    sby = tier1_report.get("sharpe_by_year", {})
    if sby:
        sby_text = "\n".join(f"  {year}: {sharpe:.2f}" for year, sharpe in sby.items())
    else:
        sby_text = "  (not available)"

    return REFINEMENT_PROMPT.format(
        current_turn=current_turn,
        max_turns=max_turns,
        sharpe_target=sharpe_target,
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
