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

### Sharpe by Year
{sharpe_by_year}
{tier2_section}
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


def format_previous_attempts_section(previous_attempts: list[dict]) -> str:
    """Format previous attempts for the refinement prompt.

    Args:
        previous_attempts: List of history dicts with turn, sharpe, action, etc.

    Returns:
        Formatted string, or empty string if no attempts.
    """
    if not previous_attempts:
        return "  (none — this is the first refinement)"

    lines = []
    for entry in previous_attempts:
        turn = entry.get("turn", "?")
        sharpe = entry.get("sharpe", 0.0)
        failure = entry.get("failure_mode", "unknown")
        action = entry.get("action", "unknown")
        hypothesis = entry.get("hypothesis", "N/A")
        params = entry.get("params_used", {})
        delta = entry.get("delta_from_prev", "N/A")

        lines.append(
            f"Turn {turn}: Sharpe {sharpe:.2f}\n"
            f"  Failure: {failure} | Action: {action}\n"
            f"  Hypothesis: \"{hypothesis}\"\n"
            f"  Params: {params}\n"
            f"  Delta: {delta}"
        )

    return "\n".join(lines)


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

    return TURN1_PROMPT.format(
        mandate_name=mandate.get("name", "unnamed"),
        strategy_archetype=mandate.get("strategy_archetype", "any"),
        sharpe_target=mandate.get("sharpe_target", 1.5),
        tickers=mandate.get("tickers", ["AAPL", "SPY"]),
        period=mandate.get("period", "2y"),
        seed_section=seed_section,
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
        sharpe_by_year=sby_text,
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
