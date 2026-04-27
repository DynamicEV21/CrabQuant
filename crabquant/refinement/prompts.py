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
{stagnation_suffix}
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

### Your Task
Invent a complete trading strategy matching the mandate.
Use the example strategies as a TEMPLATE for code structure, function signatures, \
and indicator usage patterns. Your code MUST follow the same conventions.

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

## Your Task
1. Read the CURRENT STRATEGY CODE above — understand what it's doing
2. Read the FAILURE DIAGNOSIS — understand what's wrong
3. Read PREVIOUS ATTEMPTS — don't repeat what already failed
4. Propose a targeted modification with a causal hypothesis
5. Output the COMPLETE modified strategy file

Output ONLY the JSON object with the required fields.
"""


# ── Helper Functions ─────────────────────────────────────────────────────────


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

    return TURN1_PROMPT.format(
        mandate_name=mandate.get("name", "unnamed"),
        strategy_archetype=mandate.get("strategy_archetype", "any"),
        sharpe_target=mandate.get("sharpe_target", 1.5),
        tickers=mandate.get("tickers", ["AAPL", "SPY"]),
        period=mandate.get("period", "2y"),
        seed_section=seed_section,
        strategy_catalog=catalog_text,
        strategy_examples=examples_text,
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
    )
