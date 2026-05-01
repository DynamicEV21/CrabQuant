"""Signal Analysis — Phase 6.

Analyzes strategy signals to provide the LLM with specific, actionable feedback
about why a strategy produces too few or too many signals. This replaces the
generic "too_few_trades" feedback with precise diagnosis of what's wrong with
the entry/exit conditions.

Targeted at the #1 and #3 failure modes (low_sharpe 34%, too_few_trades 23%).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Signal Density Check ──────────────────────────────────────────────────────


def analyze_signal_density(
    entries: pd.Series,
    exits: pd.Series,
    df_length: int,
) -> dict[str, Any]:
    """Analyze entry/exit signal density and identify problems.

    This runs AFTER generate_signals() but BEFORE the backtest engine,
    providing fast feedback about signal quality without waiting for the
    full backtest.

    Args:
        entries: Boolean series of entry signals (same length as df).
        exits: Boolean series of exit signals (same length as df).
        df_length: Number of bars in the backtest dataframe.

    Returns:
        Dict with keys:
            - entry_count: int — number of True entry signals
            - exit_count: int — number of True exit signals
            - entry_rate: float — fraction of bars with entry signals
            - exit_rate: float — fraction of bars with exit signals
            - estimated_trades: int — rough trade estimate (min of entries/exits)
            - diagnosis: str | None — specific diagnosis if signals are problematic
            - fix_suggestion: str | None — actionable suggestion for the LLM
            - severity: str — "ok", "warning", "critical"
            - signal_pattern: str — pattern description (e.g., "clustered", "even", "none")
            - entry_gaps: list[int] — gaps between consecutive entry signals
            - avg_entry_gap: float — average bars between entries
            - max_entry_gap: int — longest gap between entries
    """
    result: dict[str, Any] = {
        "entry_count": 0,
        "exit_count": 0,
        "entry_rate": 0.0,
        "exit_rate": 0.0,
        "estimated_trades": 0,
        "diagnosis": None,
        "fix_suggestion": None,
        "severity": "ok",
        "signal_pattern": "none",
        "entry_gaps": [],
        "avg_entry_gap": 0.0,
        "max_entry_gap": 0,
    }

    if entries is None or exits is None:
        result["diagnosis"] = "signals_are_none"
        result["fix_suggestion"] = (
            "generate_signals() returned None for entries or exits. "
            "Check that your function returns (entries, exits) as pd.Series[bool]."
        )
        result["severity"] = "critical"
        return result

    # Handle non-boolean entries (common LLM mistake)
    if not isinstance(entries, pd.Series):
        result["diagnosis"] = "entries_not_series"
        result["fix_suggestion"] = (
            f"entries is {type(entries).__name__}, not pd.Series. "
            "Convert to pd.Series[bool] before returning."
        )
        result["severity"] = "critical"
        return result

    # Handle NaN values (common when indicators produce NaN at start)
    # Use np.where to avoid pandas FutureWarning about downcasting in fillna()/where()
    entry_clean = pd.Series(
        np.where(pd.notna(entries), entries, False), dtype=bool, index=entries.index
    )
    exit_clean = pd.Series(
        np.where(pd.notna(exits), exits, False), dtype=bool, index=exits.index
    )

    entry_count = int(entry_clean.sum())
    exit_count = int(exit_clean.sum())

    result["entry_count"] = entry_count
    result["exit_count"] = exit_count
    result["entry_rate"] = entry_count / max(df_length, 1)
    result["exit_rate"] = exit_count / max(df_length, 1)

    # Estimate trades: roughly min(entries, exits) since each trade needs both
    result["estimated_trades"] = min(entry_count, exit_count)

    # Analyze entry gaps (spacing between consecutive entries)
    if entry_count > 0:
        entry_indices = np.where(entry_clean.values)[0]
        gaps = np.diff(entry_indices).tolist()
        result["entry_gaps"] = gaps
        result["avg_entry_gap"] = float(np.mean(gaps)) if gaps else 0.0
        result["max_entry_gap"] = int(np.max(gaps)) if gaps else 0

        # Detect signal pattern
        if entry_count == 1:
            result["signal_pattern"] = "single"
        elif entry_count <= 3:
            result["signal_pattern"] = "sparse"
        elif result["avg_entry_gap"] > 100:
            result["signal_pattern"] = "clustered"
        elif result["max_entry_gap"] > result["avg_entry_gap"] * 5:
            result["signal_pattern"] = "uneven"
        else:
            result["signal_pattern"] = "regular"
    else:
        result["signal_pattern"] = "none"

    # ── Diagnosis ──────────────────────────────────────────────────────────

    # Critical: zero entries
    if entry_count == 0:
        result["severity"] = "critical"
        result["diagnosis"] = "zero_entries"
        result["fix_suggestion"] = (
            "Your entry conditions produce ZERO signals. This means your "
            "thresholds are impossibly restrictive. Common causes:\n"
            "1. Requiring multiple rare conditions simultaneously (e.g., RSI < 10 AND price > SMA200)\n"
            "2. Using extremely tight thresholds (e.g., ATR ratio > 5.0)\n"
            "3. Comparing wrong data types or using .values on arrays\n"
            "FIX: Remove 1-2 conditions OR widen thresholds significantly. "
            "Start with a SINGLE simple entry condition."
        )
        return result

    # Critical: entries but zero exits (trades never close)
    if entry_count > 0 and exit_count == 0:
        result["severity"] = "critical"
        result["diagnosis"] = "zero_exits"
        result["fix_suggestion"] = (
            f"Your strategy has {entry_count} entry signals but ZERO exit signals. "
            "Trades enter but never exit, causing all capital to be deployed on the "
            "first signal. FIX: Add explicit exit conditions (e.g., opposite signal, "
            "time-based exit, trailing stop, take-profit level)."
        )
        return result

    # Warning: very few entries (< 5)
    if entry_count < 5:
        result["severity"] = "critical"
        result["diagnosis"] = "very_few_entries"
        avg_gap = result["avg_entry_gap"]
        result["fix_suggestion"] = (
            f"Only {entry_count} entry signals in {df_length} bars "
            f"(avg gap: {avg_gap:.0f} bars between entries). "
            f"This is far too few for reliable backtesting. "
            f"FIX: Widen entry thresholds or reduce the number of required conditions. "
            f"Aim for at least 20 entries (one every ~{df_length // 20} bars)."
        )
        return result

    # Warning: entries but very few exits
    if entry_count >= 5 and exit_count < 5:
        result["severity"] = "critical"
        result["diagnosis"] = "few_exits"
        result["fix_suggestion"] = (
            f"{entry_count} entries but only {exit_count} exits — "
            f"most trades never close. Add explicit exit conditions."
        )
        return result

    # Warning: too many signals (noise)
    if result["entry_rate"] > 0.3:
        result["severity"] = "warning"
        result["diagnosis"] = "excessive_entries"
        result["fix_suggestion"] = (
            f"Entry rate is {result['entry_rate']:.1%} — signals fire on {entry_count}/{df_length} bars. "
            "This is too frequent and likely means your entry conditions are too loose, "
            "capturing noise rather than genuine opportunities. Add a filter or tighten thresholds."
        )
        return result

    # Warning: estimated trades < 10
    if result["estimated_trades"] < 10:
        result["severity"] = "warning"
        result["diagnosis"] = "low_trade_estimate"
        result["fix_suggestion"] = (
            f"Estimated {result['estimated_trades']} trades ({entry_count} entries, {exit_count} exits). "
            f"This is below the 20-trade minimum for validation. "
            f"Consider widening entry conditions to generate more signals."
        )
        return result

    # Warning: clustered entries (signals only fire in bursts)
    if result["signal_pattern"] == "clustered":
        result["severity"] = "warning"
        result["diagnosis"] = "clustered_entries"
        result["fix_suggestion"] = (
            f"Entry signals are clustered — {entry_count} entries with avg gap of "
            f"{result['avg_entry_gap']:.0f} bars but max gap of {result['max_entry_gap']} bars. "
            "Your conditions only fire during specific market regimes. "
            "Consider making entry conditions less regime-dependent."
        )
        return result

    # OK — signal density looks reasonable
    result["severity"] = "ok"
    result["diagnosis"] = None
    result["fix_suggestion"] = None
    return result


def format_signal_analysis_for_prompt(analysis: dict[str, Any]) -> str:
    """Format signal analysis for injection into LLM prompt.

    Only produces output if there's a diagnosis (severity != "ok").

    Args:
        analysis: Dict from analyze_signal_density().

    Returns:
        Formatted string for LLM context. Empty string if signals are OK.
    """
    severity = analysis.get("severity", "ok")
    if severity == "ok":
        return ""

    diagnosis = analysis.get("diagnosis", "")
    fix = analysis.get("fix_suggestion", "")
    entry_count = analysis.get("entry_count", 0)
    exit_count = analysis.get("exit_count", 0)
    estimated = analysis.get("estimated_trades", 0)
    pattern = analysis.get("signal_pattern", "none")

    lines = [
        "### Signal Analysis (Pre-Backtest Check)",
        f"Entry signals: {entry_count} | Exit signals: {exit_count} | Estimated trades: {estimated}",
        f"Signal pattern: {pattern}",
    ]

    if analysis.get("avg_entry_gap"):
        lines.append(f"Average gap between entries: {analysis['avg_entry_gap']:.0f} bars")

    if diagnosis:
        lines.append(f"Diagnosis: {diagnosis}")

    if fix:
        lines.append(f"Fix: {fix}")

    return "\n".join(lines)


def check_signal_density_early_exit(
    entries: pd.Series,
    exits: pd.Series,
    df_length: int,
) -> tuple[bool, str]:
    """Quick check whether signals are so bad we should skip the backtest.

    Returns (should_skip, reason). If should_skip is True, the refinement
    loop should immediately record a failure with the given reason rather
    than running the full backtest engine.

    This saves time when strategies produce zero signals (no point running
    the engine) and gives the LLM faster, more specific feedback.

    Args:
        entries: Boolean series of entry signals.
        exits: Boolean series of exit signals.
        df_length: Number of bars in the backtest dataframe.

    Returns:
        (should_skip: bool, reason: str)
    """
    # Check for None signals
    if entries is None or exits is None:
        return True, "generate_signals returned None — check return type"

    # Handle non-Series entries
    if not isinstance(entries, pd.Series):
        return True, f"entries is {type(entries).__name__}, expected pd.Series"

    # Use np.where to avoid pandas FutureWarning about downcasting in fillna()/where()
    entry_clean = pd.Series(
        np.where(pd.notna(entries), entries, False), dtype=bool, index=entries.index
    )
    exit_clean = pd.Series(
        np.where(pd.notna(exits), exits, False), dtype=bool, index=exits.index
    )

    entry_count = int(entry_clean.sum())

    # Zero entries — definitely skip
    if entry_count == 0:
        return True, "zero entry signals — conditions are too restrictive"

    # Check for all-NaN entries (common indicator bug)
    nan_count = int(entries.isna().sum())
    if nan_count > len(entries) * 0.5:
        return True, f"{nan_count}/{len(entries)} NaN values in entries — indicator calculation bug"

    # Check exit signals
    exit_count = int(exit_clean.sum())
    if entry_count > 0 and exit_count == 0:
        return True, f"{entry_count} entries but 0 exits — trades never close"

    return False, ""
