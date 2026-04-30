"""
Stagnation detection for CrabQuant refinement pipeline.

Implements scoring formula based on consecutive failed turns, Sharpe plateau,
and lack of improvement.  Response protocol -- pivot, broaden, nuclear rewrite,
or abandon.

Phase 5.6 additions:
- detect_stagnation_trap(): identifies WHERE the LLM is stuck (Sharpe band,
  trade count trap, indicator rut, action loop)
- build_stagnation_recovery(): provides specific, actionable recovery strategies
  based on the detected trap
- track_indicator_diversity(): monitors indicator family usage across turns
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np


def compute_stagnation(history: list[dict]) -> tuple[float, str]:
    """Return (score: 0.0-1.0, trend: "improving"|"flat"|"declining").

    Score 0.0 = great progress, 1.0 = completely stuck.

    Factors (weighted):
      0.4  Sharpe trend  (linear-regression slope)
      0.3  Variance      (std of last-3 Sharpes)
      0.3  Repetition    (unique actions in last-3 turns)
    """
    if len(history) < 2:
        return (0.0, "improving")

    # Only consider entries that have a sharpe value (skip failed turns)
    sharpes = [h["sharpe"] for h in history if "sharpe" in h]

    if len(sharpes) < 2:
        return (0.0, "improving")

    # Factor 1: Sharpe trend (linear regression slope)
    slope = float(np.polyfit(range(len(sharpes)), sharpes, 1)[0])
    if slope < -0.1:
        trend_score, trend = 1.0, "declining"
    elif slope < 0.0:
        trend_score, trend = 0.8, "flat"
    else:
        trend_score, trend = 0.1, "improving"

    # Factor 2: Variance (oscillation = bad)
    variance = float(np.var(sharpes)) if len(sharpes) >= 3 else 0.0
    if variance > 0.08:
        variance_score = 1.0
    elif variance > 0.04:
        variance_score = 0.8
    elif variance > 0.02:
        variance_score = 0.5
    else:
        variance_score = 0.0

    # Factor 3: Action repetition
    actions = [h.get("action", "") for h in history[-3:]]
    if len(actions) >= 3 and all(a == "modify_params" for a in actions):
        repetition_score = 1.0
    elif len(actions) >= 2 and actions.count("modify_params") >= 2:
        repetition_score = 0.7
    else:
        repetition_score = 0.0

    # Weighted combination
    score = 0.4 * trend_score + 0.2 * variance_score + 0.4 * repetition_score
    return (round(score, 3), trend)


_FAILURE_ACTION_MISMATCHES: dict[tuple[str, str], str] = {
    ("too_few_trades", "change_exit_logic"): "Tightening exits won't create more trades",
    ("excessive_drawdown", "modify_params"): "Tweaks rarely fix drawdown",
}


def check_hypothesis_failure_alignment(
    failure_mode: str,
    addresses_failure: str,
    action: str,
) -> list[str]:
    """Warn when the LLM's proposed action is a poor fit for the diagnosed failure."""
    warnings: list[str] = []
    if failure_mode != addresses_failure:
        warnings.append(
            f"Diagnosed '{failure_mode}' but change addresses '{addresses_failure}'."
        )
    key = (failure_mode, action)
    if key in _FAILURE_ACTION_MISMATCHES:
        warnings.append(_FAILURE_ACTION_MISMATCHES[key])
    return warnings


def get_stagnation_response(iteration: int, score: float) -> dict:
    """Return prompt adjustments based on stagnation level.

    Order matters: abandon > pivot > nuclear > broaden > normal.
    """
    if iteration <= 3:
        return {"constraint": "normal", "prompt_suffix": ""}

    if score > 0.8:
        return {
            "constraint": "abandon",
            "prompt_suffix": "ABANDON: This strategy cannot converge. Archive it.",
        }

    if score > 0.7:
        return {
            "constraint": "pivot",
            "prompt_suffix": (
                "PIVOT: Your incremental changes are not working. Try adding a "
                "regime filter, switching entry signal archetype, or changing the "
                "exit logic entirely."
            ),
        }

    if score > 0.6 and iteration >= 6:
        return {
            "constraint": "nuclear",
            "prompt_suffix": (
                "NUCLEAR REWRITE: You must use a completely different indicator family "
                "and signal archetype. Do NOT modify existing parameters."
            ),
        }

    if score > 0.5:
        return {
            "constraint": "broaden",
            "prompt_suffix": (
                "BROADEN: You may change entry AND exit logic simultaneously. "
                "Consider structural changes, not just parameter tweaks."
            ),
        }

    return {"constraint": "normal", "prompt_suffix": ""}


# ── Phase 5.6: Stagnation Recovery System ─────────────────────────────────


# Indicator family classification — maps indicator function names to families
_INDICATOR_FAMILIES: dict[str, list[str]] = {
    "momentum": ["ema", "sma", "wma", "macd", "roc", "tsi", "dpo"],
    "mean_reversion": ["rsi", "bbands", "bollinger", "stoch", "cci", "willr"],
    "volatility": ["atr", "adx", "supertrend", "kelner", "keltner", "true_range"],
    "volume": ["obv", "vwap", "ad", "cmf", "mfi"],
    "trend": ["ichimoku", "dema", "tema", "psar"],
}

# Exact-match overrides: longer/more-specific names first
_EXACT_OVERRIDES: dict[str, str] = {
    "dema": "trend",
    "tema": "trend",
    "ichimoku": "trend",
    "psar": "trend",
    "supertrend": "volatility",
    "bbands": "mean_reversion",
    "bollinger": "mean_reversion",
}


def classify_indicator(indicator_name: str) -> str:
    """Classify an indicator name into its family.

    Args:
        indicator_name: Indicator function name (e.g., 'ema', 'rsi', 'atr').

    Returns:
        Family name string, or 'unknown' if not recognized.
    """
    lower = indicator_name.lower()

    # Check exact overrides first
    if lower in _EXACT_OVERRIDES:
        return _EXACT_OVERRIDES[lower]

    # Then check substring matches
    for family, members in _INDICATOR_FAMILIES.items():
        for member in members:
            if member in lower:
                return family
    return "unknown"


def extract_indicators_from_code(code: str) -> list[str]:
    """Extract indicator function names from strategy source code.

    Looks for patterns like:
    - cached_indicator("ema", ...)
    - cached_indicator('rsi', ...)

    Args:
        code: Strategy source code string.

    Returns:
        List of unique indicator function names found.
    """
    # Match cached_indicator("name", ...) or cached_indicator('name', ...)
    cached = re.findall(r'cached_indicator\s*\(\s*[\'"](\w+)', code)
    # Match ta.name(...) or pandas_ta.name(...)
    direct = re.findall(r'(?:ta|pandas_ta)\.(\w+)\s*\(', code)
    return list(dict.fromkeys(cached + direct))  # dedup preserving order


def track_indicator_diversity(
    history: list[dict],
    current_code: str = "",
) -> dict:
    """Analyze indicator diversity across turns to detect ruts.

    Counts how many different indicator families have been used and whether
    the LLM is stuck in a single family.

    Args:
        history: Turn history list with 'code_path' or 'params_used' fields.
        current_code: Current strategy source code (optional, for latest turn).

    Returns:
        Dict with:
        - families_used: set of indicator family names used
        - family_counts: dict mapping family name to usage count
        - dominant_family: the most-used family (or None)
        - is_rut: True if 80%+ of turns use the same family
        - all_indicators: list of all unique indicators seen
        - recovery_hint: suggested action if in a rut
    """
    family_counts: dict[str, int] = {}
    all_indicators: list[str] = []

    # Extract indicators from history code
    for entry in history:
        code = ""
        # Try to get code from code_path
        code_path = entry.get("code_path", "")
        if code_path:
            try:
                from pathlib import Path
                p = Path(code_path)
                if p.exists():
                    code = p.read_text()
            except (OSError, TypeError):
                pass

        # Fallback: extract from params (some params hint at indicators)
        if not code:
            params = entry.get("params_used", {})
            all_indicators.extend(
                k for k in params.keys()
                if any(fam_member in k.lower() for fam in _INDICATOR_FAMILIES.values()
                       for fam_member in fam)
            )
            continue

        indicators = extract_indicators_from_code(code)
        all_indicators.extend(indicators)
        for ind in indicators:
            family = classify_indicator(ind)
            if family != "unknown":
                family_counts[family] = family_counts.get(family, 0) + 1

    # Add current code indicators
    if current_code:
        indicators = extract_indicators_from_code(current_code)
        all_indicators.extend(indicators)
        for ind in indicators:
            family = classify_indicator(ind)
            if family != "unknown":
                family_counts[family] = family_counts.get(family, 0) + 1

    # Deduplicate indicators
    all_indicators = list(dict.fromkeys(all_indicators))

    total_family_uses = sum(family_counts.values())
    dominant_family: Optional[str] = None
    is_rut = False
    recovery_hint = ""

    if total_family_uses > 0 and family_counts:
        dominant_family = max(family_counts, key=family_counts.get)
        dominant_pct = family_counts[dominant_family] / total_family_uses
        is_rut = dominant_pct >= 0.8 and total_family_uses >= 3

        if is_rut:
            # Suggest alternative families
            alternatives = [f for f in _INDICATOR_FAMILIES if f != dominant_family]
            alt_hint = ", ".join(alternatives[:3])
            recovery_hint = (
                f"INDICATOR RUT: {dominant_pct:.0%} of turns use '{dominant_family}' "
                f"indicators. Switch to a DIFFERENT family: {alt_hint}. "
                f"Do NOT use any {dominant_family} indicators on the next turn."
            )

    return {
        "families_used": set(family_counts.keys()),
        "family_counts": family_counts,
        "dominant_family": dominant_family,
        "is_rut": is_rut,
        "all_indicators": all_indicators,
        "recovery_hint": recovery_hint,
    }


def detect_stagnation_trap(
    history: list[dict],
    best_sharpe: float = 0.0,
    sharpe_target: float = 1.5,
) -> dict:
    """Identify the specific type of stagnation trap the LLM is in.

    Instead of a single stagnation score, this identifies the SPECIFIC problem
    so the recovery can be targeted.

    Traps:
    - "zero_sharpe": All turns produced Sharpe <= 0 (completely lost)
    - "low_sharpe_plateau": Sharpe stuck in 0.0-0.3 range (wrong approach)
    - "mid_sharpe_trap": Sharpe stuck in 0.3-0.7 (needs refinement, not invention)
    - "high_sharpe_few_trades": Sharpe > target but < 20 trades (trade count trap)
    - "near_target": Sharpe 0.7-1.0 with decent trades (close, needs filter tuning)
    - "validation_loop": Keeps hitting target but failing validation (overfit)
    - "action_loop": Same action type repeated 3+ times (tweak loop)
    - "indicator_rut": Stuck using same indicator family
    - "no_trap": Making progress or too few turns to diagnose

    Args:
        history: Turn history list.
        best_sharpe: Best Sharpe achieved so far.
        sharpe_target: Target Sharpe ratio.

    Returns:
        Dict with trap, severity, description, and supporting data.
    """
    sharpes = [h.get("sharpe", 0.0) for h in history if "sharpe" in h]
    failure_modes = [h.get("failure_mode", "") for h in history[-5:]]
    actions = [h.get("action", "") for h in history[-5:]]

    if len(sharpes) < 2:
        return {
            "trap": "no_trap",
            "severity": "low",
            "sharpes": sharpes,
            "recent_failure_modes": failure_modes,
            "recent_actions": actions,
            "turns_in_trap": 0,
            "description": "Too few turns to diagnose stagnation.",
        }

    recent_sharpes = sharpes[-3:]
    avg_recent = sum(recent_sharpes) / len(recent_sharpes)

    # Count specific conditions
    turns_at_zero = sum(1 for s in sharpes if s <= 0)
    turns_high_few = sum(
        1 for h in history
        if h.get("sharpe", 0) >= sharpe_target * 0.8
        and h.get("num_trades", 100) < 20
    )
    turns_validation_fail = sum(
        1 for fm in failure_modes if fm == "validation_failed"
    )

    # Detect action loop (same action 3+ times)
    action_loop = False
    loop_action = ""
    most_common_count = 0
    if len(actions) >= 3:
        from collections import Counter
        action_counts = Counter(actions)
        most_common_action, most_common_count = action_counts.most_common(1)[0]
        if most_common_count >= 3 and most_common_action:
            action_loop = True
            loop_action = most_common_action

    # Detect indicator rut
    indicator_rut = False
    dominant_family = ""
    if len(history) >= 3:
        diversity = track_indicator_diversity(history)
        if diversity["is_rut"]:
            indicator_rut = True
            dominant_family = diversity["dominant_family"] or ""

    # Classify the trap (order matters — more specific first)
    trap = "no_trap"
    severity = "low"
    description = ""
    turns_in_trap = 0

    if turns_at_zero >= len(sharpes) - 1 and len(sharpes) >= 3:
        trap = "zero_sharpe"
        severity = "critical"
        turns_in_trap = turns_at_zero
        description = (
            "ALL recent turns produced Sharpe <= 0. The LLM is generating "
            "strategies that lose money."
        )

    elif turns_validation_fail >= 2 and best_sharpe >= sharpe_target * 0.8:
        trap = "validation_loop"
        severity = "high"
        turns_in_trap = turns_validation_fail
        description = (
            f"Sharpe reaches target ({best_sharpe:.2f}) but fails out-of-sample "
            f"validation {turns_validation_fail} times."
        )

    elif turns_high_few >= 2 and best_sharpe >= sharpe_target * 0.8:
        trap = "high_sharpe_few_trades"
        severity = "high"
        turns_in_trap = turns_high_few
        description = (
            f"Best Sharpe {best_sharpe:.2f} but only <20 trades. "
            f"The strategy is curve-fit to rare events."
        )

    elif indicator_rut:
        trap = "indicator_rut"
        severity = "medium"
        turns_in_trap = len(sharpes)
        description = (
            f"Stuck using '{dominant_family}' indicators on every turn."
        )

    elif action_loop:
        trap = "action_loop"
        severity = "medium"
        turns_in_trap = most_common_count
        description = (
            f"Action '{loop_action}' repeated {most_common_count} times."
        )

    elif avg_recent <= 0.3 and len(sharpes) >= 3:
        trap = "low_sharpe_plateau"
        severity = "high"
        turns_in_trap = sum(1 for s in reversed(sharpes) if s <= 0.3)
        description = (
            f"Sharpe stuck at {avg_recent:.2f}. The current approach isn't working."
        )

    elif avg_recent <= 0.7 and len(sharpes) >= 3 and best_sharpe < sharpe_target * 0.8:
        trap = "mid_sharpe_trap"
        severity = "medium"
        turns_in_trap = sum(1 for s in reversed(sharpes) if 0.3 < s <= 0.7)
        description = (
            f"Sharpe plateaued at {avg_recent:.2f}. Getting some signal but not enough."
        )

    elif best_sharpe >= sharpe_target * 0.7 and best_sharpe < sharpe_target:
        trap = "near_target"
        severity = "low"
        turns_in_trap = sum(1 for s in reversed(sharpes) if s >= sharpe_target * 0.5)
        description = (
            f"Close to target (best {best_sharpe:.2f} vs {sharpe_target})."
        )

    return {
        "trap": trap,
        "severity": severity,
        "sharpes": sharpes,
        "recent_failure_modes": failure_modes,
        "recent_actions": actions,
        "turns_in_trap": turns_in_trap,
        "description": description,
    }


def build_stagnation_recovery(trap_info: dict) -> str:
    """Build a targeted recovery instruction for the LLM based on the detected trap.

    This is MORE SPECIFIC than the generic stagnation suffix — it tells the LLM
    exactly WHAT to change and HOW.

    Args:
        trap_info: Dict from detect_stagnation_trap().

    Returns:
        Formatted recovery instruction string for prompt injection.
        Empty string if no trap or trap is "no_trap".
    """
    trap = trap_info.get("trap", "no_trap")
    severity = trap_info.get("severity", "low")
    turns_in_trap = trap_info.get("turns_in_trap", 0)

    if trap == "no_trap" or severity == "low":
        return ""

    recovery_map = {
        "zero_sharpe": (
            "## 🚨 STAGNATION RECOVERY: Zero Sharpe\n\n"
            "Your strategies are ALL losing money. Stop tweaking — the approach is wrong.\n\n"
            "**MANDATORY CHANGES for your next turn:**\n"
            "1. Switch to a COMPLETELY DIFFERENT indicator family\n"
            "2. Use the SIMPLEST possible signal (single indicator crossover)\n"
            "3. Do NOT add filters, conditions, or complexity\n"
            "4. Good starting points: EMA crossover, RSI < 30 oversold, MACD histogram flip\n"
            "5. Make sure your exits are simple: time stop or opposite signal\n\n"
            "Remember: a simple strategy that breaks even is better than a complex one that loses."
        ),
        "low_sharpe_plateau": (
            "## 🚨 STAGNATION RECOVERY: Low Sharpe Plateau\n\n"
            f"Sharpe has been below 0.3 for {turns_in_trap}+ turns. "
            "The current indicator family isn't capturing any edge.\n\n"
            "**MANDATORY CHANGES for your next turn:**\n"
            "1. ABANDON your current indicator approach entirely\n"
            "2. Pick a different indicator family:\n"
            "   - If using momentum (EMA/MACD/ROC) → try mean_reversion (RSI/Bollinger)\n"
            "   - If using mean_reversion → try volatility (ATR/Supertrend)\n"
            "   - If using volatility → try volume (OBV/VWAP)\n"
            "3. Use a SIMPLE signal with WIDE thresholds\n"
            "4. Aim for 30+ trades — frequency matters more than precision"
        ),
        "mid_sharpe_trap": (
            "## ⚠️ STAGNATION RECOVERY: Mid Sharpe Trap\n\n"
            f"Sharpe stuck around 0.3-0.7 for {turns_in_trap}+ turns. "
            "You're capturing some signal but not enough.\n\n"
            "**RECOMMENDED CHANGES:**\n"
            "1. Add a TREND FILTER — only trade in the direction of the dominant trend\n"
            "   (e.g., long-only when price > 200-SMA)\n"
            "2. Try different indicator PARAMETERS, not different indicators\n"
            "3. Consider combining two complementary indicators from DIFFERENT families\n"
            "4. Widen your exit conditions — you might be exiting winners too early"
        ),
        "high_sharpe_few_trades": (
            "## 🚨 STAGNATION RECOVERY: Trade Count Trap\n\n"
            "Your Sharpe is high but trade count is below 20. This is CURVE-FITTING, "
            "not a real edge. A strategy that trades 5 times with Sharpe 2.0 is WORSE "
            "than one that trades 40 times with Sharpe 0.8.\n\n"
            "**MANDATORY CHANGES for your next turn:**\n"
            "1. REMOVE at least one condition from your entry logic\n"
            "2. WIDEN all thresholds by 50% (e.g., RSI < 20 → RSI < 30)\n"
            "3. REDUCE the number of indicators to 1-2 maximum\n"
            "4. Your goal: 30+ trades. If you can't achieve this, SIMPLIFY further\n"
            "5. Do NOT try to maintain high Sharpe — trade frequency is the priority"
        ),
        "validation_loop": (
            "## 🚨 STAGNATION RECOVERY: Validation Failure Loop\n\n"
            f"Your strategy passes in-sample but fails out-of-sample {turns_in_trap}+ times. "
            "This is classic overfitting — the strategy learned the training data, not a real pattern.\n\n"
            "**MANDATORY CHANGES for your next turn:**\n"
            "1. CUT complexity in HALF — if you have 4 conditions, use 2\n"
            "2. WIDEN all thresholds by 50%\n"
            "3. Add a TIME STOP — exit after N bars regardless of signal\n"
            "4. Remove the LEAST important indicator\n"
            "5. A strategy that works in 4/6 windows at Sharpe 0.8 is better than "
            "   one that works in 1/6 windows at Sharpe 2.0"
        ),
        "action_loop": (
            "## ⚠️ STAGNATION RECOVERY: Action Loop\n\n"
            f"You've been doing the same type of change for {turns_in_trap}+ turns. "
            "It's not working.\n\n"
            "**MANDATORY CHANGE:** Try a DIFFERENT action type:\n"
            "- If you've been doing 'modify_params' → try 'replace_indicator'\n"
            "- If you've been doing 'change_entry_logic' → try 'add_filter'\n"
            "- If you've been doing 'add_filter' → try 'full_rewrite'\n"
            "- If you've been doing anything → try 'novel' (start from scratch)"
        ),
    }

    # Build indicator_rut recovery dynamically (needs dominant_family from description)
    if trap == "indicator_rut":
        dominant_family = "unknown"
        history_hint = trap_info.get("description", "")
        match = re.search(r"'(\w+)'", history_hint)
        if match:
            dominant_family = match.group(1)
        recovery_map["indicator_rut"] = (
            "## ⚠️ STAGNATION RECOVERY: Indicator Rut\n\n"
            f"You've been using '{dominant_family}' indicators "
            f"on every turn. Time to diversify.\n\n"
            "**MANDATORY CHANGE:** Use a DIFFERENT indicator family:\n"
            "- Momentum (EMA, MACD, ROC) → try RSI + Bollinger Bands\n"
            "- Mean Reversion (RSI, Stoch) → try ATR + Supertrend\n"
            "- Volatility (ATR, ADX) → try OBV + Volume SMA\n"
            "- Volume (OBV, VWAP) → try EMA crossover + MACD\n\n"
            "Do NOT use any indicators from your current family on the next turn."
        )

    return recovery_map.get(trap, "")


# ── Phase 5.6: Mandate-Aware Forced Exploration on Plateau ─────────────────


def _classify_turn_family(turn: dict) -> str:
    """Classify a single turn's dominant indicator family.

    Inspects the turn dict for a ``code`` field (source code string),
    a ``code_path`` field (path to read), or ``params_used`` keys to
    determine the primary indicator family used in that turn.

    Returns the family name or ``"unknown"``.
    """
    # 1) Inline source code
    code = turn.get("code", "")
    if not code:
        # 2) File path
        code_path = turn.get("code_path", "")
        if code_path:
            try:
                from pathlib import Path

                p = Path(code_path)
                if p.exists():
                    code = p.read_text()
            except (OSError, TypeError):
                pass

    if code:
        indicators = extract_indicators_from_code(code)
        if indicators:
            families: list[str] = [
                classify_indicator(ind) for ind in indicators
                if classify_indicator(ind) != "unknown"
            ]
            if families:
                # Return the most common family
                from collections import Counter

                return Counter(families).most_common(1)[0][0]

    # 3) Fallback: infer from params_used keys
    params = turn.get("params_used", {})
    if params:
        fam_counts: dict[str, int] = {}
        for key in params:
            key_lower = key.lower()
            for family, members in _INDICATOR_FAMILIES.items():
                for member in members:
                    if member in key_lower:
                        fam_counts[family] = fam_counts.get(family, 0) + 1
        if fam_counts:
            return max(fam_counts, key=fam_counts.get)

    return "unknown"


def check_family_plateau(
    turn_history: list[dict],
    mandate: dict,
    max_same_family: int = 3,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Check if the LLM is stuck in a strategy family rut.

    Detects when the last *max_same_family* turns all used the same
    indicator family **and** none of them had a ``"KEEP"`` status.  When
    triggered, returns a mandate-aware pivot directive:

    * **within** — the stuck family matches the mandate's
      ``strategy_archetype`` *and* ``force_diversify`` is not set.  The
      LLM is told to try different indicators *within* the same family.
    * **cross** — ``force_diversify`` is ``True`` or the stuck family
      differs from the mandate archetype.  The LLM must switch to a
      completely different family.

    Args:
        turn_history: List of turn dicts (each with ``status``, ``code``,
            ``code_path``, or ``params_used`` fields).
        mandate: Mandate dict with ``strategy_archetype`` and optionally
            ``force_diversify`` keys.
        max_same_family: Number of consecutive same-family turns that
            trigger a pivot.

    Returns:
        ``(should_pivot, pivot_type, message)`` where *pivot_type* is
        ``"within"`` or ``"cross"``, or ``None`` when no pivot is needed.
    """
    if len(turn_history) < max_same_family:
        return False, None, None

    recent = turn_history[-max_same_family:]
    families = [_classify_turn_family(t) for t in recent]
    statuses = [t.get("status", "") for t in recent]

    # All same family AND none are KEEP
    if not (len(set(families)) == 1 and families[0] != "unknown"
            and all(s.upper() != "KEEP" for s in statuses)):
        return False, None, None

    stuck_family = families[0]
    mandate_arch = mandate.get("strategy_archetype", "")
    allow_cross = mandate.get("force_diversify", False)

    if stuck_family == mandate_arch and not allow_cross:
        # Within-archetype: different indicators / logic / timeframe
        return True, "within", (
            f"STUCK: All {max_same_family} turns used {stuck_family} "
            f"indicators with no progress.\n"
            f"You MUST use a completely different set of {stuck_family} "
            f"indicators and logic structure.\n"
            f"Try a different approach within {stuck_family}: "
            f"if using MACD try ROC/TSI, if using single timeframe "
            f"try multi-timeframe, if using entry signals try a "
            f"filter-based approach.\n"
            f"Do NOT repeat your previous indicator combination."
        )

    # Cross-archetype: completely different family
    alternatives = [f for f in _INDICATOR_FAMILIES if f != stuck_family]
    alt_hint = ", ".join(alternatives[:3])
    return True, "cross", (
        f"STUCK: All {max_same_family} turns used {stuck_family} "
        f"strategies with no progress.\n"
        f"You must switch to a DIFFERENT strategy family entirely: "
        f"{alt_hint}.\n"
        f"Do NOT use any {stuck_family} indicators on the next turn."
    )
