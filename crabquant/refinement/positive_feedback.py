"""
CrabQuant Refinement Pipeline — Positive Feedback Analyzer

Complements the failure guidance system by telling the LLM what's WORKING
in the current strategy. This prevents the LLM from throwing away good
components when making modifications.

The key insight: failure guidance says "fix X" but doesn't say "keep Y".
When the LLM gets low_sharpe feedback, it often over-corrects and destroys
whatever was generating positive returns. This module identifies the
strategy's strengths so the LLM preserves them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PositiveFeedback:
    """Analysis of what's working well in a strategy.

    Attributes:
        strengths: List of identified strengths with explanations.
        preserve_warnings: List of specific components to NOT change.
        overall_assessment: One-paragraph summary of the strategy's positive aspects.
        regression_risk: Warning about what could go wrong if changes are too aggressive.
    """
    strengths: list[str]
    preserve_warnings: list[str]
    overall_assessment: str
    regression_risk: str


def analyze_positive_feedback(
    *,
    sharpe_ratio: float = 0.0,
    sharpe_target: float = 1.5,
    total_return_pct: float = 0.0,
    max_drawdown_pct: float = 0.0,
    win_rate: float = 0.0,
    profit_factor: float = 0.0,
    sortino_ratio: float = 0.0,
    calmar_ratio: float = 0.0,
    total_trades: int = 0,
    avg_holding_bars: Optional[float] = None,
    sharpe_by_year: Optional[dict] = None,
    failure_mode: str = "",
) -> PositiveFeedback:
    """Analyze backtest metrics to identify what's working.

    This is the complement to failure guidance — instead of saying "fix X",
    we say "your Y is good, keep it". This prevents regression when the LLM
    makes modifications.

    Args:
        sharpe_ratio: Current Sharpe ratio.
        sharpe_target: Target Sharpe ratio.
        total_return_pct: Total return as fraction (0.15 = 15%).
        max_drawdown_pct: Max drawdown as negative fraction (-0.15 = -15%).
        win_rate: Win rate as fraction (0.55 = 55%).
        profit_factor: Profit factor (gross profits / gross losses).
        sortino_ratio: Sortino ratio (downside-deviation-adjusted return).
        calmar_ratio: Calmar ratio (return / max drawdown).
        total_trades: Number of trades.
        avg_holding_bars: Average holding period in bars.
        sharpe_by_year: Per-year Sharpe ratios.
        failure_mode: The failure mode that was classified.

    Returns:
        PositiveFeedback with identified strengths and preservation advice.
    """
    strengths: list[str] = []
    preserve_warnings: list[str] = []

    # 1. Positive returns (even if below target)
    if total_return_pct > 0.10:
        strengths.append(
            f"**Positive returns** ({total_return_pct:.1%}) — the strategy has real alpha. "
            f"The entry/exit logic captures genuine market inefficiency."
        )
        preserve_warnings.append(
            "PRESERVE: The core entry/exit logic generates positive returns. "
            "Only modify to increase frequency or reduce drawdown, not to change the signal."
        )
    elif total_return_pct > 0:
        strengths.append(
            f"**Modest positive returns** ({total_return_pct:.1%}) — directionally correct. "
            f"Needs amplification, not replacement."
        )

    # 2. Good Sharpe (relative to target)
    if sharpe_ratio >= sharpe_target * 0.5 and sharpe_target > 0:
        pct_of_target = sharpe_ratio / sharpe_target * 100
        strengths.append(
            f"**Sharpe {sharpe_ratio:.2f} is {pct_of_target:.0f}% of target** — "
            f"the risk-adjusted return profile is partially there. "
            f"Small tweaks may close the gap."
        )
    if sharpe_ratio > 0.5:
        preserve_warnings.append(
            f"PRESERVE: Sharpe {sharpe_ratio:.2f} shows the strategy has genuine edge. "
            f"Avoid drastic changes to the signal generation logic."
        )

    # 3. Win rate analysis
    if 0.45 <= win_rate <= 0.70:
        strengths.append(
            f"**Healthy win rate** ({win_rate:.1%}) — in the optimal 45-70% range. "
            f"This is sustainable and not curve-fit."
        )
        preserve_warnings.append(
            "PRESERVE: Win rate is in the sweet spot. Don't tighten stops (would lower it) "
            "or widen targets (would lower it)."
        )
    elif win_rate > 0.70:
        strengths.append(
            f"**High win rate** ({win_rate:.1%}) — excellent but verify it's not "
            f"due to tiny winners offset by rare large losers (check profit factor)."
        )
    elif win_rate > 0.35 and win_rate < 0.45:
        strengths.append(
            f"**Acceptable win rate** ({win_rate:.1%}) — on the low end but workable "
            f"if profit factor compensates."
        )

    # 4. Profit factor
    if profit_factor > 1.5:
        strengths.append(
            f"**Strong profit factor** ({profit_factor:.2f}) — winners significantly "
            f"outweigh losers. The reward/risk balance is good."
        )
        preserve_warnings.append(
            "PRESERVE: Profit factor is strong. Don't add tight stop losses that would "
            "cut winners short."
        )
    elif profit_factor > 1.0:
        strengths.append(
            f"**Positive profit factor** ({profit_factor:.2f}) — the strategy makes "
            f"more than it loses, just needs amplification."
        )

    # 5. Sortino ratio (downside risk)
    if sortino_ratio > 1.0:
        strengths.append(
            f"**Good Sortino ratio** ({sortino_ratio:.2f}) — downside risk is well-controlled. "
            f"The strategy doesn't have catastrophic losing periods."
        )
        preserve_warnings.append(
            "PRESERVE: Downside protection is working. Don't remove any volatility filters "
            "or stop-loss mechanisms."
        )

    # 6. Drawdown control
    if max_drawdown_pct < 0 and max_drawdown_pct > -0.15:
        strengths.append(
            f"**Controlled drawdown** ({max_drawdown_pct:.1%}) — risk management is solid. "
            f"Below the 15% danger zone."
        )
        preserve_warnings.append(
            "PRESERVE: Drawdown is well-controlled. Don't remove existing risk management."
        )
    elif max_drawdown_pct <= -0.15 and max_drawdown_pct > -0.25:
        strengths.append(
            f"**Moderate drawdown** ({max_drawdown_pct:.1%}) — acceptable for the "
            f"return profile. Could improve but not critical."
        )

    # 7. Trade frequency
    if 20 <= total_trades <= 100:
        strengths.append(
            f"**Good trade frequency** ({total_trades} trades) — statistically significant "
            f"sample size, not overtrading."
        )
    elif 10 <= total_trades < 20:
        strengths.append(
            f"**Adequate trade frequency** ({total_trades} trades) — approaching the "
            f"minimum for reliable statistics. Try to increase slightly."
        )
    elif total_trades > 100:
        strengths.append(
            f"**High trade frequency** ({total_trades} trades) — the strategy fires "
            f"often. Check if transaction costs are eating into returns."
        )

    # 8. Holding period
    if avg_holding_bars is not None:
        if 3 <= avg_holding_bars <= 30:
            strengths.append(
                f"**Reasonable holding period** ({avg_holding_bars:.0f} bars avg) — "
                f"not too short (costs) and not too long (capital tie-up)."
            )
        elif avg_holding_bars > 30:
            strengths.append(
                f"**Longer holding period** ({avg_holding_bars:.0f} bars avg) — "
                f"captures larger moves but may miss quicker reversals."
            )

    # 9. Year-over-year consistency
    if sharpe_by_year and len(sharpe_by_year) >= 2:
        values = list(sharpe_by_year.values())
        positive_years = sum(1 for v in values if v > 0)
        total_years = len(values)

        if positive_years == total_years:
            strengths.append(
                f"**Profitable every year** ({total_years}/{total_years}) — "
                f"exceptional consistency across all market conditions."
            )
            preserve_warnings.append(
                "PRESERVE: The strategy works in ALL years. This is rare and valuable. "
                "Make minimal changes to preserve this consistency."
            )
        elif positive_years >= total_years * 0.7:
            strengths.append(
                f"**Profitable in most years** ({positive_years}/{total_years}) — "
                f"good consistency. The losing years may need investigation but "
                f"the core logic is sound."
            )
        elif positive_years >= 1:
            # Find the best year
            best_year = max(sharpe_by_year, key=sharpe_by_year.get)
            best_val = sharpe_by_year[best_year]
            strengths.append(
                f"**Best year: {best_year}** (Sharpe {best_val:.2f}) — "
                f"the strategy has genuine edge in some conditions. "
                f"Focus on making it work in more conditions rather than replacing it."
            )

    # 10. Calmar ratio
    if calmar_ratio > 1.0:
        strengths.append(
            f"**Good Calmar ratio** ({calmar_ratio:.2f}) — strong return relative "
            f"to drawdown risk."
        )

    # Build overall assessment
    if len(strengths) >= 3:
        overall = (
            f"This strategy has **{len(strengths)} positive attributes**. "
            f"The core logic has genuine merit — it needs refinement, not replacement. "
            f"Focus your changes on the specific failure diagnosed above while preserving "
            f"the strengths listed here."
        )
    elif len(strengths) >= 1:
        overall = (
            f"This strategy has **{len(strengths)} positive attribute(s)**. "
            f"While it didn't hit the target, there are elements worth preserving. "
            f"Target your changes at the specific failure mode."
        )
    else:
        overall = (
            "This strategy has no clearly positive attributes yet. "
            "Consider a more significant restructuring or try a different indicator family."
        )

    # Build regression risk warning
    regression_risk = _build_regression_risk(
        failure_mode=failure_mode,
        has_positive_returns=total_return_pct > 0,
        has_good_win_rate=0.45 <= win_rate <= 0.70,
        has_profit_factor=profit_factor > 1.2,
        sharpe_ratio=sharpe_ratio,
        preserve_warnings=preserve_warnings,
    )

    return PositiveFeedback(
        strengths=strengths,
        preserve_warnings=preserve_warnings,
        overall_assessment=overall,
        regression_risk=regression_risk,
    )


def _build_regression_risk(
    *,
    failure_mode: str,
    has_positive_returns: bool,
    has_good_win_rate: bool,
    has_profit_factor: bool,
    sharpe_ratio: float,
    preserve_warnings: list[str],
) -> str:
    """Build a warning about regression risk based on failure mode and strengths."""
    risks: list[str] = []

    if failure_mode == "low_sharpe" and has_positive_returns:
        risks.append(
            "⚠️ REGRESSION RISK: Your strategy HAS positive returns but Sharpe is below target. "
            "This means the risk-adjusted return needs improvement — NOT the raw signal. "
            "Do NOT change the entry/exit logic drastically. Instead:\n"
            "  - Add a trend filter to avoid bad trades (doesn't change the signal)\n"
            "  - Adjust position sizing or add a time stop\n"
            "  - Tighten stops to reduce variance (without killing win rate)"
        )

    if failure_mode == "regime_fragility" and has_positive_returns:
        risks.append(
            "⚠️ REGRESSION RISK: Your strategy works in SOME years but not others. "
            "Don't throw away the working version — ADD a regime gate on top:\n"
            "  - Detect the regime (e.g., ADX > 25 for trending)\n"
            "  - Only take signals when the regime matches\n"
            "  - This preserves the good years while skipping the bad ones"
        )

    if failure_mode == "too_few_trades" and has_good_win_rate:
        risks.append(
            "⚠️ REGRESSION RISK: Your win rate is good but you need more trades. "
            "Don't change the entry logic quality — LOOSEN it:\n"
            "  - Widen thresholds (RSI < 30 → RSI < 40)\n"
            "  - Remove one condition from multi-condition entries\n"
            "  - Shorten indicator periods (EMA 20 → EMA 10)"
        )

    if failure_mode == "too_few_trades" and has_profit_factor:
        risks.append(
            "⚠️ REGRESSION RISK: Your profit factor is good — the trades you take ARE profitable. "
            "You just need MORE of them. Widen entry thresholds, don't change the exit logic."
        )

    if failure_mode == "excessive_drawdown" and sharpe_ratio > 0.5:
        risks.append(
            "⚠️ REGRESSION RISK: Sharpe is decent but drawdown is too high. "
            "The signal works — you just need better risk management:\n"
            "  - Add ATR-based stop loss (exit when loss > 2× ATR)\n"
            "  - Add volatility filter (skip trades when ATR is high)\n"
            "  - Do NOT change the entry signal itself"
        )

    if failure_mode == "validation_failed" and sharpe_ratio > 0.8:
        risks.append(
            "⚠️ REGRESSION RISK: In-sample Sharpe is good ({:.2f}) but out-of-sample failed. "
            "This means you're OVERFIT, not wrong. Simplify:\n"
            "  - Remove the most recently added indicator/condition\n"
            "  - Widen all thresholds by 20-30%\n"
            "  - Reduce parameter count".format(sharpe_ratio)
        )

    # If no specific risk identified but there are preserve warnings
    if not risks and preserve_warnings:
        risks.append(
            "⚠️ REGRESSION RISK: This strategy has identified strengths above. "
            "Make TARGETED changes to address the failure mode — do NOT rewrite the "
            "entire strategy."
        )

    return "\n\n".join(risks)


def format_positive_feedback_for_prompt(feedback: PositiveFeedback) -> str:
    """Format positive feedback for injection into the LLM prompt.

    Args:
        feedback: PositiveFeedback from analyze_positive_feedback().

    Returns:
        Formatted string for prompt injection, or empty string if no strengths.
    """
    if not feedback.strengths:
        return ""

    parts: list[str] = []
    parts.append("### ✅ What's Working (PRESERVE these)")
    parts.append("")

    for strength in feedback.strengths:
        parts.append(f"- {strength}")

    if feedback.preserve_warnings:
        parts.append("")
        parts.append("**Preservation Rules:**")
        for warning in feedback.preserve_warnings:
            parts.append(f"- {warning}")

    parts.append("")
    parts.append(feedback.overall_assessment)

    if feedback.regression_risk:
        parts.append("")
        parts.append(feedback.regression_risk)

    return "\n".join(parts)
