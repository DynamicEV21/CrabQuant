"""
Sharpe Root Cause Analyzer — Phase 6

Analyzes backtest metrics to diagnose WHY a strategy has low Sharpe
and provides specific, actionable guidance for the LLM to fix it.

This is the key feedback loop improvement: instead of telling the LLM
"your Sharpe is low, try different things," we tell it:
"Your win rate is 35% — your entries are too noisy. Add a trend filter
or widen your thresholds to avoid false breakouts."

The function uses metrics already available in BacktestReport:
- win_rate, profit_factor, sortino_ratio, calmar_ratio
- total_return, max_drawdown, sharpe_by_year
- total_trades, avg_holding_bars (if available)
"""

from __future__ import annotations

from typing import Callable, Optional

# ── Metrics bundle for passing to individual checks ─────────────────────────

_Metrics = dict  # shorthand; all keys are keyword args to diagnose_low_sharpe


def _check_losing_money(m: _Metrics) -> str | None:
    if m["total_return_pct"] <= 0 and m["total_trades"] >= 5:
        return (
            "**Root cause: Strategy is losing money.** "
            f"Total return is {m['total_return_pct']:.1%}. The core signal direction "
            "or logic is fundamentally wrong — you're not just underperforming, "
            "you're losing.\n"
            "**Fix:** Flip the signal direction (entries ↔ exits), or switch to "
            "a completely different indicator family. Do NOT try to optimize "
            "parameters of a losing signal."
        )
    return None


def _check_very_low_winrate(m: _Metrics) -> str | None:
    if m["win_rate"] < 0.35 and m["total_trades"] >= 10 and m["total_return_pct"] > 0:
        return (
            f"**Root cause: Very low win rate ({m['win_rate']:.0%}).** "
            "Your entry signals are triggering on noise — most trades lose. "
            "You're likely getting whipsawed in choppy markets or entering "
            "at bad prices.\n"
            "**Fix:**\n"
            "- Add a TREND FILTER — only take signals aligned with the "
            "overall trend (e.g., only long when price > SMA 50)\n"
            "- WIDEN thresholds — RSI < 30 triggers too often, try RSI < 20\n"
            "- Add a VOLATILITY FILTER — avoid trading when ATR is high "
            "(choppy = false signals)\n"
            "- Increase indicator lengths — shorter periods = more noise"
        )
    return None


def _check_low_winrate(m: _Metrics) -> str | None:
    if 0.35 <= m["win_rate"] < 0.45 and m["total_trades"] >= 10 and m["total_return_pct"] > 0:
        return (
            f"**Root cause: Low win rate ({m['win_rate']:.0%}).** "
            "More than half your trades lose. Your signal lacks edge or "
            "fires at suboptimal times.\n"
            "**Fix:**\n"
            "- Add a confirmation filter — require a second indicator to "
            "agree (e.g., MACD + RSI both bullish)\n"
            "- Tighten entry conditions to only take high-conviction setups\n"
            "- Consider a different indicator family that better captures "
            "the pattern you're targeting"
        )
    return None


def _check_bad_profit_factor(m: _Metrics) -> str | None:
    if m["profit_factor"] < 1.0 and m["profit_factor"] > 0 and m["total_trades"] >= 5:
        return (
            f"**Root cause: Profit factor {m['profit_factor']:.2f} < 1.0.** "
            "Your losing trades are larger than your winners. Even if you "
            "win often, you lose more money per losing trade.\n"
            "**Fix:**\n"
            "- Add a STOP-LOSS — exit trades that go against you early "
            "(e.g., exit if price drops 2% below entry)\n"
            "- Use TRAILING STOPS to lock in profits before they reverse\n"
            "- Add a TIME STOP — exit after N bars if the trade hasn't "
            "moved in your favor (reduces prolonged losers)\n"
            "- Let winners run longer — your exits are too early"
        )
    return None


def _check_marginal_profit_factor(m: _Metrics) -> str | None:
    if 1.0 <= m["profit_factor"] < 1.3 and m["total_trades"] >= 10:
        return (
            f"**Root cause: Marginal profit factor ({m['profit_factor']:.2f}).** "
            "Your edge is thin — transaction costs and slippage will likely "
            "erase it. You need bigger winners or fewer losers.\n"
            "**Fix:**\n"
            "- Improve your EXIT logic — hold winning trades longer using "
            "trailing stops instead of fixed exits\n"
            "- Cut losers faster — a tight initial stop-loss prevents "
            "small losses from becoming big ones\n"
            "- Add a trend strength filter — only trade when ADX > 20 "
            "(weak trends produce marginal results)"
        )
    return None


def _check_excessive_drawdown(m: _Metrics) -> str | None:
    if m["max_drawdown_pct"] < -0.20 and m["total_return_pct"] > 0:
        dd_return_ratio = (
            abs(m["max_drawdown_pct"] / m["total_return_pct"])
            if m["total_return_pct"] != 0
            else 0
        )
        if dd_return_ratio > 2.0:
            return (
                f"**Root cause: Drawdown too large relative to return.** "
                f"Max DD is {abs(m['max_drawdown_pct']):.1%} but total return is "
                f"only {m['total_return_pct']:.1%} (DD/return ratio: {dd_return_ratio:.1f}x). "
                "Big drawdowns destroy risk-adjusted returns (Sharpe).\n"
                "**Fix:**\n"
                "- Add a VOLATILITY STOP — reduce position size or exit when "
                "ATR spikes (large moves against you)\n"
                "- Add a DRAWDOWN CIRCUIT BREAKER — stop trading after "
                "consecutive losses or a drawdown threshold\n"
                "- Avoid trading during HIGH VOLATILITY regimes — use "
                "ATR or VIX-based filter to sit out volatile periods"
            )
    return None


def _check_sortino_concentration(m: _Metrics) -> str | None:
    if (
        m["sortino_ratio"] < m["sharpe_ratio"] * 0.5
        and m["sortino_ratio"] > 0
        and m["total_trades"] >= 5
    ):
        return (
            f"**Root cause: Sortino ({m['sortino_ratio']:.2f}) much lower than "
            f"Sharpe ({m['sharpe_ratio']:.2f}).** Your downside risk is "
            "concentrated in a few large losing periods. Your overall "
            "returns look OK on average, but when you lose, you lose big.\n"
            "**Fix:**\n"
            "- Add a DOWNSIDE PROTECTION mechanism — exit positions when "
            "drawdown exceeds a threshold\n"
            "- Use a TRAILING STOP to cut losers before they become large\n"
            "- Reduce position size during high-volatility periods"
        )
    return None


def _check_inconsistent_returns(m: _Metrics) -> str | None:
    if m["total_return_pct"] > 0.05 and m["sharpe_ratio"] < 0.5 and m["win_rate"] >= 0.45:
        return (
            f"**Root cause: Returns are positive ({m['total_return_pct']:.1%}) but "
            f"inconsistent (Sharpe {m['sharpe_ratio']:.2f}).** Your strategy has "
            "edge but it's noisy — some periods work, others don't.\n"
            "**Fix:**\n"
            "- Add a CONVICTION FILTER — only take trades when multiple "
            "signals align (e.g., MACD crossover + RSI < 40 + price > SMA)\n"
            "- Add a REGIME FILTER — only trade in favorable market conditions\n"
            "- Reduce trade frequency — fewer, higher-quality signals\n"
            "- Use wider indicator lengths to filter out noise"
        )
    return None


def _check_whipsaw(m: _Metrics) -> str | None:
    if (
        m["avg_holding_bars"] is not None
        and m["avg_holding_bars"] < 5
        and m["win_rate"] < 0.45
        and m["total_trades"] >= 15
    ):
        return (
            f"**Root cause: Whipsaw pattern.** Average holding period is "
            f"only {m['avg_holding_bars']:.1f} bars with {m['win_rate']:.0%} win rate. "
            "You're entering and exiting too quickly — getting chopped up "
            "by market noise.\n"
            "**Fix:**\n"
            "- INCREASE indicator lengths — e.g., EMA 9/21 → EMA 20/50\n"
            "- Add a MINIMUM HOLDING PERIOD — don't exit for at least N bars\n"
            "- Use a WIDER confirmation requirement before entry\n"
            "- Filter out low-volatility periods where signals are noisy"
        )
    return None


def _check_long_holds_low_return(m: _Metrics) -> str | None:
    if (
        m["avg_holding_bars"] is not None
        and m["avg_holding_bars"] > 30
        and m["total_return_pct"] < 0.10
        and m["total_trades"] >= 5
    ):
        return (
            f"**Root cause: Long holds with little gain.** Average holding "
            f"period is {m['avg_holding_bars']:.0f} bars but return is only "
            f"{m['total_return_pct']:.1%}. Your entries are not capturing "
            "momentum — you're sitting in flat or losing positions too long.\n"
            "**Fix:**\n"
            "- Improve ENTRY TIMING — use momentum indicators to enter "
            "when a move is starting (MACD crossover, ROC turning positive)\n"
            "- Add a TIME STOP — exit after N bars if trade hasn't moved\n"
            "- Use a more RESPONSIVE exit signal instead of holding forever"
        )
    return None


def _check_regime_dependency(m: _Metrics) -> str | None:
    sharpe_by_year = m.get("sharpe_by_year") or {}
    if len(sharpe_by_year) < 2:
        return None
    sharpe_values = list(sharpe_by_year.values())
    positive_years = sum(1 for s in sharpe_values if s > 0)
    if positive_years == 0:
        return (
            "**Root cause: Strategy loses in ALL years.** No single year "
            "had a positive Sharpe. This is not a regime issue — the "
            "strategy fundamentally doesn't work.\n"
            "**Fix:** Try a completely different approach. Switch to a "
            "well-known, simple strategy type (EMA crossover, RSI "
            "mean reversion) and build from there."
        )
    if positive_years == 1 and len(sharpe_values) >= 3:
        return (
            f"**Root cause: Strategy only works in 1/{len(sharpe_values)} years.** "
            "It's heavily regime-dependent — one good year masks "
            "consistent losses.\n"
            "**Fix:**\n"
            "- Add REGIME DETECTION — only trade when conditions match "
            "the favorable regime\n"
            "- OR switch to a more robust signal that works across "
            "multiple market conditions (trend + momentum combo)"
        )
    return None


def _check_sharpe_gap(m: _Metrics) -> str | None:
    sharpe_gap = m["sharpe_target"] - m["sharpe_ratio"]
    if sharpe_gap <= 0 or m["total_trades"] < 5:
        return None
    if sharpe_gap < 0.3:
        return (
            f"**Root cause: Close to target.** Sharpe {m['sharpe_ratio']:.2f} "
            f"is only {sharpe_gap:.2f} below target {m['sharpe_target']:.1f}. "
            "Small improvements could push it over.\n"
            "**Fix:**\n"
            "- Fine-tune entry/exit thresholds (small adjustments)\n"
            "- Add a minor filter to cut the worst 10-20% of trades\n"
            "- Optimize holding period — find the sweet spot for exits"
        )
    if sharpe_gap < 1.0:
        return (
            f"**Root cause: Moderate Sharpe gap.** Strategy has some "
            f"edge (Sharpe {m['sharpe_ratio']:.2f}) but needs {sharpe_gap:.2f} "
            "improvement to hit target.\n"
            "**Fix:**\n"
            "- Add a confirmation filter to improve signal quality\n"
            "- Improve exit logic (trailing stops, time stops)\n"
            "- Reduce trade frequency to focus on high-conviction setups"
        )
    return (
        f"**Root cause: Large Sharpe gap ({sharpe_gap:.2f}).** "
        f"Strategy Sharpe {m['sharpe_ratio']:.2f} is far from target "
        f"{m['sharpe_target']:.1f}. The core approach may need rethinking.\n"
        "**Fix:**\n"
        "- Consider a FULL REWRITE with a simpler, proven approach\n"
        "- Look at the winner examples for patterns that achieve "
        "high Sharpe\n"
        "- Avoid over-engineering — start with 1-2 indicators "
        "and a clear signal"
    )


# ── Check registry — ordered list of independent diagnostic checks ────────────

_DIAGNOSTIC_CHECKS: list[Callable[[_Metrics], str | None]] = [
    _check_losing_money,
    _check_very_low_winrate,
    _check_low_winrate,
    _check_bad_profit_factor,
    _check_marginal_profit_factor,
    _check_excessive_drawdown,
    _check_sortino_concentration,
    _check_inconsistent_returns,
    _check_whipsaw,
    _check_long_holds_low_return,
    _check_regime_dependency,
    _check_sharpe_gap,
]


def diagnose_low_sharpe(
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
    avg_holding_bars: float | None = None,
    sharpe_by_year: dict | None = None,
) -> str:
    """Diagnose the root cause of low Sharpe and return actionable guidance.

    Analyzes the backtest metrics to identify the specific problem pattern
    and provides targeted advice for the LLM. Multiple diagnoses can be
    returned if several issues are detected.

    Args:
        sharpe_ratio: Current Sharpe ratio.
        sharpe_target: Target Sharpe ratio.
        total_return_pct: Total return as fraction (e.g., 0.15 for 15%).
        max_drawdown_pct: Max drawdown as negative fraction (e.g., -0.25).
        win_rate: Win rate as fraction (e.g., 0.55 for 55%).
        profit_factor: Gross profit / gross loss.
        sortino_ratio: Downside risk-adjusted return.
        calmar_ratio: Return / max drawdown.
        total_trades: Number of trades in the backtest.
        avg_holding_bars: Average holding period in bars (if available).
        sharpe_by_year: Dict mapping year to Sharpe ratio.

    Returns:
        Formatted string with root cause analysis and specific guidance.
        Returns empty string if no specific diagnosis can be made.
    """
    metrics: _Metrics = {
        "sharpe_ratio": sharpe_ratio,
        "sharpe_target": sharpe_target,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "total_trades": total_trades,
        "avg_holding_bars": avg_holding_bars,
        "sharpe_by_year": sharpe_by_year or {},
    }

    # Run all independent checks except _check_sharpe_gap (fallback).
    # The gap check only fires when no other diagnosis was found.
    diagnoses = [
        d for check in _DIAGNOSTIC_CHECKS[:-1]
        if (d := check(metrics))
    ]

    # Gap analysis — fallback when no specific diagnosis found
    if not diagnoses:
        gap = _check_sharpe_gap(metrics)
        if gap:
            diagnoses.append(gap)

    if not diagnoses:
        return ""

    sharpe_gap = sharpe_target - sharpe_ratio
    header = (
        f"### 🔍 Sharpe Root Cause Analysis\n"
        f"Current Sharpe: {sharpe_ratio:.2f} | Target: {sharpe_target:.1f} | "
        f"Gap: {sharpe_gap:.2f}\n"
        f"Win Rate: {win_rate:.0%} | Profit Factor: {profit_factor:.2f} | "
        f"Return: {total_return_pct:.1%} | MaxDD: {max_drawdown_pct:.1%}\n\n"
    )
    body = "\n\n".join(diagnoses)
    return header + body
