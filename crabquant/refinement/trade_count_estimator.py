"""
Trade Count Estimator — Phase 6

Estimates realistic trade count ranges for strategies based on backtest
period, timeframe, and strategy type. Injects guidance into LLM prompts
so the LLM knows how many trades to target, reducing too_few_trades failures
(25.5% of all backtest failures).
"""

from __future__ import annotations

import math
import re


# ── Constants ────────────────────────────────────────────────────────────────

MIN_TRADES_THRESHOLD = 10  # Minimum trades to pass validation (aligned with classifier threshold)

# Bars per year for different timeframes
BARS_PER_YEAR: dict[str, float] = {
    "daily": 252,
    "1d": 252,
    "1h": 6.5 * 252,       # ~1638
    "60m": 6.5 * 252,      # ~1638
    "4h": 2 * 252,         # ~504
    "240m": 2 * 252,       # ~504
    "15m": 26 * 252,       # ~6552
    "5m": 78 * 252,        # ~19656
}

# Default bars per year for unknown timeframes
DEFAULT_BARS_PER_YEAR = 252  # Assume daily

# Bars per trade range for each strategy type
BARS_PER_TRADE: dict[str, tuple[int, int]] = {
    "momentum": (10, 30),
    "mean_reversion": (5, 15),
    "breakout": (15, 40),
    "rsi_divergence": (8, 20),
    # Default fallback
    "default": (10, 25),
}

# Human-readable timeframe labels
TIMEFRAME_LABELS: dict[str, str] = {
    "daily": "daily",
    "1d": "daily",
    "1h": "1-hour",
    "60m": "1-hour",
    "4h": "4-hour",
    "240m": "4-hour",
    "15m": "15-minute",
    "5m": "5-minute",
}


# ── Period Parsing ───────────────────────────────────────────────────────────

_PERIOD_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*(y|mo|m|d|w)$", re.IGNORECASE)


def _parse_period_to_years(period: str) -> float:
    """Parse a period string (e.g., '2y', '6mo', '90d') to years.

    Args:
        period: Period string like '2y', '5y', '6mo', '1y', '90d', '52w'.

    Returns:
        Number of years as a float.

    Raises:
        ValueError: If the period string cannot be parsed.
    """
    period = period.strip().lower()
    match = _PERIOD_PATTERN.match(period)
    if not match:
        raise ValueError(
            f"Cannot parse period '{period}'. "
            f"Expected format: <number><unit> where unit is y, mo, m, d, or w."
        )

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "y":
        return value
    elif unit == "mo":
        return value / 12.0
    elif unit == "m":
        # Ambiguous: could be month or minute. Context: periods are typically
        # months or years. If value <= 24, treat as months. Otherwise as minutes.
        if value <= 24:
            return value / 12.0
        else:
            return value / (525600)  # minutes to years
    elif unit == "d":
        return value / 365.0
    elif unit == "w":
        return value / 52.0
    else:
        raise ValueError(f"Unknown period unit: '{unit}'")


# ── Core Estimation ──────────────────────────────────────────────────────────

def _normalize_timeframe(timeframe: str) -> str:
    """Normalize timeframe string to canonical form.

    Args:
        timeframe: Timeframe string like 'daily', '1h', '4h', '60m', etc.

    Returns:
        Normalized timeframe string.
    """
    tf = timeframe.strip().lower()
    # Normalize minute-style timeframes
    if tf in ("60m", "60min"):
        return "1h"
    if tf in ("240m", "240min"):
        return "4h"
    return tf


def estimate_trade_count(
    ticker: str,
    period: str,
    timeframe: str,
    strategy_type: str = "momentum",
) -> dict:
    """Estimate realistic trade count range for a strategy.

    Computes the expected number of bars in the backtest period, then divides
    by the typical bars-per-trade for the strategy type to get a trade count
    range.

    Args:
        ticker: Ticker symbol (e.g., 'SPY', 'AAPL'). Used for display.
        period: Backtest period string (e.g., '2y', '5y', '6mo').
        timeframe: Data timeframe (e.g., 'daily', '1h', '4h').
        strategy_type: Strategy type string. One of: 'momentum',
            'mean_reversion', 'breakout', 'rsi_divergence'. Falls back to
            'default' for unknown types.

    Returns:
        Dict with keys:
            - min_bars: Minimum expected bars in the backtest
            - max_bars: Maximum expected bars in the backtest (same as min_bars;
              kept for API consistency and future use)
            - min_trades: Minimum expected trade count (floor)
            - max_trades: Maximum expected trade count (ceiling)
            - recommended_min: Recommended minimum trades (min_trades with 20%
              safety margin), always >= MIN_TRADES_THRESHOLD (20)
            - timeframe_label: Human-readable timeframe label
            - years: Parsed number of years
            - bars_per_year: Bars per year for this timeframe
            - bars_per_trade_range: Tuple of (min, max) bars per trade
            - strategy_type: Normalized strategy type used
    """
    # Parse period to years
    try:
        years = _parse_period_to_years(period)
    except ValueError:
        years = 2.0  # Default to 2 years if parsing fails

    # Normalize timeframe
    tf_normalized = _normalize_timeframe(timeframe)

    # Get bars per year
    bars_per_year = BARS_PER_YEAR.get(tf_normalized, DEFAULT_BARS_PER_YEAR)

    # Compute total bars (with 10% tolerance for market holidays, etc.)
    min_bars = int(bars_per_year * years * 0.9)
    max_bars = int(bars_per_year * years * 1.0)

    # Get bars per trade range for strategy type
    strategy_key = strategy_type.lower().strip()
    bars_per_trade_range = BARS_PER_TRADE.get(strategy_key, BARS_PER_TRADE["default"])

    # Compute trade count range
    # More bars = more trades. Min trades from max bars-per-trade, max trades from min bars-per-trade.
    min_bpt, max_bpt = bars_per_trade_range
    min_trades = max(1, int(min_bars / max_bpt))  # Fewest trades: long hold periods
    max_trades = max(2, int(max_bars / min_bpt))  # Most trades: short hold periods

    # Recommended minimum with 20% safety margin, but always >= 20
    raw_recommended = math.ceil(min_trades * 1.2)
    recommended_min = max(raw_recommended, MIN_TRADES_THRESHOLD)

    # Timeframe label for display
    timeframe_label = TIMEFRAME_LABELS.get(tf_normalized, tf_normalized)

    return {
        "min_bars": min_bars,
        "max_bars": max_bars,
        "min_trades": min_trades,
        "max_trades": max_trades,
        "recommended_min": recommended_min,
        "timeframe_label": timeframe_label,
        "years": years,
        "bars_per_year": bars_per_year,
        "bars_per_trade_range": bars_per_trade_range,
        "strategy_type": strategy_key,
    }


# ── Prompt Formatting ───────────────────────────────────────────────────────

def format_trade_count_guidance(estimate: dict) -> str:
    """Format trade count estimate as guidance text for LLM prompts.

    Generates a concise section telling the LLM how many trades to expect
    and what the minimum threshold is.

    Args:
        estimate: Dict from estimate_trade_count().

    Returns:
        Formatted guidance string for prompt injection.
    """
    ticker = estimate.get("ticker", "")
    period = estimate.get("period", "")
    tf_label = estimate.get("timeframe_label", "daily")
    min_bars = estimate.get("min_bars", 0)
    strategy_type = estimate.get("strategy_type", "momentum")
    min_trades = estimate.get("min_trades", 0)
    max_trades = estimate.get("max_trades", 0)
    recommended_min = estimate.get("recommended_min", MIN_TRADES_THRESHOLD)

    # Build the header line
    ticker_display = f" on {ticker}" if ticker else ""
    header = f"Backtest period: {period} {tf_label}{ticker_display} (~{min_bars} bars)"

    # Strategy type display name
    display_name = strategy_type.replace("_", " ")

    # Build guidance lines
    lines = [
        "### Trade Count Expectations",
        header,
        f"For a {display_name} strategy, expect {min_trades}-{max_trades} trades.",
        f"Target at least {recommended_min} trades to pass validation.",
    ]

    # Add extra guidance for strategies that are likely to produce too few trades
    if min_trades < MIN_TRADES_THRESHOLD:
        lines.append(
            f"WARNING: This backtest period may be too short for a {display_name} strategy "
            f"to reach {MIN_TRADES_THRESHOLD} trades. Consider widening entry conditions "
            f"or using shorter holding periods."
        )

    # Always include actionable trade frequency guidance
    lines.append(
        "TRADE FREQUENCY TIPS (too_few_trades is the #1 failure mode):"
    )
    lines.append(
        "- Use SHORTER lookback windows (5-15 periods instead of 20-50) to increase signal frequency."
    )
    lines.append(
        "- Keep entry conditions SIMPLE — avoid stacking 3+ conditions, which filters out too many signals."
    )
    lines.append(
        "- Add a RE-ENTRY mechanism: after a stop-loss or exit, allow re-entry after a short cooldown (e.g., 3-5 bars)."
    )
    lines.append(
        "- Consider using FASTER indicators (e.g., EMA instead of SMA, shorter RSI periods) for more responsive signals."
    )

    return "\n".join(lines)


def build_trade_count_guidance(
    ticker: str,
    period: str,
    timeframe: str,
    strategy_type: str = "momentum",
) -> str:
    """Convenience function: estimate and format in one call.

    Args:
        ticker: Ticker symbol.
        period: Backtest period string.
        timeframe: Data timeframe.
        strategy_type: Strategy type string.

    Returns:
        Formatted guidance string for prompt injection.
    """
    estimate = estimate_trade_count(ticker, period, timeframe, strategy_type)
    # Pass through ticker and period for formatting
    estimate["ticker"] = ticker
    estimate["period"] = period
    return format_trade_count_guidance(estimate)
