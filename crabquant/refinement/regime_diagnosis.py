"""
Regime Diagnosis System — Phase 6

Analyzes per-year Sharpe data to diagnose WHICH specific market conditions
caused regime_fragility failures and provides targeted, actionable fixes.

Instead of generic "add regime detection," this tells the LLM:
"Your strategy crashed in 2022 (high-volatility year) — add a volatility
filter using ATR > 2× its 50-day SMA to reduce position size."

Key insight: Years map to known market regimes:
- 2018: Volatility spike (Feb VIX crash), late-year selloff
- 2019: Steady bull market, low volatility
- 2020: COVID crash + V-shaped recovery, extreme volatility
- 2021: Continued recovery, speculative growth, low rates
- 2022: Rate hike cycle, persistent bear market, high inflation
- 2023: Recovery rally, tech-led bull, moderation
- 2024: Continued bull, AI-driven, low vol
"""

from __future__ import annotations

# Known market regime characteristics by year.
# Used to generate contextual diagnosis when we see bad Sharpe in specific years.
_YEAR_REGIMES: dict[str, dict[str, str]] = {
    "2018": {
        "character": "High volatility — VIX spike in Feb, Q4 selloff",
        "filter": "ATR-based volatility filter",
        "fix": "Add a VOLATILITY FILTER — reduce position size or skip trades when ATR > 1.5× its 50-day average. Use a wider stop-loss to avoid getting stopped out in volatile swings.",
    },
    "2019": {
        "character": "Low volatility steady bull",
        "filter": "Trend-following filter",
        "fix": "If you lost in 2019 (a calm bull), your strategy may be too aggressive or counter-trend. Use a TREND FILTER (price > SMA 50) and avoid mean-reversion entries during steady trends.",
    },
    "2020": {
        "character": "COVID crash + extreme volatility + V-shaped recovery",
        "filter": "Drawdown circuit breaker + volatility filter",
        "fix": "2020 was extreme. Add a DRAWDOWN CIRCUIT BREAKER — stop trading after a 10% portfolio drawdown. Also add a VOLATILITY FILTER to skip the initial crash. Consider using a trailing stop to capture the recovery.",
    },
    "2021": {
        "character": "Speculative growth, low rates, steady uptrend",
        "filter": "Momentum/trend filter",
        "fix": "If you lost in 2021 (a strong bull), your strategy is likely counter-trend or mean-reversion during a trend. Consider switching to TREND-FOLLOWING or add a momentum confirmation filter.",
    },
    "2022": {
        "character": "Rate hikes, persistent bear market, high inflation volatility",
        "filter": "Macro regime filter + short-side capability",
        "fix": "2022 was a brutal bear market. Your strategy likely only goes long. Options: (1) Add SHORT entries when trend is bearish (price < SMA 200), (2) Add a REGIME FILTER that reduces long exposure during high-rate environments, (3) Use a defensive stop-loss (e.g., exit if price < 200-day SMA).",
    },
    "2023": {
        "character": "Recovery rally, tech-led bull, moderation",
        "filter": "Trend/momentum filter",
        "fix": "If you lost in 2023 (a strong recovery), your strategy may have missed the turn or been too defensive. Use faster indicator lengths (EMA 12/26 instead of 50/200) to catch trend changes earlier.",
    },
    "2024": {
        "character": "Continued bull, AI-driven, low volatility",
        "filter": "Trend-following filter",
        "fix": "If you lost in 2024 (a strong low-vol bull), your strategy may be mean-reversion during a trend or too conservative. Use a TREND FILTER and increase position sizing during confirmed uptrends.",
    },
    "2025": {
        "character": "Recent — likely mixed/moderate conditions",
        "filter": "Adaptive parameters",
        "fix": "Recent data may be incomplete. Ensure your strategy isn't overfit to a single recent pattern — test with longer lookback periods and more conservative thresholds.",
    },
}


_VOLATILE_YEARS = {"2018", "2020", "2022"}


def _classify_regime_pattern(sharpe_by_year: dict[str, float]) -> str:
    """Classify the regime dependency pattern based on per-year Sharpe.

    Returns a pattern label like 'trending_only', 'volatile_adverse',
    'single_year_fluke', etc.
    """
    if len(sharpe_by_year) < 2:
        return "unknown"

    years_sorted = sorted(sharpe_by_year.items(), key=lambda x: x[0])
    values = [v for _, v in years_sorted]
    positive = [y for y, s in years_sorted if s > 0]
    negative = [y for y, s in years_sorted if s < 0]
    good = [y for y, s in years_sorted if s >= 1.0]
    bad = [y for y, s in years_sorted if s < 0.3]

    n = len(values)
    pos_frac = len(positive) / n

    # Early exit: all negative or single-year fluke
    if not positive:
        return "always_losing"
    if len(good) == 1 and len(bad) >= 2 and n >= 3:
        return "single_year_fluke"

    # Time decay / improvement patterns
    pattern = _check_time_pattern(values)
    if pattern:
        return pattern

    # Regime clustering patterns
    pattern = _check_regime_clustering(negative, n, pos_frac)
    if pattern:
        return pattern

    return "mixed"


def _check_time_pattern(values: list[float]) -> str | None:
    """Check for time decay or improvement patterns."""
    mid = len(values) // 2
    if mid == 0:
        return None
    early_avg = sum(values[:mid]) / mid
    late_avg = sum(values[mid:]) / (len(values) - mid)
    if early_avg > 1.0 and late_avg < 0.0:
        return "time_decay"
    if early_avg < 0.0 and late_avg > 1.0:
        return "time_improvement"
    return None


def _check_regime_clustering(
    negative: list[str], n: int, pos_frac: float
) -> str | None:
    """Check if negative years cluster by regime type."""
    if not negative:
        return None

    neg_volatile = [y for y in negative if y in _VOLATILE_YEARS]
    neg_calm = [y for y in negative if y not in _VOLATILE_YEARS]
    neg_frac = len(negative) / n

    if neg_volatile and not neg_calm and neg_frac >= 0.5:
        return "volatile_adverse"
    if neg_calm and not neg_volatile and neg_frac >= 0.5:
        return "calm_adverse"
    if pos_frac >= 0.6:
        return "mostly_good_few_bad"
    if pos_frac < 0.4:
        return "mostly_bad"
    return None


def _build_year_detail(year: str, sharpe: float) -> str:
    """Build a detailed line for one year's Sharpe performance."""
    regime = _YEAR_REGIMES.get(year, {})
    character = regime.get("character", "unknown market conditions")

    if sharpe < 0:
        status = "❌ LOSING"
    elif sharpe < 0.5:
        status = "⚠️ WEAK"
    elif sharpe < 1.0:
        status = "🔶 OK"
    else:
        status = "✅ STRONG"

    return f"  {year}: Sharpe {sharpe:+.2f} [{status}] — {character}"


def _build_pattern_guidance(pattern: str, years_sorted: list[tuple[str, float]]) -> list[str]:
    """Build pattern-specific diagnosis text for a regime classification."""
    if pattern == "always_losing":
        return [
            "**Pattern: Strategy loses in ALL years.**\n"
            "This is not a regime issue — the strategy fundamentally doesn't work. "
            "The signal logic is wrong or the edge is non-existent.\n"
            "**Fix:** Start over with a simple, proven approach (EMA crossover, "
            "RSI mean reversion). Do NOT try to fix this with filters or parameters."
        ]

    if pattern == "single_year_fluke":
        good_years = [y for y, s in years_sorted if s >= 1.0]
        return [
            f"**Pattern: Single-year fluke ({', '.join(good_years)}).**\n"
            f"Your strategy only worked in {', '.join(good_years)} — it captured "
            f"one specific market move, not a repeatable edge. The other "
            f"{len(years_sorted) - len(good_years)} years all had Sharpe < 1.0.\n"
            "**Fix:** This is likely curve-fit to that year's specific conditions. "
            "Switch to a fundamentally different strategy type. A strategy that "
            "works in only 1 out of N years is not a strategy — it's luck."
        ]

    if pattern == "volatile_adverse":
        vol_years = [y for y, s in years_sorted if s < 0 and y in {"2018", "2020", "2022"}]
        vol_details = [
            f"{y} ({_YEAR_REGIMES.get(y, {}).get('character', 'volatile')})"
            for y in vol_years
        ]
        return [
            f"**Pattern: Fails in volatile/crisis years ({', '.join(vol_details)}).**\n"
            "Your strategy works in calm markets but gets destroyed when volatility "
            "spikes. This is the most common regime fragility pattern.\n"
            "**Fix:**\n"
            "- Add a VOLATILITY FILTER — compute ATR and skip/reduce trades when "
            "ATR > 1.5× its 50-day average\n"
            "- Use a DRAWDOWN CIRCUIT BREAKER — stop trading after consecutive losses\n"
            "- Widen stop-losses during volatile periods (use ATR-based stops)\n"
            "- Reduce position size when VIX-equivalent (ATR percentile) > 70th percentile"
        ]

    if pattern == "calm_adverse":
        calm_years = [y for y, s in years_sorted if s < 0 and y not in {"2018", "2020", "2022"}]
        return [
            f"**Pattern: Fails in calm/normal years ({', '.join(calm_years)}).**\n"
            "Your strategy may rely on volatility or large moves to profit. "
            "It works in crisis/trend years but loses in normal conditions.\n"
            "**Fix:**\n"
            "- If you're a mean-reversion strategy, ensure you're not fighting "
            "trends — add a trend filter (price vs SMA 200)\n"
            "- Reduce trade frequency in low-volatility environments\n"
            "- Consider that your 'edge' may just be capturing fat-tail events"
        ]

    if pattern in ("time_decay", "time_improvement"):
        early = years_sorted[:len(years_sorted) // 2]
        late = years_sorted[len(years_sorted) // 2:]
        early_avg = sum(s for _, s in early) / len(early)
        late_avg = sum(s for _, s in late) / len(late)
        if pattern == "time_decay":
            return [
                f"**Pattern: Time decay — early years avg Sharpe {early_avg:.2f}, "
                f"later years avg {late_avg:.2f}.**\n"
                "Your strategy's edge is decaying over time. The pattern it captures "
                "may have been arbitraged away or market structure changed.\n"
                "**Fix:**\n"
                "- Use SHORTER indicator lookback periods to be more adaptive\n"
                "- Add a REGIME ADAPTATION mechanism that adjusts parameters based on "
                "recent performance (rolling Sharpe over last 60 days)\n"
                "- Consider that the specific pattern may no longer exist — try a "
                "different strategy archetype entirely"
            ]
        return [
            f"**Pattern: Time improvement — early years avg Sharpe {early_avg:.2f}, "
            f"later years avg {late_avg:.2f}.**\n"
            "Your strategy is getting better over time, which suggests the market "
            "regime shifted to favor your approach. However, past improvement "
            "doesn't guarantee future performance.\n"
            "**Fix:**\n"
            "- This is a positive sign but be cautious about overfitting to "
            "recent data\n"
            "- Add a REGIME DETECTION filter to confirm conditions still match "
            "before taking trades\n"
            "- Consider using a shorter backtest window to focus on the "
            "recent favorable regime"
        ]

    if pattern == "mostly_good_few_bad":
        bad_list = [(y, s) for y, s in years_sorted if s < 0.3]
        lines = [
            f"**Pattern: Mostly good but fails in specific years "
            f"({', '.join(f'{y} ({s:+.2f})' for y, s in bad_list)}).**\n"
            "Your strategy has real edge but needs a filter for adverse conditions.\n"
        ]
        for y, s in bad_list:
            regime = _YEAR_REGIMES.get(y)
            if regime:
                lines.append(f"  - {y} (Sharpe {s:+.2f}): {regime['fix']}")
        lines.append(
            "**General fix:** Add a regime filter that detects the adverse "
            "condition and either skips trades or flips your signal. Use ATR "
            "percentile, trend direction (SMA 200), or a combination."
        )
        return lines

    if pattern == "mostly_bad":
        good_list = [(y, s) for y, s in years_sorted if s >= 1.0]
        if good_list:
            return [
                f"**Pattern: Mostly bad with rare good years "
                f"({', '.join(f'{y} ({s:+.2f})' for y, s in good_list)}).**\n"
                "Your strategy's 'edge' is not reliable — it loses in most conditions.\n"
                "**Fix:** Start over. The one good year is likely noise, not edge. "
                "Use a simpler, well-understood strategy type."
            ]
        return [
            "**Pattern: Consistently poor across years.**\n"
            "The strategy doesn't have a reliable edge.\n"
            "**Fix:** Rewrite from scratch with a different approach."
        ]

    # mixed / unknown
    return [
        "**Pattern: Mixed — no clear regime dependency.**\n"
        "Performance varies across years but doesn't cluster around a "
        "specific market condition.\n"
        "**Fix:**\n"
        "- Add a CONVICTION SCORE — only trade when multiple indicators agree\n"
        "- Use ADAPTIVE parameters that adjust to recent volatility (ATR-based)\n"
        "- Reduce trade frequency and focus on highest-conviction setups"
    ]


def diagnose_regime_fragility(
    sharpe_by_year: dict[str, float],
    *,
    sharpe_range: float | None = None,
) -> str:
    """Diagnose regime fragility and return actionable guidance.

    Analyzes per-year Sharpe ratios to identify which market conditions
    caused failure and provides specific, targeted fixes.

    Args:
        sharpe_by_year: Dict mapping year string to annual Sharpe ratio.
        sharpe_range: Optional pre-computed range (max - min Sharpe).

    Returns:
        Formatted string with year-by-year breakdown, pattern classification,
        and specific actionable fixes. Returns empty string if insufficient data.
    """
    if not sharpe_by_year or len(sharpe_by_year) < 2:
        return ""

    values = list(sharpe_by_year.values())
    if sharpe_range is None:
        sharpe_range = max(values) - min(values)

    pattern = _classify_regime_pattern(sharpe_by_year)

    # Sort years chronologically
    years_sorted = sorted(sharpe_by_year.items(), key=lambda x: x[0])

    # Identify worst and best years
    worst_year, worst_sharpe = min(sharpe_by_year.items(), key=lambda x: x[1])
    best_year, best_sharpe = max(sharpe_by_year.items(), key=lambda x: x[1])
    negative_years = [(y, s) for y, s in years_sorted if s < 0]
    positive_years = [(y, s) for y, s in years_sorted if s >= 0]

    # ── Build year-by-year breakdown ──
    year_lines = []
    for year, sharpe in years_sorted:
        year_lines.append(_build_year_detail(year, sharpe))
    year_breakdown = "\n".join(year_lines)

    # ── Build pattern-specific guidance via dispatch table ──
    diagnoses = _build_pattern_guidance(pattern, years_sorted)

    # ── Specific year-level fixes for worst years ──
    year_fixes: list[str] = []
    for y, s in years_sorted:
        if s < 0 and y in _YEAR_REGIMES:
            year_fixes.append(f"  - **{y}** (Sharpe {s:+.2f}): {_YEAR_REGIMES[y]['fix']}")

    # ── Summary stats ──
    avg_sharpe = sum(values) / len(values)
    n_positive = len(positive_years)
    n_negative = len(negative_years)

    # ── Build formatted output ──
    header = (
        f"### 🌊 Regime Fragility Diagnosis\n"
        f"Sharpe range: {sharpe_range:.1f} | Avg yearly Sharpe: {avg_sharpe:.2f} | "
        f"Positive years: {n_positive}/{len(values)} | Negative years: {n_negative}/{len(values)}\n"
        f"Worst year: {worst_year} ({worst_sharpe:+.2f}) | "
        f"Best year: {best_year} ({best_sharpe:+.2f})\n"
        f"Pattern: **{pattern}**\n\n"
        f"**Year-by-year breakdown:**\n"
        f"{year_breakdown}"
    )

    body = "\n\n".join(diagnoses)

    if year_fixes:
        fix_section = "\n\n**Specific fixes for losing years:**\n" + "\n".join(year_fixes)
    else:
        fix_section = ""

    return header + "\n\n" + body + fix_section
