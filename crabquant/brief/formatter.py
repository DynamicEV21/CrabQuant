"""
Telegram-friendly formatter for the daily brief.

Produces a clean, concise message under 800 characters.
No markdown tables — bullet lists and bold text only.
"""

from crabquant.brief.models import BriefData


def _regime_short_tag(regime_value: str) -> str:
    """Convert regime value to a short display tag. e.g. 'trending_up' → 'BULL'."""
    mapping = {
        "trending_up": "BULL",
        "trending_down": "BEAR",
        "mean_reversion": "MR",
        "high_volatility": "HVOL",
        "low_volatility": "LVOL",
        "unknown": "???",
    }
    return mapping.get(regime_value, regime_value.upper()[:5])


def format_brief(brief: BriefData) -> str:
    """
    Format brief data into a Telegram-friendly string.

    Args:
        brief: BriefData with all collected information

    Returns:
        Formatted message string (under 800 chars), or "NO_REPLY"
    """
    lines = []

    # ── Header ──
    lines.append("📊 CrabQuant Daily Brief")

    # ── Market regime ──
    regime_label = brief.regime.upper().replace("_", " ")
    parts = [regime_label]

    if brief.spy_20d_return is not None:
        sign = "+" if brief.spy_20d_return >= 0 else ""
        parts.append(f"SPY 20d: {sign}{brief.spy_20d_return:.1f}%")

    if brief.realized_vol is not None:
        vol_pct = brief.realized_vol * 100
        parts.append(f"Vol: {vol_pct:.1f}")

    lines.append("Market: " + " | ".join(parts))

    # ── Top production strategies (with regime tags) ──
    if brief.top_production:
        lines.append("")
        lines.append("🏆 Top Production:")
        for s in brief.top_production:
            sign = "+" if s["total_return"] >= 0 else ""
            regime_tag = s.get("discovery_regime", "")
            tag_str = f" [{_regime_short_tag(regime_tag)}]" if regime_tag and regime_tag != "unknown" else ""
            lines.append(
                f"• {s['ticker']}/{s['strategy_name']} — "
                f"Sharpe {s['sharpe']}, {sign}{s['total_return']}%{tag_str}"
            )

        # Suggest best strategies for current regime
        if brief.regime_strategy_suggestions:
            lines.append("")
            current_tag = _regime_short_tag(brief.regime)
            top_names = [name for name, _ in brief.regime_strategy_suggestions[:3]]
            lines.append(f"💡 Best for [{current_tag}]: {', '.join(top_names)}")
    else:
        lines.append("")
        lines.append("No production strategies yet — system still discovering")

    # ── Recent activity ──
    has_activity = (
        brief.recent_winners_count > 0
        or brief.recent_promotions_count > 0
        or brief.recent_retirements_count > 0
        or brief.total_combos_tested > 0
    )

    if has_activity:
        lines.append("")
        lines.append("📈 Last 24h:")
        if brief.recent_winners_count > 0:
            lines.append(f"• {brief.recent_winners_count} new winner{'s' if brief.recent_winners_count != 1 else ''} found")
        if brief.recent_promotions_count > 0:
            lines.append(f"• {brief.recent_promotions_count} promoted to production")
        if brief.recent_retirements_count > 0:
            lines.append(f"• {brief.recent_retirements_count} retired (failed re-validation)")
        if brief.total_combos_tested > 0:
            lines.append(f"• {brief.total_combos_tested} combos tested")
    else:
        lines.append("")
        lines.append("No new discoveries in last 24h")

    # ── Cron status ──
    cron_str = f"{brief.cron_active}/{brief.cron_total} active"
    lines.append("")
    lines.append(f"🤖 Crons: {cron_str}")

    # ── Pipeline Conversion Funnel ──
    metrics = brief.promotion_metrics
    if metrics and metrics.get("total_winners", 0) > 0:
        lines.append("")
        lines.append("🔬 Pipeline Conversion:")
        lines.append(f"• Backtest Winners: {metrics['total_winners']}")
        lines.append(f"• Walk-Forward Passed: {metrics.get('walk_forward_passed_count', 0)}")
        lines.append(f"• Confirmed: {metrics.get('confirmed_count', 0)}")
        lines.append(f"• Promoted to Registry: {metrics.get('promoted_count', 0)}")
        rate = metrics.get("promotion_rate", 0)
        lines.append(f"• Conversion Rate: {rate:.1%}")

    result = "\n".join(lines)

    # Hard cap at 800 chars — trim from bottom if needed
    if len(result) > 800:
        # Keep header + market, then trim activity
        lines = result.split("\n")
        while len("\n".join(lines)) > 800 and len(lines) > 3:
            lines.pop()
        result = "\n".join(lines)

    return result
