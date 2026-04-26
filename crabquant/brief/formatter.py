"""
Telegram-friendly formatter for the daily brief.

Produces a clean, concise message under 800 characters.
No markdown tables — bullet lists and bold text only.
"""

from crabquant.brief.models import BriefData


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

    # ── Top production strategies ──
    if brief.top_production:
        lines.append("")
        lines.append("🏆 Top Production:")
        for s in brief.top_production:
            sign = "+" if s["total_return"] >= 0 else ""
            lines.append(
                f"• {s['ticker']}/{s['strategy_name']} — "
                f"Sharpe {s['sharpe']}, {sign}{s['total_return']}%"
            )
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

    result = "\n".join(lines)

    # Hard cap at 800 chars — trim from bottom if needed
    if len(result) > 800:
        # Keep header + market, then trim activity
        lines = result.split("\n")
        while len("\n".join(lines)) > 800 and len(lines) > 3:
            lines.pop()
        result = "\n".join(lines)

    return result
