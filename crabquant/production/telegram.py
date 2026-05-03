"""
Telegram Notification System

Sends status updates, alerts, and daily briefs to a Telegram chat via the
Bot API.  Uses only the standard library (``urllib.request``) so no extra
dependencies are required.

Configuration (environment variables):
    CRABQUANT_TELEGRAM_BOT_TOKEN  — Bot token from @BotFather
    CRABQUANT_TELEGRAM_CHAT_ID    — Target chat ID (individual or group)

Usage:
    from crabquant.production.telegram import get_notifier

    notifier = get_notifier()
    notifier.send_daily_brief()
    notifier.send_alert("Strategy promoted: roc_ema_volume on SPY")
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

# ── Message formatters ───────────────────────────────────────────────────


def format_daily_brief(
    health: dict | None = None,
    api_budget: dict | None = None,
    convergence: dict | None = None,
    production_count: int = 0,
    candidate_count: int = 0,
) -> str:
    """Build a Markdown-formatted daily brief for Telegram.

    Each data source is optional — missing sections are simply omitted.

    Args:
        health: Output from ``check_health()``.
        api_budget: Output from ``ApiBudgetTracker.get_summary()``.
        convergence: Output from ``measure_convergence.compute_report()``.
        production_count: Number of strategies in production registry.
        candidate_count: Number of soft-promoted candidates.

    Returns:
        Markdown-formatted message string (Telegram MarkdownV2 compatible).
    """
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"🦀 *CrabQuant Daily Brief*")
    lines.append(f"📅 {now}")
    lines.append("")

    # ── Health section ────────────────────────────────────────────────
    if health is not None:
        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "down": "🔴",
        }.get(health.get("status", ""), "❓")

        lines.append(f"*System Status* {status_emoji}")

        daemon = health.get("daemon", {})
        if daemon.get("alive"):
            lines.append(f"  • Daemon: running (PID {daemon.get('pid', '?')})")
            wave = daemon.get("current_wave")
            mandates = daemon.get("total_mandates_run")
            promoted = daemon.get("total_promoted")
            parts = []
            if wave is not None:
                parts.append(f"wave {wave}")
            if mandates is not None:
                parts.append(f"{mandates} mandates")
            if promoted is not None:
                parts.append(f"{promoted} promoted")
            if parts:
                lines.append(f"  • Activity: {', '.join(parts)}")
        else:
            lines.append("  • Daemon: *not running*")

        sys_info = health.get("system", {})
        if sys_info:
            cpu = sys_info.get("cpu_percent", "?")
            ram_free = sys_info.get("ram_free_gb", "?")
            disk_free = sys_info.get("disk_free_gb", "?")
            lines.append(f"  • CPU: {cpu}% | RAM free: {ram_free} GB | Disk: {disk_free} GB")

        data_info = health.get("data", {})
        if data_info:
            cache_fresh = data_info.get("cache_fresh", False)
            cache_age = data_info.get("cache_age_hours", "?")
            lines.append(f"  • Cache: {'✅ fresh' if cache_fresh else '⚠️ stale'} ({cache_age}h)")

        recs = health.get("recommendations", [])
        if recs:
            lines.append("")
            lines.append("*Recommendations:*")
            for r in recs[:5]:
                lines.append(f"  • {r}")

        lines.append("")

    # ── API Budget section ────────────────────────────────────────────
    if api_budget:
        lines.append("*API Usage*")
        total_cost = api_budget.get("total_cost_usd", 0)
        total_calls = api_budget.get("total_calls", 0)
        error_rate = api_budget.get("error_rate", 0)
        avg_tokens = api_budget.get("avg_tokens_per_call", 0)
        uptime = api_budget.get("uptime", "?")

        lines.append(f"  • Cost: ${total_cost:.2f} | Calls: {total_calls}")
        lines.append(f"  • Errors: {error_rate}% | Avg tokens/call: {avg_tokens}")
        lines.append(f"  • Uptime: {uptime}")

        budget_rem = api_budget.get("budget_remaining", {})
        if budget_rem:
            parts = []
            if "cost_usd" in budget_rem:
                parts.append(f"${budget_rem['cost_usd']:.2f} budget left")
            if "calls" in budget_rem:
                parts.append(f"{budget_rem['calls']} calls left")
            if parts:
                lines.append(f"  • Budget: {', '.join(parts)}")

        lines.append("")

    # ── Convergence section ───────────────────────────────────────────
    if convergence:
        total = convergence.get("total_mandates", 0)
        rate = convergence.get("convergence_rate", 0)
        conv = convergence.get("converged", 0)
        abandoned = convergence.get("abandoned", 0)

        lines.append("*Research Progress*")
        lines.append(f"  • Mandates: {total} | Converged: {conv} ({rate}%)")
        lines.append(f"  • Abandoned: {abandoned}")

        # Top failure modes
        modes = convergence.get("failure_modes", {})
        if modes:
            top = list(modes.items())[:5]
            mode_str = ", ".join(f"{m}×{c}" for m, c in top)
            lines.append(f"  • Top failures: {mode_str}")

        # Per-archetype breakdown
        by_arch = convergence.get("by_archetype", {})
        arch_lines = []
        for arch, info in by_arch.items():
            if info.get("total", 0) > 0:
                arch_lines.append(f"{arch}: {info['success']}/{info['total']}")
        if arch_lines:
            lines.append(f"  • Archetypes: {', '.join(arch_lines[:6])}")

        lines.append("")

    # ── Strategy counts ───────────────────────────────────────────────
    lines.append("*Strategy Catalog*")
    lines.append(f"  • Production: {production_count} | Candidates: {candidate_count}")

    return "\n".join(lines)


def format_strategy_promotion_alert(
    strategy_name: str,
    ticker: str,
    sharpe: float,
    trades: int,
    composite_score: float = 0.0,
    regime: str = "",
    verdict: str = "PROMOTED",
) -> str:
    """Format a strategy promotion notification.

    Args:
        strategy_name: Name of the promoted strategy.
        ticker: Ticker it was validated on.
        sharpe: Best Sharpe ratio.
        trades: Number of trades.
        composite_score: Composite score.
        regime: Preferred regime (if any).
        verdict: Promotion verdict (PROMOTED, SOFT_PROMOTED, etc.).

    Returns:
        Markdown-formatted alert string.
    """
    emoji = "🏆" if verdict == "PROMOTED" else "📋"
    lines = [
        f"{emoji} *Strategy {verdict}*",
        "",
        f"  • Strategy: `{strategy_name}`",
        f"  • Ticker: {ticker}",
        f"  • Sharpe: {sharpe:.2f} | Trades: {trades}",
    ]
    if composite_score > 0:
        lines.append(f"  • Composite: {composite_score:.2f}")
    if regime:
        lines.append(f"  • Regime: {regime}")
    return "\n".join(lines)


def format_health_alert(
    status: str,
    previous_status: str,
    recommendations: list[str] | None = None,
) -> str:
    """Format a health status change alert.

    Only generates a message when status has actually changed.

    Args:
        status: Current health status (healthy/degraded/down).
        previous_status: Previous health status.
        recommendations: List of recommendation strings.

    Returns:
        Markdown-formatted alert string, or empty string if no change.
    """
    if status == previous_status:
        return ""

    emoji = {"healthy": "✅", "degraded": "⚠️", "down": "🔴"}.get(status, "❓")
    prev_emoji = {"healthy": "✅", "degraded": "⚠️", "down": "🔴"}.get(
        previous_status, "❓"
    )

    lines = [
        f"{emoji} *Health Status Changed*",
        "",
        f"  • {prev_emoji} → {emoji}  ({previous_status} → {status})",
    ]

    if recommendations:
        lines.append("")
        lines.append("*Actions:*")
        for r in recommendations[:5]:
            lines.append(f"  • {r}")

    return "\n".join(lines)


def format_mandate_summary(
    mandate_name: str,
    status: str,
    best_sharpe: float,
    best_composite: float,
    turns_used: int,
    max_turns: int,
    failure_mode: str = "",
) -> str:
    """Format a mandate completion summary.

    Args:
        mandate_name: Name of the mandate.
        status: Final status (success, max_turns_exhausted, abandoned).
        best_sharpe: Best Sharpe ratio achieved.
        best_composite: Best composite score achieved.
        turns_used: Number of turns actually used.
        max_turns: Maximum turns configured.
        failure_mode: Primary failure mode (if any).

    Returns:
        Markdown-formatted summary string.
    """
    emoji = {"success": "🎯", "max_turns_exhausted": "⏰", "abandoned": "🚫"}.get(
        status, "❓"
    )

    lines = [
        f"{emoji} *Mandate Complete: {mandate_name}*",
        "",
        f"  • Status: {status}",
        f"  • Best Sharpe: {best_sharpe:.2f} | Composite: {best_composite:.2f}",
        f"  • Turns: {turns_used}/{max_turns}",
    ]

    if failure_mode:
        lines.append(f"  • Failure mode: {failure_mode}")

    return "\n".join(lines)


# ── Notifier class ───────────────────────────────────────────────────────


class TelegramNotifier:
    """Sends formatted messages to Telegram.

    All methods silently return False on configuration or network errors
    so they never crash the calling code.

    Args:
        bot_token: Telegram Bot API token.
        chat_id: Target chat ID.
        timeout_seconds: HTTP request timeout.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout_seconds: int = 10,
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout_seconds = timeout_seconds
        self._api_url = TELEGRAM_API.format(token=bot_token)

    # ── Low-level send ────────────────────────────────────────────────

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a text message to the configured chat.

        Args:
            text: Message body.
            parse_mode: Telegram parse mode (Markdown, MarkdownV2, HTML).

        Returns:
            True if the message was sent successfully.
        """
        if not self.bot_token or not self.chat_id:
            logger.debug("Telegram not configured — skipping message")
            return False

        # Telegram message size limit: 4096 characters
        if len(text) > 4096:
            text = text[:4050] + "\n\n[truncated]"

        payload = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": "true",
        }).encode()

        req = urllib.request.Request(
            self._api_url,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = json.loads(resp.read().decode())
                if not body.get("ok"):
                    logger.warning("Telegram API error: %s", body.get("description"))
                    return False
                return True
        except urllib.error.HTTPError as e:
            logger.warning("Telegram HTTP %s: %s", e.code, e.reason)
            return False
        except urllib.error.URLError as e:
            logger.warning("Telegram network error: %s", e.reason)
            return False
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
            return False

    # ── High-level convenience methods ────────────────────────────────

    def send_daily_brief(self, **kwargs) -> bool:
        """Build and send a daily brief. See ``format_daily_brief`` for kwargs."""
        # Gather data from project modules if not provided
        kwargs.setdefault("health", self._load_health())
        kwargs.setdefault("api_budget", self._load_api_budget())
        kwargs.setdefault("convergence", self._load_convergence())
        kwargs.setdefault("production_count", self._count_production())
        kwargs.setdefault("candidate_count", self._count_candidates())

        text = format_daily_brief(**kwargs)
        return self.send_message(text)

    def send_promotion_alert(self, **kwargs) -> bool:
        """Send a strategy promotion alert. See ``format_strategy_promotion_alert``."""
        text = format_strategy_promotion_alert(**kwargs)
        return self.send_message(text)

    def send_health_alert(self, **kwargs) -> bool:
        """Send a health status change alert. Returns False if no change."""
        text = format_health_alert(**kwargs)
        if not text:
            return False
        return self.send_message(text)

    def send_mandate_summary(self, **kwargs) -> bool:
        """Send a mandate completion summary. See ``format_mandate_summary``."""
        text = format_mandate_summary(**kwargs)
        return self.send_message(text)

    def send_raw(self, text: str) -> bool:
        """Send an arbitrary text message."""
        return self.send_message(text)

    # ── Data gathering helpers ────────────────────────────────────────

    def _load_health(self) -> dict | None:
        try:
            from crabquant.production.health import check_health
            return check_health()
        except Exception as e:
            logger.debug("Could not load health data: %s", e)
            return None

    def _load_api_budget(self) -> dict | None:
        try:
            from crabquant.refinement.api_budget import get_global_tracker
            return get_global_tracker().get_summary()
        except Exception as e:
            logger.debug("Could not load API budget: %s", e)
            return None

    def _load_convergence(self) -> dict | None:
        try:
            base = Path(__file__).resolve().parent.parent.parent
            state_dir = base / "refinement_runs"
            if not state_dir.is_dir():
                return None
            from scripts.measure_convergence import load_runs, compute_report
            runs = load_runs(state_dir, since=None)
            if not runs:
                return None
            return compute_report(runs)
        except Exception as e:
            logger.debug("Could not load convergence data: %s", e)
            return None

    def _count_production(self) -> int:
        try:
            from crabquant.production import get_production_strategies
            return len(get_production_strategies())
        except Exception:
            return 0

    def _count_candidates(self) -> int:
        try:
            base = Path(__file__).resolve().parent.parent.parent
            candidates_dir = base / "results" / "candidates"
            if not candidates_dir.is_dir():
                return 0
            return len(list(candidates_dir.glob("*.json")))
        except Exception:
            return 0

    # ── Health change detection ───────────────────────────────────────

    def check_and_alert_health(self) -> bool:
        """Check health, detect status changes, and alert if needed.

        Stores previous status in ``results/telegram_prev_status.json``.

        Returns:
            True if an alert was sent.
        """
        health = self._load_health()
        if not health:
            return False

        current_status = health.get("status", "unknown")

        # Load previous status
        base = Path(__file__).resolve().parent.parent.parent
        prev_path = base / "results" / "telegram_prev_status.json"
        prev_status = "unknown"
        try:
            if prev_path.exists():
                prev_status = json.loads(prev_path.read_text()).get("status", "unknown")
        except (json.JSONDecodeError, OSError):
            pass

        # Save current status
        try:
            prev_path.parent.mkdir(parents=True, exist_ok=True)
            prev_path.write_text(json.dumps({"status": current_status}))
        except OSError:
            pass

        # Alert on change
        if current_status != prev_status:
            return self.send_health_alert(
                status=current_status,
                previous_status=prev_status,
                recommendations=health.get("recommendations", []),
            )

        return False


# ── Singleton / factory ──────────────────────────────────────────────────

_notifier: Optional[TelegramNotifier] = None


def get_notifier(
    bot_token: str | None = None,
    chat_id: str | None = None,
) -> TelegramNotifier:
    """Get or create the global Telegram notifier.

    Uses environment variables if no arguments are provided:
        CRABQUANT_TELEGRAM_BOT_TOKEN
        CRABQUANT_TELEGRAM_CHAT_ID

    Returns a *no-op* notifier if configuration is missing (all sends
    silently return False).
    """
    global _notifier
    if _notifier is None:
        token = bot_token or os.environ.get("CRABQUANT_TELEGRAM_BOT_TOKEN", "")
        chat = chat_id or os.environ.get("CRABQUANT_TELEGRAM_CHAT_ID", "")
        _notifier = TelegramNotifier(bot_token=token, chat_id=chat)
    return _notifier


def is_configured() -> bool:
    """Check if Telegram is configured with valid credentials."""
    token = os.environ.get("CRABQUANT_TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("CRABQUANT_TELEGRAM_CHAT_ID", "")
    return bool(token and chat_id)
