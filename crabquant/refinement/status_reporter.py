"""StatusReporter — generates daily Telegram-friendly status reports for the
CrabQuant daemon.

Aggregates data from:
- DaemonState (daemon health)
- run_history.jsonl (mandate completion stats)
- ApiBudgetTracker (API budget)
- ResourceLimiter (system resources)
- winning_strategies/ directory (top strategies)

All data sources are optional — missing data shows as "N/A".
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TELEGRAM_CHAR_LIMIT = 4096
_PACIFIC_TZ = "America/Los_Angeles"


def _pacific_now() -> datetime:
    """Return current datetime in Pacific timezone."""
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo(_PACIFIC_TZ))


def _try_import_api_budget() -> Any | None:
    """Try importing ApiBudgetTracker.  Returns the class or None."""
    try:
        from crabquant.refinement.api_budget import ApiBudgetTracker

        return ApiBudgetTracker
    except Exception:
        return None


def _try_import_resource_limiter() -> Any | None:
    """Try importing ResourceLimiter.  Returns the class or None."""
    try:
        from crabquant.refinement.resource_limiter import ResourceLimiter

        return ResourceLimiter
    except Exception:
        return None


class StatusReporter:
    """Generates a daily status report for CrabQuant.

    Parameters
    ----------
    results_dir:
        Path to the ``results/`` directory containing daemon state, run
        history, budget tracker, and winning strategies.
    """

    def __init__(self, results_dir: str = "results") -> None:
        self._results_dir = Path(results_dir)

    # ── public API ────────────────────────────────────────────────────────

    def generate_report(self) -> str:
        """Generate a full status report as Telegram-friendly markdown."""
        daemon = self._get_daemon_health()
        mandates = self._get_mandate_stats()
        budget = self._get_budget_status()
        resources = self._get_resource_status()
        strategies = self._get_top_strategies()

        sections: dict[str, dict[str, Any]] = {
            "daemon": daemon,
            "mandates": mandates,
            "budget": budget,
            "resources": resources,
            "strategies": strategies,
        }

        issues = self._collect_issues(budget, resources, daemon)
        sections["issues"] = {"items": issues}

        return self._format_telegram(sections)

    # ── data collectors ───────────────────────────────────────────────────

    def _get_daemon_health(self) -> dict[str, Any]:
        """Read daemon state and health info."""
        try:
            from crabquant.refinement.state import DaemonState

            state_path = self._results_dir / "daemon_state.json"
            state: Optional[DaemonState] = DaemonState.load(str(state_path))
        except Exception:
            state = None

        if state is None:
            return {"status": "N/A", "available": False}

        # Derive uptime from started_at
        uptime_str = "N/A"
        try:
            started = datetime.fromisoformat(state.started_at)
            now = datetime.now(timezone.utc)
            delta = now - started
            total_seconds = int(delta.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            uptime_str = f"{hours}h {minutes}m"
        except Exception:
            pass

        # Determine health status
        if state.shutdown_requested or state.last_error:
            status_label = "degraded ⚠️" if state.last_error else "down 🔴"
        else:
            status_label = "healthy ✅"

        return {
            "status": status_label,
            "wave": state.current_wave,
            "total_mandates": state.total_mandates_run,
            "promoted": state.total_strategies_promoted,
            "uptime": uptime_str,
            "last_error": state.last_error,
            "available": True,
        }

    def _get_mandate_stats(self) -> dict[str, Any]:
        """Read mandate completion stats from run_history.jsonl."""
        history_path = self._results_dir / "run_history.jsonl"
        if not history_path.exists():
            return {"completed": 0, "failed": 0, "total": 0, "available": False}

        completed = 0
        failed = 0
        try:
            with open(history_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("success") is True:
                        completed += 1
                    elif entry.get("success") is False:
                        failed += 1
        except Exception:
            return {"completed": "N/A", "failed": "N/A", "total": "N/A", "available": False}

        return {
            "completed": completed,
            "failed": failed,
            "total": completed + failed,
            "available": True,
        }

    def _get_budget_status(self) -> dict[str, Any]:
        """Read API budget tracker status."""
        cls = _try_import_api_budget()
        if cls is None:
            return {"available": False}

        try:
            budget_file = str(self._results_dir / "api_budget.json")
            tracker = cls(budget_file=budget_file)
            summary = tracker.get_usage_summary()
            summary["available"] = True
            return summary
        except Exception:
            return {"available": False}

    def _get_resource_status(self) -> dict[str, Any]:
        """Read resource limiter status."""
        cls = _try_import_resource_limiter()
        if cls is None:
            return {"available": False}

        try:
            limiter = cls()
            summary = limiter.get_status_summary()
            summary["available"] = True
            return summary
        except Exception:
            return {"available": False}

    def _get_top_strategies(self) -> list[dict[str, Any]]:
        """Read winning strategies directory, sorted by Sharpe ratio."""
        winners_dir = self._results_dir / "winning_strategies"
        if not winners_dir.exists():
            return []

        strategies: list[dict[str, Any]] = []
        try:
            for fpath in winners_dir.iterdir():
                if not fpath.is_file():
                    continue
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    sharpe = float(data.get("sharpe", data.get("sharpe_ratio", 0)))
                    name = data.get("strategy_name", data.get("name", fpath.stem))
                    ticker = data.get("ticker", data.get("symbol", ""))
                    strategies.append(
                        {"name": name, "ticker": ticker, "sharpe": sharpe}
                    )
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        except Exception:
            return []

        strategies.sort(key=lambda s: s["sharpe"], reverse=True)
        return strategies

    # ── issue collection ──────────────────────────────────────────────────

    def _collect_issues(
        self,
        budget: dict[str, Any],
        resources: dict[str, Any],
        daemon: dict[str, Any],
    ) -> list[str]:
        """Collect notable issues for the warnings section."""
        issues: list[str] = []

        # Budget alerts
        if budget.get("available"):
            weekly_pct = budget.get("weekly_pct", 0)
            if weekly_pct >= 0.90:
                issues.append(
                    f"API budget at {int(weekly_pct * 100)}% weekly — "
                    "consider reducing mandate rate"
                )
            daily_pct = budget.get("daily_pct", 0)
            if daily_pct >= 0.90:
                issues.append(
                    f"API budget at {int(daily_pct * 100)}% daily — throttling active"
                )
            if budget.get("throttled"):
                issues.append("API throttling is active — using fallback model")

        # Resource alerts
        if resources.get("available"):
            if resources.get("should_pause"):
                issues.append("System resources critically low — workers paused")
            cpu = resources.get("cpu_percent", 0)
            if cpu > 85:
                issues.append(f"High CPU usage: {int(cpu)}%")

        # Daemon alerts
        if daemon.get("available"):
            if daemon.get("last_error"):
                issues.append(f"Last daemon error: {daemon['last_error'][:80]}")
            if daemon.get("status", "").startswith("down"):
                issues.append("Daemon appears to be down")

        return issues

    # ── formatting ────────────────────────────────────────────────────────

    def _format_telegram(self, sections: dict[str, dict[str, Any]]) -> str:
        """Format all sections into Telegram markdown, respecting char limit."""
        lines: list[str] = []

        # Header
        now = _pacific_now()
        date_str = now.strftime("%B %d, %Y")
        lines.append("🦀 CrabQuant Daily Report")
        lines.append(f"📅 {date_str}")
        lines.append("")

        # Overall status line
        daemon = sections.get("daemon", {})
        overall = "N/A"
        if daemon.get("available"):
            raw = daemon.get("status", "N/A")
            overall = raw.split()[0] if raw else "N/A"
        lines.append(f"**Overall:** {overall}")
        lines.append("")

        # Daemon section
        lines.append("📊 **Daemon**")
        if daemon.get("available"):
            lines.append(f"• Status: {daemon.get('status', 'N/A')}")
            lines.append(
                f"• Wave: {daemon.get('wave', 0)} | "
                f"Mandates: {daemon.get('total_mandates', 0)} | "
                f"Promoted: {daemon.get('promoted', 0)}"
            )
            lines.append(f"• Uptime: {daemon.get('uptime', 'N/A')}")
        else:
            lines.append("• No daemon state found")
        lines.append("")

        # Mandate stats
        mandates = sections.get("mandates", {})
        if mandates.get("available"):
            lines.append("📋 **Mandates**")
            lines.append(
                f"• Completed: {mandates.get('completed', 0)} | "
                f"Failed: {mandates.get('failed', 0)} | "
                f"Total: {mandates.get('total', 0)}"
            )
            lines.append("")

        # API Budget section
        budget = sections.get("budget", {})
        lines.append("💰 **API Budget**")
        if budget.get("available"):
            daily_count = budget.get("daily_count", 0)
            daily_limit = budget.get("daily_limit", 0)
            daily_pct = budget.get("daily_pct", 0)
            weekly_count = budget.get("weekly_count", 0)
            weekly_limit = budget.get("weekly_limit", 0)
            weekly_pct = budget.get("weekly_pct", 0)

            daily_warning = " ⚠️" if daily_pct >= 0.80 else ""
            weekly_warning = " ⚠️" if weekly_pct >= 0.80 else ""

            lines.append(
                f"• Today: {daily_count}/{daily_limit} prompts "
                f"({int(daily_pct * 100)}%){daily_warning}"
            )
            lines.append(
                f"• Weekly: {weekly_count:,}/{weekly_limit:,} "
                f"({int(weekly_pct * 100)}%){weekly_warning}"
            )
            lines.append(f"• Throttled: {'Yes' if budget.get('throttled') else 'No'}")
        else:
            lines.append("• N/A")
        lines.append("")

        # Resources section
        resources = sections.get("resources", {})
        lines.append("💻 **Resources**")
        if resources.get("available"):
            cpu = resources.get("cpu_percent", 0)
            ram_free = resources.get("ram_free_gb", 0)
            parallel = resources.get("recommended_parallel", "N/A")
            lines.append(f"• CPU: {int(cpu)}% | RAM: {ram_free} GB free")
            lines.append(f"• Parallel workers: {parallel}")
        else:
            lines.append("• N/A")
        lines.append("")

        # Top Strategies
        strategies = sections.get("strategies", [])
        if strategies:
            lines.append("🎯 **Top Strategies**")
            # Budget roughly 200 chars for this section minus header
            remaining = _TELEGRAM_CHAR_LIMIT - len("\n".join(lines)) - 200
            count = 0
            for strat in strategies:
                entry = (
                    f"{count + 1}. {strat['name']}"
                    + (f"|{strat['ticker']}" if strat.get("ticker") else "")
                    + f" — Sharpe {strat['sharpe']:.2f}"
                )
                if len(entry) > remaining:
                    break
                lines.append(entry)
                remaining -= len(entry) + 1
                count += 1
            if count < len(strategies):
                lines.append(f"  … and {len(strategies) - count} more")
            lines.append("")
        else:
            lines.append("🎯 **Top Strategies**")
            lines.append("• No winning strategies yet")
            lines.append("")

        # Issues
        issues = sections.get("issues", {}).get("items", [])
        if issues:
            lines.append("⚠️ **Issues**")
            for issue in issues:
                lines.append(f"• {issue}")
            lines.append("")

        report = "\n".join(lines)

        # Truncate to Telegram limit if still too long
        if len(report) > _TELEGRAM_CHAR_LIMIT:
            report = report[: _TELEGRAM_CHAR_LIMIT - 20] + "\n\n… (truncated)"

        return report
