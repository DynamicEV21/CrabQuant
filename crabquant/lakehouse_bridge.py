"""Lakehouse Bridge — connects CrabQuant to the central DuckDB lakehouse.

CrabQuant writes strategies and backtest results to the lakehouse via
CrabQuantConnector (called from agentic-quant-os backfill). This bridge
provides the READ path — getting regime data and strategy metrics from
the lakehouse for use inside the CrabQuant pipeline.

No direct imports from other quant projects — everything via QuantClient.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_AQOS_PATH = Path("/home/Zev/development/agentic-quant-os")

_client = None


def _get_client():
    """Lazy-init QuantClient."""
    global _client
    if _client is not None:
        return _client
    try:
        if str(_AQOS_PATH) not in sys.path:
            sys.path.insert(0, str(_AQOS_PATH))
        from src.client import QuantClient
        _client = QuantClient()
        return _client
    except Exception:
        logger.warning("Failed to init lakehouse client", exc_info=True)
        return None


def get_lakehouse_regime(ticker: str) -> Optional[dict]:
    """Get current regime from lakehouse for strategy regime filtering."""
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get_regime(ticker)
    except Exception:
        return None


def get_lakehouse_regime_label(ticker: str) -> tuple[str, float]:
    """Convenience: (label, confidence) with defaults."""
    r = get_lakehouse_regime(ticker)
    if r is None:
        return "Unknown", 0.5
    return r.get("composite_regime", "Unknown"), r.get("confidence", 0.5)


def get_strategy_history(strategy_name: str) -> list[dict]:
    """Get backtest history for a strategy from the lakehouse."""
    client = _get_client()
    if client is None:
        return []
    try:
        rows = client.query(
            "SELECT * FROM backtest_results WHERE strategy_name = ? ORDER BY created_at DESC LIMIT 20",
            [strategy_name],
        )
        return rows
    except Exception:
        return []


def write_scan_result(
    strategy_name: str,
    cycle: int,
    metrics: dict[str, float],
    ticker: str = "UNKNOWN",
    verdict: str = "pending",
) -> bool:
    """Write a CrabQuant scan/verify result to the lakehouse."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.write_backtest_result({
            "id": f"cq-scan-{strategy_name}-c{cycle}",
            "strategy_name": strategy_name,
            "ticker": ticker,
            "sharpe": metrics.get("sharpe"),
            "sortino": metrics.get("sortino"),
            "total_return": metrics.get("total_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "win_rate": metrics.get("win_rate"),
            "num_trades": metrics.get("num_trades"),
            "profit_factor": metrics.get("profit_factor"),
            "score": metrics.get("score"),
            "passed": verdict in ("winner", "validated", "confirmed"),
            "regime_label": metrics.get("regime"),
            "source_repo": "CrabQuant",
        })

        # Also write as a signal for time-series tracking
        client.write_signal({
            "source": "CrabQuant",
            "signal_type": f"scan.{strategy_name}",
            "ticker": ticker,
            "value": metrics.get("sharpe", 0.0),
            "confidence": metrics.get("sortino", 0.0) / 3.0,
            "metadata_json": json.dumps({
                "cycle": cycle,
                "verdict": verdict,
                **metrics,
            }),
        })
        return True
    except Exception:
        logger.warning("Failed to write scan result for %s", strategy_name, exc_info=True)
        return False
