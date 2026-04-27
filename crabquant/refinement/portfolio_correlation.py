"""Portfolio Correlation — Phase 3.

Compute equity curve correlation across winning strategies.
Identify redundant (highly correlated) and diversifying (low correlation)
strategy pairs to inform portfolio construction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _load_equity_for_winner(winner: dict) -> pd.Series | None:
    """Load equity curve data for a single winner entry.

    In production this would read equity curve files from the run directory.
    For now it returns None, signalling no embedded equity data.
    """
    return None


def load_winners_equity_curves(winners_path: str | Path) -> dict[str, pd.Series]:
    """Load equity curves from a winners.json file.

    Each winner entry is looked up via ``_load_equity_for_winner``.  Entries
    without equity data are silently skipped.

    Args:
        winners_path: Path to the winners.json file.

    Returns:
        Dict mapping strategy name to equity curve (pd.Series).
    """
    path = Path(winners_path)
    if not path.exists():
        return {}

    try:
        winners = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(winners, list) or len(winners) == 0:
        return {}

    curves: dict[str, pd.Series] = {}
    for winner in winners:
        equity = _load_equity_for_winner(winner)
        if equity is not None:
            name = winner.get("strategy", "unknown")
            curves[name] = equity

    return curves


def compute_correlation_matrix(
    equity_curves: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of equity curves.

    Args:
        equity_curves: Dict mapping strategy name to equity curve Series.

    Returns:
        Correlation matrix as a DataFrame (strategies × strategies).
    """
    if not equity_curves:
        return pd.DataFrame()

    df = pd.DataFrame(equity_curves)
    # Kendall's tau — robust to non-stationarity in equity curves.
    # Pearson on levels produces spurious correlation due to shared trends.
    return df.corr(method='kendall')


def identify_redundant_strategies(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.9,
) -> list[tuple[str, str]]:
    """Find strategy pairs with correlation above the threshold.

    Only returns pairs where (i < j) to avoid duplicates.

    Args:
        correlation_matrix: Correlation DataFrame from ``compute_correlation_matrix``.
        threshold: Minimum correlation to be considered redundant.

    Returns:
        List of (strategy_a, strategy_b) tuples.
    """
    if correlation_matrix.empty:
        return []

    redundant = []
    names = list(correlation_matrix.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr = correlation_matrix.iloc[i, j]
            if corr >= threshold:
                redundant.append((names[i], names[j]))

    return redundant


def identify_diversifying_strategies(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.3,
) -> list[tuple[str, str]]:
    """Find strategy pairs with correlation below the threshold.

    Only returns pairs where (i < j) to avoid duplicates.

    Args:
        correlation_matrix: Correlation DataFrame from ``compute_correlation_matrix``.
        threshold: Maximum correlation to be considered diversifying.

    Returns:
        List of (strategy_a, strategy_b) tuples.
    """
    if correlation_matrix.empty:
        return []

    diversifying = []
    names = list(correlation_matrix.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) <= threshold:
                diversifying.append((names[i], names[j]))

    return diversifying


def generate_correlation_report(
    equity_curves: dict[str, pd.Series],
    redundancy_threshold: float = 0.9,
    diversifying_threshold: float = 0.3,
) -> dict[str, Any]:
    """Generate a full correlation report.

    Args:
        equity_curves: Dict mapping strategy name to equity curve Series.
        redundancy_threshold: Threshold for redundant pair detection.
        diversifying_threshold: Threshold for diversifying pair detection.

    Returns:
        Dict with keys: correlation_matrix, redundant_pairs,
        diversifying_pairs, n_strategies.
    """
    matrix = compute_correlation_matrix(equity_curves)

    return {
        "correlation_matrix": matrix.to_dict() if not matrix.empty else {},
        "redundant_pairs": identify_redundant_strategies(matrix, redundancy_threshold),
        "diversifying_pairs": identify_diversifying_strategies(matrix, diversifying_threshold),
        "n_strategies": len(equity_curves),
    }
