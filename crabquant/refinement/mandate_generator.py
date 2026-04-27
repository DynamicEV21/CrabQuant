"""Mandate Generator — auto-generate mandates from strategy catalog + market analysis.

Scans existing strategy files to extract metadata (description, params, archetype),
then generates varied mandate JSON configs with different tickers, timeframes,
and Sharpe targets.  Each mandate seeds from an existing strategy for the LLM
refinement loop to improve upon.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Archetype keywords for auto-detection from strategy descriptions
_ARCHETYPE_KEYWORDS: dict[str, list[str]] = {
    "momentum": ["momentum", "roc", "rate of change", "acceleration", "velocity"],
    "mean_reversion": ["mean reversion", "rsi", "bollinger", "squeeze", "oversold", "overbought"],
    "breakout": ["breakout", "channel", "atr", "range expansion", "volatility expansion"],
    "trend": ["trend", "ichimoku", "ribbon", "moving average", "ema", "sma", "downtrend", "uptrend"],
}

# Default tickers to use when none provided
_DEFAULT_TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]
_DEFAULT_PERIODS = ["1y", "2y", "3y"]
_DEFAULT_SHARPE_TARGETS = [1.0, 1.5, 2.0, 2.5]


def scan_strategy_catalog(strategies_dir: Path | str) -> list[dict[str, Any]]:
    """Scan a directory of strategy .py files and extract metadata.

    For each strategy file, extracts:
      - name (filename without .py)
      - description (from DESCRIPTION variable)
      - default_params (from DEFAULT_PARAMS dict)
      - param_grid (from PARAM_GRID dict)
      - archetype (auto-detected from description)

    Args:
        strategies_dir: Path to directory containing strategy .py files.

    Returns:
        List of dicts with strategy metadata.
    """
    strategies_dir = Path(strategies_dir)
    if not strategies_dir.is_dir():
        return []

    catalog: list[dict[str, Any]] = []

    for py_file in sorted(strategies_dir.glob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning("Skipping %s: %s", py_file.name, e)
            continue

        # Extract top-level assignments (DESCRIPTION, DEFAULT_PARAMS, PARAM_GRID)
        description = ""
        default_params: dict = {}
        param_grid: dict = {}

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "DESCRIPTION":
                    description = ast.literal_eval(node.value) if isinstance(node.value, ast.Constant) else ""
                elif target.id == "DEFAULT_PARAMS":
                    default_params = ast.literal_eval(node.value) if isinstance(node.value, ast.Dict) else {}
                elif target.id == "PARAM_GRID":
                    param_grid = ast.literal_eval(node.value) if isinstance(node.value, ast.Dict) else {}

        # Fallback: extract DESCRIPTION from docstring
        if not description and ast.get_docstring(tree):
            description = ast.get_docstring(tree) or ""

        entry = {
            "name": py_file.stem,
            "description": description,
            "default_params": default_params,
            "param_grid": param_grid,
            "archetype": detect_archetype(description),
            "file_path": str(py_file),
        }
        catalog.append(entry)

    return catalog


def detect_archetype(description: str) -> str:
    """Auto-detect strategy archetype from its description.

    Uses keyword matching against known archetype patterns.
    Returns the best-matching archetype or "other" if no match.
    """
    if not description:
        return "other"

    desc_lower = description.lower()
    best_archetype = "other"
    best_count = 0

    for archetype, keywords in _ARCHETYPE_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in desc_lower)
        if count > best_count:
            best_count = count
            best_archetype = archetype

    return best_archetype


def generate_mandates(
    strategies_dir: Path | str,
    tickers: list[str] | None = None,
    count: int = 5,
    sharpe_targets: list[float] | None = None,
    periods: list[str] | None = None,
    max_turns: int = 7,
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate a list of mandate JSON configs from the strategy catalog.

    Each mandate picks a seed strategy, assigns a primary ticker and secondary
    tickers, sets a Sharpe target, and includes constraints.  The tickers,
    targets, and periods are rotated to produce varied mandates.

    Args:
        strategies_dir: Directory of strategy .py files to scan.
        tickers: Pool of tickers to choose from.  Defaults to major US stocks.
        count: Number of mandates to generate.
        sharpe_targets: Pool of Sharpe targets.  Defaults to [1.0, 1.5, 2.0, 2.5].
        periods: Pool of backtest periods.  Defaults to ["1y", "2y", "3y"].
        max_turns: Max refinement turns per mandate.
        constraints: Optional constraint overrides applied to all mandates.

    Returns:
        List of mandate dicts ready for JSON serialization.
    """
    catalog = scan_strategy_catalog(strategies_dir)
    if not catalog:
        return []

    ticker_pool = tickers or _DEFAULT_TICKERS
    sharpe_pool = sharpe_targets or _DEFAULT_SHARPE_TARGETS
    period_pool = periods or _DEFAULT_PERIODS
    base_constraints = constraints or {
        "max_parameters": 8,
        "required_indicators": [],
        "forbidden_indicators": [],
        "min_trades": 5,
        "max_drawdown_pct": 25,
    }

    mandates: list[dict[str, Any]] = []

    for i in range(count):
        # Cycle through strategies, tickers, targets, and periods
        strategy = catalog[i % len(catalog)]
        primary_ticker = ticker_pool[i % len(ticker_pool)]
        sharpe_target = sharpe_pool[i % len(sharpe_pool)]
        period = period_pool[i % len(period_pool)]

        # Pick 2-3 secondary tickers (excluding primary)
        secondary_tickers = [t for t in ticker_pool if t != primary_ticker]
        secondary = secondary_tickers[: min(3, len(secondary_tickers))]
        all_tickers = [primary_ticker] + secondary

        mandate = {
            "name": f"{strategy['archetype']}_{primary_ticker.lower()}_{i + 1}",
            "description": f"Refinement of {strategy['name']}: {strategy['description']}",
            "strategy_archetype": strategy["archetype"],
            "tickers": all_tickers,
            "primary_ticker": primary_ticker,
            "period": period,
            "sharpe_target": sharpe_target,
            "max_turns": max_turns,
            "seed_strategy": strategy["name"],
            "seed_params": strategy["default_params"],
            "constraints": dict(base_constraints),
        }
        mandates.append(mandate)

    return mandates


def save_mandates(
    mandates: list[dict[str, Any]],
    output_dir: Path | str,
) -> list[Path]:
    """Save mandate dicts to individual JSON files in output_dir.

    Each file is named ``{mandate_name}.json``.  The directory is created
    if it does not exist.  Existing files with the same name are overwritten.

    Args:
        mandates: List of mandate dicts.
        output_dir: Directory to write JSON files into.

    Returns:
        List of paths to the written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for mandate in mandates:
        name = mandate["name"].replace(" ", "_").lower()
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(mandate, indent=2))
        paths.append(path)

    return paths
