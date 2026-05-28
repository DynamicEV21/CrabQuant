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
import random
from collections import Counter
from pathlib import Path
from typing import Any

from crabquant.refinement.config import DIVERSITY_CONFIG

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


def _load_winners_history(winners_file: Path | str | None) -> list[dict[str, Any]]:
    """Load winners history from a JSON file.

    Args:
        winners_file: Path to winners.json.  If ``None`` or the file does not
            exist, returns an empty list.

    Returns:
        List of winner dicts (each with at least ``"strategy"`` and
        ``"ticker"`` keys).
    """
    if winners_file is None:
        return []
    path = Path(winners_file)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load winners history from %s: %s", path, exc)
    return []


def diversity_score(
    mandate: dict[str, Any],
    winners_history: list[dict[str, Any]],
    registry_keys: set[str],
) -> float:
    """Score a candidate mandate for diversity.  Lower is better.

    Penalises mandates that duplicate already-explored (strategy, ticker)
    combinations and strategies already present in ``STRATEGY_REGISTRY``.

    Args:
        mandate: Candidate mandate dict.  Must contain ``"seed_strategy"``
            and ``"primary_ticker"`` keys.
        winners_history: List of winner dicts from ``winners.json`` (each
            with ``"strategy"`` and ``"ticker"``).
        registry_keys: Set of strategy names in ``STRATEGY_REGISTRY``.

    Returns:
        Numeric score (float).  Lower = more diverse / less explored.
    """
    strategy = mandate.get("seed_strategy", "")
    ticker = mandate.get("primary_ticker", "")

    # Count how many winners already exist for this specific combo
    combo_count = sum(
        1
        for w in winners_history
        if w.get("strategy") == strategy and w.get("ticker") == ticker
    )

    # Count how many winners exist for this ticker across ALL strategies
    ticker_count = sum(1 for w in winners_history if w.get("ticker") == ticker)

    # Penalty if the strategy is already in STRATEGY_REGISTRY
    registry_penalty = 20 if strategy in registry_keys else 0

    # Formula: ticker_count * 2 + combo_count * 5 + registry_penalty
    score = ticker_count * 2 + combo_count * 5 + registry_penalty
    return float(score)


def _get_registry_keys() -> set[str]:
    """Return the set of strategy names currently in STRATEGY_REGISTRY.

    Imports lazily to avoid circular-import issues in test environments
    where the strategies package may not be available.
    """
    try:
        from crabquant.strategies import STRATEGY_REGISTRY

        return set(STRATEGY_REGISTRY.keys())
    except Exception:
        logger.debug("Could not import STRATEGY_REGISTRY; skipping registry check")
        return set()


def generate_mandates(
    strategies_dir: Path | str,
    tickers: list[str] | None = None,
    count: int = 5,
    sharpe_targets: list[float] | None = None,
    periods: list[str] | None = None,
    max_turns: int = 7,
    constraints: dict[str, Any] | None = None,
    winners_file: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Generate a list of mandate JSON configs from the strategy catalog.

    Each mandate picks a seed strategy, assigns a primary ticker and secondary
    tickers, sets a Sharpe target, and includes constraints.  The tickers,
    targets, and periods are rotated to produce varied mandates.

    When a ``winners_file`` is available (or
    ``DIVERSITY_CONFIG["winners_file"]`` exists), a **diversity scoring**
    pass selects the most under-explored combinations first, ensuring
    broad coverage across tickers and strategy archetypes.

    Args:
        strategies_dir: Directory of strategy .py files to scan.
        tickers: Pool of tickers to choose from.  Defaults to major US stocks.
        count: Number of mandates to generate.
        sharpe_targets: Pool of Sharpe targets.  Defaults to [1.0, 1.5, 2.0, 2.5].
        periods: Pool of backtest periods.  Defaults to ["1y", "2y", "3y"].
        max_turns: Max refinement turns per mandate.
        constraints: Optional constraint overrides applied to all mandates.
        winners_file: Path to winners.json for diversity scoring.
            Defaults to ``DIVERSITY_CONFIG["winners_file"]``.

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

    # ── Diversity-aware generation ─────────────────────────────────────
    winners_file = winners_file or DIVERSITY_CONFIG.get("winners_file")
    winners_history = _load_winners_history(winners_file)
    registry_keys = _get_registry_keys()
    has_diversity_data = bool(winners_history or registry_keys)

    if has_diversity_data:
        return _generate_diverse_mandates(
            catalog=catalog,
            ticker_pool=ticker_pool,
            sharpe_pool=sharpe_pool,
            period_pool=period_pool,
            base_constraints=base_constraints,
            count=count,
            max_turns=max_turns,
            winners_history=winners_history,
            registry_keys=registry_keys,
        )

    # Fallback: original simple cycling (backward-compatible)
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


def _generate_diverse_mandates(
    catalog: list[dict[str, Any]],
    ticker_pool: list[str],
    sharpe_pool: list[float],
    period_pool: list[str],
    base_constraints: dict[str, Any],
    count: int,
    max_turns: int,
    winners_history: list[dict[str, Any]],
    registry_keys: set[str],
) -> list[dict[str, Any]]:
    """Generate diversity-scored mandates from a candidate pool.

    Creates a larger pool of candidate mandates, scores each one with
    :func:`diversity_score`, then selects the top-*N* while enforcing
    minimum ticker and archetype diversity constraints.
    """
    cfg = DIVERSITY_CONFIG
    max_combo = cfg.get("max_winners_per_combo", 5)
    min_tickers = cfg.get("min_ticker_diversity", 3)
    min_archetypes = cfg.get("min_archetype_diversity", 3)

    # Build a lookup of combo -> existing winner count
    combo_winner_counts: dict[tuple[str, str], int] = Counter()
    for w in winners_history:
        key = (w.get("strategy", ""), w.get("ticker", ""))
        combo_winner_counts[key] += 1

    # ── Generate candidate pool (3× the requested count) ──────────────
    pool_size = max(count * 3, count + 10)
    candidates: list[dict[str, Any]] = []

    for i in range(pool_size):
        strategy = catalog[i % len(catalog)]
        primary_ticker = ticker_pool[i % len(ticker_pool)]
        sharpe_target = sharpe_pool[i % len(sharpe_pool)]
        period = period_pool[i % len(period_pool)]

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
            "_diversity_score": diversity_score(
                mandate={"seed_strategy": strategy["name"], "primary_ticker": primary_ticker},
                winners_history=winners_history,
                registry_keys=registry_keys,
            ),
        }
        candidates.append(mandate)

    # ── Filter out saturated combos ───────────────────────────────────
    eligible = [
        c for c in candidates
        if combo_winner_counts.get((c["seed_strategy"], c["primary_ticker"]), 0) < max_combo
    ]

    if not eligible:
        logger.warning(
            "All candidate combos have %d+ existing winners; "
            "falling back to full candidate pool",
            max_combo,
        )
        eligible = candidates

    # ── Sort by diversity score (ascending = most diverse first) ───────
    eligible.sort(key=lambda c: c.get("_diversity_score", 0))

    # ── Greedy selection enforcing diversity constraints ───────────────
    selected: list[dict[str, Any]] = []
    used_tickers: set[str] = set()
    used_archetypes: set[str] = set()

    # Phase 1: pick one mandate per archetype until diversity met
    for c in eligible:
        if len(selected) >= count:
            break
        arch = c["strategy_archetype"]
        tick = c["primary_ticker"]
        if arch not in used_archetypes:
            selected.append(c)
            used_tickers.add(tick)
            used_archetypes.add(arch)

    # Phase 2: pick one mandate per new ticker until ticker diversity met
    for c in eligible:
        if len(selected) >= count:
            break
        tick = c["primary_ticker"]
        if tick not in used_tickers:
            selected.append(c)
            used_tickers.add(tick)
            used_archetypes.add(c["strategy_archetype"])

    # Phase 3: fill remaining slots by score
    selected_names = {id(m) for m in selected}
    for c in eligible:
        if len(selected) >= count:
            break
        if id(c) not in selected_names:
            selected.append(c)
            selected_names.add(id(c))
            used_tickers.add(c["primary_ticker"])
            used_archetypes.add(c["strategy_archetype"])

    # ── Log warnings if diversity constraints not met ─────────────────
    if len(used_tickers) < min_tickers:
        logger.warning(
            "Diversity constraint not met: only %d unique tickers (minimum %d). "
            "Consider adding more strategies or tickers.",
            len(used_tickers),
            min_tickers,
        )
    if len(used_archetypes) < min_archetypes:
        logger.warning(
            "Diversity constraint not met: only %d unique archetypes (minimum %d). "
            "Consider adding more strategy types.",
            len(used_archetypes),
            min_archetypes,
        )

    # ── Strip internal scoring field before returning ─────────────────
    for m in selected:
        m.pop("_diversity_score", None)

    # Renumber names sequentially
    for idx, m in enumerate(selected):
        m["name"] = f"{m['strategy_archetype']}_{m['primary_ticker'].lower()}_{idx + 1}"

    return selected


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
