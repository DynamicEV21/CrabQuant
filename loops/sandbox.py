"""
CrabQuant Loop Sandbox — Helpers for the diversity-explorer loop.

Provides functions for loading registry data, computing diversity metrics,
identifying coverage gaps, creating mandates, running the refinement pipeline,
and evaluating results. Designed to be called by an agent following program.md.

CRITICAL: This module does NOT modify crabquant/ source code, backtest engine,
or validation logic. It only reads data, creates mandate files, and invokes
the existing refinement pipeline via crabquant_cron.py.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Project Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "strategies" / "production" / "registry.json"
SWEEP_RESULTS_PATH = PROJECT_ROOT / "results" / "sweep_results_cycle13.json"
WINNERS_PATH = PROJECT_ROOT / "results" / "winners" / "winners.json"
MANDATES_DIR = PROJECT_ROOT / "refinement" / "mandates"
RUNS_DIR = PROJECT_ROOT / "refinement_runs"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
CRON_SCRIPT = PROJECT_ROOT / "scripts" / "crabquant_cron.py"

# Diversity-explorer working directories
EXPLORER_DIR = PROJECT_ROOT / "loops" / "diversity-explorer"
GAPS_LOG_PATH = EXPLORER_DIR / "gaps_log.tsv"
EXPLORER_MANDATES_DIR = EXPLORER_DIR / "mandates"

# ── Ticker Universe ──────────────────────────────────────────────────────────

TICKER_UNIVERSE: list[str] = [
    "AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA",
    "BAC", "JPM", "V",
    "ABBV", "LLY", "UNH",
    "COST", "HD", "PEP", "PG", "WMT",
    "AMD", "AVGO", "CRM", "INTC", "ORCL",
    "CVX", "XOM",
    "CAT", "HON",
    "DIS", "NFLX",
    "IWM", "QQQ", "SPY",
    "GLD", "SLV",
]

# ── Archetype Definitions ────────────────────────────────────────────────────

ARCHETYPES: dict[str, dict[str, Any]] = {
    "momentum": {
        "regime_affinity": "trending",
        "typical_indicators": ["ema", "sma", "roc", "macd", "adx", "supertrend"],
        "trade_freq_range": (15, 40),
    },
    "mean_reversion": {
        "regime_affinity": "ranging",
        "typical_indicators": ["rsi", "bbands", "cci", "stoch", "willr"],
        "trade_freq_range": (30, 60),
    },
    "breakout": {
        "regime_affinity": "volatile",
        "typical_indicators": ["atr", "bbands", "donchian", "keltner", "adx"],
        "trade_freq_range": (10, 25),
    },
    "volatility": {
        "regime_affinity": "volatile",
        "typical_indicators": ["atr", "bbands", "vix", "cci", "keltner"],
        "trade_freq_range": (10, 20),
    },
    "statistical_arb": {
        "regime_affinity": "ranging",
        "typical_indicators": ["sma", "rsi", "bbands", "cci", "obv"],
        "trade_freq_range": (20, 50),
    },
    "multi_signal_ensemble": {
        "regime_affinity": "any",
        "typical_indicators": ["ema", "sma", "rsi", "macd", "adx", "obv"],
        "trade_freq_range": (20, 40),
    },
}

REGIMES: list[str] = ["ranging", "trending", "volatile"]

INDICATOR_FAMILIES: dict[str, list[str]] = {
    "momentum_indicators": ["ema", "sma", "roc", "macd", "adx", "supertrend"],
    "mean_reversion_indicators": ["rsi", "bbands", "cci", "stoch", "willr"],
    "volatility_indicators": ["atr", "vix", "keltner"],
    "volume_indicators": ["obv", "volume", "vwap"],
    "trend_indicators": ["ema", "sma", "donchian", "supertrend"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_registry() -> list[dict]:
    """Load the production strategy registry.

    Returns:
        List of strategy entry dicts from registry.json.
    """
    if not REGISTRY_PATH.exists():
        logger.warning("Registry not found: %s", REGISTRY_PATH)
        return []
    data = json.loads(REGISTRY_PATH.read_text())
    if isinstance(data, list):
        return data
    logger.warning("Registry is not a list: %s", type(data))
    return []


def load_sweep_results() -> dict:
    """Load sweep results from the latest cycle.

    Returns:
        Dict with 'passed' and optionally 'failed' keys, each containing lists.
    """
    if not SWEEP_RESULTS_PATH.exists():
        logger.warning("Sweep results not found: %s", SWEEP_RESULTS_PATH)
        return {}
    data = json.loads(SWEEP_RESULTS_PATH.read_text())
    return data if isinstance(data, dict) else {}


def load_winners() -> list[dict]:
    """Load the winners database.

    Returns:
        List of winner entry dicts.
    """
    if not WINNERS_PATH.exists():
        logger.warning("Winners not found: %s", WINNERS_PATH)
        return []
    data = json.loads(WINNERS_PATH.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Some formats nest winners under a key
        return data.get("winners", data.get("entries", []))
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# DIVERSITY COVERAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def _infer_archetype(strategy_name: str) -> str:
    """Infer archetype from strategy name using keyword matching.

    Falls back to 'unknown' if no match found.
    """
    name_lower = strategy_name.lower()
    archetype_keywords: dict[str, list[str]] = {
        "momentum": ["momentum", "trend", "roc", "ema_cross", "macd"],
        "mean_reversion": ["mean_rev", "reversion", "rsi_", "bollinger", "bbands",
                           "oversold", "overbought", "zscore", "z_score"],
        "breakout": ["breakout", "channel", "range_exp", "donchian"],
        "volatility": ["volatility", "vol_regime", "vol_break", "squeeze",
                       "atr_ratio", "vix"],
        "statistical_arb": ["stat_arb", "zscore", "pairs", "spread"],
        "multi_signal_ensemble": ["ensemble", "multi_signal", "consensus"],
    }
    for archetype, keywords in archetype_keywords.items():
        for kw in keywords:
            if kw in name_lower:
                return archetype
    return "unknown"


def _classify_trade_frequency(trades_per_year: int | float) -> str:
    """Classify trade frequency into low/medium/high."""
    if trades_per_year < 20:
        return "low"
    elif trades_per_year <= 60:
        return "medium"
    else:
        return "high"


def _classify_sharpe_tier(sharpe: float) -> str:
    """Classify Sharpe ratio into quality tiers."""
    if sharpe > 2.0:
        return "elite"
    elif sharpe >= 1.0:
        return "strong"
    elif sharpe >= 0.5:
        return "viable"
    else:
        return "marginal"


def _estimate_trades_per_year(trades: int, period_years: float = 2.0) -> float:
    """Estimate annualized trade frequency."""
    return trades / max(period_years, 0.1)


def compute_diversity_coverage(
    registry: list[dict],
) -> dict[str, Any]:
    """Analyze portfolio diversity coverage across multiple dimensions.

    Computes coverage matrices for:
      - archetype × ticker
      - archetype × regime
      - ticker × sharpe_tier
      - trade_frequency distribution
      - per-archetype strategy counts
      - per-ticker strategy counts

    Args:
        registry: List of registry entries.

    Returns:
        Coverage dict with nested breakdowns.
    """
    coverage: dict[str, Any] = {
        "total_strategies": len(registry),
        "by_archetype": {},
        "by_ticker": {},
        "by_regime": {},
        "by_trade_freq": {"low": 0, "medium": 0, "high": 0, "unknown": 0},
        "by_sharpe_tier": {"elite": 0, "strong": 0, "viable": 0, "marginal": 0, "unknown": 0},
        "archetype_ticker_matrix": {},
        "archetype_regime_matrix": {},
        "ticker_sharpe_matrix": {},
        "gaps": [],
    }

    for entry in registry:
        name = entry.get("strategy_name", "")
        ticker = entry.get("ticker", "UNKNOWN")
        sharpe = entry.get("sharpe", 0) or 0
        trades = entry.get("trades", 0) or 0
        source = entry.get("source", "")
        verdict = entry.get("verdict", "")

        # Only count robust entries
        if verdict and verdict != "ROBUST":
            continue

        archetype = _infer_archetype(name)
        trade_freq = _classify_trade_frequency(_estimate_trades_per_year(trades))
        sharpe_tier = _classify_sharpe_tier(sharpe)

        # Archetype info
        if archetype not in coverage["by_archetype"]:
            coverage["by_archetype"][archetype] = {"count": 0, "tickers": set(),
                                                    "sharpes": [], "trades": []}
        coverage["by_archetype"][archetype]["count"] += 1
        coverage["by_archetype"][archetype]["tickers"].add(ticker)
        coverage["by_archetype"][archetype]["sharpes"].append(sharpe)
        coverage["by_archetype"][archetype]["trades"].append(trades)

        # Ticker info
        if ticker not in coverage["by_ticker"]:
            coverage["by_ticker"][ticker] = {"count": 0, "archetypes": set(),
                                              "sharpes": []}
        coverage["by_ticker"][ticker]["count"] += 1
        coverage["by_ticker"][ticker]["archetypes"].add(archetype)
        coverage["by_ticker"][ticker]["sharpes"].append(sharpe)

        # Regime (from archetype definition, since strategies don't have explicit regime tags)
        regime = ARCHETYPES.get(archetype, {}).get("regime_affinity", "unknown")
        if regime not in coverage["by_regime"]:
            coverage["by_regime"][regime] = 0
        coverage["by_regime"][regime] += 1

        # Trade frequency
        coverage["by_trade_freq"][trade_freq] = (
            coverage["by_trade_freq"].get(trade_freq, 0) + 1
        )

        # Sharpe tier
        coverage["by_sharpe_tier"][sharpe_tier] = (
            coverage["by_sharpe_tier"].get(sharpe_tier, 0) + 1
        )

        # Cross matrices
        at_key = f"{archetype}|{ticker}"
        coverage["archetype_ticker_matrix"][at_key] = entry

        ar_key = f"{archetype}|{regime}"
        if ar_key not in coverage["archetype_regime_matrix"]:
            coverage["archetype_regime_matrix"][ar_key] = []
        coverage["archetype_regime_matrix"][ar_key].append(sharpe)

        ts_key = f"{ticker}|{sharpe_tier}"
        if ts_key not in coverage["ticker_sharpe_matrix"]:
            coverage["ticker_sharpe_matrix"][ts_key] = 0
        coverage["ticker_sharpe_matrix"][ts_key] += 1

    # Convert sets to sorted lists for JSON serialization
    for arch_data in coverage["by_archetype"].values():
        arch_data["tickers"] = sorted(arch_data["tickers"])
    for tick_data in coverage["by_ticker"].values():
        tick_data["archetypes"] = sorted(tick_data["archetypes"])

    return coverage


# ═══════════════════════════════════════════════════════════════════════════════
# GAP IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


def identify_gaps(
    coverage: dict[str, Any],
    min_per_archetype: int = 2,
    min_per_ticker: int = 1,
    min_per_archetype_ticker: int = 1,
    min_per_regime: int = 5,
) -> list[dict]:
    """Identify under-covered combinations in the portfolio.

    A gap is a (archetype, ticker, regime) combination where the portfolio
    has fewer robust strategies than the minimum threshold.

    Args:
        coverage: Output from compute_diversity_coverage().
        min_per_archetype: Min strategies per archetype.
        min_per_ticker: Min strategies per ticker.
        min_per_archetype_ticker: Min strategies per archetype-ticker pair.
        min_per_regime: Min strategies per regime.

    Returns:
        Sorted list of gap dicts with priority scoring.
    """
    gaps: list[dict] = []
    seen: set[str] = set()

    # Gap type 1: Archetypes with too few strategies
    for archetype, data in coverage["by_archetype"].items():
        if archetype == "unknown":
            continue
        if data["count"] < min_per_archetype:
            gap_key = f"archetype:{archetype}"
            if gap_key not in seen:
                seen.add(gap_key)
                gaps.append({
                    "type": "archetype_deficit",
                    "archetype": archetype,
                    "ticker": "any",
                    "regime": ARCHETYPES.get(archetype, {}).get("regime_affinity", "any"),
                    "current_count": data["count"],
                    "min_required": min_per_archetype,
                    "priority": (min_per_archetype - data["count"]) * 10,
                    "description": (
                        f"Only {data['count']} {archetype} strategies (need {min_per_archetype})"
                    ),
                })

    # Gap type 2: Tickers with no strategies at all
    for ticker in TICKER_UNIVERSE:
        ticker_data = coverage["by_ticker"].get(ticker)
        if ticker_data is None or ticker_data["count"] == 0:
            gap_key = f"ticker:{ticker}"
            if gap_key not in seen:
                seen.add(gap_key)
                gaps.append({
                    "type": "ticker_deficit",
                    "archetype": "any",
                    "ticker": ticker,
                    "regime": "any",
                    "current_count": 0,
                    "min_required": min_per_ticker,
                    "priority": min_per_ticker * 15,
                    "description": f"No robust strategies for {ticker}",
                })

    # Gap type 3: Archetype × Ticker combinations with no coverage
    for archetype in ARCHETYPES:
        if archetype == "multi_signal_ensemble":
            # Ensembles are composed — don't mandate them directly
            continue
        for ticker in TICKER_UNIVERSE:
            key = f"{archetype}|{ticker}"
            if key not in coverage.get("archetype_ticker_matrix", {}):
                gap_key = f"combo:{key}"
                if gap_key not in seen:
                    seen.add(gap_key)
                    regime = ARCHETYPES[archetype]["regime_affinity"]
                    gaps.append({
                        "type": "archetype_ticker_gap",
                        "archetype": archetype,
                        "ticker": ticker,
                        "regime": regime,
                        "current_count": 0,
                        "min_required": min_per_archetype_ticker,
                        "priority": 5,
                        "description": (
                            f"No {archetype} strategy for {ticker} "
                            f"(regime: {regime})"
                        ),
                    })

    # Gap type 4: Regimes with too few strategies
    for regime in REGIMES:
        count = coverage["by_regime"].get(regime, 0)
        if count < min_per_regime:
            gap_key = f"regime:{regime}"
            if gap_key not in seen:
                seen.add(gap_key)
                gaps.append({
                    "type": "regime_deficit",
                    "archetype": "any",
                    "ticker": "any",
                    "regime": regime,
                    "current_count": count,
                    "min_required": min_per_regime,
                    "priority": (min_per_regime - count) * 8,
                    "description": (
                        f"Only {count} strategies for {regime} regime "
                        f"(need {min_per_regime})"
                    ),
                })

    # Sort by priority (highest first)
    gaps.sort(key=lambda g: g["priority"], reverse=True)
    return gaps


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY & NOVELTY SCORING
# ═══════════════════════════════════════════════════════════════════════════════


def compute_quality_score(
    sharpe: float,
    trades: int,
    max_drawdown: float,
) -> float:
    """Compute CrabQuant's composite quality score.

    Formula: Sharpe * sqrt(trades/20) * (1 - |max_drawdown|)

    This matches the scoring used by the backtest engine.

    Args:
        sharpe: Sharpe ratio.
        trades: Number of trades.
        max_drawdown: Max drawdown as negative fraction (e.g., -0.15).

    Returns:
        Composite quality score (float).
    """
    trade_factor = math.sqrt(max(trades, 0) / 20.0)
    dd_penalty = max(1.0 - abs(max_drawdown), 0.0)
    return sharpe * trade_factor * dd_penalty


def compute_novelty_score(
    strategy_features: dict[str, Any],
    population_features: list[dict[str, Any]],
) -> float:
    """Compute how different a strategy is from the existing population.

    Novelty is based on feature-space distance. A strategy that occupies
    an empty cell in the (archetype, ticker, regime, trade_freq) space
    gets novelty = 1.0. A strategy identical to an existing one gets 0.0.

    Args:
        strategy_features: Dict with archetype, ticker, regime, trade_freq keys.
        population_features: List of feature dicts for existing strategies.

    Returns:
        Novelty score between 0.0 and 1.0.
    """
    if not population_features:
        return 1.0

    # Count how many existing strategies match on each dimension
    archetype_matches = sum(
        1 for p in population_features
        if p.get("archetype") == strategy_features.get("archetype")
    )
    ticker_matches = sum(
        1 for p in population_features
        if p.get("ticker") == strategy_features.get("ticker")
    )
    regime_matches = sum(
        1 for p in population_features
        if p.get("regime") == strategy_features.get("regime")
    )
    trade_freq_matches = sum(
        1 for p in population_features
        if p.get("trade_freq") == strategy_features.get("trade_freq")
    )

    n = len(population_features)
    # Exact cell match (all 4 dimensions)
    exact_matches = sum(
        1 for p in population_features
        if (p.get("archetype") == strategy_features.get("archetype")
            and p.get("ticker") == strategy_features.get("ticker")
            and p.get("regime") == strategy_features.get("regime")
            and p.get("trade_freq") == strategy_features.get("trade_freq"))
    )

    if exact_matches == 0:
        return 1.0  # Completely novel cell

    # Weighted novelty: penalize matches on each dimension
    archetype_novelty = 1.0 - (archetype_matches / n)
    ticker_novelty = 1.0 - (ticker_matches / n)
    regime_novelty = 1.0 - (regime_matches / n)
    trade_freq_novelty = 1.0 - (trade_freq_matches / n)

    # Weighted average
    novelty = (0.3 * archetype_novelty
               + 0.3 * ticker_novelty
               + 0.2 * regime_novelty
               + 0.2 * trade_freq_novelty)

    return round(max(min(novelty, 1.0), 0.0), 4)


def compute_qd_score(quality: float, novelty: float) -> float:
    """Compute Quality-Diversity score.

    Args:
        quality: Composite quality score.
        novelty: Novelty score (0-1).

    Returns:
        QD-Score = quality * novelty.
    """
    return quality * novelty


# ═══════════════════════════════════════════════════════════════════════════════
# MANDATE CREATION
# ═══════════════════════════════════════════════════════════════════════════════


def create_mandate_for_gap(
    gap: dict[str, Any],
    output_dir: str | Path | None = None,
) -> Path:
    """Generate a mandate JSON targeting a specific coverage gap.

    The mandate instructs the refinement pipeline to invent a strategy
    for the under-covered (archetype, ticker, regime) combination.

    Args:
        gap: A gap dict from identify_gaps().
        output_dir: Directory to write the mandate file.
                   Defaults to EXPLORER_MANDATES_DIR.

    Returns:
        Path to the created mandate file.
    """
    if output_dir is None:
        output_dir = EXPLORER_MANDATES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archetype = gap["archetype"]
    ticker = gap["ticker"]
    regime = gap["regime"]
    description = gap["description"]

    # Build mandate name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if archetype != "any" and ticker != "any":
        mandate_name = f"diversity_{archetype}_{ticker.lower()}_{timestamp}"
    elif archetype != "any":
        mandate_name = f"diversity_{archetype}_any_{timestamp}"
    elif ticker != "any":
        mandate_name = f"diversity_any_{ticker.lower()}_{timestamp}"
    else:
        mandate_name = f"diversity_regime_{regime}_{timestamp}"

    mandate = {
        "name": f"Diversity Mandate: {description}",
        "description": (
            f"Fill portfolio gap: {description}. "
            f"Archetype: {archetype}, Ticker: {ticker}, Regime: {regime}. "
            f"Created by diversity-explorer loop."
        ),
        "tickers": [ticker] if ticker != "any" else ["SPY", "QQQ"],
        "primary_ticker": ticker if ticker != "any" else "SPY",
        "period": "2y",
        "strategy_archetype": archetype if archetype != "any" else None,
        "target_regime": regime if regime != "any" else None,
        "max_turns": 7,
        "sharpe_target": 1.0,
        "constraints": {
            "min_trades": 5,
            "max_drawdown_pct": 25.0,
        },
        "backtest_config": {
            "start_date": "2023-01-01",
            "initial_capital": 100000,
            "commission": 0.001,
            "slippage": 0.0001,
        },
        "diversity_source": "diversity-explorer",
        "diversity_gap_type": gap["type"],
        "diversity_gap_priority": gap["priority"],
    }

    mandate_path = output_dir / f"{mandate_name}.json"
    mandate_path.write_text(json.dumps(mandate, indent=2))
    logger.info("Created mandate: %s", mandate_path)

    return mandate_path


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════


def run_mandate(
    mandate_path: str | Path,
    timeout: int = 900,
) -> dict[str, Any]:
    """Run the CrabQuant refinement pipeline on a mandate.

    Executes crabquant_cron.py targeting the specific mandate file.
    Uses the project's venv Python.

    Args:
        mandate_path: Path to the mandate JSON file.
        timeout: Maximum execution time in seconds (default 15 min).

    Returns:
        Dict with 'success', 'returncode', 'stdout', 'stderr', 'duration' keys.
    """
    mandate_path = Path(mandate_path)
    if not mandate_path.exists():
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Mandate not found: {mandate_path}",
            "duration": 0,
        }

    python = str(VENV_PYTHON)
    cron_script = str(CRON_SCRIPT)

    cmd = [
        python, cron_script,
        "--mandates-dir", str(mandate_path.parent),
        "--runs-dir", str(RUNS_DIR),
    ]

    logger.info("Running mandate: %s", mandate_path.name)
    start = datetime.now(timezone.utc)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
            "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
            "duration": duration,
        }
    except subprocess.TimeoutExpired:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Timeout after {timeout}s",
            "duration": duration,
        }
    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "duration": duration,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_result(run_dir: str | Path) -> dict[str, Any]:
    """Parse state.json and report from a refinement run directory.

    Looks for state.json in the run directory and extracts key metrics.

    Args:
        run_dir: Path to the refinement run directory.

    Returns:
        Dict with parsed metrics or error info.
    """
    run_dir = Path(run_dir)
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "found": False,
        "status": "unknown",
    }

    # Try state.json
    state_path = run_dir / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            result["found"] = True
            result["status"] = state.get("status", "unknown")
            result["total_turns"] = state.get("total_turns", 0)
            result["best_sharpe"] = state.get("best_sharpe", 0)
            result["best_score"] = state.get("best_score", 0)
            result["converged"] = state.get("converged", False)
            result["strategy_name"] = state.get("strategy_name", "")
            result["failure_mode"] = state.get("failure_mode", "")
            return result
        except (json.JSONDecodeError, OSError) as e:
            result["error"] = f"Failed to parse state.json: {e}"

    # Try report.json
    report_path = run_dir / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            result["found"] = True
            result.update(report)
            return result
        except (json.JSONDecodeError, OSError) as e:
            result["error"] = f"Failed to parse report.json: {e}"

    # Check if directory exists at all
    if not run_dir.exists():
        result["error"] = f"Run directory not found: {run_dir}"
    else:
        result["error"] = "No state.json or report.json found in run directory"
        # List contents for debugging
        result["contents"] = [f.name for f in run_dir.iterdir()] if run_dir.is_dir() else []

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DIVERSITY REPORT (HUMAN-READABLE)
# ═══════════════════════════════════════════════════════════════════════════════


def diversity_report(registry: list[dict]) -> str:
    """Generate a human-readable diversity analysis report.

    Args:
        registry: List of registry entries.

    Returns:
        Multi-line report string.
    """
    coverage = compute_diversity_coverage(registry)
    gaps = identify_gaps(coverage)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("CRABQUANT PORTFOLIO DIVERSITY REPORT")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    lines.append(f"Total robust strategies: {coverage['total_strategies']}")
    lines.append(f"Tickers covered: {len(coverage['by_ticker'])} / {len(TICKER_UNIVERSE)}")
    lines.append(f"Top coverage gaps: {len(gaps)}")
    lines.append("")

    # Per-archetype breakdown
    lines.append("-" * 50)
    lines.append("BY ARCHETYPE")
    lines.append("-" * 50)
    for arch in ["momentum", "mean_reversion", "breakout", "volatility",
                  "statistical_arb", "multi_signal_ensemble", "unknown"]:
        data = coverage["by_archetype"].get(arch)
        if data is None:
            lines.append(f"  {arch:25s}  0 strategies")
            continue
        avg_sharpe = (sum(data["sharpes"]) / len(data["sharpes"])
                      if data["sharpes"] else 0)
        lines.append(
            f"  {arch:25s}  {data['count']:3d} strategies  "
            f"avg Sharpe: {avg_sharpe:.2f}  "
            f"tickers: {', '.join(data['tickers'][:5])}"
            + (f" +{len(data['tickers'])-5} more" if len(data["tickers"]) > 5 else "")
        )

    lines.append("")

    # Per-ticker breakdown (top/bottom)
    lines.append("-" * 50)
    lines.append("BY TICKER (sorted by strategy count)")
    lines.append("-" * 50)
    sorted_tickers = sorted(
        coverage["by_ticker"].items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )
    for ticker, data in sorted_tickers[:15]:
        avg_sharpe = (sum(data["sharpes"]) / len(data["sharpes"])
                      if data["sharpes"] else 0)
        lines.append(
            f"  {ticker:6s}  {data['count']:3d} strategies  "
            f"avg Sharpe: {avg_sharpe:.2f}  "
            f"archetypes: {', '.join(data['archetypes'])}"
        )

    # Show tickers with NO strategies
    uncovered = [t for t in TICKER_UNIVERSE if t not in coverage["by_ticker"]]
    if uncovered:
        lines.append("")
        lines.append(f"  UNCOVERED TICKERS ({len(uncovered)}):")
        for t in uncovered:
            lines.append(f"    {t}")

    lines.append("")

    # Trade frequency distribution
    lines.append("-" * 50)
    lines.append("TRADE FREQUENCY DISTRIBUTION")
    lines.append("-" * 50)
    total = sum(coverage["by_trade_freq"].values())
    for freq, count in sorted(coverage["by_trade_freq"].items()):
        pct = (count / total * 100) if total > 0 else 0
        bar = "#" * int(pct / 2)
        lines.append(f"  {freq:10s}  {count:4d}  ({pct:5.1f}%)  {bar}")

    lines.append("")

    # Sharpe tier distribution
    lines.append("-" * 50)
    lines.append("SHARPE TIER DISTRIBUTION")
    lines.append("-" * 50)
    for tier in ["elite", "strong", "viable", "marginal", "unknown"]:
        count = coverage["by_sharpe_tier"].get(tier, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = "#" * int(pct / 2)
        lines.append(f"  {tier:10s}  {count:4d}  ({pct:5.1f}%)  {bar}")

    lines.append("")

    # Regime coverage
    lines.append("-" * 50)
    lines.append("REGIME COVERAGE")
    lines.append("-" * 50)
    for regime in REGIMES:
        count = coverage["by_regime"].get(regime, 0)
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"  {regime:12s}  {count:4d}  ({pct:5.1f}%)")

    lines.append("")

    # Top gaps
    lines.append("-" * 50)
    lines.append("TOP COVERAGE GAPS (by priority)")
    lines.append("-" * 50)
    for i, gap in enumerate(gaps[:20]):
        lines.append(
            f"  {i+1:2d}. [{gap['type']}] {gap['description']} "
            f"(priority: {gap['priority']})"
        )

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# GIT RATCHET
# ═══════════════════════════════════════════════════════════════════════════════


def git_commit(description: str) -> str:
    """Stage and commit changes with a descriptive message.

    Args:
        description: Commit message.

    Returns:
        Commit hash on success, empty string on failure.
    """
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        result = subprocess.run(
            ["git", "commit", "-m", description],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Extract commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            return hash_result.stdout.strip()
        else:
            logger.warning("git commit failed: %s", result.stderr)
            return ""
    except Exception as e:
        logger.error("git_commit error: %s", e)
        return ""


def git_revert(files: list[str]) -> bool:
    """Revert specified files to their last committed state.

    Args:
        files: List of file paths relative to project root.

    Returns:
        True if revert succeeded.
    """
    try:
        for f in files:
            subprocess.run(
                ["git", "checkout", "--", f],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
        return True
    except Exception as e:
        logger.error("git_revert error: %s", e)
        return False


def git_current_commit() -> str:
    """Get the current git commit hash (short form).

    Returns:
        Short commit hash string.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def git_has_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes.

    Returns:
        True if working tree is dirty.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# GAPS LOG
# ═══════════════════════════════════════════════════════════════════════════════


def log_gap_attempt(
    gap: dict[str, Any],
    mandate_path: str | Path | None = None,
    result_sharpe: float | None = None,
    result_wf_windows: str | None = None,
    status: str = "pending",
    notes: str = "",
) -> None:
    """Append a gap exploration attempt to the gaps log TSV.

    Args:
        gap: Gap dict from identify_gaps().
        mandate_path: Path to created mandate (if any).
        result_sharpe: Sharpe of the resulting strategy (if available).
        result_wf_windows: Walk-forward windows passed (e.g., "5/6").
        status: One of pending, running, success, failed, skipped.
        notes: Free-text notes.
    """
    EXPLORER_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mandate_file = Path(mandate_path).name if mandate_path else ""
    sharpe_str = f"{result_sharpe:.4f}" if result_sharpe is not None else ""
    wf_str = result_wf_windows or ""

    row = (
        f"{timestamp}\t"
        f"{gap.get('description', '')}\t"
        f"{gap.get('archetype', '')}\t"
        f"{gap.get('ticker', '')}\t"
        f"{gap.get('regime', '')}\t"
        f"{mandate_file}\t"
        f"{sharpe_str}\t"
        f"{wf_str}\t"
        f"{status}\t"
        f"{notes}"
    )

    # Write header if file doesn't exist
    if not GAPS_LOG_PATH.exists():
        header = (
            "timestamp\tgap_description\tarchetype\tticker\tregime\t"
            "mandate_created\tresult_sharpe\tresult_wf_windows\tstatus\tnotes"
        )
        GAPS_LOG_PATH.write_text(header + "\n")

    with open(GAPS_LOG_PATH, "a") as f:
        f.write(row + "\n")


def load_gaps_log() -> list[dict]:
    """Load the gaps log TSV as a list of dicts.

    Returns:
        List of dicts with column-name keys.
    """
    if not GAPS_LOG_PATH.exists():
        return []

    lines = GAPS_LOG_PATH.read_text().strip().split("\n")
    if len(lines) < 2:
        return []

    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        values = line.split("\t")
        row = dict(zip(headers, values))
        rows.append(row)

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY FEATURE EXTRACTION (for novelty scoring)
# ═══════════════════════════════════════════════════════════════════════════════


def extract_strategy_features(entry: dict) -> dict[str, Any]:
    """Extract diversity features from a registry entry.

    Args:
        entry: A registry entry dict.

    Returns:
        Feature dict with archetype, ticker, regime, trade_freq, sharpe_tier keys.
    """
    name = entry.get("strategy_name", "")
    ticker = entry.get("ticker", "")
    sharpe = entry.get("sharpe", 0) or 0
    trades = entry.get("trades", 0) or 0

    archetype = _infer_archetype(name)
    regime = ARCHETYPES.get(archetype, {}).get("regime_affinity", "unknown")
    trade_freq = _classify_trade_frequency(_estimate_trades_per_year(trades))
    sharpe_tier = _classify_sharpe_tier(sharpe)

    return {
        "archetype": archetype,
        "ticker": ticker,
        "regime": regime,
        "trade_freq": trade_freq,
        "sharpe_tier": sharpe_tier,
    }


def build_population_features(registry: list[dict]) -> list[dict[str, Any]]:
    """Build population feature list from the entire registry.

    Args:
        registry: List of registry entries.

    Returns:
        List of feature dicts for all robust strategies.
    """
    features = []
    for entry in registry:
        if entry.get("verdict") != "ROBUST":
            continue
        features.append(extract_strategy_features(entry))
    return features


# ── Shared Loop Infrastructure ─────────────────────────────────────────────


def list_loops() -> list[dict]:
    """List all registered loops from registry.yaml.

    Returns:
        List of dicts with keys: name, description, metric, direction, category.
    """
    registry = load_loop_registry()
    result = []
    for name, meta in sorted(registry.items()):
        result.append({
            "name": name,
            "description": meta.get("description", ""),
            "metric": meta.get("metric", "unknown"),
            "direction": meta.get("direction", "unknown"),
            "category": meta.get("category", "unknown"),
        })
    return result


def load_loop_registry() -> dict:
    """Load the loop registry.yaml.

    Returns:
        Dict of loop_name → metadata, or empty dict on failure.
    """
    registry_path = Path(__file__).resolve().parent / "registry.yaml"
    if not registry_path.exists():
        return {}
    try:
        import yaml
        with open(registry_path) as f:
            data = yaml.safe_load(f)
        return data.get("loops", {}) if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load registry.yaml: {e}")
        return {}


def validate_loop(loop_name: str) -> dict:
    """Validate a loop has required files.

    Checks:
        - Loop is registered in registry.yaml
        - Loop directory exists
        - program.md exists and is non-empty
        - TSV log (if configured) has valid headers

    Args:
        loop_name: Name of the loop.

    Returns:
        Dict with 'valid' (bool) and 'errors' (list[str]).
    """
    errors = []
    loops_dir = Path(__file__).resolve().parent

    # Check registry
    registry = load_loop_registry()
    if loop_name not in registry:
        errors.append(f"Loop '{loop_name}' not found in registry.yaml")

    # Check directory
    loop_dir = loops_dir / loop_name
    if not loop_dir.exists():
        errors.append(f"Loop directory '{loop_name}' does not exist")
        return {"valid": False, "errors": errors}

    # Check program.md
    program_path = loop_dir / "program.md"
    if not program_path.exists():
        errors.append("program.md is missing")
    elif program_path.stat().st_size == 0:
        errors.append("program.md is empty")

    # Check TSV log if configured
    tsv_log = registry.get(loop_name, {}).get("tsv_log")
    if tsv_log:
        tsv_path = loops_dir / tsv_log
        if tsv_path.exists():
            try:
                content = tsv_path.read_text()
                lines = [l for l in content.splitlines() if l.strip()]
                if len(lines) < 1:
                    errors.append(f"{tsv_path.name} exists but is empty (no header)")
                elif "\t" not in lines[0]:
                    errors.append(f"{tsv_path.name} header is not valid TSV")
            except Exception as e:
                errors.append(f"{tsv_path.name} is not readable: {e}")

    return {"valid": len(errors) == 0, "errors": errors}


def log_experiment(
    tsv_path: Path | str,
    metric: float,
    status: str,
    description: str,
    extra: str = "",
) -> None:
    """Append a row to a TSV experiment log. Auto-detects headers from first line.

    If the TSV file doesn't exist, creates it with a header row:
    timestamp, metric, status, description, extra

    If the file exists, appends a data row matching the existing header
    (fills missing columns with empty strings).

    Args:
        tsv_path: Path to the TSV file.
        metric: Numeric metric value.
        status: Status string (e.g., 'keep', 'discard', 'crash').
        description: Human-readable description of the experiment.
        extra: Optional extra information.
    """
    tsv_path = Path(tsv_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not tsv_path.exists():
        # Create with standard header
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        header = "timestamp\tmetric\tstatus\tdescription\textra\n"
        row = f"{timestamp}\t{metric}\t{status}\t{description}\t{extra}\n"
        tsv_path.write_text(header + row)
        return

    # Read existing header to match column count
    content = tsv_path.read_text()
    lines = content.splitlines()
    header_line = ""
    for line in lines:
        if line.strip():
            header_line = line
            break
    num_cols = len(header_line.split("\t")) if header_line else 5

    # Build row with matching column count
    values = [timestamp, str(metric), status, description, extra]
    while len(values) < num_cols:
        values.append("")
    row = "\t".join(values[:num_cols]) + "\n"

    with open(tsv_path, "a") as f:
        f.write(row)


def generate_report(tsv_path: Path | str) -> str:
    """Parse a TSV log and generate a summary report string.

    Args:
        tsv_path: Path to the TSV file.

    Returns:
        Formatted multi-line report string.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        return f"No TSV log found at {tsv_path}."

    import csv as csv_mod

    total = 0
    statuses: dict[str, int] = {}
    metric_values: list[float] = []
    last_entry = ""
    status_col_idx = None
    metric_col_idx = None
    description_col_idx = None

    with open(tsv_path, newline="") as f:
        reader = csv_mod.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return f"{tsv_path.name} is empty (no header)."

        # Try to find metric/status/description columns
        for i, col in enumerate(header):
            col_lower = col.lower()
            if col_lower in ("status", "result_status"):
                status_col_idx = i
            if col_lower in ("metric", "result_sharpe", "sharpe_ratio"):
                metric_col_idx = i
            if col_lower in ("description", "gap_description", "strategy_name", "insight"):
                description_col_idx = i

        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            total += 1

            if status_col_idx is not None and status_col_idx < len(row):
                status = row[status_col_idx].strip()
                statuses[status] = statuses.get(status, 0) + 1

            if metric_col_idx is not None and metric_col_idx < len(row):
                try:
                    metric_values.append(float(row[metric_col_idx]))
                except (ValueError, TypeError):
                    pass

            if description_col_idx is not None and description_col_idx < len(row):
                last_entry = row[description_col_idx].strip()
            elif row:
                last_entry = row[-1].strip()[:80]

    if total == 0:
        return f"No data rows in {tsv_path.name}."

    lines = [
        f"{'=' * 50}",
        f"  Report: {tsv_path.name}",
        f"{'=' * 50}",
        f"  Total entries:     {total}",
    ]

    if statuses:
        for status, count in sorted(statuses.items()):
            pct = count / total * 100
            lines.append(f"  {status:20s} {count:4d}  ({pct:.1f}%)")

    if metric_values:
        best = max(metric_values)
        avg = sum(metric_values) / len(metric_values)
        worst = min(metric_values)
        lines.append(f"  Best metric:       {best:.6f}")
        lines.append(f"  Avg metric:        {avg:.6f}")
        lines.append(f"  Worst metric:      {worst:.6f}")

    if last_entry:
        lines.append(f"  Last entry:        {last_entry[:80]}")

    lines.append(f"{'=' * 50}")
    return "\n".join(lines)


def parse_mandate(mandate_path: Path | str) -> dict:
    """Read and validate a mandate JSON file.

    Args:
        mandate_path: Path to the mandate JSON file.

    Returns:
        Dict with 'valid' (bool), 'data' (dict or None), 'error' (str or None).
    """
    mandate_path = Path(mandate_path)
    data = safe_json_load(mandate_path)
    if data is None:
        return {"valid": False, "data": None, "error": f"Failed to load {mandate_path}"}

    # Check required fields
    required = ["strategy_name"]
    missing = [f for f in required if f not in data]
    if missing:
        return {"valid": False, "data": data, "error": f"Missing required fields: {missing}"}

    return {"valid": True, "data": data, "error": None}


def parse_run_state(run_dir: Path | str) -> dict | None:
    """Read state.json from a refinement run directory.

    Args:
        run_dir: Path to the refinement run directory.

    Returns:
        State dict, or None if file doesn't exist or can't be parsed.
    """
    run_dir = Path(run_dir)
    state_path = run_dir / "state.json"
    return safe_json_load(state_path)


def safe_json_load(path: Path | str) -> Any:
    """Load JSON with error handling, returns None on failure.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data, or None on any error.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load JSON from {path}: {e}")
        return None


def safe_json_save(path: Path | str, data: Any) -> bool:
    """Save JSON with error handling, returns True on success.

    Args:
        path: Path to write the JSON file.
        data: Data to serialize.

    Returns:
        True if saved successfully, False otherwise.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except (OSError, TypeError) as e:
        logger.warning(f"Failed to save JSON to {path}: {e}")
        return False
