"""Promotion — full validation on success and auto-promotion to STRATEGY_REGISTRY.

Component 7 (full_validation):
    Before promoting a strategy as a "winner", run walk-forward validation
    (train/test split) and cross-ticker validation (test on different tickers).
    Only promote if both pass minimum Sharpe thresholds.

Component 9 (auto_promotion):
    Strategies that pass full_validation auto-register in STRATEGY_REGISTRY.
    Save strategy source + metadata (Sharpe, regime performance, date).
    Prevent duplicate registrations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crabquant.validation import WalkForwardResult, CrossTickerResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal path helpers (mockable in tests)
# ---------------------------------------------------------------------------

def _get_strategies_dir() -> Path:
    """Return the directory where strategy .py files live."""
    # Try relative to this file first, then absolute
    candidates = [
        Path(__file__).parent.parent / "strategies",
        Path("crabquant/strategies"),
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


def _get_winners_path() -> Path:
    """Return the path to the winners.json file."""
    return Path("results/winners/winners.json")


# ---------------------------------------------------------------------------
# Component 7: Full Validation Check
# ---------------------------------------------------------------------------

def run_full_validation_check(
    strategy_fn,
    params: dict,
    discovery_ticker: str,
    validation_tickers: list[str],
    min_walk_forward_sharpe: float = 0.5,
    min_cross_ticker_sharpe: float = 0.5,
) -> dict[str, Any]:
    """Run walk-forward + cross-ticker validation before promoting.

    Uses the existing ``crabquant.validation`` module for the heavy lifting.
    Returns a result dict with ``passed`` flag and detailed metrics.

    Args:
        strategy_fn: Callable (df, params) -> (entries, exits).
        params: Strategy parameters to use.
        discovery_ticker: Ticker where strategy was discovered.
        validation_tickers: All tickers including discovery.
        min_walk_forward_sharpe: Min OOS Sharpe for walk-forward to pass.
        min_cross_ticker_sharpe: Min avg Sharpe for cross-ticker to pass.

    Returns:
        Dict with keys: passed, walk_forward_robust, cross_ticker_robust,
        walk_forward, cross_ticker, error.
    """
    from crabquant.validation import walk_forward_test, cross_ticker_validation

    result: dict[str, Any] = {
        "passed": False,
        "walk_forward_robust": False,
        "cross_ticker_robust": False,
        "walk_forward": None,
        "cross_ticker": None,
        "error": None,
    }

    try:
        # Walk-forward on discovery ticker
        wf = walk_forward_test(strategy_fn, discovery_ticker, params)
        result["walk_forward"] = {
            "test_sharpe": wf.test_sharpe,
            "train_sharpe": wf.train_sharpe,
            "degradation": wf.degradation,
            "robust": wf.robust,
            "notes": wf.notes,
            "regime_shift": wf.regime_shift,
            "test_regime": wf.test_regime,
            "train_regime": wf.train_regime,
        }

        # Cross-ticker (exclude discovery ticker from OOS set)
        oos_tickers = [t for t in validation_tickers if t != discovery_ticker]
        if oos_tickers:
            ct = cross_ticker_validation(strategy_fn, params, oos_tickers)
            result["cross_ticker"] = {
                "avg_sharpe": ct.avg_sharpe,
                "median_sharpe": ct.median_sharpe,
                "robust": ct.robust,
                "tickers_profitable": ct.tickers_profitable,
                "tickers_tested": ct.tickers_tested,
                "notes": ct.notes,
            }

        # Evaluate
        wf_pass = wf.robust and wf.test_sharpe >= min_walk_forward_sharpe
        ct_pass = True
        if result["cross_ticker"]:
            ct_pass = ct.robust and ct.avg_sharpe >= min_cross_ticker_sharpe
        elif not oos_tickers:
            # No cross-ticker tickers available — pass by default
            ct_pass = True

        result["walk_forward_robust"] = wf_pass
        result["cross_ticker_robust"] = ct_pass
        result["passed"] = wf_pass and ct_pass

    except Exception as e:
        logger.error("Full validation error: %s", e)
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Component 7: Promote to Winner
# ---------------------------------------------------------------------------

def promote_to_winner(
    strategy_code: str,
    result: Any,  # BacktestResult
    validation: dict[str, Any],
    state: Any,  # RunState
    strategy_module: Any = None,
) -> dict[str, Any]:
    """Promote a validated strategy to the winners file.

    Writes the strategy metadata to winners.json.  Does NOT register in
    STRATEGY_REGISTRY (that's ``auto_promote``'s job).

    Args:
        strategy_code: Full Python source of the strategy.
        result: BacktestResult from the engine.
        validation: Dict from ``run_full_validation_check``.
        state: RunState for the refinement run.
        strategy_module: Loaded strategy module (for metadata extraction).

    Returns:
        Dict with promotion details.
    """
    strategy_name = f"refined_{state.mandate_name}"
    winners_path = _get_winners_path()

    # Determine validation_status based on validation result
    validation_status = "backtest_only"
    if validation.get("passed", False):
        validation_status = "walk_forward_passed"

    entry = {
        "strategy": strategy_name,
        "ticker": result.ticker,
        "sharpe": result.sharpe,
        "return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "trades": result.num_trades,
        "params": result.params if result.params else {},
        "refinement_run": state.run_id,
        "refinement_turns": state.current_turn,
        "validation": validation,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": validation_status,
    }

    winners_path.parent.mkdir(parents=True, exist_ok=True)
    winners: list[dict] = []
    if winners_path.exists():
        try:
            winners = json.loads(winners_path.read_text())
        except (json.JSONDecodeError, OSError):
            winners = []

    winners.append(entry)
    winners_path.write_text(json.dumps(winners, indent=2))

    logger.info("Promoted %s to winners (Sharpe %.2f)", strategy_name, result.sharpe)
    return {"strategy_name": strategy_name, "winners_file": str(winners_path)}


# ---------------------------------------------------------------------------
# Component 9: Auto-Promotion
# ---------------------------------------------------------------------------

def is_already_registered(strategy_name: str) -> bool:
    """Check if a strategy is already in STRATEGY_REGISTRY."""
    from crabquant.strategies import STRATEGY_REGISTRY
    return strategy_name in STRATEGY_REGISTRY


def register_strategy(
    strategy_name: str,
    strategy_module: Any,
    strategy_code: str | None = None,
) -> bool:
    """Register a strategy in STRATEGY_REGISTRY.

    Writes the .py file to the strategies directory and inserts into the
    in-memory registry.  Prevents duplicate registrations.

    Args:
        strategy_name: Name for the strategy (e.g., "refined_momentum_spy_1").
        strategy_module: Loaded module with generate_signals, DEFAULT_PARAMS, etc.
        strategy_code: Optional source code to write to file.

    Returns:
        True if registered successfully, False if duplicate or error.
    """
    from crabquant.strategies import STRATEGY_REGISTRY

    # Prevent duplicate
    if strategy_name in STRATEGY_REGISTRY:
        logger.warning("Strategy %s already registered, skipping.", strategy_name)
        return False

    try:
        # Write strategy file
        if strategy_code:
            strategies_dir = _get_strategies_dir()
            strategies_dir.mkdir(parents=True, exist_ok=True)
            strategy_file = strategies_dir / f"{strategy_name}.py"
            strategy_file.write_text(strategy_code)

        # Insert into registry
        STRATEGY_REGISTRY[strategy_name] = (
            strategy_module.generate_signals,
            getattr(strategy_module, "DEFAULT_PARAMS", {}),
            getattr(strategy_module, "PARAM_GRID", {}),
            getattr(strategy_module, "DESCRIPTION", ""),
            getattr(strategy_module, "generate_signals_matrix", None),
        )

        logger.info("Registered strategy %s in STRATEGY_REGISTRY", strategy_name)
        return True

    except Exception as e:
        logger.error("Failed to register strategy %s: %s", strategy_name, e)
        return False


def auto_promote(
    strategy_code: str,
    strategy_module: Any,
    result: Any,  # BacktestResult
    validation: dict[str, Any],
    state: Any,  # RunState
) -> dict[str, Any]:
    """Auto-promote a validated strategy to STRATEGY_REGISTRY + winners.

    Only promotes if validation["passed"] is True.  Prevents duplicates.

    Args:
        strategy_code: Full Python source of the strategy.
        strategy_module: Loaded strategy module.
        result: BacktestResult from the engine.
        validation: Dict from ``run_full_validation_check``.
        state: RunState for the refinement run.

    Returns:
        Dict with keys: registered, strategy_name, error.
    """
    strategy_name = f"refined_{state.mandate_name}"
    response: dict[str, Any] = {
        "registered": False,
        "strategy_name": strategy_name,
        "error": None,
    }

    # Check validation passed
    if not validation.get("passed", False):
        response["error"] = "Validation did not pass — skipping promotion."
        logger.info("Skipping promotion for %s: validation failed.", strategy_name)
        return response

    # Check for duplicate
    if is_already_registered(strategy_name):
        response["error"] = f"Strategy {strategy_name} already registered."
        logger.info("Skipping promotion for %s: already registered.", strategy_name)
        return response

    # Register in STRATEGY_REGISTRY
    reg_ok = register_strategy(strategy_name, strategy_module, strategy_code=strategy_code)
    if not reg_ok:
        response["error"] = f"Failed to register {strategy_name}."
        return response

    # Promote to winners
    try:
        promote_to_winner(strategy_code, result, validation, state, strategy_module)
        # Update validation_status to "promoted" in winners.json
        _update_winner_status(strategy_name, "promoted")
    except Exception as e:
        logger.warning("Failed to write to winners file: %s", e)

    response["registered"] = True
    logger.info("Auto-promoted %s (Sharpe %.2f)", strategy_name, result.sharpe)
    return response


def _update_winner_status(strategy_name: str, status: str) -> None:
    """Update the validation_status of a winner entry in winners.json.

    Finds the most recent entry matching ``strategy_name`` and sets its
    ``validation_status`` to the given value.
    """
    winners_path = _get_winners_path()
    if not winners_path.exists():
        return

    try:
        winners = json.loads(winners_path.read_text())
    except (json.JSONDecodeError, OSError):
        return

    updated = False
    # Walk in reverse to update the most recent match
    for entry in reversed(winners):
        if entry.get("strategy") == strategy_name:
            entry["validation_status"] = status
            updated = True
            break

    if updated:
        winners_path.write_text(json.dumps(winners, indent=2))
