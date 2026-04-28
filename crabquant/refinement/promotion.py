"""Promotion — full validation on success and auto-promotion to STRATEGY_REGISTRY.

Component 7 (full_validation):
    Before promoting a strategy as a "winner", run rolling walk-forward validation
    (multiple overlapping train/test windows) and cross-ticker validation.
    Only promote if both pass minimum thresholds.  Rolling windows handle regime
    shifts naturally — a strategy that performs consistently across multiple windows
    is robust regardless of whether the regime changes between train and test.

Component 9 (auto_promotion):
    Strategies that pass full_validation auto-register in STRATEGY_REGISTRY
    with regime tags (preferred_regimes) computed from per-regime Sharpe analysis.
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
    *,
    use_rolling: bool = True,
    rolling_config: dict | None = None,
    is_regime_specific: bool = False,
) -> dict[str, Any]:
    """Run rolling walk-forward + cross-ticker validation before promoting.

    Uses rolling walk-forward windows by default (18mo train / 6mo test / 6mo
    step).  This handles regime shifts naturally — a strategy that holds up
    across multiple overlapping windows is robust regardless of regime changes.
    Falls back to single-split walk-forward if ``use_rolling=False``.

    Args:
        strategy_fn: Callable (df, params) -> (entries, exits).
        params: Strategy parameters to use.
        discovery_ticker: Ticker where strategy was discovered.
        validation_tickers: All tickers including discovery.
        min_walk_forward_sharpe: Min avg OOS Sharpe for walk-forward to pass.
        min_cross_ticker_sharpe: Min avg Sharpe for cross-ticker to pass.
        use_rolling: Use rolling walk-forward (default True).  Set False for
            the old single-split behaviour.
        rolling_config: Override rolling window params.  Keys: train_window,
            test_window, step, min_avg_test_sharpe, min_windows_passed.
        is_regime_specific: If True, apply relaxed thresholds (regime-specific
            strategies don't need to work everywhere — just in their regime).

    Returns:
        Dict with keys: passed, walk_forward_robust, cross_ticker_robust,
        walk_forward, cross_ticker, error, validation_method, is_regime_specific.
    """
    from crabquant.validation import (
        walk_forward_test,
        cross_ticker_validation,
        rolling_walk_forward,
    )
    from crabquant.refinement.config import VALIDATION_CONFIG

    # ── Adjust thresholds for regime-specific strategies ──
    if is_regime_specific:
        wf_factor = VALIDATION_CONFIG.get("regime_specific_wf_sharpe_factor", 0.6)
        ct_factor = VALIDATION_CONFIG.get("regime_specific_ct_sharpe_factor", 0.7)
        soft_floor = VALIDATION_CONFIG.get("soft_promote_test_sharpe", 0.3)
        min_walk_forward_sharpe = max(min_walk_forward_sharpe * wf_factor, soft_floor)
        min_cross_ticker_sharpe = max(min_cross_ticker_sharpe * ct_factor, soft_floor)

    result: dict[str, Any] = {
        "passed": False,
        "walk_forward_robust": False,
        "cross_ticker_robust": False,
        "walk_forward": None,
        "cross_ticker": None,
        "error": None,
        "validation_method": "rolling" if use_rolling else "single_split",
        "is_regime_specific": is_regime_specific,
    }

    # Merge rolling config from VALIDATION_CONFIG + caller overrides
    rcfg = {
        "train_window": "18mo",
        "test_window": "6mo",
        "step": "6mo",
        "min_avg_test_sharpe": 0.5,
        "min_windows_passed": 2,
    }
    rcfg.update(VALIDATION_CONFIG.get("rolling", {}))
    if rolling_config:
        rcfg.update(rolling_config)

    try:
        if use_rolling:
            # ── Rolling walk-forward (multiple overlapping windows) ──
            rwf = rolling_walk_forward(
                strategy_fn,
                discovery_ticker,
                params,
                train_window=rcfg["train_window"],
                test_window=rcfg["test_window"],
                step=rcfg["step"],
                min_avg_test_sharpe=rcfg["min_avg_test_sharpe"],
                min_windows_passed=rcfg["min_windows_passed"],
            )
            result["walk_forward"] = {
                "avg_test_sharpe": rwf.avg_test_sharpe,
                "min_test_sharpe": rwf.min_test_sharpe,
                "avg_degradation": rwf.avg_degradation,
                "num_windows": rwf.num_windows,
                "windows_passed": rwf.windows_passed,
                "robust": rwf.robust,
                "notes": rwf.notes,
                "window_results": rwf.window_results,
            }
            wf_pass = rwf.robust and rwf.avg_test_sharpe >= min_walk_forward_sharpe
        else:
            # ── Single-split walk-forward (legacy) ──
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
            wf_pass = wf.robust and wf.test_sharpe >= min_walk_forward_sharpe

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
    *,
    regime_tags: dict | None = None,
) -> bool:
    """Register a strategy in STRATEGY_REGISTRY.

    Writes the .py file to the strategies directory and inserts into the
    in-memory registry with optional regime tags.  Prevents duplicates.

    Args:
        strategy_name: Name for the strategy (e.g., "refined_momentum_spy_1").
        strategy_module: Loaded module with generate_signals, DEFAULT_PARAMS, etc.
        strategy_code: Optional source code to write to file.
        regime_tags: Optional dict from ``compute_strategy_regime_tags()`` with
            keys preferred_regimes, acceptable_regimes, weak_regimes,
            regime_sharpes, is_regime_specific.

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

        # Insert into registry with regime metadata
        registry_entry = {
            "fn": strategy_module.generate_signals,
            "defaults": getattr(strategy_module, "DEFAULT_PARAMS", {}),
            "grid": getattr(strategy_module, "PARAM_GRID", {}),
            "description": getattr(strategy_module, "DESCRIPTION", ""),
            "matrix_fn": getattr(strategy_module, "generate_signals_matrix", None),
        }

        # Attach regime tags if available
        if regime_tags:
            registry_entry["preferred_regimes"] = regime_tags.get("preferred_regimes", [])
            registry_entry["acceptable_regimes"] = regime_tags.get("acceptable_regimes", [])
            registry_entry["weak_regimes"] = regime_tags.get("weak_regimes", [])
            registry_entry["regime_sharpes"] = regime_tags.get("regime_sharpes", {})
            registry_entry["is_regime_specific"] = regime_tags.get("is_regime_specific", False)
        else:
            registry_entry["preferred_regimes"] = []
            registry_entry["acceptable_regimes"] = []
            registry_entry["weak_regimes"] = []
            registry_entry["regime_sharpes"] = {}
            registry_entry["is_regime_specific"] = False

        STRATEGY_REGISTRY[strategy_name] = registry_entry

        logger.info(
            "Registered strategy %s in STRATEGY_REGISTRY (regimes: %s)",
            strategy_name,
            regime_tags.get("preferred_regimes", []) if regime_tags else "not tagged",
        )
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
    Computes regime tags before registration so the strategy is queryable
    by regime in the execution layer.

    Args:
        strategy_code: Full Python source of the strategy.
        strategy_module: Loaded strategy module.
        result: BacktestResult from the engine.
        validation: Dict from ``run_full_validation_check``.
        state: RunState for the refinement run.

    Returns:
        Dict with keys: registered, strategy_name, regime_tags, error.
    """
    strategy_name = f"refined_{state.mandate_name}"
    response: dict[str, Any] = {
        "registered": False,
        "strategy_name": strategy_name,
        "regime_tags": None,
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

    # Compute regime tags before registration
    regime_tags = None
    try:
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags
        regime_tags = compute_strategy_regime_tags(
            strategy_module.generate_signals,
            result.params if result.params else {},
            ticker=result.ticker,
        )
        logger.info(
            "Regime tags for %s: preferred=%s, specific=%s",
            strategy_name,
            regime_tags["preferred_regimes"],
            regime_tags["is_regime_specific"],
        )
    except Exception as e:
        logger.warning("Regime tagging failed for %s: %s (registering without tags)", strategy_name, e)

    # Register in STRATEGY_REGISTRY with regime tags
    reg_ok = register_strategy(
        strategy_name, strategy_module,
        strategy_code=strategy_code,
        regime_tags=regime_tags,
    )
    if not reg_ok:
        response["error"] = f"Failed to register {strategy_name}."
        return response

    response["regime_tags"] = regime_tags

    # Promote to winners
    try:
        promote_to_winner(strategy_code, result, validation, state, strategy_module)
        # Update validation_status to "promoted" in winners.json
        _update_winner_status(strategy_name, "promoted")
        # Add regime tags to the winner entry
        if regime_tags:
            _update_winner_regime_tags(strategy_name, regime_tags)
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


def _update_winner_regime_tags(strategy_name: str, regime_tags: dict) -> None:
    """Add regime tags to a winner entry in winners.json.

    Finds the most recent entry matching ``strategy_name`` and adds
    ``preferred_regimes``, ``weak_regimes``, and ``is_regime_specific`` fields.
    """
    winners_path = _get_winners_path()
    if not winners_path.exists():
        return

    try:
        winners = json.loads(winners_path.read_text())
    except (json.JSONDecodeError, OSError):
        return

    updated = False
    for entry in reversed(winners):
        if entry.get("strategy") == strategy_name:
            entry["regime_tags"] = {
                "preferred_regimes": regime_tags.get("preferred_regimes", []),
                "acceptable_regimes": regime_tags.get("acceptable_regimes", []),
                "weak_regimes": regime_tags.get("weak_regimes", []),
                "is_regime_specific": regime_tags.get("is_regime_specific", False),
            }
            updated = True
            break

    if updated:
        winners_path.write_text(json.dumps(winners, indent=2))
