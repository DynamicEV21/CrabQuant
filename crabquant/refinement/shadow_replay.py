"""
Shadow Replay — Enhancement 6 for CrabQuant.

Re-runs past winning strategies on new data to detect when they stop working.
Pairs with :func:`~crabquant.refinement.stagnation.label_strategy_family`
for family-level analytics.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def shadow_replay(
    winners: list[dict],
    new_data: pd.DataFrame,
    *,
    min_sharpe: float = 0.3,
    engine=None,
) -> list[dict]:
    """Re-run past winning strategies on *new_data* and detect degradation.

    Each entry in *winners* must have at least:
    - ``strategy_code`` (str): Python source code defining a ``simulate``
      or ``generate_signals`` callable.
    - ``params`` (dict): Parameters for the strategy.
    - ``ticker`` (str): Ticker symbol (used for engine metadata).
    - ``old_sharpe`` (float, optional): The Sharpe achieved when the
      strategy was originally promoted.  Falls back to 0.0.

    The function dynamically compiles and executes the strategy code,
    runs a single backtest on *new_data*, and compares the new Sharpe
    to the old one.  A strategy is **degraded** when its new Sharpe
    drops below 50 % of the original value.

    Args:
        winners: List of winner dicts.
        new_data: OHLCV DataFrame for the replay window.
        min_sharpe: Minimum Sharpe to consider a result meaningful
            (default 0.3).  Strategies below this on new data are
            flagged as degraded regardless of the ratio.
        engine: Optional :class:`BacktestEngine` instance.  Created
            with defaults when *None*.

    Returns:
        List of dicts, one per winner, with keys:
        ``name``, ``ticker``, ``old_sharpe``, ``new_sharpe``,
        ``degraded`` (bool), ``error`` (str or empty).
    """
    from crabquant.engine import BacktestEngine

    if engine is None:
        engine = BacktestEngine()

    results: list[dict] = []

    for winner in winners:
        name = winner.get("name", winner.get("strategy_name", "unnamed"))
        ticker = winner.get("ticker", "UNKNOWN")
        old_sharpe = float(winner.get("old_sharpe", 0.0))
        params = winner.get("params", {})
        code = winner.get("strategy_code", "")

        entry: dict = {
            "name": name,
            "ticker": ticker,
            "old_sharpe": old_sharpe,
            "new_sharpe": 0.0,
            "degraded": True,
            "error": "",
        }

        if not code.strip():
            entry["error"] = "No strategy_code provided"
            results.append(entry)
            continue

        try:
            strategy_fn = _compile_strategy(code, name)
            entries, exits = strategy_fn(new_data, params)
            bt_result = engine.run(
                new_data, entries, exits, name, ticker, params=params,
            )
            new_sharpe = bt_result.sharpe
            entry["new_sharpe"] = new_sharpe

            # Degraded if new Sharpe < 50% of old, or below min_sharpe
            if old_sharpe > 0:
                entry["degraded"] = (
                    new_sharpe < old_sharpe * 0.5 or new_sharpe < min_sharpe
                )
            else:
                # No meaningful old Sharpe — use min_sharpe as sole gate
                entry["degraded"] = new_sharpe < min_sharpe

        except Exception as exc:
            logger.warning("Shadow replay failed for %s: %s", name, exc)
            entry["error"] = str(exc)
            entry["degraded"] = True

        results.append(entry)

    return results


def _compile_strategy(code: str, name: str = "strategy"):
    """Compile strategy source code and return the callable.

    Looks for a function named ``simulate`` or ``generate_signals`` in
    the compiled module namespace.

    Args:
        code: Strategy source code.
        name: Strategy name (used in error messages).

    Returns:
        Callable ``(df, params) -> (entries, exits)``.

    Raises:
        RuntimeError: If no suitable function is found in *code*.
    """
    import types

    ns: dict = {}
    try:
        exec(compile(code, f"<{name}>", "exec"), ns)
    except SyntaxError as exc:
        raise RuntimeError(f"Syntax error in {name}: {exc}") from exc

    for fn_name in ("simulate", "generate_signals"):
        fn = ns.get(fn_name)
        if callable(fn):
            return fn

    available = [k for k in ns if callable(ns[k],) and not k.startswith("_")]
    raise RuntimeError(
        f"No 'simulate' or 'generate_signals' function found in {name}. "
        f"Available callables: {available or 'none'}"
    )
