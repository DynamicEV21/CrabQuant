"""
Gate 3: Smoke backtest validation — catch overtrading/undertrading and
suspicious metrics before committing to a full backtest.

Runs a quick 6-month backtest with timeout. Checks for:
- NaN/Inf metrics (corrupted computation)
- Suspiciously high Sharpe (> 5.0) — likely lookahead bias
- Zero trades (dead strategy)
- Excessive trades (overtrading)
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

# ── Thresholds ───────────────────────────────────────────────────────────────

_MAX_SHARPE = 5.0        # Suspiciously perfect — likely lookahead
_MAX_TRADES = 300         # Overtrading threshold for 6-month window
_TIMEOUT_DEFAULT = 10     # Seconds


# ── Internal helpers (mockable) ─────────────────────────────────────────────

def _load_strategy_module(code: str) -> ModuleType | None:
    """Load strategy code as a module using the same temp-file approach as Gate 2."""
    from crabquant.refinement.validation_gates import _load_module_from_code

    module, _err = _load_module_from_code(code)
    return module


def _run_smoke_backtest(
    module: ModuleType,
    df: Any,
    *,
    timeout: int = _TIMEOUT_DEFAULT,
) -> Any:
    """Execute the actual smoke backtest. Returns a BacktestResult-like object."""
    from crabquant.engine.backtest import BacktestEngine

    entries, exits = module.generate_signals(df, module.DEFAULT_PARAMS)
    engine = BacktestEngine()
    result = engine.run(
        df, entries, exits,
        strategy_name="smoke_gate3",
        ticker="smoke",
        params=module.DEFAULT_PARAMS,
    )
    return result


# ── Public API ───────────────────────────────────────────────────────────────

def gate_smoke_backtest(
    code: str,
    ticker: str = "AAPL",
    *,
    timeout: int = _TIMEOUT_DEFAULT,
) -> tuple[bool, list[str]]:
    """Gate 3: quick 6-month smoke backtest validation.

    Returns:
        (passed, errors) — errors empty when passed is True.
    """
    errors: list[str] = []

    module = _load_strategy_module(code)
    if module is None:
        return (False, ["Failed to load strategy module — import/exec error"])

    # Load 6 months of data
    try:
        from crabquant.data import load_data

        df = load_data(ticker, period="6mo")
        if df is None or df.empty:
            return (False, [f"No data available for {ticker} (6mo)"])
    except Exception as e:
        return (False, [f"Data load error: {e}"])

    # Run backtest
    try:
        result = _run_smoke_backtest(module, df, timeout=timeout)
    except Exception as e:
        return (False, [f"Backtest error: {e}"])

    # ── Validation checks ──

    # NaN / Inf metrics
    key_metrics = {
        "sharpe": getattr(result, "sharpe", None),
        "total_return": getattr(result, "total_return", None),
        "max_drawdown": getattr(result, "max_drawdown", None),
    }
    nan_inf_metrics = [
        name for name, val in key_metrics.items()
        if val is not None and (np.isnan(val) or np.isinf(val))
    ]
    if nan_inf_metrics:
        errors.append(
            f"NaN/Inf metrics: {', '.join(f'{n}={key_metrics[n]}' for n in nan_inf_metrics)}"
        )

    # Suspiciously perfect Sharpe
    sharpe = getattr(result, "sharpe", None)
    if sharpe is not None and sharpe > _MAX_SHARPE:
        errors.append(
            f"Suspiciously high Sharpe ({sharpe:.1f}) — likely lookahead bias"
        )

    # Zero trades
    num_trades = getattr(result, "num_trades", 0)
    if num_trades == 0:
        errors.append("Zero trades generated — strategy produced no signals")

    # Overtrading
    if num_trades > _MAX_TRADES:
        errors.append(
            f"Excessive trades: {num_trades} > {_MAX_TRADES} — likely overtrading"
        )

    return (len(errors) == 0, errors)
