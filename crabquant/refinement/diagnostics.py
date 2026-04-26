"""CrabQuant Refinement — Diagnostics Helpers."""

import hashlib
import logging
from types import ModuleType

import pandas as pd

from crabquant.data import load_data
from crabquant.engine.backtest import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


def run_backtest_safely(
    strategy_module: ModuleType,
    ticker: str,
    period: str = "2y",
    return_portfolio: bool = False,
) -> tuple:
    """Load data, generate signals, and run a backtest.

    Returns (BacktestResult, df, portfolio) on success, or (None, None, None) on any
    error.  When return_portfolio=True the engine is called with that kwarg; if the
    engine does not yet support it (TypeError) the call is retried without it and
    portfolio is returned as None.
    """
    try:
        df = load_data(ticker, period)
        if df is None or df.empty:
            logger.warning("No data for %s/%s", ticker, period)
            return None, None, None

        params = strategy_module.DEFAULT_PARAMS
        entries, exits = strategy_module.generate_signals(df, params)

        strategy_name = getattr(strategy_module, "DESCRIPTION", "unknown")[:30]
        engine = BacktestEngine()

        portfolio = None
        if return_portfolio:
            try:
                result, portfolio = engine.run(
                    df, entries, exits,
                    strategy_name=strategy_name,
                    ticker=ticker,
                    return_portfolio=True,
                )
            except TypeError:
                result = engine.run(
                    df, entries, exits,
                    strategy_name=strategy_name,
                    ticker=ticker,
                )
        else:
            result = engine.run(
                df, entries, exits,
                strategy_name=strategy_name,
                ticker=ticker,
            )

        return result, df, portfolio

    except ValueError as e:
        logger.warning("Data error for %s: %s", ticker, e)
        return None, None, None
    except Exception as e:
        logger.warning("Backtest error for %s: %s", ticker, e)
        return None, None, None


def compute_sharpe_by_year(portfolio) -> dict:
    """Compute annualised Sharpe ratio per calendar year from a vbt.Portfolio.

    Returns {"2022": 1.23, "2023": 0.87, ...} or {} on any error.
    Years with fewer than 10 bars are skipped.
    """
    try:
        returns = portfolio.returns()
        idx = returns.index
        result = {}
        for year in sorted(set(idx.year)):
            mask = idx.year == year
            yr = returns[mask]
            if len(yr) < 10:
                continue
            std = yr.std()
            if std > 1e-10:
                result[str(year)] = round(yr.mean() / std * (252 ** 0.5), 4)
            else:
                result[str(year)] = 0.0
        return result
    except Exception as e:
        logger.warning("Sharpe-by-year error: %s", e)
        return {}


def compute_strategy_hash(code: str) -> str:
    """Deterministic 12-char SHA-256 hash of strategy source code.

    Normalises by stripping blank lines and per-line leading/trailing whitespace
    so trivial formatting differences don't produce different hashes.
    """
    normalized = "\n".join(line.strip() for line in code.split("\n") if line.strip())
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]
