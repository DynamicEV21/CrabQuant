"""
CrabQuant Validation Suite

Walk-forward testing and cross-ticker validation.
This is what separates real strategies from curve-fitted ones.
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from crabquant.data import load_data
from crabquant.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""
    strategy_name: str
    ticker: str
    train_sharpe: float
    train_return: float
    test_sharpe: float
    test_return: float
    test_max_dd: float
    degradation: float  # How much Sharpe dropped from train to test
    robust: bool  # True if strategy holds up out-of-sample
    notes: str


@dataclass
class CrossTickerResult:
    """Results from cross-ticker validation."""
    strategy_name: str
    params: dict
    tickers_tested: int
    tickers_profitable: int
    tickers_passed: int
    avg_sharpe: float
    median_sharpe: float
    sharpe_std: float
    avg_return: float
    avg_max_dd: float
    win_rate_across_tickers: float
    robust: bool
    notes: str


def walk_forward_test(
    strategy_fn,
    ticker: str,
    params: dict,
    train_months: int = 18,
    test_months: int = 6,
    engine: Optional[BacktestEngine] = None,
) -> WalkForwardResult:
    """
    Walk-forward test: train on one period, validate on the next.

    Args:
        strategy_fn: Strategy function (df, params) -> (entries, exits)
        ticker: Ticker to test
        params: Strategy parameters
        train_months: Months of training data
        test_months: Months of out-of-sample data
        engine: BacktestEngine instance

    Returns:
        WalkForwardResult with train vs test comparison
    """
    if engine is None:
        engine = BacktestEngine()

    # Load full dataset
    try:
        full_df = load_data(ticker, period="3y")
    except Exception as e:
        return WalkForwardResult(
            strategy_name=strategy_fn.__name__,
            ticker=ticker,
            train_sharpe=0, train_return=0,
            test_sharpe=0, test_return=0, test_max_dd=0,
            degradation=1.0, robust=False,
            notes=f"Data error: {e}",
        )

    # Split into train/test
    split_idx = len(full_df) - int(test_months * 21)  # ~21 trading days/month
    train_df = full_df.iloc[:split_idx]
    test_df = full_df.iloc[split_idx:]

    if len(train_df) < 100 or len(test_df) < 50:
        return WalkForwardResult(
            strategy_name=strategy_fn.__name__,
            ticker=ticker,
            train_sharpe=0, train_return=0,
            test_sharpe=0, test_return=0, test_max_dd=0,
            degradation=1.0, robust=False,
            notes="Insufficient data for split",
        )

    # Train period backtest
    train_entries, train_exits = strategy_fn(train_df, params)
    train_result = engine.run(train_df, train_entries, train_exits,
                              strategy_fn.__name__, ticker, params=params)

    # Test period backtest (OUT OF SAMPLE)
    test_entries, test_exits = strategy_fn(test_df, params)
    test_result = engine.run(test_df, test_entries, test_exits,
                             strategy_fn.__name__, ticker, params=params)

    # Calculate degradation
    if train_result.sharpe > 0:
        degradation = (train_result.sharpe - test_result.sharpe) / train_result.sharpe
    else:
        degradation = 1.0

    # Robust if test Sharpe > 0.5 and degradation < 50%
    robust = test_result.sharpe > 0.5 and degradation < 0.5

    notes_parts = [
        f"Train: Sharpe {train_result.sharpe:.2f}, Return {train_result.total_return:.1%}",
        f"Test: Sharpe {test_result.sharpe:.2f}, Return {test_result.total_return:.1%}",
        f"Degradation: {degradation:.1%}",
    ]
    if test_result.num_trades < 3:
        notes_parts.append(f"⚠️ Only {test_result.num_trades} OOS trades")

    return WalkForwardResult(
        strategy_name=strategy_fn.__name__,
        ticker=ticker,
        train_sharpe=train_result.sharpe,
        train_return=train_result.total_return,
        test_sharpe=test_result.sharpe,
        test_return=test_result.total_return,
        test_max_dd=test_result.max_drawdown,
        degradation=degradation,
        robust=robust,
        notes=" | ".join(notes_parts),
    )


def cross_ticker_validation(
    strategy_fn,
    params: dict,
    tickers: list[str],
    engine: Optional[BacktestEngine] = None,
) -> CrossTickerResult:
    """
    Test a strategy across multiple tickers to check generalization.

    Args:
        strategy_fn: Strategy function
        params: Strategy parameters
        tickers: List of tickers to test
        engine: BacktestEngine instance

    Returns:
        CrossTickerResult with aggregate statistics
    """
    if engine is None:
        engine = BacktestEngine()

    results = []
    for ticker in tickers:
        try:
            df = load_data(ticker, period="2y")
            entries, exits = strategy_fn(df, params)
            result = engine.run(df, entries, exits, strategy_fn.__name__, ticker, params=params)
            if result.num_trades > 0 and not (result.sharpe == 0 and result.total_return == 0):
                results.append(result)
        except Exception as e:
            logger.warning(f"Cross-ticker {ticker} failed: {e}")

    if not results:
        return CrossTickerResult(
            strategy_name=strategy_fn.__name__,
            params=params,
            tickers_tested=len(tickers),
            tickers_profitable=0, tickers_passed=0,
            avg_sharpe=0, median_sharpe=0, sharpe_std=0,
            avg_return=0, avg_max_dd=0,
            win_rate_across_tickers=0, robust=False,
            notes="No valid results",
        )

    sharpes = [r.sharpe for r in results]
    returns = [r.total_return for r in results]
    max_dds = [abs(r.max_drawdown) for r in results]
    profitable = sum(1 for r in results if r.total_return > 0)
    passed = sum(1 for r in results if r.passed)

    avg_sharpe = sum(sharpes) / len(sharpes)
    median_sharpe = sorted(sharpes)[len(sharpes) // 2]
    sharpe_std = (sum((s - avg_sharpe) ** 2 for s in sharpes) / len(sharpes)) ** 0.5
    avg_return = sum(returns) / len(returns)
    avg_max_dd = sum(max_dds) / len(max_dds)

    # Robust if >40% of tickers are profitable and avg Sharpe > 0.5
    robust = (profitable / len(results) > 0.4) and avg_sharpe > 0.5

    notes_parts = [
        f"Tested {len(results)}/{len(tickers)} tickers",
        f"Profitable: {profitable}/{len(results)} ({profitable/len(results):.0%})",
        f"Passed: {passed}/{len(results)}",
        f"Avg Sharpe: {avg_sharpe:.2f} (σ={sharpe_std:.2f})",
        f"Median Sharpe: {median_sharpe:.2f}",
        f"Avg Return: {avg_return:.1%}",
    ]

    return CrossTickerResult(
        strategy_name=strategy_fn.__name__,
        params=params,
        tickers_tested=len(tickers),
        tickers_profitable=profitable,
        tickers_passed=passed,
        avg_sharpe=avg_sharpe,
        median_sharpe=median_sharpe,
        sharpe_std=sharpe_std,
        avg_return=avg_return,
        avg_max_dd=avg_max_dd,
        win_rate_across_tickers=profitable / len(results) if results else 0,
        robust=robust,
        notes=" | ".join(notes_parts),
    )


def full_validation(
    strategy_fn,
    params: dict,
    discovery_ticker: str,
    validation_tickers: list[str],
) -> dict:
    """
    Full validation pipeline: walk-forward + cross-ticker.

    Args:
        strategy_fn: Strategy function
        params: Strategy parameters
        discovery_ticker: Ticker where strategy was discovered (excluded from OOS)
        validation_tickers: Tickers for cross-ticker validation (should exclude discovery)

    Returns:
        Dict with walk_forward and cross_ticker results
    """
    engine = BacktestEngine()

    # Walk-forward on discovery ticker
    wf = walk_forward_test(strategy_fn, discovery_ticker, params, engine=engine)

    # Cross-ticker on validation tickers (exclude discovery ticker)
    oos_tickers = [t for t in validation_tickers if t != discovery_ticker]
    ct = cross_ticker_validation(strategy_fn, params, oos_tickers, engine=engine)

    return {
        "walk_forward": asdict(wf),
        "cross_ticker": asdict(ct),
        "overall_robust": wf.robust and ct.robust,
        "timestamp": datetime.now().isoformat(),
    }
