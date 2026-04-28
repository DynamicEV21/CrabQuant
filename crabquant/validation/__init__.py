"""
CrabQuant Validation Suite

Walk-forward testing and cross-ticker validation.
This is what separates real strategies from curve-fitted ones.
"""

import logging
from dataclasses import dataclass, asdict, field
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
    train_regime: str = ""  # Regime during train period
    test_regime: str = ""   # Regime during test period
    regime_shift: bool = False  # True if train and test regimes differ


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


def _detect_regime_for_period(df: pd.DataFrame, full_spy_df: pd.DataFrame | None = None) -> str:
    """Detect market regime for a given time period.
    
    Uses SPY data sliced to match the period of the given DataFrame.
    Falls back to the DataFrame's own close column if SPY data unavailable.
    Returns the regime value as a string.
    """
    try:
        from crabquant.regime import detect_regime
        
        if full_spy_df is not None and len(full_spy_df) > 0 and hasattr(df.index, 'min'):
            # Slice SPY to match the DataFrame's date range
            start = df.index.min()
            end = df.index.max()
            spy_slice = full_spy_df.loc[start:end]
            if len(spy_slice) >= 20:
                regime, _ = detect_regime(spy_slice)
                return regime.value
        
        # Fallback: use the ticker's own data as a proxy
        if len(df) >= 20:
            regime, _ = detect_regime(df)
            return regime.value
    except Exception:
        pass
    return "unknown"


def walk_forward_test(
    strategy_fn,
    ticker: str,
    params: dict,
    train_months: int = 18,
    test_months: int = 6,
    engine: Optional[BacktestEngine] = None,
    *,
    min_train_bars: int = 252,
    min_test_sharpe: float = 0.3,
    min_test_trades: int = 10,
    max_degradation: float = 0.7,
) -> WalkForwardResult:
    """
    Walk-forward test: train on one period, validate on the next.

    Args:
        strategy_fn: Strategy function (df, params) -> (entries, exits)
        ticker: Ticker to test
        params: Strategy parameters
        train_months: Months of training data (default 18 → 75% of 3y)
        test_months: Months of out-of-sample data (default 6 → 25% of 3y)
        engine: BacktestEngine instance
        min_train_bars: Minimum bars required in train split (default 252 ≈ 1 year)
        min_test_sharpe: Minimum OOS Sharpe to be considered robust (default 0.3)
        min_test_trades: Minimum trades in test period to be considered robust (default 10)
        max_degradation: Max allowed degradation ratio.  Test Sharpe must be
            >= train_sharpe * (1 - max_degradation).  E.g. 0.7 means if train
            Sharpe is 2.0, test must be >= 0.6.  Default 0.7.

    Returns:
        WalkForwardResult with train vs test comparison, including regime context
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

    if len(train_df) < min_train_bars or len(test_df) < 50:
        return WalkForwardResult(
            strategy_name=strategy_fn.__name__,
            ticker=ticker,
            train_sharpe=0, train_return=0,
            test_sharpe=0, test_return=0, test_max_dd=0,
            degradation=1.0, robust=False,
            notes=f"Insufficient data for split (train={len(train_df)}, need>={min_train_bars})",
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
        degradation = max(0.0, degradation)  # Clamp: negative degradation = improvement
    elif test_result.sharpe > 0:
        degradation = 0.0  # Negative train but positive test = improvement
    else:
        degradation = 1.0  # Both negative

    # Detect regimes for train and test periods
    train_regime = ""
    test_regime = ""
    regime_shift = False
    try:
        full_spy_df = load_data("SPY", period="3y")
        train_regime = _detect_regime_for_period(train_df, full_spy_df)
        test_regime = _detect_regime_for_period(test_df, full_spy_df)
        regime_shift = train_regime != test_regime and train_regime != "unknown" and test_regime != "unknown"
    except Exception:
        pass

    # Robustness check using configurable thresholds:
    #   1. test_sharpe >= min_test_sharpe
    #   2. test_sharpe >= train_sharpe * (1 - max_degradation)  (degradation gate)
    #   3. Enough trades: num_test_trades >= min_test_trades
    # Regime shifts get extra leniency on the Sharpe floor (halved).
    sharpe_floor = min_test_sharpe * 0.5 if regime_shift else min_test_sharpe
    degradation_ok = degradation <= max_degradation
    sharpe_ok = test_result.sharpe >= sharpe_floor
    trades_ok = test_result.num_trades >= min_test_trades
    robust = sharpe_ok and degradation_ok and trades_ok

    notes_parts = [
        f"Train: Sharpe {train_result.sharpe:.2f}, Return {train_result.total_return:.1%}",
        f"Test: Sharpe {test_result.sharpe:.2f}, Return {test_result.total_return:.1%}",
        f"Degradation: {degradation:.1%}",
    ]
    if train_regime:
        notes_parts.append(f"Train regime: {train_regime}")
    if test_regime:
        notes_parts.append(f"Test regime: {test_regime}")
    if regime_shift:
        notes_parts.append("⚠️ Regime shift detected")
    if test_result.num_trades < min_test_trades:
        notes_parts.append(f"⚠️ Only {test_result.num_trades} OOS trades (need >= {min_test_trades})")
    if not degradation_ok:
        notes_parts.append(f"⚠️ Degradation {degradation:.1%} exceeds max {max_degradation:.0%}")

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
        train_regime=train_regime,
        test_regime=test_regime,
        regime_shift=regime_shift,
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


def _parse_duration(s: str) -> int:
    """Parse a duration string like '18mo' or '6mo' into trading-day bars.

    Assumes ~21 trading days per month.
    """
    s = s.strip().lower()
    if s.endswith("mo"):
        return int(s[:-2]) * 21
    if s.endswith("d"):
        return int(s[:-1])
    if s.endswith("y"):
        return int(s[:-1]) * 252
    return int(s)


@dataclass
class RollingWalkForwardResult:
    """Results from rolling walk-forward validation."""
    strategy_name: str
    ticker: str
    num_windows: int
    windows_passed: int
    avg_test_sharpe: float
    min_test_sharpe: float
    avg_degradation: float
    robust: bool
    notes: str
    window_results: list = field(default_factory=list)


def rolling_walk_forward(
    strategy_fn,
    ticker: str,
    params: dict,
    *,
    train_window: str = "18mo",
    test_window: str = "6mo",
    step: str = "6mo",
    min_avg_test_sharpe: float = 0.5,
    min_windows_passed: int = 2,
    engine: Optional[BacktestEngine] = None,
) -> RollingWalkForwardResult:
    """Validate across multiple rolling walk-forward windows.

    Instead of a single train/test split, this slides overlapping windows
    across the data and runs a backtest on each.  A strategy is robust if
    it meets the Sharpe floor on enough windows.

    Args:
        strategy_fn: Strategy function (df, params) -> (entries, exits)
        ticker: Ticker to test
        params: Strategy parameters
        train_window: Training duration (e.g. "18mo", "252d", "1y")
        test_window: Test duration (e.g. "6mo")
        step: Step between window starts (e.g. "6mo")
        min_avg_test_sharpe: Minimum average test Sharpe across windows
        min_windows_passed: Minimum windows that must individually pass
        engine: BacktestEngine instance

    Returns:
        RollingWalkForwardResult compatible with the WalkForwardResult interface
        (has ``robust``, ``notes``, and key metrics).
    """
    if engine is None:
        engine = BacktestEngine()

    try:
        full_df = load_data(ticker, period="5y")
    except Exception as e:
        return RollingWalkForwardResult(
            strategy_name=strategy_fn.__name__,
            ticker=ticker,
            num_windows=0, windows_passed=0,
            avg_test_sharpe=0, min_test_sharpe=0, avg_degradation=1.0,
            robust=False,
            notes=f"Data error: {e}",
        )

    train_bars = _parse_duration(train_window)
    test_bars = _parse_duration(test_window)
    step_bars = _parse_duration(step)
    window_size = train_bars + test_bars

    if len(full_df) < window_size:
        return RollingWalkForwardResult(
            strategy_name=strategy_fn.__name__,
            ticker=ticker,
            num_windows=0, windows_passed=0,
            avg_test_sharpe=0, min_test_sharpe=0, avg_degradation=1.0,
            robust=False,
            notes=f"Data too short ({len(full_df)} bars, need {window_size})",
        )

    # Generate window start indices
    starts = list(range(0, len(full_df) - window_size + 1, step_bars))
    if not starts:
        starts = [0]

    window_results: list[dict] = []
    for i, start in enumerate(starts):
        end = start + window_size
        train_df = full_df.iloc[start:start + train_bars]
        test_df = full_df.iloc[start + train_bars:end]

        try:
            train_entries, train_exits = strategy_fn(train_df, params)
            train_result = engine.run(train_df, train_entries, train_exits,
                                      strategy_fn.__name__, ticker, params=params)
            test_entries, test_exits = strategy_fn(test_df, params)
            test_result = engine.run(test_df, test_entries, test_exits,
                                     strategy_fn.__name__, ticker, params=params)

            if train_result.sharpe > 0:
                degradation = (train_result.sharpe - test_result.sharpe) / train_result.sharpe
                degradation = max(0.0, degradation)  # Clamp: negative degradation = improvement
            elif test_result.sharpe > 0:
                # Negative train but positive test = strategy works out-of-sample despite bad train
                degradation = 0.0
            else:
                # Both negative — can't meaningfully measure degradation
                degradation = 1.0

            window_passed = test_result.sharpe >= 0.3 and degradation <= 0.7
            window_results.append({
                "window": i + 1,
                "train_sharpe": train_result.sharpe,
                "test_sharpe": test_result.sharpe,
                "degradation": degradation,
                "passed": window_passed,
            })
        except Exception as exc:
            logger.warning(f"Rolling window {i+1} failed: {exc}")
            window_results.append({
                "window": i + 1, "train_sharpe": 0, "test_sharpe": 0,
                "degradation": 1.0, "passed": False, "error": str(exc),
            })

    if not window_results:
        return RollingWalkForwardResult(
            strategy_name=strategy_fn.__name__,
            ticker=ticker,
            num_windows=0, windows_passed=0,
            avg_test_sharpe=0, min_test_sharpe=0, avg_degradation=1.0,
            robust=False,
            notes="No valid windows produced",
        )

    test_sharpes = [w["test_sharpe"] for w in window_results]
    degradations = [w["degradation"] for w in window_results]
    windows_passed = sum(1 for w in window_results if w["passed"])

    avg_test_sharpe = sum(test_sharpes) / len(test_sharpes)
    min_test_sharpe = min(test_sharpes)
    avg_degradation = sum(degradations) / len(degradations)

    robust = (
        avg_test_sharpe >= min_avg_test_sharpe
        and windows_passed >= min_windows_passed
    )

    notes_parts = [
        f"Windows: {len(window_results)}, passed: {windows_passed}",
        f"Avg test Sharpe: {avg_test_sharpe:.2f} (min: {min_test_sharpe:.2f})",
        f"Avg degradation: {avg_degradation:.1%}",
    ]
    if not robust:
        if avg_test_sharpe < min_avg_test_sharpe:
            notes_parts.append(f"⚠️ Avg test Sharpe {avg_test_sharpe:.2f} < {min_avg_test_sharpe}")
        if windows_passed < min_windows_passed:
            notes_parts.append(f"⚠️ Only {windows_passed}/{len(window_results)} windows passed (need >= {min_windows_passed})")

    return RollingWalkForwardResult(
        strategy_name=strategy_fn.__name__,
        ticker=ticker,
        num_windows=len(window_results),
        windows_passed=windows_passed,
        avg_test_sharpe=avg_test_sharpe,
        min_test_sharpe=min_test_sharpe,
        avg_degradation=avg_degradation,
        robust=robust,
        notes=" | ".join(notes_parts),
        window_results=window_results,
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
