"""
Confirmation Runner

Runs backtesting.py backtests for strategies that passed VectorBT screening.
Extracts stats and returns ConfirmationResult.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from crabquant.confirm import ConfirmationResult, CONFIRM_THRESHOLDS
from crabquant.confirm.strategy_converter import convert_strategy
from crabquant.data import load_data

logger = logging.getLogger(__name__)


def _slippage_commission(size, price, _slippage_pct):
    """Custom commission function that adds slippage to entries.
    
    backtesting.py calls commission(size, price) where:
    - size > 0 means buying (entry)
    - size < 0 means selling (exit)
    We add slippage cost only on entries (buying at worse price).
    """
    # Base commission (0.1%)
    base = abs(size) * price * 0.001
    # Slippage: only on entries (size > 0 means we're buying)
    if size > 0:
        slip_cost = abs(size) * price * _slippage_pct
        return base + slip_cost
    return base


def _compute_profit_factor(trades_df: pd.DataFrame) -> float:
    """Compute profit factor from backtesting.py trades DataFrame."""
    if trades_df is None or len(trades_df) == 0:
        return 0.0

    # In backtesting.py, each trade row has 'PnL' column
    pnl_col = None
    for col in ["PnL", "pnl", "Profit"]:
        if col in trades_df.columns:
            pnl_col = col
            break

    if pnl_col is None:
        return 0.0

    wins = trades_df[trades_df[pnl_col] > 0][pnl_col].sum()
    losses = abs(trades_df[trades_df[pnl_col] < 0][pnl_col].sum())

    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def _compute_expectancy(trades_df: pd.DataFrame) -> float:
    """Average profit per trade."""
    if trades_df is None or len(trades_df) == 0:
        return 0.0

    pnl_col = None
    for col in ["PnL", "pnl", "Profit"]:
        if col in trades_df.columns:
            pnl_col = col
            break

    if pnl_col is None:
        return 0.0

    return trades_df[pnl_col].mean()


def run_confirmation(
    strategy_name: str,
    ticker: str,
    params: dict,
    df: Optional[pd.DataFrame] = None,
    period: str = "2y",
    cash: float = 10000,
    commission: float = 0.001,
    slippage_pct: float = 0.001,
    position_pct: float = 0.95,
) -> ConfirmationResult:
    """
    Run a confirmation backtest using backtesting.py.

    Args:
        strategy_name: Strategy name from STRATEGY_REGISTRY
        ticker: Stock ticker symbol
        params: Strategy parameters dict
        df: Optional pre-loaded DataFrame (lowercase OHLCV columns)
        period: Data period if df is None
        cash: Starting cash
        commission: Commission per trade (as fraction)
        slippage_pct: Slippage as fraction (0.001 = 0.1%)
        position_pct: Fraction of portfolio per trade

    Returns:
        ConfirmationResult with all metrics and pass/fail status
    """
    notes = []

    # Load data
    if df is None:
        try:
            df = load_data(ticker, period=period)
        except Exception as e:
            notes.append(f"Failed to load data for {ticker}: {e}")
            return ConfirmationResult(notes=notes, passed=False)

    # Ensure we have enough data
    if len(df) < 100:
        notes.append(f"Insufficient data: {len(df)} bars (need ≥100)")
        return ConfirmationResult(notes=notes, passed=False)

    # Convert strategy
    try:
        strategy_class = convert_strategy(
            strategy_name, params,
            position_pct=position_pct,
            slippage_pct=slippage_pct,
        )
    except ValueError as e:
        notes.append(str(e))
        return ConfirmationResult(notes=notes, passed=False)

    # Prepare data for backtesting.py (needs capital column names)
    bt_df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }).copy()

    # Ensure index is DatetimeIndex
    if not isinstance(bt_df.index, pd.DatetimeIndex):
        bt_df.index = pd.to_datetime(bt_df.index)

    # Remove any timezone info (backtesting.py doesn't like it)
    if bt_df.index.tz is not None:
        bt_df.index = bt_df.index.tz_localize(None)

    # Create slippage-aware commission function
    slip = slippage_pct
    slippage_fn = lambda size, price: _slippage_commission(size, price, slip)

    # Run backtest
    # - commission: custom function that adds slippage on entries
    # - exclusive_orders: can't be in two positions at once
    # - hedging=False: long-only, no short selling
    # - finalize_trades: close open positions at end for accurate stats
    try:
        bt = Backtest(
            bt_df,
            strategy_class,
            cash=cash,
            commission=slippage_fn,
            exclusive_orders=True,
            hedging=False,
            finalize_trades=True,
        )
        stats = bt.run()
    except Exception as e:
        notes.append(f"Backtest failed: {e}")
        logger.warning(f"Confirmation backtest failed for {strategy_name}/{ticker}: {e}")
        return ConfirmationResult(notes=notes, passed=False)

    # Extract stats
    # backtesting.py stats dict keys may vary; use safe access
    sharpe = stats.get("Sharpe Ratio", 0.0) or 0.0
    total_return = stats.get("Return [%]", 0.0) or 0.0
    if isinstance(total_return, (int, float)):
        total_return = total_return / 100.0  # Convert from percentage
    max_dd = stats.get("Max. Drawdown [%]", 0.0) or 0.0
    if isinstance(max_dd, (int, float)):
        max_dd = max_dd / 100.0  # Convert from percentage (negative value)
    num_trades = stats.get("# Trades", 0) or 0
    win_rate = stats.get("Win Rate [%]", 0.0) or 0.0
    if isinstance(win_rate, (int, float)):
        win_rate = win_rate / 100.0

    # Get trades DataFrame for profit factor and expectancy
    trades_df = stats.get("_trades", None)
    profit_factor = _compute_profit_factor(trades_df)
    expectancy = _compute_expectancy(trades_df)

    # Check thresholds
    t = CONFIRM_THRESHOLDS
    checks = {
        f"sharpe {sharpe:.2f} ≥ {t['sharpe']}": sharpe >= t["sharpe"],
        f"max_dd {abs(max_dd):.2%} ≤ {t['max_drawdown']:.0%}": abs(max_dd) <= t["max_drawdown"],
        f"return {total_return:.2%} ≥ {t['total_return']:.0%}": total_return >= t["total_return"],
        f"trades {num_trades} ≥ {t['min_trades']}": num_trades >= t["min_trades"],
        f"expectancy {expectancy:.2f} ≥ {t['expectancy']}": expectancy >= t["expectancy"],
    }

    passed = all(checks.values())
    notes.extend(
        f"{'✓' if v else '✗'} {k}" for k, v in checks.items()
    )

    return ConfirmationResult(
        sharpe=round(sharpe, 4),
        total_return=round(total_return, 4),
        max_dd=round(max_dd, 4),
        trades=int(num_trades),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        expectancy=round(expectancy, 2),
        passed=passed,
        verdict="PASSED" if passed else "FAILED",
        notes=notes,
    )
