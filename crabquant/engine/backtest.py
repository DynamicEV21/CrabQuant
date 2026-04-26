"""
CrabQuant Backtest Engine

VectorBT-powered backtesting with comprehensive metrics.
"""

import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import vectorbt as vbt

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete backtest result with all metrics."""
    ticker: str
    strategy_name: str
    iteration: int
    sharpe: float
    total_return: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_trade_return: float
    calmar_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_holding_bars: float
    best_trade: float
    worst_trade: float
    passed: bool
    score: float  # Composite score
    notes: str
    params: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class BacktestEngine:
    """
    VectorBT backtest engine with configurable criteria.
    """

    def __init__(
        self,
        initial_cash: float = 100_000,
        commission: float = 0.001,
        sharpe_target: float = 1.5,
        max_drawdown_limit: float = 0.25,
        min_total_return: float = 0.10,
        min_trades: int = 5,
    ):
        self.initial_cash = initial_cash
        self.commission = commission
        self.sharpe_target = sharpe_target
        self.max_drawdown_limit = max_drawdown_limit
        self.min_total_return = min_total_return
        self.min_trades = min_trades

    def run(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        strategy_name: str,
        ticker: str,
        iteration: int = 0,
        params: dict | None = None,
    ) -> BacktestResult:
        """
        Run a backtest and compute all metrics.

        Args:
            df: OHLCV DataFrame
            entries: Boolean Series of entry signals
            exits: Boolean Series of exit signals
            strategy_name: Name of the strategy
            ticker: Ticker symbol
            iteration: Iteration number
            params: Parameters used

        Returns:
            BacktestResult with all metrics
        """
        try:
            pf = vbt.Portfolio.from_signals(
                close=df["close"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                entries=entries,
                exits=exits,
                init_cash=self.initial_cash,
                fees=self.commission,
                freq="1D",
                accumulate=False,
                call_seq="auto",
            )

            stats = pf.stats()
            trades = pf.trades.records_readable

            sharpe = float(stats.get("Sharpe Ratio", 0))
            total_return = float(stats.get("Total Return [%]", 0)) / 100
            max_dd = float(stats.get("Max Drawdown [%]", 0)) / 100
            win_rate = float(stats.get("Win Rate [%]", 0)) / 100
            num_trades = int(stats.get("Total Trades", 0))
            calmar = float(stats.get("Calmar Ratio", 0))
            sortino = float(stats.get("Sortino Ratio", 0))
            avg_trade = float(stats.get("Avg Winning Trade [%]", 0)) / 100 if "Avg Winning Trade [%]" in stats else 0
            avg_hold = float(stats.get("Avg Holding Duration [# Bars]", 0)) if "Avg Holding Duration [# Bars]" in stats else 0

            # Profit factor
            if len(trades) > 0 and "PnL" in trades.columns:
                winning = trades[trades["PnL"] > 0]["PnL"].sum()
                losing = abs(trades[trades["PnL"] < 0]["PnL"].sum())
                profit_factor = winning / losing if losing > 0 else 999.0
                best_trade = float(trades["PnL"].max()) if len(trades) > 0 else 0
                worst_trade = float(trades["PnL"].min()) if len(trades) > 0 else 0
            else:
                profit_factor = 0.0
                best_trade = 0.0
                worst_trade = 0.0

            # Composite score: reward consistency and risk management
            # score = sharpe * sqrt(trades/20) * (1 - abs(max_dd))
            # This penalizes low-trade-count and high-drawdown strategies
            trade_factor = np.sqrt(min(num_trades, 100) / 20)
            dd_penalty = max(0, 1 - abs(max_dd))
            score = sharpe * trade_factor * dd_penalty

            # Handle edge cases
            if np.isinf(sharpe) or np.isnan(sharpe):
                sharpe = 0.0
                score = 0.0
            if np.isnan(max_dd):
                max_dd = 0.0

            passed = (
                sharpe >= self.sharpe_target
                and max_dd >= -self.max_drawdown_limit
                and total_return >= self.min_total_return
                and num_trades >= self.min_trades
            )

            notes = self._build_notes(sharpe, max_dd, num_trades, total_return)

            return BacktestResult(
                ticker=ticker,
                strategy_name=strategy_name,
                iteration=iteration,
                sharpe=sharpe,
                total_return=total_return,
                max_drawdown=max_dd,
                win_rate=win_rate,
                num_trades=num_trades,
                avg_trade_return=avg_trade,
                calmar_ratio=calmar,
                sortino_ratio=sortino,
                profit_factor=profit_factor,
                avg_holding_bars=avg_hold,
                best_trade=best_trade,
                worst_trade=worst_trade,
                passed=passed,
                score=score,
                notes=notes,
                params=params or {},
            )

        except Exception as e:
            logger.error(f"Backtest error ({ticker}/{strategy_name}): {e}")
            return BacktestResult(
                ticker=ticker,
                strategy_name=strategy_name,
                iteration=iteration,
                sharpe=0, total_return=0, max_drawdown=0, win_rate=0,
                num_trades=0, avg_trade_return=0, calmar_ratio=0,
                sortino_ratio=0, profit_factor=0, avg_holding_bars=0,
                best_trade=0, worst_trade=0,
                passed=False, score=0, notes=f"ERROR: {e}",
                params=params or {},
            )

    def _build_notes(self, sharpe: float, max_dd: float, num_trades: int, total_return: float) -> str:
        """Build human-readable pass/fail notes."""
        notes = []
        if sharpe >= self.sharpe_target:
            notes.append(f"Sharpe {sharpe:.2f} >= {self.sharpe_target}")
        else:
            notes.append(f"Sharpe {sharpe:.2f} < {self.sharpe_target}")
        if max_dd >= -self.max_drawdown_limit:
            notes.append(f"MaxDD {max_dd:.1%} OK")
        else:
            notes.append(f"MaxDD {max_dd:.1%} > {self.max_drawdown_limit:.0%}")
        if num_trades >= self.min_trades:
            notes.append(f"{num_trades} trades")
        else:
            notes.append(f"Only {num_trades} trades")
        if total_return >= self.min_total_return:
            notes.append(f"Return {total_return:.1%}")
        else:
            notes.append(f"Return {total_return:.1%} < {self.min_total_return:.0%}")
        return " | ".join(notes)
