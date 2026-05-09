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
from itertools import product

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
    expected_value: float = 0.0
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
        return_portfolio: bool = False,
    ) -> BacktestResult | tuple[BacktestResult, "vbt.Portfolio"]:
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
            )

            stats = pf.stats()
            trades = pf.trades.records_readable

            sharpe = float(stats.get("Sharpe Ratio", 0))
            total_return = float(stats.get("Total Return [%]", 0)) / 100
            # Max Drawdown [%] from stats is positive (e.g. 23.6); negate to match
            # VectorBT's pf.max_drawdown() convention of negative values (e.g. -0.236)
            max_dd = -float(stats.get("Max Drawdown [%]", 0)) / 100
            num_trades = int(stats.get("Total Trades", 0))
            calmar = float(stats.get("Calmar Ratio", 0))
            sortino = float(stats.get("Sortino Ratio", 0))

            # Trade-level metrics computed directly from records for consistency
            # with run_vectorized() and to avoid pf.stats() quirks (e.g. Win Rate
            # counts differently, Holding Duration returns 0 for daily freq)
            if len(trades) > 0 and "PnL" in trades.columns:
                winners = trades[trades["PnL"] > 0]
                losers = trades[trades["PnL"] < 0]
                win_rate = len(winners) / len(trades) if len(trades) > 0 else 0
                avg_trade = float(winners["Return"].mean()) if len(winners) > 0 and "Return" in winners.columns else 0
                profit_factor = float(winners["PnL"].sum()) / abs(losers["PnL"].sum()) if len(losers) > 0 and losers["PnL"].sum() != 0 else (999.0 if len(winners) > 0 else 0.0)
                best_trade = float(trades["PnL"].max())
                worst_trade = float(trades["PnL"].min())
                # Holding duration in bars (trading days)
                if "Entry Timestamp" in trades.columns and "Exit Timestamp" in trades.columns:
                    hold_days = (trades["Exit Timestamp"] - trades["Entry Timestamp"]).dt.days
                    avg_hold = float(hold_days.mean()) if len(hold_days) > 0 else 0
                else:
                    avg_hold = 0.0
            else:
                win_rate = 0.0
                avg_trade = 0.0
                profit_factor = 0.0
                best_trade = 0.0
                worst_trade = 0.0
                avg_hold = 0.0

            # Expected Value: (win_rate * avg_win_size) - ((1 - win_rate) * avg_loss_size)
            if len(trades) > 0 and "PnL" in trades.columns:
                winners_pnl = trades[trades["PnL"] > 0]["PnL"]
                losers_pnl = trades[trades["PnL"] < 0]["PnL"]
                avg_win_size = float(winners_pnl.mean()) if len(winners_pnl) > 0 else 0.0
                avg_loss_size = float(losers_pnl.abs().mean()) if len(losers_pnl) > 0 else 0.0
                expected_value = (win_rate * avg_win_size) - ((1 - win_rate) * avg_loss_size)
            else:
                expected_value = 0.0

            # Handle edge cases BEFORE computing score to avoid RuntimeWarning
            if np.isinf(sharpe) or np.isnan(sharpe):
                sharpe = 0.0
            if np.isnan(max_dd):
                max_dd = 0.0

            # Composite score: reward consistency, risk management, and edge quality
            # score = (sortino_weighted + ev_weighted) * robustness_factor
            trade_factor = np.sqrt(min(num_trades, 100) / 20)
            dd_penalty = max(0, 1 - abs(max_dd))
            robustness_factor = trade_factor * dd_penalty
            sortino_safe = max(sortino, 0.0) if not (np.isinf(sortino) or np.isnan(sortino)) else 0.0
            ev_weighted = np.sign(expected_value) * min(abs(expected_value) / 100.0, 1.0)  # normalise EV to [-1, 1] scale
            sortino_weighted = min(sortino_safe / 3.0, 1.0)  # normalise: sortino 3.0 → 1.0
            score = (sortino_weighted + ev_weighted) * robustness_factor

            passed = (
                sharpe >= self.sharpe_target
                and max_dd >= -self.max_drawdown_limit
                and total_return >= self.min_total_return
                and num_trades >= self.min_trades
            )

            notes = self._build_notes(sharpe, max_dd, num_trades, total_return)

            result = BacktestResult(
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
                expected_value=expected_value,
                profit_factor=profit_factor,
                avg_holding_bars=avg_hold,
                best_trade=best_trade,
                worst_trade=worst_trade,
                passed=passed,
                score=score,
                notes=notes,
                params=params or {},
            )
            return (result, pf) if return_portfolio else result

        except Exception as e:
            logger.error(f"Backtest error ({ticker}/{strategy_name}): {e}")
            error_result = BacktestResult(
                ticker=ticker,
                strategy_name=strategy_name,
                iteration=iteration,
                sharpe=0, total_return=0, max_drawdown=0, win_rate=0,
                num_trades=0, avg_trade_return=0, calmar_ratio=0,
                sortino_ratio=0, expected_value=0, profit_factor=0, avg_holding_bars=0,
                best_trade=0, worst_trade=0,
                passed=False, score=0, notes=f"ERROR: {e}",
                params=params or {},
            )
            return (error_result, None) if return_portfolio else error_result

    def run_vectorized(
        self,
        df: pd.DataFrame,
        entries_df: pd.DataFrame,
        exits_df: pd.DataFrame,
        param_list: list[dict],
        strategy_name: str,
        ticker: str,
    ) -> list[BacktestResult]:
        """
        Run backtests for ALL param combos in one vectorized Portfolio.from_signals call.

        Uses batch metrics (pf.sharpe_ratio(), pf.total_return(), etc.) instead of
        per-column stats() calls for maximum speed. Only trade-level details (PnL,
        holding duration) are computed from the shared records DataFrame.

        Args:
            df: OHLCV DataFrame
            entries_df: Multi-column DataFrame of boolean entry signals (one column per combo)
            exits_df: Multi-column DataFrame of boolean exit signals (one column per combo)
            param_list: List of param dicts, one per column (maps index to params)
            strategy_name: Name of the strategy
            ticker: Ticker symbol

        Returns:
            List of BacktestResult, one per param combo
        """
        try:
            pf = vbt.Portfolio.from_signals(
                close=df["close"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                entries=entries_df,
                exits=exits_df,
                init_cash=self.initial_cash,
                fees=self.commission,
                freq="1D",
                accumulate=False,
                call_seq="auto",
            )

            # ── Batch metrics: one call per metric, returns per-column Series ──
            sharpes = pf.sharpe_ratio().fillna(0).replace([np.inf, -np.inf], 0)
            total_returns = pf.total_return().fillna(0)
            max_drawdowns = pf.max_drawdown().fillna(0)
            trade_counts = pf.trades.count().fillna(0).astype(int)
            calmars = pf.calmar_ratio().fillna(0).replace([np.inf, -np.inf], 0)
            sortinos = pf.sortino_ratio().fillna(0).replace([np.inf, -np.inf], 0)

            # ── Trade records (shared across all metrics) ──
            records = pf.trades.records_readable

            # Profit factor — manual calculation to avoid read-only array bug in vbt
            profit_factors = pd.Series(0.0, index=entries_df.columns)
            # Win rate per column (count-based: wins / total trades)
            win_rates_per_col = pd.Series(0.0, index=entries_df.columns)
            if len(records) > 0 and "PnL" in records.columns and "Column" in records.columns:
                win_pnl = records[records["PnL"] > 0].groupby("Column")["PnL"].sum()
                lose_pnl = records[records["PnL"] < 0].groupby("Column")["PnL"].sum().abs()
                for col in entries_df.columns:
                    w = float(win_pnl.get(col, 0))
                    l = float(lose_pnl.get(col, 0))
                    profit_factors[col] = w / l if l > 0 else (999.0 if w > 0 else 0.0)

                # Win rate: count of trades with PnL > 0 / total trades per column
                trade_counts = records.groupby("Column").size()
                win_counts = records[records["PnL"] > 0].groupby("Column").size()
                for col in entries_df.columns:
                    tc = int(trade_counts.get(col, 0))
                    wc = int(win_counts.get(col, 0))
                    win_rates_per_col[col] = wc / tc if tc > 0 else 0.0

            # ── Trade-level details via groupby ──
            trade_details = {}
            if len(records) > 0 and "PnL" in records.columns and "Column" in records.columns:
                grouped = records.groupby("Column")
                trade_details["best"] = grouped["PnL"].max().to_dict()
                trade_details["worst"] = grouped["PnL"].min().to_dict()
                # avg_trade_return: mean Return (pct) of winning trades per column
                trade_details["avg_win_pct"] = (
                    records[records["PnL"] > 0].groupby("Column")["Return"].mean().to_dict()
                    if "Return" in records.columns and (records["PnL"] > 0).any() else {}
                )
                # avg_holding_bars: mean holding days per column
                trade_details["avg_hold"] = {}
                if "Entry Timestamp" in records.columns and "Exit Timestamp" in records.columns:
                    records_copy = records.copy()
                    records_copy["_hold"] = (
                        records_copy["Exit Timestamp"] - records_copy["Entry Timestamp"]
                    ).dt.days
                    trade_details["avg_hold"] = records_copy.groupby("Column")["_hold"].mean().to_dict()

                # Build lookup with defaults for columns that had no trades
                col_trades = {}
                for col in entries_df.columns:
                    # Expected Value per column
                    col_records = records[records["Column"] == col] if "Column" in records.columns else pd.DataFrame()
                    if len(col_records) > 0 and "PnL" in col_records.columns:
                        wr = win_rates_per_col.get(col, 0.0)
                        w_pnl = col_records[col_records["PnL"] > 0]["PnL"]
                        l_pnl = col_records[col_records["PnL"] < 0]["PnL"]
                        avg_win = float(w_pnl.mean()) if len(w_pnl) > 0 else 0.0
                        avg_loss = float(l_pnl.abs().mean()) if len(l_pnl) > 0 else 0.0
                        ev = (wr * avg_win) - ((1 - wr) * avg_loss)
                    else:
                        ev = 0.0
                    col_trades[col] = {
                        "best_trade": float(trade_details["best"].get(col, 0.0)),
                        "worst_trade": float(trade_details["worst"].get(col, 0.0)),
                        "avg_trade_return": float(trade_details["avg_win_pct"].get(col, 0.0)),
                        "avg_holding_bars": float(trade_details["avg_hold"].get(col, 0.0)),
                        "expected_value": ev,
                    }
            else:
                col_trades = {col: {"best_trade": 0.0, "worst_trade": 0.0, "avg_trade_return": 0.0, "avg_holding_bars": 0.0, "expected_value": 0.0} for col in entries_df.columns}

            # ── Build results ──
            results = []
            for i, (col_name, params) in enumerate(zip(entries_df.columns, param_list)):
                sharpe = float(sharpes.iloc[i]) if i < len(sharpes) else 0.0
                total_return = float(total_returns.iloc[i]) if i < len(total_returns) else 0.0
                max_dd = float(max_drawdowns.iloc[i]) if i < len(max_drawdowns) else 0.0
                win_rate = float(win_rates_per_col.iloc[i]) if i < len(win_rates_per_col) else 0.0
                num_trades = int(trade_counts.iloc[i]) if i < len(trade_counts) else 0
                calmar = float(calmars.iloc[i]) if i < len(calmars) else 0.0
                sortino = float(sortinos.iloc[i]) if i < len(sortinos) else 0.0
                pf_val = float(profit_factors.iloc[i]) if i < len(profit_factors) else 0.0

                ct = col_trades.get(col_name, {})
                expected_value = ct.get("expected_value", 0.0)

                # Composite score: (sortino_weighted + ev_weighted) * robustness_factor
                trade_factor = np.sqrt(min(num_trades, 100) / 20)
                dd_penalty = max(0, 1 - abs(max_dd))
                robustness_factor = trade_factor * dd_penalty
                sortino_safe = max(sortino, 0.0) if not (np.isinf(sortino) or np.isnan(sortino)) else 0.0
                ev_weighted = np.sign(expected_value) * min(abs(expected_value) / 100.0, 1.0)
                sortino_weighted = min(sortino_safe / 3.0, 1.0)
                score = (sortino_weighted + ev_weighted) * robustness_factor

                if np.isnan(score):
                    score = 0.0

                passed = (
                    sharpe >= self.sharpe_target
                    and max_dd >= -self.max_drawdown_limit
                    and total_return >= self.min_total_return
                    and num_trades >= self.min_trades
                )

                notes = self._build_notes(sharpe, max_dd, num_trades, total_return)

                results.append(BacktestResult(
                    ticker=ticker,
                    strategy_name=strategy_name,
                    iteration=i,
                    sharpe=sharpe,
                    total_return=total_return,
                    max_drawdown=max_dd,
                    win_rate=win_rate,
                    num_trades=num_trades,
                    avg_trade_return=ct.get("avg_trade_return", 0.0),
                    calmar_ratio=calmar,
                    sortino_ratio=sortino,
                    expected_value=expected_value,
                    profit_factor=pf_val,
                    avg_holding_bars=ct.get("avg_holding_bars", 0.0),
                    best_trade=ct.get("best_trade", 0.0),
                    worst_trade=ct.get("worst_trade", 0.0),
                    passed=passed,
                    score=score,
                    notes=notes,
                    params=params,
                ))

            return results

        except Exception as e:
            logger.error(f"Vectorized backtest error ({ticker}/{strategy_name}): {e}")
            # Fall back to individual runs if the batch call fails
            logger.info("Falling back to sequential runs...")
            return [
                self.run(df, entries_df[col], exits_df[col], strategy_name, ticker, i, params)
                for i, (col, params) in enumerate(zip(entries_df.columns, param_list))
            ]

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
