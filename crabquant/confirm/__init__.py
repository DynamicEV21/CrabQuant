"""
CrabQuant Confirmation Module

Two-phase validation: VectorBT fast screening → backtesting.py slow confirmation.
This module handles the second phase — realistic bar-by-bar simulation
of strategies that pass VectorBT thresholds.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfirmationResult:
    """Result of a confirmation backtest."""

    sharpe: float = 0.0
    total_return: float = 0.0
    max_dd: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    passed: bool = False
    verdict: str = "FAILED"  # ROBUST, FRAGILE, FAILED
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sharpe": self.sharpe,
            "total_return": self.total_return,
            "max_dd": self.max_dd,
            "max_drawdown": self.max_dd,
            "trades": self.trades,
            "num_trades": self.trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "passed": self.passed,
            "verdict": self.verdict,
            "notes": self.notes,
        }

    @property
    def num_trades(self) -> int:
        return self.trades

    @property
    def max_drawdown(self) -> float:
        return self.max_dd


# Confirmation thresholds — more lenient than VectorBT because realistic fills are harsher
CONFIRM_THRESHOLDS = {
    "sharpe": 1.0,        # vs VectorBT 1.5
    "max_drawdown": 0.30,  # vs VectorBT 0.25
    "total_return": 0.05,  # vs VectorBT 0.10
    "min_trades": 5,
    "expectancy": 0.0,     # must be positive
}


def confirm_strategy(
    strategy_name: str,
    ticker: str,
    params: dict,
    df: Optional[object] = None,
    period: str = "2y",
) -> ConfirmationResult:
    """
    Run a single confirmation backtest for a strategy.

    Args:
        strategy_name: Strategy name from STRATEGY_REGISTRY
        ticker: Stock ticker symbol
        params: Strategy parameters dict
        df: Optional pre-loaded DataFrame (loads from yfinance if None)
        period: Data period if df is None

    Returns:
        ConfirmationResult with all metrics and pass/fail
    """
    from crabquant.confirm.runner import run_confirmation
    return run_confirmation(strategy_name, ticker, params, df=df, period=period)


__all__ = [
    "ConfirmationResult",
    "CONFIRM_THRESHOLDS",
    "confirm_strategy",
    "run_confirmation",
    "batch_confirm",
]
