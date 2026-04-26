"""
Production Strategy Report

A StrategyReport dataclass that holds all metrics for a production strategy
and can render a human-readable markdown report.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional


@dataclass
class SlippageResult:
    """Result at a specific slippage level."""
    slippage_pct: float
    sharpe: float
    total_return: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    passed: bool


@dataclass
class PeriodResult:
    """Result for a specific time period."""
    period: str
    sharpe: float
    total_return: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    passed: bool


@dataclass
class RegimeInfo:
    """Regime affinity information for a strategy."""
    best_regime: str = ""
    works_in: list = field(default_factory=list)
    avoid_in: list = field(default_factory=list)


@dataclass
class StrategyReport:
    """Full production report for a promoted strategy."""

    # Identity
    strategy_name: str = ""
    ticker: str = ""
    params: dict = field(default_factory=dict)
    date_promoted: str = ""
    verdict: str = ""

    # VectorBT discovery metrics
    vbt_sharpe: float = 0.0
    vbt_total_return: float = 0.0
    vbt_max_drawdown: float = 0.0
    vbt_num_trades: int = 0
    vbt_win_rate: float = 0.0
    vbt_score: float = 0.0

    # Confirmation metrics (backtesting.py, 2y primary)
    confirm_sharpe: float = 0.0
    confirm_total_return: float = 0.0
    confirm_max_drawdown: float = 0.0
    confirm_num_trades: int = 0
    confirm_win_rate: float = 0.0
    confirm_profit_factor: float = 0.0
    confirm_expectancy: float = 0.0

    # Slippage sensitivity
    slippage_results: list = field(default_factory=list)  # list[SlippageResult]

    # Period performance
    period_results: list = field(default_factory=list)  # list[PeriodResult]

    # Regime affinity
    regime_info: RegimeInfo = field(default_factory=RegimeInfo)

    # Key for dedup
    key: str = ""

    def to_markdown(self) -> str:
        """Render a human-readable markdown report."""
        lines = []
        lines.append(f"# {self.ticker} / {self.strategy_name} — PRODUCTION")
        lines.append(f"**Promoted:** {self.date_promoted}")
        lines.append(f"**Verdict:** {self.verdict}")
        lines.append("")

        # VectorBT Results
        lines.append("## VectorBT Results")
        lines.append(
            f"- Sharpe: {self.vbt_sharpe:.2f} | "
            f"Return: {self._pct(self.vbt_total_return)} | "
            f"MaxDD: {self._pct(self.vbt_max_drawdown)} | "
            f"Trades: {self.vbt_num_trades} | "
            f"Win Rate: {self._pct(self.vbt_win_rate)}"
        )
        lines.append(f"- Composite Score: {self.vbt_score:.2f}")
        lines.append("")

        # Confirmation Results
        lines.append("## Confirmation Results (backtesting.py)")
        lines.append(
            f"- Sharpe: {self.confirm_sharpe:.2f} | "
            f"Return: {self._pct(self.confirm_total_return)} | "
            f"MaxDD: {self._pct(self.confirm_max_drawdown)} | "
            f"Trades: {self.confirm_num_trades} | "
            f"Win Rate: {self._pct(self.confirm_win_rate)}"
        )

        # Fill degradation
        if self.vbt_total_return > 0:
            ret_degrade = (self.confirm_total_return - self.vbt_total_return) / self.vbt_total_return * 100
            sharpe_degrade = self.confirm_sharpe - self.vbt_sharpe
            lines.append(
                f"- Realistic fill degradation: "
                f"{ret_degrade:+.0f}% return, {sharpe_degrade:+.2f} Sharpe"
            )
        lines.append("")

        # Slippage Sensitivity
        if self.slippage_results:
            lines.append("## Slippage Sensitivity")
            for sr in self.slippage_results:
                icon = "✅" if sr.passed else "❌"
                lines.append(
                    f"- {sr.slippage_pct*100:.1f}% slippage: {icon} Sharpe {sr.sharpe:.2f}"
                )
            lines.append("")

        # Period Performance
        if self.period_results:
            lines.append("## Period Performance")
            for pr in self.period_results:
                lines.append(
                    f"- {pr.period}: Sharpe {pr.sharpe:.2f}, Return {self._pct(pr.total_return)}"
                )
            lines.append("")

        # Strategy Parameters
        if self.params:
            lines.append("## Strategy Parameters")
            for k, v in self.params.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        # Regime
        if self.regime_info and self.regime_info.best_regime:
            lines.append("## Regime")
            lines.append(f"- Best in: {self.regime_info.best_regime}")
            if self.regime_info.works_in:
                lines.append(f"- Works in: {', '.join(self.regime_info.works_in)}")
            if self.regime_info.avoid_in:
                lines.append(f"- Avoid in: {', '.join(self.regime_info.avoid_in)}")
            lines.append("")

        # Embed metadata for programmatic reading
        lines.append("<!-- METADATA")
        lines.append(json.dumps(self.to_dict()))
        lines.append("-->")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        d = asdict(self)
        # Convert nested dataclasses to dicts
        if isinstance(d.get("slippage_results"), list):
            d["slippage_results"] = [
                asdict(s) if hasattr(s, '__dataclass_fields__') else s
                for s in d["slippage_results"]
            ]
        if isinstance(d.get("period_results"), list):
            d["period_results"] = [
                asdict(p) if hasattr(p, '__dataclass_fields__') else p
                for p in d["period_results"]
            ]
        if isinstance(d.get("regime_info"), RegimeInfo):
            d["regime_info"] = asdict(d["regime_info"])
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyReport":
        """Deserialize from dict."""
        # Reconstruct nested dataclasses
        slippage = [
            SlippageResult(**s) for s in data.pop("slippage_results", [])
        ]
        periods = [
            PeriodResult(**p) for p in data.pop("period_results", [])
        ]
        regime_data = data.pop("regime_info", {})
        regime = RegimeInfo(**regime_data) if regime_data else RegimeInfo()

        return cls(
            **data,
            slippage_results=slippage,
            period_results=periods,
            regime_info=regime,
        )

    @staticmethod
    def _pct(value: float) -> str:
        """Format a ratio as percentage."""
        return f"{value:.1%}"
