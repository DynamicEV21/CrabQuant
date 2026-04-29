"""Tests for crabquant.production.report"""

import json
from dataclasses import asdict

import pytest

from crabquant.production.report import (
    StrategyReport,
    SlippageResult,
    PeriodResult,
    RegimeInfo,
)


class TestSlippageResult:
    def test_creation(self):
        sr = SlippageResult(
            slippage_pct=0.001, sharpe=1.2, total_return=0.15,
            max_drawdown=-0.08, num_trades=30, win_rate=0.55, passed=True,
        )
        assert sr.slippage_pct == 0.001
        assert sr.passed is True

    def test_defaults(self):
        sr = SlippageResult(0.0, 0.0, 0.0, 0.0, 0, 0.0, False)
        assert sr.slippage_pct == 0.0
        assert sr.passed is False


class TestPeriodResult:
    def test_creation(self):
        pr = PeriodResult(
            period="2y", sharpe=1.5, total_return=0.20,
            max_drawdown=-0.10, num_trades=40, win_rate=0.58, passed=True,
        )
        assert pr.period == "2y"
        assert pr.sharpe == 1.5

    def test_different_periods(self):
        for period in ["2y", "1y", "6mo"]:
            pr = PeriodResult(period, 0.0, 0.0, 0.0, 0, 0.0, False)
            assert pr.period == period


class TestRegimeInfo:
    def test_defaults(self):
        ri = RegimeInfo()
        assert ri.best_regime == ""
        assert ri.works_in == []
        assert ri.avoid_in == []

    def test_with_values(self):
        ri = RegimeInfo(
            best_regime="TRENDING",
            works_in=["TRENDING", "VOLATILE"],
            avoid_in=["RANGING"],
        )
        assert ri.best_regime == "TRENDING"
        assert len(ri.works_in) == 2
        assert "RANGING" in ri.avoid_in


class TestStrategyReport:
    def _make_report(self, **overrides):
        defaults = dict(
            strategy_name="test_strat",
            ticker="SPY",
            params={"fast": 10, "slow": 20},
            date_promoted="2026-04-28",
            verdict="ROBUST",
            vbt_sharpe=2.0,
            vbt_total_return=0.30,
            vbt_max_drawdown=-0.08,
            vbt_num_trades=40,
            vbt_win_rate=0.60,
            vbt_score=3.0,
            confirm_sharpe=1.5,
            confirm_total_return=0.25,
            confirm_max_drawdown=-0.10,
            confirm_num_trades=50,
            confirm_win_rate=0.55,
            confirm_profit_factor=1.3,
            confirm_expectancy=0.02,
        )
        defaults.update(overrides)
        return StrategyReport(**defaults)

    def test_basic_creation(self):
        r = self._make_report()
        assert r.strategy_name == "test_strat"
        assert r.ticker == "SPY"
        assert r.vbt_sharpe == 2.0
        assert r.confirm_sharpe == 1.5

    def test_empty_defaults(self):
        r = StrategyReport()
        assert r.strategy_name == ""
        assert r.vbt_sharpe == 0.0
        assert r.slippage_results == []
        assert r.period_results == []

    def test_slippage_results(self):
        slippages = [
            SlippageResult(0.0, 1.5, 0.20, -0.08, 50, 0.55, True),
            SlippageResult(0.001, 1.3, 0.15, -0.09, 48, 0.54, True),
            SlippageResult(0.002, 0.8, 0.05, -0.15, 45, 0.50, False),
        ]
        r = self._make_report(slippage_results=slippages)
        assert len(r.slippage_results) == 3
        assert r.slippage_results[0].passed is True
        assert r.slippage_results[2].passed is False

    def test_period_results(self):
        periods = [
            PeriodResult("2y", 1.5, 0.20, -0.08, 50, 0.55, True),
            PeriodResult("1y", 1.2, 0.15, -0.10, 25, 0.52, True),
            PeriodResult("6mo", 0.8, 0.08, -0.12, 12, 0.50, False),
        ]
        r = self._make_report(period_results=periods)
        assert len(r.period_results) == 3
        assert r.period_results[0].period == "2y"

    def test_regime_info(self):
        ri = RegimeInfo(
            best_regime="TRENDING",
            works_in=["TRENDING", "VOLATILE"],
            avoid_in=["RANGING"],
        )
        r = self._make_report(regime_info=ri, discovery_regime="trending")
        assert r.regime_info.best_regime == "TRENDING"
        assert r.discovery_regime == "trending"

    def test_to_markdown_basic(self):
        r = self._make_report()
        md = r.to_markdown()
        assert "# SPY / test_strat — PRODUCTION" in md
        assert "**Verdict:** ROBUST" in md
        assert "## VectorBT Results" in md
        assert "## Confirmation Results" in md

    def test_to_markdown_sharpe_display(self):
        r = self._make_report(vbt_sharpe=2.5, vbt_num_trades=100)
        md = r.to_markdown()
        assert "Sharpe: 2.50" in md
        assert "Trades: 100" in md

    def test_to_markdown_slippage_section(self):
        slippages = [
            SlippageResult(0.0, 1.5, 0.20, -0.08, 50, 0.55, True),
            SlippageResult(0.001, 1.3, 0.15, -0.09, 48, 0.54, True),
            SlippageResult(0.002, 0.8, 0.05, -0.15, 45, 0.50, False),
        ]
        r = self._make_report(slippage_results=slippages)
        md = r.to_markdown()
        assert "## Slippage Sensitivity" in md
        assert "✅" in md
        assert "❌" in md

    def test_to_markdown_period_section(self):
        periods = [
            PeriodResult("2y", 1.5, 0.20, -0.08, 50, 0.55, True),
            PeriodResult("1y", 1.2, 0.15, -0.10, 25, 0.52, True),
        ]
        r = self._make_report(period_results=periods)
        md = r.to_markdown()
        assert "## Period Performance" in md
        assert "2y:" in md
        assert "1y:" in md

    def test_to_markdown_params_section(self):
        r = self._make_report(params={"fast": 10, "slow": 20, "period": 14})
        md = r.to_markdown()
        assert "## Strategy Parameters" in md
        assert "fast: 10" in md
        assert "slow: 20" in md

    def test_to_markdown_no_params(self):
        r = self._make_report(params={})
        md = r.to_markdown()
        assert "## Strategy Parameters" not in md

    def test_to_markdown_regime_section(self):
        ri = RegimeInfo(
            best_regime="TRENDING",
            works_in=["TRENDING", "VOLATILE"],
            avoid_in=["RANGING"],
        )
        r = self._make_report(
            regime_info=ri,
            discovery_regime="trending",
            validation_regime="volatile",
        )
        md = r.to_markdown()
        assert "## Regime" in md
        assert "Discovered in: **TRENDING**" in md
        assert "Confirmed in: **VOLATILE**" in md
        assert "TRENDING" in md
        assert "VOLATILE" in md
        assert "RANGING" in md

    def test_to_markdown_no_regime(self):
        ri = RegimeInfo()  # empty
        r = self._make_report(regime_info=ri)
        md = r.to_markdown()
        assert "## Regime" not in md

    def test_to_markdown_metadata(self):
        r = self._make_report()
        md = r.to_markdown()
        assert "<!-- METADATA" in md
        assert "-->" in md
        # Extract metadata JSON
        start = md.index("<!-- METADATA\n") + len("<!-- METADATA\n")
        end = md.index("\n-->")
        metadata = json.loads(md[start:end])
        assert metadata["strategy_name"] == "test_strat"
        assert metadata["ticker"] == "SPY"

    def test_to_markdown_fill_degradation(self):
        r = self._make_report(
            vbt_total_return=0.30,
            confirm_total_return=0.20,
        )
        md = r.to_markdown()
        assert "Realistic fill degradation" in md

    def test_to_markdown_no_fill_degradation_when_zero_return(self):
        r = self._make_report(vbt_total_return=0.0, confirm_total_return=0.0)
        md = r.to_markdown()
        assert "Realistic fill degradation" not in md

    def test_to_dict(self):
        slippages = [SlippageResult(0.0, 1.5, 0.20, -0.08, 50, 0.55, True)]
        periods = [PeriodResult("2y", 1.5, 0.20, -0.08, 50, 0.55, True)]
        ri = RegimeInfo(best_regime="TRENDING")
        r = self._make_report(
            slippage_results=slippages,
            period_results=periods,
            regime_info=ri,
        )
        d = r.to_dict()
        assert d["strategy_name"] == "test_strat"
        assert isinstance(d["slippage_results"], list)
        assert isinstance(d["period_results"], list)
        assert isinstance(d["regime_info"], dict)
        assert d["regime_info"]["best_regime"] == "TRENDING"
        assert d["slippage_results"][0]["passed"] is True

    def test_from_dict_roundtrip(self):
        slippages = [SlippageResult(0.0, 1.5, 0.20, -0.08, 50, 0.55, True)]
        periods = [PeriodResult("2y", 1.5, 0.20, -0.08, 50, 0.55, True)]
        ri = RegimeInfo(best_regime="TRENDING", works_in=["TRENDING"], avoid_in=["RANGING"])
        r1 = self._make_report(
            slippage_results=slippages,
            period_results=periods,
            regime_info=ri,
        )
        d = r1.to_dict()
        r2 = StrategyReport.from_dict(d)
        assert r2.strategy_name == r1.strategy_name
        assert r2.ticker == r1.ticker
        assert r2.vbt_sharpe == r1.vbt_sharpe
        assert r2.confirm_sharpe == r1.confirm_sharpe
        assert len(r2.slippage_results) == 1
        assert r2.slippage_results[0].passed is True
        assert len(r2.period_results) == 1
        assert r2.period_results[0].period == "2y"
        assert r2.regime_info.best_regime == "TRENDING"
        assert r2.regime_info.works_in == ["TRENDING"]
        assert r2.regime_info.avoid_in == ["RANGING"]

    def test_from_dict_empty_nested(self):
        r = StrategyReport.from_dict({"strategy_name": "test"})
        assert r.strategy_name == "test"
        assert r.slippage_results == []
        assert r.period_results == []
        assert r.regime_info.best_regime == ""

    def test_pct_formatting(self):
        assert StrategyReport._pct(0.55) == "55.0%"
        assert StrategyReport._pct(0.0) == "0.0%"
        assert StrategyReport._pct(-0.10) == "-10.0%"
        assert StrategyReport._pct(1.0) == "100.0%"
