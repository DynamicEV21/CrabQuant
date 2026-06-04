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


# ── Extended _pct tests ─────────────────────────────────────────────────


class TestPctFormattingExtended:

    @pytest.mark.parametrize("value,expected", [
        (0.001, "0.1%"),
        (0.999, "99.9%"),
        (0.12345, "12.3%"),
        (-0.555, "-55.5%"),
        (-1.5, "-150.0%"),
        (2.5, "250.0%"),
        (0.005, "0.5%"),
        (0.0001, "0.0%"),
    ])
    def test_pct_various_values(self, value, expected):
        assert StrategyReport._pct(value) == expected


# ── Extended to_markdown tests ───────────────────────────────────────────


class TestToMarkdownExtended:

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

    def test_markdown_empty_report(self):
        """Empty/default report should still render without errors."""
        r = StrategyReport()
        md = r.to_markdown()
        assert "#  /  — PRODUCTION" in md
        assert "## VectorBT Results" in md
        assert "## Confirmation Results" in md

    def test_markdown_fill_degradation_positive(self):
        """When confirm > vbt return, degradation should be positive."""
        r = self._make_report(
            vbt_total_return=0.10,
            confirm_total_return=0.20,
            confirm_sharpe=2.0,
        )
        md = r.to_markdown()
        assert "Realistic fill degradation" in md
        assert "+100%" in md  # (0.20 - 0.10) / 0.10 * 100

    def test_markdown_fill_degradation_negative(self):
        """When confirm < vbt return, degradation should be negative."""
        r = self._make_report(
            vbt_total_return=0.30,
            confirm_total_return=0.15,
            confirm_sharpe=1.0,
        )
        md = r.to_markdown()
        assert "Realistic fill degradation" in md
        assert "-50%" in md  # (0.15 - 0.30) / 0.30 * 100

    def test_markdown_composite_score(self):
        r = self._make_report(vbt_score=4.56)
        md = r.to_markdown()
        assert "4.56" in md

    def test_markdown_no_slippage(self):
        """Without slippage results, section should not appear."""
        r = self._make_report(slippage_results=[])
        md = r.to_markdown()
        assert "## Slippage Sensitivity" not in md

    def test_markdown_no_period_results(self):
        """Without period results, section should not appear."""
        r = self._make_report(period_results=[])
        md = r.to_markdown()
        assert "## Period Performance" not in md

    def test_markdown_slippage_all_passed(self):
        slippages = [
            SlippageResult(0.0, 1.5, 0.20, -0.08, 50, 0.55, True),
            SlippageResult(0.001, 1.3, 0.15, -0.09, 48, 0.54, True),
        ]
        r = self._make_report(slippage_results=slippages)
        md = r.to_markdown()
        assert "✅" in md
        assert "❌" not in md

    def test_markdown_slippage_all_failed(self):
        slippages = [
            SlippageResult(0.0, 0.5, 0.02, -0.20, 50, 0.40, False),
            SlippageResult(0.001, 0.3, 0.01, -0.25, 48, 0.35, False),
        ]
        r = self._make_report(slippage_results=slippages)
        md = r.to_markdown()
        assert "❌" in md
        assert "✅" not in md

    def test_markdown_regime_no_discovery(self):
        """Regime section with no discovery/validation regime."""
        ri = RegimeInfo(best_regime="TRENDING", works_in=["TRENDING"])
        r = self._make_report(regime_info=ri)
        md = r.to_markdown()
        assert "## Regime" in md
        assert "Discovered in:" not in md
        assert "Confirmed in:" not in md

    def test_markdown_regime_no_works_avoid(self):
        ri = RegimeInfo(best_regime="TRENDING")
        r = self._make_report(regime_info=ri, discovery_regime="trending")
        md = r.to_markdown()
        assert "## Regime" in md
        assert "Works in:" not in md
        assert "Avoid in:" not in md

    def test_markdown_promoted_date(self):
        r = self._make_report(date_promoted="2026-01-15")
        md = r.to_markdown()
        assert "2026-01-15" in md

    def test_markdown_metadata_contains_all_fields(self):
        r = self._make_report()
        md = r.to_markdown()
        start = md.index("<!-- METADATA\n") + len("<!-- METADATA\n")
        end = md.index("\n-->")
        metadata = json.loads(md[start:end])
        assert "strategy_name" in metadata
        assert "ticker" in metadata
        assert "verdict" in metadata
        assert "vbt_sharpe" in metadata
        assert "confirm_sharpe" in metadata
        assert "key" in metadata
        assert "discovery_regime" in metadata


# ── Extended to_dict / from_dict tests ───────────────────────────────────


class TestDictRoundTripExtended:

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

    def test_full_roundtrip_all_fields(self):
        """to_dict → from_dict → to_dict should be identical."""
        slippages = [
            SlippageResult(0.0, 1.5, 0.20, -0.08, 50, 0.55, True),
            SlippageResult(0.002, 0.8, 0.05, -0.15, 45, 0.50, False),
        ]
        periods = [
            PeriodResult("2y", 1.5, 0.20, -0.08, 50, 0.55, True),
            PeriodResult("1y", 1.2, 0.15, -0.10, 25, 0.52, True),
        ]
        ri = RegimeInfo(
            best_regime="TRENDING",
            works_in=["TRENDING", "VOLATILE"],
            avoid_in=["RANGING"],
        )
        r1 = self._make_report(
            slippage_results=slippages,
            period_results=periods,
            regime_info=ri,
            discovery_regime="trending",
            validation_regime="volatile",
            key="SPY_test_strat_abc123",
        )
        d1 = r1.to_dict()
        r2 = StrategyReport.from_dict(d1.copy())  # copy because from_dict pops keys
        d2 = r2.to_dict()

        assert d1 == d2

    def test_roundtrip_empty_report(self):
        r1 = StrategyReport()
        d1 = r1.to_dict()
        r2 = StrategyReport.from_dict(d1.copy())
        d2 = r2.to_dict()
        assert d1 == d2

    def test_roundtrip_preserves_confirm_metrics(self):
        r1 = self._make_report(
            confirm_profit_factor=2.5,
            confirm_expectancy=0.05,
        )
        r2 = StrategyReport.from_dict(r1.to_dict())
        assert r2.confirm_profit_factor == 2.5
        assert r2.confirm_expectancy == 0.05

    def test_to_dict_nested_are_dicts(self):
        """Nested dataclasses should be plain dicts in to_dict output."""
        slippages = [SlippageResult(0.0, 1.5, 0.20, -0.08, 50, 0.55, True)]
        periods = [PeriodResult("2y", 1.5, 0.20, -0.08, 50, 0.55, True)]
        ri = RegimeInfo(best_regime="TRENDING")
        r = self._make_report(
            slippage_results=slippages,
            period_results=periods,
            regime_info=ri,
        )
        d = r.to_dict()
        assert isinstance(d["slippage_results"][0], dict)
        assert isinstance(d["period_results"][0], dict)
        assert isinstance(d["regime_info"], dict)
        assert not hasattr(d["slippage_results"][0], '__dataclass_fields__')

    def test_from_dict_missing_optional_fields(self):
        """from_dict should handle missing optional fields gracefully."""
        data = {"strategy_name": "test"}
        r = StrategyReport.from_dict(data)
        assert r.strategy_name == "test"
        assert r.ticker == ""
        assert r.vbt_sharpe == 0.0
        assert r.slippage_results == []
        assert r.period_results == []
        assert r.regime_info == RegimeInfo()

    def test_to_dict_preserves_params(self):
        params = {"fast": 5, "slow": 25, "period": 30, "type": "EMA"}
        r = self._make_report(params=params)
        d = r.to_dict()
        assert d["params"] == params


# ── Edge case tests ──────────────────────────────────────────────────────


class TestEdgeCases:

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

    def test_zero_values_render(self):
        """Report with all zero metrics should render without error."""
        r = StrategyReport(
            strategy_name="zero_strat",
            ticker="ZERO",
            vbt_sharpe=0.0,
            vbt_total_return=0.0,
            vbt_max_drawdown=0.0,
            vbt_num_trades=0,
            vbt_win_rate=0.0,
            vbt_score=0.0,
            confirm_sharpe=0.0,
            confirm_total_return=0.0,
            confirm_max_drawdown=0.0,
            confirm_num_trades=0,
            confirm_win_rate=0.0,
        )
        md = r.to_markdown()
        assert "0.00" in md  # sharpe formatted
        assert "0.0%" in md  # pct formatted

    def test_negative_sharpe(self):
        """Negative sharpe should render correctly."""
        r = self._make_report(vbt_sharpe=-1.5, confirm_sharpe=-0.8)
        md = r.to_markdown()
        assert "-1.50" in md
        assert "-0.80" in md

    def test_very_large_numbers(self):
        """Very large metric values should render."""
        r = self._make_report(
            vbt_total_return=5.0,
            vbt_num_trades=100000,
            vbt_score=99.9,
        )
        md = r.to_markdown()
        assert "100000" in md
        assert "99.90" in md

    def test_negative_return(self):
        """Negative total return should render correctly."""
        r = self._make_report(
            vbt_total_return=-0.25,
            confirm_total_return=-0.30,
        )
        md = r.to_markdown()
        # No fill degradation when vbt_total_return < 0 (the condition is > 0)
        assert "Realistic fill degradation" not in md

    def test_win_rate_above_one_hundred_percent(self):
        """Win rate > 1.0 (100%) should still render."""
        r = self._make_report(vbt_win_rate=1.5, confirm_win_rate=1.2)
        md = r.to_markdown()
        assert "150.0%" in md
        assert "120.0%" in md

    def test_negative_max_drawdown(self):
        """Max drawdown should typically be negative."""
        r = self._make_report(vbt_max_drawdown=-0.50, confirm_max_drawdown=-0.60)
        md = r.to_markdown()
        assert "-50.0%" in md
        assert "-60.0%" in md

    def test_positive_max_drawdown(self):
        """Unusual positive drawdown should still render."""
        r = self._make_report(vbt_max_drawdown=0.05)
        md = r.to_markdown()
        assert "5.0%" in md

    def test_empty_params_dict(self):
        r = self._make_report(params={})
        md = r.to_markdown()
        assert "## Strategy Parameters" not in md

    def test_regime_info_with_only_avoid(self):
        ri = RegimeInfo(best_regime="RANGING", avoid_in=["TRENDING"])
        r = self._make_report(regime_info=ri)
        md = r.to_markdown()
        assert "## Regime" in md
        assert "Avoid in:" in md
        assert "Works in:" not in md

    def test_key_field(self):
        r = self._make_report(key="SPY_test_strat_hash123")
        assert r.key == "SPY_test_strat_hash123"
        d = r.to_dict()
        assert d["key"] == "SPY_test_strat_hash123"

    def test_from_dict_preserves_key(self):
        r1 = self._make_report(key="ABC_123")
        r2 = StrategyReport.from_dict(r1.to_dict())
        assert r2.key == "ABC_123"

    def test_slippage_result_with_large_values(self):
        sr = SlippageResult(
            slippage_pct=0.05, sharpe=10.0, total_return=5.0,
            max_drawdown=-0.01, num_trades=10000, win_rate=0.95, passed=True,
        )
        assert sr.sharpe == 10.0
        assert sr.num_trades == 10000

    def test_period_result_various_periods(self):
        for period in ["3mo", "1mo", "5y", "YTD", "MTD"]:
            pr = PeriodResult(period, 1.0, 0.10, -0.05, 20, 0.55, True)
            assert pr.period == period
