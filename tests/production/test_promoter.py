"""Tests for crabquant.production.promoter"""

import json
import hashlib
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from crabquant.production.report import (
    StrategyReport,
    SlippageResult,
    PeriodResult,
    RegimeInfo,
)
from crabquant.production.promoter import (
    _params_hash,
    _make_key,
    get_promotion_metrics,
    promote_strategy,
)


class TestParamsHash:
    def test_deterministic(self):
        h1 = _params_hash({"a": 1, "b": 2})
        h2 = _params_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_params(self):
        h1 = _params_hash({"a": 1})
        h2 = _params_hash({"a": 2})
        assert h1 != h2

    def test_empty_params(self):
        h = _params_hash({})
        assert isinstance(h, str)
        assert len(h) == 12

    def test_nested_params(self):
        h = _params_hash({"nested": {"key": "value"}})
        assert isinstance(h, str)
        assert len(h) == 12


class TestMakeKey:
    def test_format(self):
        key = _make_key("ema_cross", "SPY", {"fast": 10, "slow": 20})
        assert key.startswith("ema_cross|SPY|")
        assert len(key.split("|")) == 3

    def test_unique_for_different_params(self):
        k1 = _make_key("strat", "SPY", {"a": 1})
        k2 = _make_key("strat", "SPY", {"a": 2})
        assert k1 != k2

    def test_unique_for_different_tickers(self):
        k1 = _make_key("strat", "SPY", {"a": 1})
        k2 = _make_key("strat", "QQQ", {"a": 1})
        assert k1 != k2


class TestGetPromotionMetrics:
    def test_empty_file(self, tmp_path):
        metrics = get_promotion_metrics(tmp_path / "nonexistent.json")
        assert metrics["total_winners"] == 0
        assert metrics["promotion_rate"] == 0.0
        assert metrics["backtest_only_count"] == 0
        assert metrics["promoted_count"] == 0

    def test_all_backtest_only(self, tmp_path):
        winners = [
            {"strategy": "a", "validation_status": "backtest_only"},
            {"strategy": "b", "validation_status": "backtest_only"},
        ]
        p = tmp_path / "winners.json"
        p.write_text(json.dumps(winners))
        metrics = get_promotion_metrics(p)
        assert metrics["total_winners"] == 2
        assert metrics["backtest_only_count"] == 2
        assert metrics["promotion_rate"] == 0.0

    def test_mixed_statuses(self, tmp_path):
        winners = [
            {"strategy": "a", "validation_status": "backtest_only"},
            {"strategy": "b", "validation_status": "walk_forward_passed"},
            {"strategy": "c", "validation_status": "confirmed"},
            {"strategy": "d", "validation_status": "promoted"},
        ]
        p = tmp_path / "winners.json"
        p.write_text(json.dumps(winners))
        metrics = get_promotion_metrics(p)
        assert metrics["total_winners"] == 4
        assert metrics["backtest_only_count"] == 1
        assert metrics["walk_forward_passed_count"] == 1
        assert metrics["confirmed_count"] == 1
        assert metrics["promoted_count"] == 1
        assert metrics["promotion_rate"] == 0.25

    def test_unknown_status_treated_as_backtest(self, tmp_path):
        winners = [
            {"strategy": "a", "validation_status": "unknown_status"},
        ]
        p = tmp_path / "winners.json"
        p.write_text(json.dumps(winners))
        metrics = get_promotion_metrics(p)
        assert metrics["backtest_only_count"] == 1

    def test_missing_status_defaults_to_backtest(self, tmp_path):
        winners = [
            {"strategy": "a"},
        ]
        p = tmp_path / "winners.json"
        p.write_text(json.dumps(winners))
        metrics = get_promotion_metrics(p)
        assert metrics["backtest_only_count"] == 1

    def test_cross_check_with_registry(self, tmp_path):
        """When a winner's strategy is in STRATEGY_REGISTRY, it counts as promoted."""
        winners = [
            {"strategy": "a", "validation_status": "backtest_only"},
            {"strategy": "b", "validation_status": "confirmed"},
        ]
        p = tmp_path / "winners.json"
        p.write_text(json.dumps(winners))

        # The function imports STRATEGY_REGISTRY inside a try/except.
        # We mock it at the source module level.
        fake_registry = {"a": MagicMock(), "b": MagicMock()}
        with patch.dict("sys.modules", {
            "crabquant.strategies": MagicMock(STRATEGY_REGISTRY=fake_registry)
        }):
            # Clear any cached import
            import importlib
            import crabquant.production.promoter as mod
            importlib.reload(mod)
            try:
                metrics = mod.get_promotion_metrics(p)
                # "a" and "b" are both in the registry → both promoted
                assert metrics["total_winners"] == 2
                assert metrics["promoted_count"] == 2
                assert metrics["promotion_rate"] == 1.0
            finally:
                importlib.reload(mod)

    def test_invalid_json_returns_empty(self, tmp_path):
        p = tmp_path / "winners.json"
        p.write_text("not json")
        metrics = get_promotion_metrics(p)
        assert metrics["total_winners"] == 0


class TestPromoteStrategy:
    def _mock_confirm_result(self, **overrides):
        """Create a confirm result that supports both attribute and dict access."""
        class ConfirmResult:
            def __init__(self):
                self.verdict = "ROBUST"
                self.sharpe = 1.5
                self.total_return = 0.25
                self.max_dd = -0.10
                self.trades = 50
                self.win_rate = 0.55
                self.profit_factor = 1.3
                self.expectancy = 0.02
                self.notes = []

            def get(self, key, default=None):
                mapping = {
                    "verdict": self.verdict,
                    "sharpe": self.sharpe,
                    "total_return": self.total_return,
                    "max_dd": self.max_dd,
                    "trades": self.trades,
                    "win_rate": self.win_rate,
                    "profit_factor": self.profit_factor,
                    "expectancy": self.expectancy,
                    "notes": self.notes,
                    "confirm_sharpe": self.sharpe,
                    "confirm_return": self.total_return,
                    "confirm_max_dd": self.max_dd,
                    "confirm_trades": self.trades,
                    "confirm_win_rate": self.win_rate,
                    "confirm_profit_factor": self.profit_factor,
                    "confirm_expectancy": self.expectancy,
                    "validation_regime": "",
                }
                return mapping.get(key, default)

        result = ConfirmResult()
        for k, v in overrides.items():
            setattr(result, k, v)
        return result

    def test_basic_promotion(self, tmp_path):
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm_result()
            report = promote_strategy(
                strategy_name="ema_cross",
                ticker="SPY",
                params={"fast": 10, "slow": 20},
                vbt_result={
                    "sharpe": 2.0, "return": 0.30, "max_dd": -0.08,
                    "trades": 40, "score": 3.0, "win_rate": 0.60, "regime": "trending",
                },
                confirm_result=confirm,
            )
            assert isinstance(report, StrategyReport)
            assert report.strategy_name == "ema_cross"
            assert report.ticker == "SPY"
            assert report.verdict == "ROBUST"
            assert report.vbt_sharpe == 2.0
            assert report.confirm_sharpe == 1.5
            assert report.key == _make_key("ema_cross", "SPY", {"fast": 10, "slow": 20})

    def test_promotion_creates_registry_file(self, tmp_path):
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm_result()
            promote_strategy(
                strategy_name="test_strat",
                ticker="AAPL",
                params={"period": 14},
                vbt_result={"sharpe": 1.8, "return": 0.20, "max_dd": -0.12, "trades": 30, "score": 2.5, "win_rate": 0.55},
                confirm_result=confirm,
            )
            registry_file = tmp_path / "registry.json"
            assert registry_file.exists()
            registry = json.loads(registry_file.read_text())
            assert len(registry) == 1
            assert registry[0]["strategy_name"] == "test_strat"
            assert registry[0]["ticker"] == "AAPL"

    def test_promotion_creates_markdown_report(self, tmp_path):
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm_result()
            promote_strategy(
                strategy_name="test_strat",
                ticker="SPY",
                params={},
                vbt_result={"sharpe": 1.5, "return": 0.15, "max_dd": -0.10, "trades": 25, "score": 2.0, "win_rate": 0.52},
                confirm_result=confirm,
            )
            report_file = tmp_path / "test_strat_SPY.md"
            assert report_file.exists()
            content = report_file.read_text()
            assert "# SPY / test_strat" in content
            assert "PRODUCTION" in content

    def test_rejects_non_robust(self, tmp_path):
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm_result()
            confirm.verdict = "FAILED"
            with pytest.raises(ValueError, match="not ROBUST"):
                promote_strategy(
                    strategy_name="test", ticker="SPY", params={},
                    vbt_result={}, confirm_result=confirm,
                )

    def test_rejects_duplicate(self, tmp_path):
        registry_file = tmp_path / "registry.json"
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", registry_file), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm_result()
            promote_strategy(
                strategy_name="test", ticker="SPY", params={"a": 1},
                vbt_result={}, confirm_result=confirm,
            )
            with pytest.raises(ValueError, match="already promoted"):
                promote_strategy(
                    strategy_name="test", ticker="SPY", params={"a": 1},
                    vbt_result={}, confirm_result=confirm,
                )

    def test_dict_style_confirm_result(self, tmp_path):
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = {
                "verdict": "ROBUST",
                "confirm_sharpe": 1.2,
                "confirm_return": 0.18,
                "confirm_max_dd": -0.09,
                "confirm_trades": 35,
                "confirm_win_rate": 0.54,
                "confirm_profit_factor": 1.2,
                "confirm_expectancy": 0.015,
                "notes": [],
            }
            report = promote_strategy(
                strategy_name="test", ticker="SPY", params={},
                vbt_result={"sharpe": 1.8, "return": 0.22, "max_dd": -0.07, "trades": 30, "score": 2.0, "win_rate": 0.58},
                confirm_result=confirm,
            )
            assert report.confirm_sharpe == 1.2
            assert report.confirm_num_trades == 35

    def test_slippage_extraction_from_notes(self, tmp_path):
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm_result()
            confirm.notes = [
                "2y @ 0.0% slip | Sharpe 1.5: PASS",
                "2y @ 0.1% slip | Sharpe 1.3: PASS",
                "2y @ 0.2% slip | Sharpe 0.8: FAIL",
            ]
            report = promote_strategy(
                strategy_name="test", ticker="SPY", params={},
                vbt_result={}, confirm_result=confirm,
            )
            assert len(report.slippage_results) == 3
            assert report.slippage_results[0].passed is True
            assert report.slippage_results[2].passed is False

    def test_period_extraction(self, tmp_path):
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm_result()
            report = promote_strategy(
                strategy_name="test", ticker="SPY", params={},
                vbt_result={}, confirm_result=confirm,
            )
            assert len(report.period_results) >= 1
            primary = report.period_results[0]
            assert primary.period == "2y"
            assert primary.sharpe == 1.5
            assert primary.passed is True
