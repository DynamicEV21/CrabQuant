"""Tests for crabquant.production.promoter"""

import json
import hashlib
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

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
    _load_registry,
    _save_registry,
    _infer_regime,
    _extract_slippage_results,
    _extract_period_results,
    get_promotion_metrics,
    promote_strategy,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class _MockConfirmResult:
    """Confirm result that supports both attribute and dict-style access."""

    def __init__(self, **overrides):
        self.verdict = "ROBUST"
        self.sharpe = 1.5
        self.total_return = 0.25
        self.max_dd = -0.10
        self.trades = 50
        self.win_rate = 0.55
        self.profit_factor = 1.3
        self.expectancy = 0.02
        self.notes: list[str] = []
        for k, v in overrides.items():
            setattr(self, k, v)

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


# ---------------------------------------------------------------------------
# _params_hash
# ---------------------------------------------------------------------------

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

    def test_always_12_chars(self):
        """Hash is always truncated to 12 characters regardless of input size."""
        for params in [
            {"a": 1},
            {"x": list(range(100))},
            {"k" * 200: "v" * 200},
            {},
        ]:
            assert len(_params_hash(params)) == 12

    def test_string_values(self):
        h = _params_hash({"period": "daily", "method": "sma"})
        assert isinstance(h, str)
        assert len(h) == 12

    def test_list_values(self):
        h = _params_hash({"windows": [5, 10, 20]})
        assert isinstance(h, str)
        assert len(h) == 12

    def test_numeric_zero_values(self):
        h1 = _params_hash({"value": 0})
        h2 = _params_hash({"value": 0.0})
        # JSON serialization differs: 0 vs 0.0
        assert isinstance(h1, str)
        assert isinstance(h2, str)

    def test_special_characters_in_values(self):
        h = _params_hash({"key": "value with spaces & symbols!"})
        assert isinstance(h, str)
        assert len(h) == 12


# ---------------------------------------------------------------------------
# _make_key
# ---------------------------------------------------------------------------

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

    def test_unique_for_different_strategy_names(self):
        k1 = _make_key("strat_a", "SPY", {"a": 1})
        k2 = _make_key("strat_b", "SPY", {"a": 1})
        assert k1 != k2

    def test_empty_params(self):
        key = _make_key("s", "SPY", {})
        parts = key.split("|")
        assert len(parts) == 3
        assert parts[0] == "s"
        assert parts[1] == "SPY"
        assert len(parts[2]) == 12

    def test_same_params_same_key(self):
        k1 = _make_key("s", "SPY", {"a": 1})
        k2 = _make_key("s", "SPY", {"a": 1})
        assert k1 == k2

    def test_hash_part_matches_params_hash(self):
        params = {"fast": 10, "slow": 20}
        key = _make_key("s", "SPY", params)
        expected_hash = _params_hash(params)
        assert key == f"s|SPY|{expected_hash}"


# ---------------------------------------------------------------------------
# _load_registry / _save_registry
# ---------------------------------------------------------------------------

class TestLoadSaveRegistry:
    def test_load_nonexistent_returns_empty(self, tmp_path):
        with patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "nope.json"):
            assert _load_registry() == []

    def test_save_and_load_roundtrip(self, tmp_path):
        reg_file = tmp_path / "registry.json"
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", reg_file):
            entries = [{"key": "a|SPY|h1", "strategy_name": "a"}]
            _save_registry(entries)
            assert reg_file.exists()
            loaded = _load_registry()
            assert loaded == entries

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        reg_file = nested / "registry.json"
        with patch("crabquant.production.promoter.PRODUCTION_DIR", nested), \
             patch("crabquant.production.promoter.REGISTRY_FILE", reg_file):
            _save_registry([])
            assert nested.exists()

    def test_save_overwrites_existing(self, tmp_path):
        reg_file = tmp_path / "registry.json"
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", reg_file):
            _save_registry([{"key": "old"}])
            _save_registry([{"key": "new"}])
            loaded = _load_registry()
            assert loaded == [{"key": "new"}]

    def test_save_with_default_str_serializer(self, tmp_path):
        """_save_registry uses default=str for non-serializable types."""
        import datetime
        reg_file = tmp_path / "registry.json"
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", reg_file):
            entry = {"key": "a", "date": datetime.date(2026, 1, 1)}
            _save_registry([entry])
            loaded = json.loads(reg_file.read_text())
            assert loaded[0]["date"] == "2026-01-01"


# ---------------------------------------------------------------------------
# _infer_regime
# ---------------------------------------------------------------------------

class TestInferRegime:
    def test_returns_empty_regime_info_on_import_error(self):
        with patch.dict("sys.modules", {"crabquant.regime": None}):
            # Force reimport
            import importlib
            from crabquant.production import promoter as mod
            importlib.reload(mod)
            try:
                result = mod._infer_regime("any_strategy")
                assert isinstance(result, RegimeInfo)
                assert result.best_regime == ""
                assert result.works_in == []
                assert result.avoid_in == []
            finally:
                importlib.reload(mod)

    def test_returns_empty_when_strategy_not_found(self):
        """When strategy not in REGIME_STRATEGY_AFFINITY, return empty RegimeInfo."""
        mock_regime_mod = MagicMock()
        mock_regime_mod.REGIME_STRATEGY_AFFINITY = {}
        mock_regime_mod.MarketRegime = MagicMock()

        with patch.dict("sys.modules", {"crabquant.regime": mock_regime_mod}):
            import importlib
            from crabquant.production import promoter as mod
            importlib.reload(mod)
            try:
                result = mod._infer_regime("nonexistent_strategy")
                assert isinstance(result, RegimeInfo)
            finally:
                importlib.reload(mod)

    def test_ranks_regimes_and_sets_best(self):
        from enum import Enum

        class FakeRegime(Enum):
            TRENDING = "trending"
            MEAN_REV = "mean_rev"
            VOLATILE = "volatile"

        mock_regime_mod = MagicMock()
        mock_regime_mod.REGIME_STRATEGY_AFFINITY = {
            FakeRegime.TRENDING: {"my_strat": 0.8},
            FakeRegime.MEAN_REV: {"my_strat": 0.3},
            FakeRegime.VOLATILE: {"my_strat": 0.7},
        }
        mock_regime_mod.MarketRegime = FakeRegime

        with patch.dict("sys.modules", {"crabquant.regime": mock_regime_mod}):
            import importlib
            from crabquant.production import promoter as mod
            importlib.reload(mod)
            try:
                result = mod._infer_regime("my_strat")
                assert result.best_regime == "TRENDING"
                assert "TRENDING" in result.works_in  # 0.8 >= 0.6
                assert "VOLATILE" in result.works_in  # 0.7 >= 0.6
                assert "MEAN_REV" in result.avoid_in  # 0.3 < 0.5
            finally:
                importlib.reload(mod)


# ---------------------------------------------------------------------------
# _extract_slippage_results
# ---------------------------------------------------------------------------

class TestExtractSlippageResults:
    def test_empty_notes(self):
        confirm = _MockConfirmResult()
        assert _extract_slippage_results(confirm) == []

    def test_no_notes_attribute(self):
        """confirm_result with no notes at all — uses a plain dict."""
        confirm = {"verdict": "ROBUST"}
        result = _extract_slippage_results(confirm)
        assert result == []

    def test_single_slippage_note(self):
        confirm = _MockConfirmResult()
        confirm.notes = ["2y @ 0.0% slip | Sharpe 1.5: PASS"]
        results = _extract_slippage_results(confirm)
        assert len(results) == 1
        assert results[0].slippage_pct == 0.0
        assert results[0].passed is True

    def test_all_three_slippage_levels(self):
        confirm = _MockConfirmResult()
        confirm.notes = [
            "2y @ 0.0% slip | Sharpe 1.5: PASS",
            "2y @ 0.1% slip | Sharpe 1.3: PASS",
            "2y @ 0.2% slip | Sharpe 0.8: FAIL",
        ]
        results = _extract_slippage_results(confirm)
        assert len(results) == 3
        assert results[0].passed is True
        assert results[1].passed is True
        assert results[2].passed is False
        assert results[0].slippage_pct == 0.0
        assert results[1].slippage_pct == 0.001
        assert results[2].slippage_pct == 0.002

    def test_slippage_with_fail(self):
        confirm = _MockConfirmResult()
        confirm.notes = ["2y @ 0.1% slip | Sharpe 0.5: FAIL"]
        results = _extract_slippage_results(confirm)
        assert len(results) == 1
        assert results[0].passed is False

    def test_dict_style_notes(self):
        confirm = {
            "verdict": "ROBUST",
            "notes": ["2y @ 0.0% slip | Sharpe 1.0: PASS"],
        }
        results = _extract_slippage_results(confirm)
        assert len(results) == 1
        assert results[0].passed is True

    def test_non_slippage_notes_ignored(self):
        """Notes that don't match the slippage pattern should be ignored."""
        confirm = _MockConfirmResult()
        confirm.notes = [
            "Some random note",
            "Another unrelated note",
        ]
        results = _extract_slippage_results(confirm)
        assert results == []

    def test_robust_verdict_no_slippage_notes_returns_empty(self):
        """ROBUST verdict with no slippage notes returns empty list (with warning)."""
        confirm = _MockConfirmResult()
        confirm.notes = []
        results = _extract_slippage_results(confirm)
        assert results == []

    def test_non_robust_verdict_no_slippage_notes(self):
        confirm = _MockConfirmResult(verdict="FAILED")
        confirm.notes = []
        results = _extract_slippage_results(confirm)
        assert results == []

    def test_slippage_result_fields(self):
        """Slippage results should have default zero numeric fields."""
        confirm = _MockConfirmResult()
        confirm.notes = ["2y @ 0.0% slip | Sharpe 1.5: PASS"]
        results = _extract_slippage_results(confirm)
        sr = results[0]
        assert sr.sharpe == 0.0
        assert sr.total_return == 0.0
        assert sr.max_drawdown == 0.0
        assert sr.num_trades == 0
        assert sr.win_rate == 0.0


# ---------------------------------------------------------------------------
# _extract_period_results
# ---------------------------------------------------------------------------

class TestExtractPeriodResults:
    def test_default_period_result(self):
        """Without notes, should still return the primary 2y period."""
        confirm = _MockConfirmResult()
        results = _extract_period_results(confirm)
        assert len(results) >= 1
        primary = results[0]
        assert primary.period == "2y"
        assert primary.sharpe == 1.5
        assert primary.total_return == 0.25
        assert primary.passed is True

    def test_dict_style_confirm_period(self):
        confirm = {
            "confirm_sharpe": 2.0,
            "confirm_return": 0.30,
            "confirm_max_dd": -0.05,
            "confirm_trades": 60,
            "confirm_win_rate": 0.60,
        }
        results = _extract_period_results(confirm)
        assert len(results) >= 1
        primary = results[0]
        assert primary.sharpe == 2.0
        assert primary.total_return == 0.30

    def test_period_notes_extraction(self):
        confirm = _MockConfirmResult()
        confirm.notes = [
            "2y @ 0.0% slip | Sharpe 1.5: PASS",
            "1y @ 0.0% slip | Sharpe 1.2: PASS",
            "6mo @ 0.0% slip | Sharpe 0.9: FAIL",
        ]
        results = _extract_period_results(confirm)
        assert len(results) == 3
        assert results[0].period == "2y"
        assert results[1].period == "1y"
        assert results[2].period == "6mo"
        # Notes-based results have zero values (primary 2y is overridden by note)
        assert results[1].passed is True
        assert results[2].passed is False

    def test_partial_period_notes(self):
        """Only some periods in notes; primary 2y still included."""
        confirm = _MockConfirmResult()
        confirm.notes = [
            "1y @ 0.0% slip | Sharpe 1.2: PASS",
        ]
        results = _extract_period_results(confirm)
        periods = [r.period for r in results]
        assert "2y" in periods
        assert "1y" in periods

    def test_primary_period_actual_values(self):
        """When no 2y note present, primary period uses confirm_result values."""
        confirm = _MockConfirmResult()
        results = _extract_period_results(confirm)
        primary = results[0]
        assert primary.period == "2y"
        assert primary.sharpe == 1.5
        assert primary.total_return == 0.25
        assert primary.max_drawdown == -0.10
        assert primary.num_trades == 50
        assert primary.win_rate == 0.55

    def test_empty_notes_primary_fallback(self):
        confirm = _MockConfirmResult()
        confirm.notes = []
        results = _extract_period_results(confirm)
        assert len(results) == 1
        assert results[0].period == "2y"


# ---------------------------------------------------------------------------
# get_promotion_metrics
# ---------------------------------------------------------------------------

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

        fake_registry = {"a": MagicMock(), "b": MagicMock()}
        with patch.dict("sys.modules", {
            "crabquant.strategies": MagicMock(STRATEGY_REGISTRY=fake_registry)
        }):
            import importlib
            import crabquant.production.promoter as mod
            importlib.reload(mod)
            try:
                metrics = mod.get_promotion_metrics(p)
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

    def test_empty_winners_list(self, tmp_path):
        """Valid JSON but empty list."""
        p = tmp_path / "winners.json"
        p.write_text("[]")
        metrics = get_promotion_metrics(p)
        assert metrics["total_winners"] == 0
        assert metrics["promotion_rate"] == 0.0

    def test_all_promoted(self, tmp_path):
        winners = [
            {"strategy": "a", "validation_status": "promoted"},
            {"strategy": "b", "validation_status": "promoted"},
        ]
        p = tmp_path / "winners.json"
        p.write_text(json.dumps(winners))
        metrics = get_promotion_metrics(p)
        assert metrics["promoted_count"] == 2
        assert metrics["promotion_rate"] == 1.0

    def test_missing_strategy_field(self, tmp_path):
        """Winner entry missing 'strategy' key should still work."""
        winners = [{"validation_status": "promoted"}]
        p = tmp_path / "winners.json"
        p.write_text(json.dumps(winners))
        metrics = get_promotion_metrics(p)
        assert metrics["total_winners"] == 1
        assert metrics["promoted_count"] == 1


# ---------------------------------------------------------------------------
# promote_strategy
# ---------------------------------------------------------------------------

class TestPromoteStrategy:
    def _mock_confirm(self, **overrides):
        return _MockConfirmResult(**overrides)

    def test_basic_promotion(self, tmp_path):
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm()
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
            confirm = self._mock_confirm()
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
            confirm = self._mock_confirm()
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
            confirm = self._mock_confirm(verdict="FAILED")
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
            confirm = self._mock_confirm()
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
            confirm = self._mock_confirm()
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
            confirm = self._mock_confirm()
            report = promote_strategy(
                strategy_name="test", ticker="SPY", params={},
                vbt_result={}, confirm_result=confirm,
            )
            assert len(report.period_results) >= 1
            primary = report.period_results[0]
            assert primary.period == "2y"
            assert primary.sharpe == 1.5
            assert primary.passed is True

    def test_rejects_weak_verdict(self, tmp_path):
        """Test that various non-ROBUST verdicts are rejected."""
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            for verdict in ("FAILED", "WEAK", "CONDITIONAL", ""):
                confirm = self._mock_confirm(verdict=verdict)
                with pytest.raises(ValueError, match="not ROBUST"):
                    promote_strategy(
                        strategy_name="test", ticker="SPY", params={},
                        vbt_result={}, confirm_result=confirm,
                    )

    def test_discovery_regime_from_vbt(self, tmp_path):
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm()
            report = promote_strategy(
                strategy_name="test", ticker="SPY", params={},
                vbt_result={"regime": "volatile"},
                confirm_result=confirm,
            )
            assert report.discovery_regime == "volatile"

    def test_validation_regime_from_dict_confirm(self, tmp_path):
        """Dict-style confirm_result can have validation_regime."""
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = {
                "verdict": "ROBUST",
                "confirm_sharpe": 1.0, "confirm_return": 0.1,
                "confirm_max_dd": -0.05, "confirm_trades": 10,
                "confirm_win_rate": 0.5, "confirm_profit_factor": 1.1,
                "confirm_expectancy": 0.01, "notes": [],
                "validation_regime": "trending",
            }
            report = promote_strategy(
                strategy_name="test", ticker="SPY", params={},
                vbt_result={}, confirm_result=confirm,
            )
            assert report.validation_regime == "trending"

    def test_vbt_result_defaults(self, tmp_path):
        """Missing keys in vbt_result should default to 0."""
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm()
            report = promote_strategy(
                strategy_name="test", ticker="SPY", params={},
                vbt_result={},  # Empty dict — all keys missing
                confirm_result=confirm,
            )
            assert report.vbt_sharpe == 0
            assert report.vbt_total_return == 0
            assert report.vbt_max_drawdown == 0
            assert report.vbt_num_trades == 0
            assert report.vbt_win_rate == 0
            assert report.vbt_score == 0

    def test_dict_confirm_missing_defaults(self, tmp_path):
        """Dict-style confirm with missing keys should default to 0."""
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = {"verdict": "ROBUST", "notes": []}
            report = promote_strategy(
                strategy_name="test", ticker="SPY", params={},
                vbt_result={}, confirm_result=confirm,
            )
            assert report.confirm_sharpe == 0
            assert report.confirm_profit_factor == 0
            assert report.confirm_expectancy == 0

    def test_registry_entry_structure(self, tmp_path):
        """Verify all expected fields in the registry entry."""
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm()
            promote_strategy(
                strategy_name="my_strat", ticker="TSLA",
                params={"fast": 5},
                vbt_result={"sharpe": 1.0},
                confirm_result=confirm,
            )
            registry = json.loads((tmp_path / "registry.json").read_text())
            entry = registry[0]
            assert "key" in entry
            assert "strategy_name" in entry
            assert "ticker" in entry
            assert "params_hash" in entry
            assert "params" in entry
            assert "promoted_at" in entry
            assert "verdict" in entry
            assert "report_file" in entry
            assert entry["verdict"] == "ROBUST"
            assert entry["params"] == {"fast": 5}

    def test_multiple_promotions_append(self, tmp_path):
        """Promoting multiple strategies appends to registry."""
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            promote_strategy(
                strategy_name="s1", ticker="SPY", params={"a": 1},
                vbt_result={}, confirm_result=self._mock_confirm(),
            )
            promote_strategy(
                strategy_name="s2", ticker="QQQ", params={"b": 2},
                vbt_result={}, confirm_result=self._mock_confirm(),
            )
            registry = json.loads((tmp_path / "registry.json").read_text())
            assert len(registry) == 2
            assert registry[0]["strategy_name"] == "s1"
            assert registry[1]["strategy_name"] == "s2"

    def test_report_markdown_filename(self, tmp_path):
        """Report filename is {strategy_name}_{ticker}.md."""
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            promote_strategy(
                strategy_name="rsi_strat", ticker="MSFT",
                params={}, vbt_result={},
                confirm_result=self._mock_confirm(),
            )
            assert (tmp_path / "rsi_strat_MSFT.md").exists()

    def test_promoted_at_is_today(self, tmp_path):
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", tmp_path / "registry.json"), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm()
            report = promote_strategy(
                strategy_name="test", ticker="SPY", params={},
                vbt_result={}, confirm_result=confirm,
            )
            assert report.date_promoted == date.today().isoformat()

    def test_duplicate_error_includes_date(self, tmp_path):
        """Duplicate promotion error message includes the original promoted_at date."""
        registry_file = tmp_path / "registry.json"
        with patch("crabquant.production.promoter.PRODUCTION_DIR", tmp_path), \
             patch("crabquant.production.promoter.REGISTRY_FILE", registry_file), \
             patch("crabquant.production.promoter._infer_regime", return_value=RegimeInfo()):
            confirm = self._mock_confirm()
            promote_strategy(
                strategy_name="test", ticker="SPY", params={"a": 1},
                vbt_result={}, confirm_result=confirm,
            )
            # The date should be today's date
            from datetime import date as _date
            today = _date.today().isoformat()
            with pytest.raises(ValueError, match=today):
                promote_strategy(
                    strategy_name="test", ticker="SPY", params={"a": 1},
                    vbt_result={}, confirm_result=confirm,
                )
