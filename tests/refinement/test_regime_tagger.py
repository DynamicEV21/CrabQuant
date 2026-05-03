"""Tests for regime_tagger.py — compute preferred_regimes for strategies."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestRegimeTagger:
    """Test strategy regime tagging."""

    def test_compute_strategy_regime_tags_returns_expected_keys(self):
        """Result has all expected keys."""
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.num_trades = 10
        mock_result.sharpe = 1.5
        mock_engine.run.return_value = mock_result

        mock_df = pd.DataFrame({
            "close": [100 + i * 0.5 for i in range(300)],
            "high": [101 + i * 0.5 for i in range(300)],
            "low": [99 + i * 0.5 for i in range(300)],
            "open": [100 + i * 0.5 for i in range(300)],
            "volume": [1000000] * 300,
        }, index=pd.date_range("2023-01-01", periods=300))

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df), \
             patch("crabquant.refinement.regime_tagger.BacktestEngine", return_value=mock_engine):
            result = compute_strategy_regime_tags(mock_fn, {}, ticker="SPY")

        assert "preferred_regimes" in result
        assert "acceptable_regimes" in result
        assert "weak_regimes" in result
        assert "regime_sharpes" in result
        assert "is_regime_specific" in result

    def test_compute_strategy_regime_tags_empty_on_insufficient_data(self):
        """Returns empty result when data is too short."""
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")

        mock_df = pd.DataFrame({
            "close": [100] * 50,
            "high": [101] * 50,
            "low": [99] * 50,
            "open": [100] * 50,
            "volume": [1000000] * 50,
        }, index=pd.date_range("2023-01-01", periods=50))

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df):
            result = compute_strategy_regime_tags(mock_fn, {}, ticker="SPY")

        assert result["preferred_regimes"] == []
        assert result["is_regime_specific"] is False

    def test_compute_strategy_regime_tags_handles_load_error(self):
        """Returns empty result when data loading fails."""
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")

        with patch("crabquant.refinement.regime_tagger.load_data", side_effect=Exception("API down")):
            result = compute_strategy_regime_tags(mock_fn, {}, ticker="BAD")

        assert result["preferred_regimes"] == []
        assert result["regime_sharpes"] == {}

    def test_get_regime_strategies_filters_by_regime(self):
        """Filters strategies by regime Sharpe threshold."""
        from crabquant.refinement.regime_tagger import get_regime_strategies

        registry = {
            "trend_strategy": {
                "fn": MagicMock(),
                "description": "Trend following",
                "regime_sharpes": {"trending_up": 1.5, "high_volatility": -0.3},
            },
            "range_strategy": {
                "fn": MagicMock(),
                "description": "Range bound",
                "regime_sharpes": {"range_bound": 1.2, "trending_up": 0.1},
            },
            "universal_strategy": {
                "fn": MagicMock(),
                "description": "Works everywhere",
                "regime_sharpes": {"trending_up": 2.0, "range_bound": 1.8, "high_volatility": 1.0},
            },
        }

        # Get strategies that work in trending_up
        results = get_regime_strategies("trending_up", registry, min_sharpe=0.5)

        names = [r["name"] for r in results]
        assert "trend_strategy" in names
        assert "universal_strategy" in names
        assert "range_strategy" not in names  # only 0.1 in trending_up

    def test_get_regime_strategies_handles_legacy_tuples(self):
        """Legacy tuple entries are included without regime filtering."""
        from crabquant.refinement.regime_tagger import get_regime_strategies

        registry = {
            "new_strategy": {
                "fn": MagicMock(),
                "description": "New format",
                "regime_sharpes": {"trending_up": 1.5},
            },
            "old_strategy": (
                MagicMock(),  # fn
                {},  # defaults
                {},  # grid
                "Old format strategy",  # description
                None,  # matrix_fn
            ),
        }

        results = get_regime_strategies("trending_up", registry, min_sharpe=0.5)

        # New strategy with known regime Sharpe comes first
        assert results[0]["name"] == "new_strategy"
        assert results[0]["regime_sharpe"] == 1.5
        # Old strategy has no regime data, included but sorted last
        assert results[1]["name"] == "old_strategy"
        assert results[1]["regime_sharpe"] is None


class TestEmptyResult:
    """Tests for the _empty_result helper."""

    def test_returns_dict(self):
        from crabquant.refinement.regime_tagger import _empty_result
        result = _empty_result()
        assert isinstance(result, dict)

    def test_has_all_expected_keys(self):
        from crabquant.refinement.regime_tagger import _empty_result
        result = _empty_result()
        expected = {"preferred_regimes", "acceptable_regimes", "weak_regimes", "regime_sharpes", "is_regime_specific"}
        assert set(result.keys()) == expected

    def test_all_lists_are_empty(self):
        from crabquant.refinement.regime_tagger import _empty_result
        result = _empty_result()
        assert result["preferred_regimes"] == []
        assert result["acceptable_regimes"] == []
        assert result["weak_regimes"] == []

    def test_regime_sharpes_is_empty_dict(self):
        from crabquant.refinement.regime_tagger import _empty_result
        result = _empty_result()
        assert result["regime_sharpes"] == {}

    def test_is_regime_specific_is_false(self):
        from crabquant.refinement.regime_tagger import _empty_result
        result = _empty_result()
        assert result["is_regime_specific"] is False

    def test_returns_new_dict_each_call(self):
        from crabquant.refinement.regime_tagger import _empty_result
        r1 = _empty_result()
        r2 = _empty_result()
        r1["preferred_regimes"].append("test")
        assert r2["preferred_regimes"] == []


class TestComputeStrategyRegimeTagsExpanded:
    """Expanded tests for compute_strategy_regime_tags."""

    def _make_mock_df(self, n=300, trend=0.5):
        import numpy as np
        prices = 100 + np.cumsum(np.random.default_rng(42).normal(0, 1, n)) * trend
        return pd.DataFrame({
            "close": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "open": prices,
            "volume": [1000000] * n,
        }, index=pd.date_range("2023-01-01", periods=n))

    def test_returns_empty_when_strategy_raises(self):
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        def bad_strategy(df, params):
            raise RuntimeError("strategy failed")

        mock_df = self._make_mock_df()
        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df), \
             patch("crabquant.refinement.regime_tagger.BacktestEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_result = MagicMock()
            mock_result.num_trades = 10
            mock_engine.run.return_value = mock_result
            # Strategy itself raises
            result = compute_strategy_regime_tags(bad_strategy, {})

        assert result["preferred_regimes"] == []
        assert result["is_regime_specific"] is False

    def test_returns_empty_when_few_trades(self):
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")
        mock_df = self._make_mock_df()

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df), \
             patch("crabquant.refinement.regime_tagger.BacktestEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_result = MagicMock()
            mock_result.num_trades = 2  # < 5
            mock_engine.run.return_value = mock_result

            result = compute_strategy_regime_tags(mock_fn, {})

        assert result["preferred_regimes"] == []
        assert result["regime_sharpes"] == {}

    def test_returns_empty_when_few_trades_boundary(self):
        """Exactly 4 trades should return empty (need >= 5)."""
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")
        mock_df = self._make_mock_df()

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df), \
             patch("crabquant.refinement.regime_tagger.BacktestEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_result = MagicMock()
            mock_result.num_trades = 4
            mock_engine.run.return_value = mock_result

            result = compute_strategy_regime_tags(mock_fn, {})

        assert result["preferred_regimes"] == []

    def test_uses_provided_engine(self):
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy", side_effect=lambda df, params: (pd.Series(False, index=df.index), pd.Series(False, index=df.index)))
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.num_trades = 10
        mock_engine.run.return_value = mock_result
        mock_df = self._make_mock_df()

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df):
            compute_strategy_regime_tags(mock_fn, {}, engine=mock_engine)

        mock_engine.run.assert_called_once()

    def test_custom_ticker_and_period(self):
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.num_trades = 10
        mock_engine.run.return_value = mock_result
        mock_df = self._make_mock_df()

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df) as mock_load:
            compute_strategy_regime_tags(mock_fn, {}, ticker="AAPL", period="5y", engine=mock_engine)

        # load_data called with custom ticker
        mock_load.assert_any_call("AAPL", period="5y")

    def test_spy_load_failure_does_not_crash(self):
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.num_trades = 10
        mock_engine.run.return_value = mock_result
        mock_df = self._make_mock_df()

        call_count = [0]
        def load_side_effect(ticker, **kwargs):
            call_count[0] += 1
            if ticker == "SPY" and call_count[0] > 1:
                raise Exception("SPY unavailable")
            return mock_df

        with patch("crabquant.refinement.regime_tagger.load_data", side_effect=load_side_effect):
            result = compute_strategy_regime_tags(mock_fn, {}, engine=mock_engine)

        # Should not crash; result may be empty or have data
        assert "preferred_regimes" in result

    def test_data_length_exactly_200(self):
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.num_trades = 10
        mock_engine.run.return_value = mock_result
        mock_df = self._make_mock_df(n=200)

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df):
            result = compute_strategy_regime_tags(mock_fn, {}, engine=mock_engine)

        assert "preferred_regimes" in result

    def test_data_length_199_returns_empty(self):
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")
        mock_df = self._make_mock_df(n=199)

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df):
            result = compute_strategy_regime_tags(mock_fn, {})

        assert result["preferred_regimes"] == []

    def test_preferred_regimes_are_sorted(self):
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.num_trades = 10
        mock_engine.run.return_value = mock_result
        mock_df = self._make_mock_df()

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df):
            result = compute_strategy_regime_tags(mock_fn, {}, engine=mock_engine)

        assert result["preferred_regimes"] == sorted(result["preferred_regimes"])
        assert result["acceptable_regimes"] == sorted(result["acceptable_regimes"])
        assert result["weak_regimes"] == sorted(result["weak_regimes"])

    def test_regime_sharpes_is_dict(self):
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.num_trades = 10
        mock_engine.run.return_value = mock_result
        mock_df = self._make_mock_df()

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df):
            result = compute_strategy_regime_tags(mock_fn, {}, engine=mock_engine)

        assert isinstance(result["regime_sharpes"], dict)

    def test_is_regime_specific_is_bool(self):
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.num_trades = 10
        mock_engine.run.return_value = mock_result
        mock_df = self._make_mock_df()

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df):
            result = compute_strategy_regime_tags(mock_fn, {}, engine=mock_engine)

        assert isinstance(result["is_regime_specific"], bool)


class TestLabelRegimes:
    """Tests for _label_regimes helper."""

    def test_returns_series(self):
        from crabquant.refinement.regime_tagger import _label_regimes
        df = pd.DataFrame({"close": [100] * 100}, index=pd.date_range("2023-01-01", periods=100))
        result = _label_regimes(df, None)
        assert isinstance(result, pd.Series)

    def test_index_matches_input(self):
        from crabquant.refinement.regime_tagger import _label_regimes
        df = pd.DataFrame({"close": [100] * 100}, index=pd.date_range("2023-01-01", periods=100))
        result = _label_regimes(df, None)
        assert result.index.equals(df.index)

    def test_default_label_is_unknown(self):
        from crabquant.refinement.regime_tagger import _label_regimes
        df = pd.DataFrame({"close": [100] * 30}, index=pd.date_range("2023-01-01", periods=30))
        result = _label_regimes(df, None)
        # Bars before lookback should be "unknown"
        assert (result.iloc[:50] == "unknown").all() if len(result) >= 50 else (result == "unknown").all()

    def test_with_spy_df_fallback_to_ticker(self):
        from crabquant.refinement.regime_tagger import _label_regimes
        df = pd.DataFrame({"close": [100] * 100}, index=pd.date_range("2023-01-01", periods=100))
        # spy_df=None should use df itself
        result = _label_regimes(df, None)
        assert isinstance(result, pd.Series)

    def test_with_empty_spy_df_uses_ticker(self):
        from crabquant.refinement.regime_tagger import _label_regimes
        df = pd.DataFrame({"close": [100] * 100}, index=pd.date_range("2023-01-01", periods=100))
        empty_spy = pd.DataFrame()
        result = _label_regimes(df, empty_spy)
        assert isinstance(result, pd.Series)

    def test_short_dataframe(self):
        from crabquant.refinement.regime_tagger import _label_regimes
        df = pd.DataFrame({"close": [100] * 10}, index=pd.date_range("2023-01-01", periods=10))
        result = _label_regimes(df, None)
        assert (result == "unknown").all()

    def test_with_different_spy_index_aligns(self):
        from crabquant.refinement.regime_tagger import _label_regimes
        idx = pd.date_range("2023-01-01", periods=100)
        spy_idx = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame({"close": [100] * 100}, index=idx)
        spy_df = pd.DataFrame({"close": [100] * 100}, index=spy_idx)
        result = _label_regimes(df, spy_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)


class TestComputePerRegimeSharpe:
    """Tests for _compute_per_regime_sharpe helper."""

    def _make_df(self, n=300):
        import numpy as np
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0.1, 1, n))
        return pd.DataFrame({
            "close": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "open": prices,
            "volume": [1000000] * n,
        }, index=pd.date_range("2023-01-01", periods=n))

    def test_returns_dict(self):
        from crabquant.refinement.regime_tagger import _compute_per_regime_sharpe
        df = self._make_df()
        entries = pd.Series(False, index=df.index)
        result_mock = MagicMock()
        with patch("crabquant.refinement.regime_tagger.load_data", return_value=df):
            result = _compute_per_regime_sharpe(df, entries, result_mock)
        assert isinstance(result, dict)

    def test_with_all_false_entries(self):
        from crabquant.refinement.regime_tagger import _compute_per_regime_sharpe
        df = self._make_df()
        entries = pd.Series(False, index=df.index)
        with patch("crabquant.refinement.regime_tagger.load_data", return_value=df):
            result = _compute_per_regime_sharpe(df, entries, MagicMock())
        # All flat positions → returns should be 0 → std=0 → skipped
        assert isinstance(result, dict)

    def test_with_all_true_entries(self):
        from crabquant.refinement.regime_tagger import _compute_per_regime_sharpe
        df = self._make_df()
        entries = pd.Series(True, index=df.index)
        with patch("crabquant.refinement.regime_tagger.load_data", return_value=df):
            result = _compute_per_regime_sharpe(df, entries, MagicMock())
        assert isinstance(result, dict)

    def test_spy_load_failure_handled(self):
        from crabquant.refinement.regime_tagger import _compute_per_regime_sharpe
        df = self._make_df()
        entries = pd.Series(True, index=df.index)
        with patch("crabquant.refinement.regime_tagger.load_data", side_effect=Exception("no SPY")):
            result = _compute_per_regime_sharpe(df, entries, MagicMock())
        assert isinstance(result, dict)

    def test_short_dataframe(self):
        from crabquant.refinement.regime_tagger import _compute_per_regime_sharpe
        df = self._make_df(n=30)
        entries = pd.Series(True, index=df.index)
        with patch("crabquant.refinement.regime_tagger.load_data", return_value=df):
            result = _compute_per_regime_sharpe(df, entries, MagicMock())
        assert isinstance(result, dict)


class TestGetRegimeStrategiesExpanded:
    """Expanded tests for get_regime_strategies."""

    def test_empty_registry_returns_empty(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        results = get_regime_strategies("trending_up", {})
        assert results == []

    def test_no_strategies_meet_threshold(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "weak_strat": {
                "description": "Weak",
                "regime_sharpes": {"trending_up": 0.1},
            }
        }
        results = get_regime_strategies("trending_up", registry, min_sharpe=0.5)
        assert results == []

    def test_all_strategies_meet_threshold(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "strat_a": {"description": "A", "regime_sharpes": {"trending_up": 1.5}},
            "strat_b": {"description": "B", "regime_sharpes": {"trending_up": 2.0}},
        }
        results = get_regime_strategies("trending_up", registry, min_sharpe=0.5)
        assert len(results) == 2

    def test_sorted_by_sharpe_descending(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "low": {"description": "Low", "regime_sharpes": {"trending_up": 0.6}},
            "high": {"description": "High", "regime_sharpes": {"trending_up": 2.0}},
            "mid": {"description": "Mid", "regime_sharpes": {"trending_up": 1.2}},
        }
        results = get_regime_strategies("trending_up", registry, min_sharpe=0.5)
        sharpes = [r["regime_sharpe"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_unknown_regime_returns_empty_for_dict_entries(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "strat_a": {"description": "A", "regime_sharpes": {"trending_up": 1.5}},
        }
        results = get_regime_strategies("unknown_regime", registry, min_sharpe=0.5)
        # Dict entries with no sharpe for unknown regime → 0 < 0.5 → excluded
        assert results == []

    def test_min_sharpe_zero_includes_all_dict_entries_with_data(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "strat_a": {"description": "A", "regime_sharpes": {"trending_up": 0.1}},
            "strat_b": {"description": "B", "regime_sharpes": {"trending_up": 0.3}},
        }
        results = get_regime_strategies("trending_up", registry, min_sharpe=0.0)
        assert len(results) == 2

    def test_negative_min_sharpe_includes_negative_sharpe_strategies(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "strat_a": {"description": "A", "regime_sharpes": {"trending_up": -0.3}},
        }
        results = get_regime_strategies("trending_up", registry, min_sharpe=-1.0)
        assert len(results) == 1

    def test_legacy_tuple_without_description(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "old_strat": (MagicMock(), {}, {}, None, None),
        }
        results = get_regime_strategies("trending_up", registry, min_sharpe=0.5)
        assert len(results) == 1
        assert results[0]["description"] == ""

    def test_legacy_tuple_with_description(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "old_strat": (MagicMock(), {}, {}, "Good strategy", None),
        }
        results = get_regime_strategies("trending_up", registry, min_sharpe=0.5)
        assert results[0]["description"] == "Good strategy"

    def test_legacy_tuple_shorter_than_4_elements_ignored(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "short_tuple": (MagicMock(), {},),  # only 2 elements
        }
        results = get_regime_strategies("trending_up", registry, min_sharpe=0.5)
        assert results == []

    def test_mixed_dict_and_legacy_entries(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "new_strat": {
                "description": "New",
                "regime_sharpes": {"trending_up": 1.5},
            },
            "old_strat": (MagicMock(), {}, {}, "Old", None),
        }
        results = get_regime_strategies("trending_up", registry, min_sharpe=0.5)
        assert len(results) == 2
        # Dict entry (known sharpe) should come first
        assert results[0]["name"] == "new_strat"
        assert results[0]["regime_sharpe"] == 1.5
        assert results[1]["name"] == "old_strat"
        assert results[1]["regime_sharpe"] is None

    def test_result_dict_has_expected_keys(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "strat_a": {
                "description": "A",
                "regime_sharpes": {"trending_up": 1.5, "high_volatility": 0.8},
            }
        }
        results = get_regime_strategies("trending_up", registry)
        assert len(results) == 1
        r = results[0]
        assert "name" in r
        assert "regime_sharpe" in r
        assert "all_regime_sharpes" in r
        assert "description" in r

    def test_all_regime_sharpes_copied_to_result(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        sharpes = {"trending_up": 1.5, "high_volatility": 0.8, "range_bound": 0.2}
        registry = {"strat_a": {"description": "A", "regime_sharpes": sharpes}}
        results = get_regime_strategies("trending_up", registry)
        assert results[0]["all_regime_sharpes"] == sharpes

    def test_regime_sharpe_default_zero_when_regime_missing(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "strat_a": {"description": "A", "regime_sharpes": {"range_bound": 1.0}},
        }
        results = get_regime_strategies("trending_up", registry, min_sharpe=0.0)
        assert len(results) == 1
        assert results[0]["regime_sharpe"] == 0

    def test_non_dict_non_tuple_entry_ignored(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "string_entry": "just a string",
            "int_entry": 42,
        }
        results = get_regime_strategies("trending_up", registry)
        assert results == []

    def test_multiple_legacy_tuples_all_included(self):
        from crabquant.refinement.regime_tagger import get_regime_strategies
        registry = {
            "old1": (MagicMock(), {}, {}, "Old 1", None),
            "old2": (MagicMock(), {}, {}, "Old 2", None),
            "old3": (MagicMock(), {}, {}, "Old 3", None),
        }
        results = get_regime_strategies("trending_up", registry)
        assert len(results) == 3
        names = [r["name"] for r in results]
        assert "old1" in names
        assert "old2" in names
        assert "old3" in names


class TestConstants:
    """Verify module-level constants."""

    def test_sharpe_good_threshold(self):
        from crabquant.refinement.regime_tagger import SHARPE_GOOD_THRESHOLD
        assert SHARPE_GOOD_THRESHOLD == 0.8

    def test_sharpe_acceptable_threshold(self):
        from crabquant.refinement.regime_tagger import SHARPE_ACCEPTABLE_THRESHOLD
        assert SHARPE_ACCEPTABLE_THRESHOLD == 0.3

    def test_min_bars_per_regime(self):
        from crabquant.refinement.regime_tagger import MIN_BARS_PER_REGIME
        assert MIN_BARS_PER_REGIME == 20

    def test_good_greater_than_acceptable(self):
        from crabquant.refinement.regime_tagger import SHARPE_GOOD_THRESHOLD, SHARPE_ACCEPTABLE_THRESHOLD
        assert SHARPE_GOOD_THRESHOLD > SHARPE_ACCEPTABLE_THRESHOLD
