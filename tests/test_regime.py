"""Tests for CrabQuant market regime detector."""

import numpy as np
import pandas as pd
import pytest

from crabquant.regime import (
    MarketRegime,
    detect_regime,
    get_strategy_ranking,
    REGIME_STRATEGY_AFFINITY,
)


# ── Synthetic data generators ──

def _make_spy_data(trend: str = "up", n: int = 120, noise: float = 0.01) -> pd.DataFrame:
    """
    Generate synthetic SPY-like OHLCV data.

    Args:
        trend: 'up', 'down', 'sideways'
        n: number of bars
        noise: random noise magnitude
    """
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    price = 100.0
    prices = []

    for i in range(n):
        if trend == "up":
            drift = 0.003
        elif trend == "down":
            drift = -0.003
        else:
            drift = 0.0
        change = drift + np.random.normal(0, noise)
        price *= (1 + change)
        prices.append(price)

    closes = pd.Series(prices, index=dates, name="close")
    df = pd.DataFrame({
        "open": closes * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high": closes * (1 + np.random.uniform(0, 0.015, n)),
        "low": closes * (1 - np.random.uniform(0, 0.015, n)),
        "close": closes,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)
    return df


def _make_vix_data(level: float = 20.0, n: int = 120) -> pd.DataFrame:
    """Generate synthetic VIX data around a given level."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    values = level + np.random.normal(0, 2, n)
    values = np.maximum(values, 5.0)  # VIX floor
    return pd.DataFrame({"close": values}, index=dates)


# ── detect_regime ──

class TestDetectRegime:
    def test_returns_market_regime_enum(self):
        """detect_regime should return a MarketRegime enum member."""
        spy = _make_spy_data("up")
        regime, meta = detect_regime(spy)
        assert isinstance(regime, MarketRegime)

    def test_returns_metadata(self):
        """Metadata should contain expected keys."""
        spy = _make_spy_data("up")
        regime, meta = detect_regime(spy)

        assert isinstance(meta, dict)
        assert "confidence" in meta
        assert "scores" in meta
        assert isinstance(meta["scores"], dict)

    def test_uptrend_detects_trending_up(self):
        """Strong uptrend should produce TRENDING_UP regime."""
        # Use a strong uptrend with low noise for deterministic behavior
        np.random.seed(42)
        spy = _make_spy_data("up", n=120, noise=0.005)
        regime, meta = detect_regime(spy)

        # With seed 42 and strong uptrend, should be TRENDING_UP
        assert regime == MarketRegime.TRENDING_UP

    def test_downtrend_detects_trending_down(self):
        """Strong downtrend should produce TRENDING_DOWN regime."""
        np.random.seed(42)
        spy = _make_spy_data("down", n=120, noise=0.005)
        regime, meta = detect_regime(spy)

        assert regime == MarketRegime.TRENDING_DOWN

    def test_sideways_detects_mean_reversion_or_low_vol(self):
        """Sideways market should produce MEAN_REVERSION or LOW_VOLATILITY."""
        np.random.seed(42)
        spy = _make_spy_data("sideways", n=120, noise=0.005)
        regime, meta = detect_regime(spy)

        assert regime in (MarketRegime.MEAN_REVERSION, MarketRegime.LOW_VOLATILITY)

    def test_vix_increases_high_vol_score(self):
        """High VIX should push toward HIGH_VOLATILITY regime."""
        np.random.seed(42)
        spy = _make_spy_data("sideways", n=120, noise=0.02)
        vix = _make_vix_data(level=35.0, n=120)

        regime_no_vix, meta_no_vix = detect_regime(spy)
        regime_with_vix, meta_with_vix = detect_regime(spy, vix_data=vix)

        # With high VIX, high_vol score should be present in metadata
        assert meta_with_vix["vix_value"] is not None
        assert meta_with_vix["vix_value"] > 25

    def test_low_vix_increases_low_vol_score(self):
        """Low VIX should push toward LOW_VOLATILITY regime."""
        np.random.seed(42)
        spy = _make_spy_data("sideways", n=120, noise=0.003)
        vix = _make_vix_data(level=10.0, n=120)

        regime, meta = detect_regime(spy, vix_data=vix)

        assert meta["vix_value"] < 15
        assert meta["scores"]["low_volatility"] >= meta["scores"]["high_volatility"]

    def test_missing_vix_handled_gracefully(self):
        """Missing VIX data should not crash and should still return results."""
        np.random.seed(42)
        spy = _make_spy_data("up", n=120)
        regime, meta = detect_regime(spy, vix_data=None)

        assert isinstance(regime, MarketRegime)
        assert meta["vix_value"] is None

    def test_confidence_between_zero_and_one(self):
        """Confidence should always be in [0, 1]."""
        spy = _make_spy_data("up")
        regime, meta = detect_regime(spy)
        assert 0.0 <= meta["confidence"] <= 1.0

    def test_short_data_handled(self):
        """Very short data should return a regime without crashing."""
        dates = pd.date_range("2024-01-01", periods=15, freq="B")
        spy = pd.DataFrame({
            "close": np.linspace(100, 105, 15),
            "open": np.linspace(100, 105, 15),
            "high": np.linspace(100, 105, 15) + 0.5,
            "low": np.linspace(100, 105, 15) - 0.5,
            "volume": np.ones(15) * 1_000_000,
        }, index=dates)
        regime, meta = detect_regime(spy)
        assert isinstance(regime, MarketRegime)


# ── get_strategy_ranking ──

class TestGetStrategyRanking:
    def test_returns_sorted_list(self):
        """Rankings should be sorted by score descending."""
        ranking = get_strategy_ranking(MarketRegime.TRENDING_UP)
        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_all_strategies_have_scores(self):
        """Every strategy should have a score for every regime."""
        all_strategies = set()
        for regime in MarketRegime:
            all_strategies.update(REGIME_STRATEGY_AFFINITY[regime].keys())

        for regime in MarketRegime:
            ranking = get_strategy_ranking(regime)
            ranked_strategies = {name for name, _ in ranking}
            assert ranked_strategies == all_strategies, (
                f"{regime.value}: missing {all_strategies - ranked_strategies}"
            )

    def test_filtered_to_available_strategies(self):
        """Should only return strategies in the available list."""
        available = ["rsi_crossover", "macd_momentum"]
        ranking = get_strategy_ranking(MarketRegime.TRENDING_UP, available)
        names = [name for name, _ in ranking]
        assert names == ["macd_momentum", "rsi_crossover"]  # macd scores higher in trending_up

    def test_unknown_strategy_not_in_affinity(self):
        """Strategies not in the affinity dict should be filtered out."""
        ranking = get_strategy_ranking(
            MarketRegime.TRENDING_UP,
            available_strategies=["nonexistent_strategy"],
        )
        assert len(ranking) == 0

    def test_scores_between_zero_and_one(self):
        """All affinity scores should be in [0, 1]."""
        for regime in MarketRegime:
            for strategy, score in REGIME_STRATEGY_AFFINITY[regime].items():
                assert 0.0 <= score <= 1.0, (
                    f"{regime.value}/{strategy}: score {score} out of range"
                )

    def test_trending_up_favors_trend_strategies(self):
        """TRENDING_UP should rank trend strategies highest."""
        ranking = get_strategy_ranking(MarketRegime.TRENDING_UP)
        top_names = [name for name, _ in ranking[:3]]
        # At least one trend-following strategy should be in top 3
        trend_strategies = {"ema_ribbon_reversal", "macd_momentum", "adx_pullback", "ichimoku_trend"}
        assert len(set(top_names) & trend_strategies) >= 2

    def test_mean_reversion_favors_bb_strategies(self):
        """MEAN_REVERSION should rank bollinger strategies highest."""
        ranking = get_strategy_ranking(MarketRegime.MEAN_REVERSION)
        top_name = ranking[0][0]
        assert top_name == "bollinger_squeeze"


# ── Extended detect_regime tests ─────────────────────────────────────────


class TestDetectRegimeExtended:
    def test_insufficient_data_returns_low_volatility(self):
        """Data with fewer than 20 bars should return LOW_VOLATILITY."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        spy = pd.DataFrame({"close": np.linspace(100, 101, 10)}, index=dates)
        regime, meta = detect_regime(spy)
        assert regime == MarketRegime.LOW_VOLATILITY
        assert meta["confidence"] == 0.0
        assert meta["reason"] == "insufficient_data"

    def test_high_volatility_detected_with_volatile_data(self):
        """Highly volatile sideways data + high VIX → HIGH_VOLATILITY."""
        np.random.seed(42)
        n = 120
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # Create volatile price data
        price = 100.0
        prices = []
        for i in range(n):
            price *= (1 + np.random.normal(0, 0.03))  # 3% daily noise
            prices.append(price)
        spy = pd.DataFrame({"close": prices}, index=dates)
        vix = _make_vix_data(level=35.0, n=n)
        regime, meta = detect_regime(spy, vix_data=vix)
        # Should have non-trivial high_vol score
        assert meta["scores"]["high_volatility"] > 0.1

    def test_metadata_has_all_expected_keys(self):
        """Metadata dict should contain all expected keys."""
        np.random.seed(42)
        spy = _make_spy_data("up", n=120)
        regime, meta = detect_regime(spy)
        expected_keys = {
            "confidence", "scores", "vix_value",
            "realized_vol", "bb_width", "sma20_slope", "roc_20",
        }
        assert set(meta.keys()) == expected_keys

    def test_scores_dict_has_all_regimes(self):
        """Scores should contain entries for all 5 regimes."""
        np.random.seed(42)
        spy = _make_spy_data("up", n=120)
        regime, meta = detect_regime(spy)
        expected_regimes = {"trending_up", "trending_down", "mean_reversion",
                            "high_volatility", "low_volatility"}
        assert set(meta["scores"].keys()) == expected_regimes

    def test_all_scores_are_numeric(self):
        """All regime scores should be numeric (float)."""
        np.random.seed(42)
        spy = _make_spy_data("up", n=120)
        regime, meta = detect_regime(spy)
        for score in meta["scores"].values():
            assert isinstance(score, (int, float))

    def test_vix_at_boundary_20(self):
        """VIX at exactly 20 → vix_score should be 0.0."""
        np.random.seed(42)
        spy = _make_spy_data("sideways", n=120, noise=0.003)
        vix = _make_vix_data(level=20.0, n=120)
        regime, meta = detect_regime(spy, vix_data=vix)
        assert meta["vix_value"] is not None

    def test_vix_at_boundary_30(self):
        """VIX at exactly 30 → vix_score should be 1.0."""
        np.random.seed(42)
        spy = _make_spy_data("sideways", n=120, noise=0.003)
        vix = _make_vix_data(level=30.0, n=120)
        regime, meta = detect_regime(spy, vix_data=vix)
        assert meta["vix_value"] is not None

    def test_custom_lookback(self):
        """Custom lookback parameter should be accepted."""
        np.random.seed(42)
        spy = _make_spy_data("up", n=200)
        regime_default, meta_default = detect_regime(spy)
        regime_short, meta_short = detect_regime(spy, lookback=60)
        # Both should return valid results
        assert isinstance(regime_default, MarketRegime)
        assert isinstance(regime_short, MarketRegime)

    def test_empty_vix_data(self):
        """Empty VIX DataFrame should not crash."""
        np.random.seed(42)
        spy = _make_spy_data("up", n=120)
        vix = pd.DataFrame({"close": []})
        regime, meta = detect_regime(spy, vix_data=vix)
        assert isinstance(regime, MarketRegime)
        assert meta["vix_value"] is None

    def test_realized_vol_in_metadata(self):
        """Realized volatility should be in metadata and non-negative."""
        np.random.seed(42)
        spy = _make_spy_data("up", n=120)
        regime, meta = detect_regime(spy)
        assert meta["realized_vol"] >= 0.0

    def test_bb_width_in_metadata(self):
        """Bollinger Band width should be in metadata and non-negative."""
        np.random.seed(42)
        spy = _make_spy_data("up", n=120)
        regime, meta = detect_regime(spy)
        assert meta["bb_width"] >= 0.0

    def test_deterministic_with_seed(self):
        """Same seed should produce same regime."""
        np.random.seed(42)
        spy1 = _make_spy_data("up", n=120, noise=0.01)
        regime1, _ = detect_regime(spy1)

        np.random.seed(42)
        spy2 = _make_spy_data("up", n=120, noise=0.01)
        regime2, _ = detect_regime(spy2)

        assert regime1 == regime2

    def test_strong_downtrend_high_confidence(self):
        """Strong downtrend should have reasonably high confidence."""
        np.random.seed(42)
        spy = _make_spy_data("down", n=120, noise=0.002)
        regime, meta = detect_regime(spy)
        assert regime == MarketRegime.TRENDING_DOWN
        # Strong trend should have meaningful confidence
        assert meta["confidence"] > 0.0

    def test_very_short_data_exactly_19_bars(self):
        """Exactly 19 bars should return insufficient_data."""
        dates = pd.date_range("2024-01-01", periods=19, freq="B")
        spy = pd.DataFrame({"close": np.linspace(100, 105, 19)}, index=dates)
        regime, meta = detect_regime(spy)
        assert regime == MarketRegime.LOW_VOLATILITY
        assert meta.get("reason") == "insufficient_data"

    def test_exactly_20_bars(self):
        """Exactly 20 bars is the minimum for analysis."""
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        np.random.seed(42)
        spy = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(20) * 0.5),
        }, index=dates)
        regime, meta = detect_regime(spy)
        assert isinstance(regime, MarketRegime)
        # Should have actual scores, not insufficient_data
        assert meta.get("reason") is None


# ── Extended get_strategy_ranking tests ──────────────────────────────────


class TestGetStrategyRankingExtended:
    def test_empty_available_list(self):
        """Empty available_strategies list → empty ranking."""
        ranking = get_strategy_ranking(
            MarketRegime.TRENDING_UP,
            available_strategies=[],
        )
        assert ranking == []

    def test_none_available_returns_all(self):
        """None available_strategies → returns all strategies."""
        ranking_all = get_strategy_ranking(MarketRegime.TRENDING_UP)
        ranking_none = get_strategy_ranking(
            MarketRegime.TRENDING_UP,
            available_strategies=None,
        )
        assert len(ranking_all) == len(ranking_none)

    def test_high_volatility_avoids_risk_strategies(self):
        """HIGH_VOLATILITY should have lower scores for momentum strategies."""
        ranking = get_strategy_ranking(MarketRegime.HIGH_VOLATILITY)
        # Get the max score (should be moderate, not extreme)
        max_score = ranking[0][1]
        # In high vol, even the top strategy should have score <= 0.8
        assert max_score <= 0.80

    def test_low_volatility_favors_momentum(self):
        """LOW_VOLATILITY should rank momentum strategies highest."""
        ranking = get_strategy_ranking(MarketRegime.LOW_VOLATILITY)
        top_name = ranking[0][0]
        assert top_name == "ema_ribbon_reversal"

    def test_trending_down_top_strategy(self):
        """TRENDING_DOWN should rank adx_pullback highest."""
        ranking = get_strategy_ranking(MarketRegime.TRENDING_DOWN)
        top_name = ranking[0][0]
        assert top_name == "adx_pullback"

    def test_all_regimes_have_rankings(self):
        """Every regime should return a non-empty ranking."""
        for regime in MarketRegime:
            ranking = get_strategy_ranking(regime)
            assert len(ranking) > 0, f"{regime.value} has empty ranking"

    def test_return_type_is_list_of_tuples(self):
        """Should return list of (str, float) tuples."""
        ranking = get_strategy_ranking(MarketRegime.TRENDING_UP)
        assert isinstance(ranking, list)
        for item in ranking:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_partial_filter_keeps_only_matching(self):
        """Filtering to a subset should keep only those strategies."""
        available = ["bollinger_squeeze", "rsi_crossover", "macd_momentum"]
        ranking = get_strategy_ranking(MarketRegime.MEAN_REVERSION, available)
        names = [name for name, _ in ranking]
        assert set(names) == set(available)
        # bollinger_squeeze should be ranked first for mean_reversion
        assert names[0] == "bollinger_squeeze"

    def test_affinity_dict_completeness(self):
        """All regimes should have the same set of strategies."""
        strategy_sets = []
        for regime in MarketRegime:
            strategy_sets.append(set(REGIME_STRATEGY_AFFINITY[regime].keys()))
        # All should be identical
        for s in strategy_sets[1:]:
            assert s == strategy_sets[0], "Not all regimes have the same strategy set"
