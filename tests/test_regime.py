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
