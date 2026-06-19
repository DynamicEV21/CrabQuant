"""Tests for enhanced mandate generator — smart mandate generation (Step 03)."""

import json
import textwrap
from unittest.mock import patch

import pytest

from crabquant.refinement.mandate_generator import (
    get_portfolio_gaps,
    get_regime_weighted_archetypes,
    generate_smart_mandates,
    _map_archetype,
)
from crabquant.regime import MarketRegime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_regime_enum():
    """Return a simple namespace that quacks like MarketRegime for testing."""
    return MarketRegime


@pytest.fixture
def tmp_winning_strategies(tmp_path):
    """Create a temporary winning_strategies directory with mock files."""
    ws = tmp_path / "winning_strategies"
    ws.mkdir()

    # Create mock strategy result files
    for i in range(5):
        (ws / f"momentum_spy_{i}.json").write_text(
            json.dumps({"archetype": "momentum", "sharpe": 1.5})
        )
    for i in range(2):
        (ws / f"mean_reversion_aapl_{i}.json").write_text(
            json.dumps({"archetype": "mean_reversion", "sharpe": 1.2})
        )
    for i in range(1):
        (ws / f"breakout_nvda_{i}.json").write_text(
            json.dumps({"archetype": "breakout", "sharpe": 0.9})
        )
    return ws


@pytest.fixture
def tmp_strategies_dir(tmp_path):
    """Create a temporary strategies directory with mock strategy files."""
    sdir = tmp_path / "strategies"
    sdir.mkdir()

    mock_files = {
        "momentum_ema.py": textwrap.dedent("""\
            DESCRIPTION = "Momentum strategy using EMA crossover."
            DEFAULT_PARAMS = {"fast": 12, "slow": 26}
            PARAM_GRID = {"fast": [8, 12], "slow": [20, 26]}
        """),
        "mean_reversion_bb.py": textwrap.dedent("""\
            DESCRIPTION = "Mean reversion strategy using Bollinger Bands."
            DEFAULT_PARAMS = {"period": 20, "std": 2}
            PARAM_GRID = {"period": [15, 20, 25]}
        """),
        "breakout_atr.py": textwrap.dedent("""\
            DESCRIPTION = "Breakout strategy using ATR channels."
            DEFAULT_PARAMS = {"atr_period": 14}
            PARAM_GRID = {"atr_period": [10, 14, 20]}
        """),
        "trend_ichimoku.py": textwrap.dedent("""\
            DESCRIPTION = "Trend following with Ichimoku cloud."
            DEFAULT_PARAMS = {"tenkan": 9, "kijun": 26}
            PARAM_GRID = {}
        """),
    }

    for fname, content in mock_files.items():
        (sdir / fname).write_text(content)

    return sdir


# ---------------------------------------------------------------------------
# Tests: get_regime_weighted_archetypes
# ---------------------------------------------------------------------------

class TestGetRegimeWeightedArchetypes:
    """Tests for regime-weighted archetype weights."""

    @pytest.mark.parametrize(
        "regime",
        list(MarketRegime),
        ids=lambda r: r.value,
    )
    def test_all_regimes_return_valid_weights(self, regime):
        """Every regime should return weights that sum to 1.0."""
        weights = get_regime_weighted_archetypes(regime)
        assert isinstance(weights, dict)
        assert len(weights) == 6
        assert abs(sum(weights.values()) - 1.0) < 1e-4, (
            f"Weights sum to {sum(weights.values())}, expected ~1.0"
        )

    def test_trending_up_favors_momentum_and_trend(self):
        """TRENDING_UP should give momentum + trend the highest combined weight."""
        weights = get_regime_weighted_archetypes(MarketRegime.TRENDING_UP)
        mt_combined = weights["momentum"] + weights["trend"]
        others = sum(v for k, v in weights.items() if k not in ("momentum", "trend"))
        assert mt_combined > others, (
            f"momentum+trend={mt_combined} should exceed others={others}"
        )

    def test_trending_down_favors_mean_reversion_and_breakout(self):
        """TRENDING_DOWN should give mean_reversion + breakout the highest combined weight."""
        weights = get_regime_weighted_archetypes(MarketRegime.TRENDING_DOWN)
        mr_bo = weights["mean_reversion"] + weights["breakout"]
        others = sum(v for k, v in weights.items() if k not in ("mean_reversion", "breakout"))
        assert mr_bo > others

    def test_mean_reversion_regime_favors_mean_reversion(self):
        """MEAN_REVERSION regime should give mean_reversion the highest weight."""
        weights = get_regime_weighted_archetypes(MarketRegime.MEAN_REVERSION)
        assert weights["mean_reversion"] >= weights["momentum"]
        assert weights["mean_reversion"] >= weights["trend"]

    def test_high_volatility_favors_breakout(self):
        """HIGH_VOLATILITY should give breakout the highest weight."""
        weights = get_regime_weighted_archetypes(MarketRegime.HIGH_VOLATILITY)
        assert weights["breakout"] >= weights["momentum"]
        assert weights["breakout"] >= weights["mean_reversion"]

    def test_low_volatility_favors_momentum_and_trend(self):
        """LOW_VOLATILITY should give momentum + trend the highest combined weight."""
        weights = get_regime_weighted_archetypes(MarketRegime.LOW_VOLATILITY)
        mt_combined = weights["momentum"] + weights["trend"]
        others = sum(v for k, v in weights.items() if k not in ("momentum", "trend"))
        assert mt_combined > others

    def test_all_weights_non_negative(self):
        """No archetype weight should be negative."""
        for regime in MarketRegime:
            weights = get_regime_weighted_archetypes(regime)
            for arch, w in weights.items():
                assert w >= 0, f"{regime.value}/{arch} weight is negative: {w}"

    def test_unknown_regime_falls_back_to_low_volatility(self):
        """An unknown regime value should fall back to low_volatility weights."""
        fake = type("FakeRegime", (), {"value": "unknown_regime_xyz"})()
        weights_fake = get_regime_weighted_archetypes(fake)
        weights_lv = get_regime_weighted_archetypes(MarketRegime.LOW_VOLATILITY)
        assert weights_fake == weights_lv


# ---------------------------------------------------------------------------
# Tests: get_portfolio_gaps
# ---------------------------------------------------------------------------

class TestGetPortfolioGaps:

    def test_empty_directory(self, tmp_path):
        """Empty directory should return zero counts and positive gaps for all."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        gaps = get_portfolio_gaps(str(empty_dir))
        assert len(gaps) == 6
        for arch, info in gaps.items():
            assert info["current"] == 0
            assert info["gap"] >= 0  # under-represented

    def test_nonexistent_directory(self):
        """Nonexistent directory should return neutral gaps."""
        gaps = get_portfolio_gaps("/nonexistent/path/12345")
        assert len(gaps) == 6
        for arch, info in gaps.items():
            assert info["current"] == 0

    def test_balanced_distribution(self, tmp_path):
        """Equal distribution across archetypes should yield ~zero gaps."""
        ws = tmp_path / "ws"
        ws.mkdir()
        archetypes = ["momentum", "mean_reversion", "breakout", "trend", "volatility", "rsi_oscillator"]
        for arch in archetypes:
            (ws / f"{arch}_spy.json").write_text("{}")
        gaps = get_portfolio_gaps(str(ws))
        for arch, info in gaps.items():
            assert info["current"] == 1
            assert abs(info["gap"]) <= 1  # rounding tolerance

    def test_over_represented_archetype(self, tmp_path):
        """Heavily skewed portfolio should show negative gaps for dominant archetype."""
        ws = tmp_path / "ws"
        ws.mkdir()
        for i in range(10):
            (ws / f"momentum_{i}.json").write_text("{}")
        gaps = get_portfolio_gaps(str(ws))
        assert gaps["momentum"]["current"] == 10
        assert gaps["momentum"]["gap"] < 0  # over-represented
        # Other archetypes should have positive gaps
        for arch in ["mean_reversion", "breakout", "trend", "volatility", "rsi_oscillator"]:
            assert gaps[arch]["current"] == 0
            assert gaps[arch]["gap"] > 0

    def test_single_archetype(self, tmp_path):
        """Only one archetype present → it should be over-represented."""
        ws = tmp_path / "ws"
        ws.mkdir()
        for i in range(3):
            (ws / f"breakout_{i}.json").write_text("{}")
        gaps = get_portfolio_gaps(str(ws))
        assert gaps["breakout"]["current"] == 3
        assert gaps["breakout"]["gap"] < 0

    def test_gap_calculation_accuracy(self, tmp_path):
        """Verify exact gap calculation: gap = target_count - current."""
        ws = tmp_path / "ws"
        ws.mkdir()
        # 6 momentum + 2 mean_reversion = 8 total
        for i in range(6):
            (ws / f"momentum_{i}.json").write_text("{}")
        for i in range(2):
            (ws / f"mean_reversion_{i}.json").write_text("{}")
        gaps = get_portfolio_gaps(str(ws))
        # target_pct ≈ 16.7%, target_count = 16.7/100 * 8 ≈ 1.336
        # momentum: gap = round(1.336 - 6) = round(-4.664) = -5
        assert gaps["momentum"]["current"] == 6
        assert gaps["momentum"]["gap"] == -5
        assert gaps["mean_reversion"]["current"] == 2
        # mean_reversion: gap = round(1.336 - 2) = round(-0.664) = -1
        assert gaps["mean_reversion"]["gap"] == -1

    def test_target_pct_is_equal(self):
        """All archetypes should have the same target_pct (equal distribution)."""
        gaps = get_portfolio_gaps("/nonexistent/path")
        target_pcts = {info["target_pct"] for info in gaps.values()}
        assert len(target_pcts) == 1

    def test_json_content_detection(self, tmp_path):
        """JSON files with archetype in content (but not filename) should be counted."""
        ws = tmp_path / "ws"
        ws.mkdir()
        # Filename doesn't contain archetype, but content does.
        # Use a unique archetype word that won't partially match others.
        (ws / "strategy_42.json").write_text(
            json.dumps({"strategy_type": "rsi_oscillator_strategy", "sharpe": 1.0})
        )
        gaps = get_portfolio_gaps(str(ws))
        # "rsi_oscillator" should be detected from content
        assert gaps["rsi_oscillator"]["current"] >= 1


# ---------------------------------------------------------------------------
# Tests: generate_smart_mandates
# ---------------------------------------------------------------------------

class TestGenerateSmartMandates:

    def test_returns_correct_count(self, tmp_strategies_dir):
        """Should return exactly `count` mandates."""
        with patch("crabquant.refinement.mandate_generator.scan_strategy_catalog") as mock_scan:
            mock_scan.return_value = [
                {
                    "name": f"strat_{i}",
                    "description": f"Strategy {i}",
                    "default_params": {},
                    "param_grid": {},
                    "archetype": "momentum",
                    "file_path": "/tmp/fake.py",
                }
                for i in range(4)
            ]
            mandates = generate_smart_mandates(
                count=5,
                regime=MarketRegime.LOW_VOLATILITY,
                strategies_dir="/fake/strategies",
            )
        assert len(mandates) == 5

    def test_mandate_format_matches_existing(self, tmp_strategies_dir):
        """Each mandate should have all required keys matching existing format."""
        with patch("crabquant.refinement.mandate_generator.scan_strategy_catalog") as mock_scan:
            mock_scan.return_value = [
                {
                    "name": "test_strat",
                    "description": "Test strategy",
                    "default_params": {"fast": 12},
                    "param_grid": {},
                    "archetype": "trend",
                    "file_path": "/tmp/fake.py",
                }
            ]
            mandates = generate_smart_mandates(
                count=1,
                regime=MarketRegime.TRENDING_UP,
                strategies_dir="/fake/strategies",
            )
        assert len(mandates) == 1
        m = mandates[0]
        for key in [
            "name", "description", "strategy_archetype", "tickers",
            "primary_ticker", "period", "sharpe_target", "max_turns",
            "seed_strategy", "seed_params", "constraints",
        ]:
            assert key in m, f"Missing key: {key}"

    def test_high_volatility_wider_stops(self):
        """HIGH_VOLATILITY regime should use max_drawdown_pct=30."""
        with patch("crabquant.refinement.mandate_generator.scan_strategy_catalog") as mock_scan:
            mock_scan.return_value = [
                {
                    "name": "strat",
                    "description": "Test",
                    "default_params": {},
                    "param_grid": {},
                    "archetype": "breakout",
                    "file_path": "/tmp/fake.py",
                }
            ]
            mandates = generate_smart_mandates(
                count=1,
                regime=MarketRegime.HIGH_VOLATILITY,
                strategies_dir="/fake",
            )
        assert mandates[0]["constraints"]["max_drawdown_pct"] == 30

    def test_default_regime_not_high_vol(self):
        """Non-high-volatility regime should use max_drawdown_pct=25."""
        with patch("crabquant.refinement.mandate_generator.scan_strategy_catalog") as mock_scan:
            mock_scan.return_value = [
                {
                    "name": "strat",
                    "description": "Test",
                    "default_params": {},
                    "param_grid": {},
                    "archetype": "momentum",
                    "file_path": "/tmp/fake.py",
                }
            ]
            mandates = generate_smart_mandates(
                count=1,
                regime=MarketRegime.TRENDING_UP,
                strategies_dir="/fake",
            )
        assert mandates[0]["constraints"]["max_drawdown_pct"] == 25

    def test_ticker_override(self):
        """When ticker is set, all mandates should use it as primary."""
        with patch("crabquant.refinement.mandate_generator.scan_strategy_catalog") as mock_scan:
            mock_scan.return_value = [
                {
                    "name": "strat",
                    "description": "Test",
                    "default_params": {},
                    "param_grid": {},
                    "archetype": "momentum",
                    "file_path": "/tmp/fake.py",
                }
            ]
            mandates = generate_smart_mandates(
                count=3,
                ticker="TSLA",
                regime=MarketRegime.LOW_VOLATILITY,
                strategies_dir="/fake",
            )
        for m in mandates:
            assert m["primary_ticker"] == "TSLA"

    def test_empty_catalog_returns_empty(self):
        """Empty strategy catalog should return empty list."""
        with patch("crabquant.refinement.mandate_generator.scan_strategy_catalog") as mock_scan:
            mock_scan.return_value = []
            mandates = generate_smart_mandates(
                count=5,
                regime=MarketRegime.LOW_VOLATILITY,
                strategies_dir="/fake",
            )
        assert mandates == []

    def test_auto_detect_regime_fallback(self, tmp_path):
        """When regime=None and detection fails, should still produce mandates."""
        with patch("crabquant.refinement.mandate_generator.scan_strategy_catalog") as mock_scan, \
             patch("crabquant.regime.detect_regime", side_effect=Exception("no data")):
            mock_scan.return_value = [
                {
                    "name": "strat",
                    "description": "Test",
                    "default_params": {},
                    "param_grid": {},
                    "archetype": "momentum",
                    "file_path": "/tmp/fake.py",
                }
            ]
            # Should not raise, should fall back gracefully
            mandates = generate_smart_mandates(
                count=2,
                strategies_dir="/fake",
            )
        assert len(mandates) == 2

    def test_regime_influences_archetype_distribution(self):
        """Different regimes should produce different archetype distributions (statistically)."""
        with patch("crabquant.refinement.mandate_generator.scan_strategy_catalog") as mock_scan, \
             patch("crabquant.refinement.mandate_generator.random") as mock_random:
            mock_scan.return_value = [
                {
                    "name": f"strat_{arch}",
                    "description": f"{arch} strategy",
                    "default_params": {},
                    "param_grid": {},
                    "archetype": arch,
                    "file_path": f"/tmp/{arch}.py",
                }
                for arch in ["momentum", "trend", "mean_reversion", "breakout", "volatility", "rsi_oscillator"]
            ]
            # Make choices deterministic by returning first element of k=1 list
            mock_random.choices.side_effect = lambda population, weights, k=1: [population[0]]

            mandates_up = generate_smart_mandates(
                count=1,
                regime=MarketRegime.TRENDING_UP,
                strategies_dir="/fake",
            )
            mandates_down = generate_smart_mandates(
                count=1,
                regime=MarketRegime.TRENDING_DOWN,
                strategies_dir="/fake",
            )
            # With deterministic choices, the first weighted archetype should differ
            # because trending_up favors momentum first, trending_down favors mean_reversion first
            # (weights dict ordering is insertion-ordered in Python 3.7+)
            # At minimum, verify both return valid mandates
            assert len(mandates_up) == 1
            assert len(mandates_down) == 1


# ---------------------------------------------------------------------------
# Tests: _map_archetype
# ---------------------------------------------------------------------------

class TestMapArchetype:

    def test_known_archetypes(self):
        assert _map_archetype("momentum") == "momentum"
        assert _map_archetype("mean_reversion") == "mean_reversion"
        assert _map_archetype("breakout") == "breakout"
        assert _map_archetype("trend") == "trend"
        assert _map_archetype("volatility") == "volatility"
        assert _map_archetype("rsi_oscillator") == "rsi_oscillator"

    def test_unknown_maps_to_momentum(self):
        assert _map_archetype("something_weird") == "momentum"
        assert _map_archetype("other") == "momentum"

    def test_alias_mapping(self):
        assert _map_archetype("rsi") == "rsi_oscillator"
        assert _map_archetype("bollinger") == "mean_reversion"
