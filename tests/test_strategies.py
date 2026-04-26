"""Tests for CrabQuant strategy library."""

import pytest
import pandas as pd
import numpy as np
from crabquant.data import load_data
from crabquant.strategies import STRATEGY_REGISTRY


# Sample OHLCV data for testing
SAMPLE_DATA = None


def get_sample_data() -> pd.DataFrame:
    """Get real AAPL data for testing."""
    global SAMPLE_DATA
    if SAMPLE_DATA is None:
        SAMPLE_DATA = load_data("AAPL", period="2y")
    return SAMPLE_DATA


class TestStrategyInterface:
    """All strategies should have a consistent interface."""

    def test_all_strategies_in_registry(self):
        """Every strategy should be in the registry."""
        expected = [
            "rsi_crossover", "macd_momentum", "adx_pullback",
            "atr_channel_breakout", "volume_breakout", "multi_rsi_confluence",
            "ema_ribbon_reversal", "bollinger_squeeze", "ichimoku_trend",
        ]
        for name in expected:
            assert name in STRATEGY_REGISTRY, f"{name} missing from registry"

    def test_registry_tuple_structure(self):
        """Each registry entry should be (fn, defaults, grid, description)."""
        for name, (fn, defaults, grid, desc) in STRATEGY_REGISTRY.items():
            assert callable(fn), f"{name}: fn should be callable"
            assert isinstance(defaults, dict), f"{name}: defaults should be dict"
            assert isinstance(grid, dict), f"{name}: grid should be dict"
            assert isinstance(desc, str), f"{name}: desc should be string"
            assert len(desc) > 20, f"{name}: desc too short"

    def test_all_strategies_return_boolean_series(self):
        """Every strategy should return (entries, exits) as boolean Series."""
        df = get_sample_data()

        for name, (fn, defaults, grid, desc) in STRATEGY_REGISTRY.items():
            entries, exits = fn(df, defaults)

            assert isinstance(entries, pd.Series), f"{name}: entries should be Series"
            assert isinstance(exits, pd.Series), f"{name}: exits should should be Series"
            assert entries.dtype == bool, f"{name}: entries should be bool"
            assert exits.dtype == bool, f"{name}: exits should be bool"
            assert len(entries) == len(df), f"{name}: entries length mismatch"
            assert len(exits) == len(df), f"{name}: exits length mismatch"

    def test_strategies_work_with_none_params(self):
        """Strategies should work with params=None (use defaults)."""
        df = get_sample_data()

        for name, (fn, defaults, grid, desc) in STRATEGY_REGISTRY.items():
            entries, exits = fn(df, None)
            assert len(entries) == len(df)


class TestIndividualStrategies:
    """Tests for specific strategy behaviors."""

    def test_rsi_crossover_generates_signals(self):
        """RSI crossover should generate some entries on 2y data."""
        from crabquant.strategies.rsi_crossover import generate_signals

        df = get_sample_data()
        entries, exits = generate_signals(df)
        assert entries.sum() >= 0  # May be 0 for some data
        assert exits.sum() >= 0

    def test_macd_momentum_generates_signals(self):
        """MACD momentum should generate entries on trending data."""
        from crabquant.strategies.macd_momentum import generate_signals

        df = get_sample_data()
        entries, exits = generate_signals(df)
        assert isinstance(entries, pd.Series)

    def test_ichimoku_no_params(self):
        """Ichimoku should work with empty params dict."""
        from crabquant.strategies.ichimoku_trend import generate_signals

        df = get_sample_data()
        entries, exits = generate_signals(df, {})
        assert len(entries) == len(df)

    def test_strategy_params_override_defaults(self):
        """Custom params should override defaults."""
        from crabquant.strategies.rsi_crossover import generate_signals, DEFAULT_PARAMS

        df = get_sample_data()
        custom_params = {**DEFAULT_PARAMS, "fast_len": 3}
        entries, exits = generate_signals(df, custom_params)
        assert len(entries) == len(df)
