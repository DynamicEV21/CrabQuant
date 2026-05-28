"""Tests for mandate_generator — auto-generate mandates from strategy catalog + market analysis."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures — mock strategy files that mimic real strategy modules
# ---------------------------------------------------------------------------

MOCK_STRATEGY_1 = textwrap.dedent("""\
    # Momentum strategy for SPY
    DESCRIPTION = "Momentum strategy using EMA crossover with volume confirmation."
    DEFAULT_PARAMS = {"fast": 12, "slow": 26}
    PARAM_GRID = {"fast": [8, 12, 16], "slow": [20, 26, 50]}

    def generate_signals(df, params):
        import pandas as pd
        fast = params["fast"]
        slow = params["slow"]
        ema_fast = df["close"].ewm(span=fast).mean()
        ema_slow = df["close"].ewm(span=slow).mean()
        entries = (ema_fast > ema_slow) & (ema_fast.shift() <= ema_slow.shift())
        exits = (ema_fast < ema_slow) & (ema_fast.shift() >= ema_slow.shift())
        return entries.fillna(False), exits.fillna(False)

    def generate_signals_matrix(df, param_grid):
        return pd.DataFrame(), pd.DataFrame(), []
""")

MOCK_STRATEGY_2 = textwrap.dedent("""\
    # Mean reversion using RSI
    DESCRIPTION = "Mean reversion strategy based on RSI oversold/overbought."
    DEFAULT_PARAMS = {"rsi_period": 14, "oversold": 30, "overbought": 70}
    PARAM_GRID = {"rsi_period": [10, 14, 20], "oversold": [25, 30], "overbought": [70, 75]}

    def generate_signals(df, params):
        import pandas as pd
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        return entries, exits

    def generate_signals_matrix(df, param_grid):
        return pd.DataFrame(), pd.DataFrame(), []
""")

MOCK_STRATEGY_3 = textwrap.dedent("""\
    # Breakout strategy
    DESCRIPTION = "ATR channel breakout with trend filter."
    DEFAULT_PARAMS = {"atr_period": 14, "mult": 2.0}
    PARAM_GRID = {"atr_period": [10, 14, 20], "mult": [1.5, 2.0, 2.5]}

    def generate_signals(df, params):
        import pandas as pd
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        return entries, exits

    def generate_signals_matrix(df, param_grid):
        return pd.DataFrame(), pd.DataFrame(), []
""")


@pytest.fixture
def mock_strategies_dir(tmp_path):
    """Create a temporary directory with mock strategy files."""
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()
    (strategies_dir / "ema_momentum.py").write_text(MOCK_STRATEGY_1)
    (strategies_dir / "rsi_mean_reversion.py").write_text(MOCK_STRATEGY_2)
    (strategies_dir / "atr_breakout.py").write_text(MOCK_STRATEGY_3)
    return strategies_dir


# ---------------------------------------------------------------------------
# Tests — scan_strategy_catalog
# ---------------------------------------------------------------------------

class TestScanStrategyCatalog:
    """Test scanning strategy files and extracting metadata."""

    def test_scan_finds_all_strategies(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import scan_strategy_catalog
        catalog = scan_strategy_catalog(mock_strategies_dir)
        assert len(catalog) == 3
        names = [s["name"] for s in catalog]
        assert "ema_momentum" in names
        assert "rsi_mean_reversion" in names
        assert "atr_breakout" in names

    def test_scan_extracts_descriptions(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import scan_strategy_catalog
        catalog = scan_strategy_catalog(mock_strategies_dir)
        ema = next(s for s in catalog if s["name"] == "ema_momentum")
        assert "Momentum" in ema["description"]

    def test_scan_extracts_params(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import scan_strategy_catalog
        catalog = scan_strategy_catalog(mock_strategies_dir)
        rsi = next(s for s in catalog if s["name"] == "rsi_mean_reversion")
        assert rsi["default_params"]["rsi_period"] == 14

    def test_scan_empty_dir(self, tmp_path):
        from crabquant.refinement.mandate_generator import scan_strategy_catalog
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        catalog = scan_strategy_catalog(empty_dir)
        assert catalog == []

    def test_scan_skips_non_py_files(self, mock_strategies_dir):
        (mock_strategies_dir / "README.md").write_text("# not a strategy")
        from crabquant.refinement.mandate_generator import scan_strategy_catalog
        catalog = scan_strategy_catalog(mock_strategies_dir)
        assert len(catalog) == 3


# ---------------------------------------------------------------------------
# Tests — detect_archetype
# ---------------------------------------------------------------------------

class TestDetectArchetype:
    """Test automatic archetype detection from strategy description."""

    def test_detect_momentum(self):
        from crabquant.refinement.mandate_generator import detect_archetype
        assert detect_archetype("Momentum strategy using EMA crossover") == "momentum"

    def test_detect_mean_reversion(self):
        from crabquant.refinement.mandate_generator import detect_archetype
        assert detect_archetype("Mean reversion strategy based on RSI") == "mean_reversion"

    def test_detect_breakout(self):
        from crabquant.refinement.mandate_generator import detect_archetype
        assert detect_archetype("ATR channel breakout with trend filter") == "breakout"

    def test_detect_trend(self):
        from crabquant.refinement.mandate_generator import detect_archetype
        assert detect_archetype("Ichimoku trend following strategy") == "trend"

    def test_detect_unknown(self):
        from crabquant.refinement.mandate_generator import detect_archetype
        assert detect_archetype("Some random strategy") == "other"


# ---------------------------------------------------------------------------
# Tests — generate_mandates
# ---------------------------------------------------------------------------

class TestGenerateMandates:
    """Test the main mandate generation function."""

    def test_generates_multiple_mandates(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY", "AAPL", "NVDA"],
            count=5,
        )
        assert len(mandates) == 5

    def test_mandate_has_required_fields(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=1,
        )
        m = mandates[0]
        assert "name" in m
        assert "strategy_archetype" in m
        assert "tickers" in m
        assert "primary_ticker" in m
        assert "sharpe_target" in m
        assert "max_turns" in m
        assert "period" in m

    def test_varied_tickers(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY", "AAPL", "NVDA", "TSLA"],
            count=6,
        )
        primary_tickers = [m["primary_ticker"] for m in mandates]
        # Should use different tickers across mandates
        assert len(set(primary_tickers)) >= 2

    def test_varied_sharpe_targets(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=4,
            sharpe_targets=[1.0, 1.5, 2.0, 2.5],
        )
        targets = [m["sharpe_target"] for m in mandates]
        assert len(set(targets)) >= 2

    def test_seed_strategy_set(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=1,
        )
        m = mandates[0]
        assert "seed_strategy" in m
        assert m["seed_strategy"] is not None

    def test_valid_json_roundtrip(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=3,
        )
        # Should be serializable to JSON
        json_str = json.dumps(mandates)
        parsed = json.loads(json_str)
        assert len(parsed) == 3

    def test_count_larger_than_catalog(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        # Only 3 strategies, request 10 mandates (should cycle/permute)
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY", "AAPL"],
            count=10,
        )
        assert len(mandates) == 10

    def test_varied_periods(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=6,
            periods=["1y", "2y", "3y"],
        )
        periods = [m["period"] for m in mandates]
        assert len(set(periods)) >= 2


# ---------------------------------------------------------------------------
# Tests — save_mandates
# ---------------------------------------------------------------------------

class TestSaveMandates:
    """Test saving mandates to JSON files."""

    def test_saves_to_directory(self, mock_strategies_dir, tmp_path):
        from crabquant.refinement.mandate_generator import generate_mandates, save_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=3,
        )
        output_dir = tmp_path / "mandates"
        save_mandates(mandates, output_dir)

        files = list(output_dir.glob("*.json"))
        assert len(files) == 3

        for f in files:
            data = json.loads(f.read_text())
            assert "name" in data

    def test_overwrite_existing(self, mock_strategies_dir, tmp_path):
        from crabquant.refinement.mandate_generator import generate_mandates, save_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=2,
        )
        output_dir = tmp_path / "mandates"
        save_mandates(mandates, output_dir)
        save_mandates(mandates, output_dir)

        files = list(output_dir.glob("*.json"))
        assert len(files) == 2
