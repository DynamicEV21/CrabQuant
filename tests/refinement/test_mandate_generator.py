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

MOCK_STRATEGY_NO_DESC = textwrap.dedent("""\
    \"\"\"Trend following with Ichimoku cloud.\"\"\"
    DEFAULT_PARAMS = {"tenkan": 9, "kijun": 26}
    PARAM_GRID = {}

    def generate_signals(df, params):
        return None, None

    def generate_signals_matrix(df, param_grid):
        return None, None
""")

MOCK_STRATEGY_SYNTAX_ERROR = textwrap.dedent("""\
    DESCRIPTION = "Broken strategy"
    def generate_signals(  -- this is invalid Python
""")

MOCK_STRATEGY_4 = textwrap.dedent("""\
    DESCRIPTION = "Trend following using moving average ribbon."
    DEFAULT_PARAMS = {"short": 10, "medium": 50, "long": 200}
    PARAM_GRID = {"short": [5, 10, 20]}

    def generate_signals(df, params):
        return None, None

    def generate_signals_matrix(df, param_grid):
        return None, None
""")

MOCK_STRATEGY_5 = textwrap.dedent("""\
    DESCRIPTION = "Volatility breakout using Bollinger Bands squeeze."
    DEFAULT_PARAMS = {"period": 20, "std": 2.0}
    PARAM_GRID = {"period": [15, 20, 25]}

    def generate_signals(df, params):
        return None, None

    def generate_signals_matrix(df, param_grid):
        return None, None
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


@pytest.fixture
def diverse_strategies_dir(tmp_path):
    """Create a directory with 5 strategies across different archetypes."""
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()
    (strategies_dir / "ema_momentum.py").write_text(MOCK_STRATEGY_1)
    (strategies_dir / "rsi_mean_reversion.py").write_text(MOCK_STRATEGY_2)
    (strategies_dir / "atr_breakout.py").write_text(MOCK_STRATEGY_3)
    (strategies_dir / "ichimoku_trend.py").write_text(MOCK_STRATEGY_NO_DESC)
    (strategies_dir / "ma_ribbon_trend.py").write_text(MOCK_STRATEGY_4)
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

    def test_scan_falls_back_to_docstring(self, diverse_strategies_dir):
        from crabquant.refinement.mandate_generator import scan_strategy_catalog
        catalog = scan_strategy_catalog(diverse_strategies_dir)
        ichi = next(s for s in catalog if s["name"] == "ichimoku_trend")
        assert "Ichimoku" in ichi["description"]

    def test_scan_skips_syntax_error_files(self, diverse_strategies_dir):
        (diverse_strategies_dir / "broken.py").write_text(MOCK_STRATEGY_SYNTAX_ERROR)
        from crabquant.refinement.mandate_generator import scan_strategy_catalog
        catalog = scan_strategy_catalog(diverse_strategies_dir)
        names = [s["name"] for s in catalog]
        assert "broken" not in names

    def test_scan_extracts_param_grid(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import scan_strategy_catalog
        catalog = scan_strategy_catalog(mock_strategies_dir)
        ema = next(s for s in catalog if s["name"] == "ema_momentum")
        assert "fast" in ema["param_grid"]
        assert 8 in ema["param_grid"]["fast"]

    def test_scan_nonexistent_dir(self):
        from crabquant.refinement.mandate_generator import scan_strategy_catalog
        catalog = scan_strategy_catalog("/nonexistent/dir/abc")
        assert catalog == []

    def test_scan_includes_file_path(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import scan_strategy_catalog
        catalog = scan_strategy_catalog(mock_strategies_dir)
        assert all("file_path" in s for s in catalog)


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

    def test_detect_empty_string(self):
        from crabquant.refinement.mandate_generator import detect_archetype
        assert detect_archetype("") == "other"

    def test_detect_case_insensitive(self):
        from crabquant.refinement.mandate_generator import detect_archetype
        assert detect_archetype("BOLLINGER SQUEEZE STRATEGY") == "mean_reversion"

    def test_detect_multiple_keywords_picks_best(self):
        from crabquant.refinement.mandate_generator import detect_archetype
        # "momentum" matches momentum, "ema" matches trend — but "momentum" keyword wins
        assert detect_archetype("momentum ema crossover") == "momentum"


# ---------------------------------------------------------------------------
# Tests — _load_winners_history
# ---------------------------------------------------------------------------

class TestLoadWinnersHistory:
    """Test loading winners history from JSON."""

    def test_none_returns_empty(self):
        from crabquant.refinement.mandate_generator import _load_winners_history
        assert _load_winners_history(None) == []

    def test_nonexistent_file_returns_empty(self):
        from crabquant.refinement.mandate_generator import _load_winners_history
        assert _load_winners_history("/nonexistent/winners.json") == []

    def test_valid_json_loaded(self, tmp_path):
        from crabquant.refinement.mandate_generator import _load_winners_history
        data = [{"strategy": "ema_momentum", "ticker": "SPY"}]
        f = tmp_path / "winners.json"
        f.write_text(json.dumps(data))
        result = _load_winners_history(str(f))
        assert len(result) == 1
        assert result[0]["strategy"] == "ema_momentum"

    def test_corrupt_json_returns_empty(self, tmp_path):
        from crabquant.refinement.mandate_generator import _load_winners_history
        f = tmp_path / "bad.json"
        f.write_text("NOT JSON{{{")
        result = _load_winners_history(str(f))
        assert result == []

    def test_non_list_json_returns_empty(self, tmp_path):
        from crabquant.refinement.mandate_generator import _load_winners_history
        f = tmp_path / "dict.json"
        f.write_text(json.dumps({"key": "value"}))
        result = _load_winners_history(str(f))
        assert result == []


# ---------------------------------------------------------------------------
# Tests — diversity_score
# ---------------------------------------------------------------------------

class TestDiversityScore:
    """Test diversity scoring for candidate mandates."""

    def test_no_overlap_zero_score(self):
        from crabquant.refinement.mandate_generator import diversity_score
        score = diversity_score(
            mandate={"seed_strategy": "new_strat", "primary_ticker": "NEW"},
            winners_history=[],
            registry_keys=set(),
        )
        assert score == 0.0

    def test_registry_penalty(self):
        from crabquant.refinement.mandate_generator import diversity_score
        score = diversity_score(
            mandate={"seed_strategy": "existing_strat", "primary_ticker": "SPY"},
            winners_history=[],
            registry_keys={"existing_strat"},
        )
        assert score == 20.0

    def test_ticker_overlap(self):
        from crabquant.refinement.mandate_generator import diversity_score
        winners = [
            {"strategy": "other", "ticker": "SPY"},
            {"strategy": "another", "ticker": "SPY"},
        ]
        score = diversity_score(
            mandate={"seed_strategy": "new", "primary_ticker": "SPY"},
            winners_history=winners,
            registry_keys=set(),
        )
        # ticker_count=2 => 2*2=4
        assert score == 4.0

    def test_combo_overlap(self):
        from crabquant.refinement.mandate_generator import diversity_score
        winners = [
            {"strategy": "ema_momentum", "ticker": "SPY"},
            {"strategy": "ema_momentum", "ticker": "SPY"},
        ]
        score = diversity_score(
            mandate={"seed_strategy": "ema_momentum", "primary_ticker": "SPY"},
            winners_history=winners,
            registry_keys=set(),
        )
        # ticker_count=2 => 4, combo_count=2 => 10, total=14
        assert score == 14.0

    def test_full_overlap(self):
        from crabquant.refinement.mandate_generator import diversity_score
        winners = [
            {"strategy": "ema", "ticker": "SPY"},
        ]
        score = diversity_score(
            mandate={"seed_strategy": "ema", "primary_ticker": "SPY"},
            winners_history=winners,
            registry_keys={"ema"},
        )
        # ticker=2, combo=5, registry=20 => 27
        assert score == 27.0


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

    def test_empty_catalog_returns_empty(self, tmp_path):
        from crabquant.refinement.mandate_generator import generate_mandates
        empty_dir = tmp_path / "nonexistent_strategies"
        mandates = generate_mandates(strategies_dir=str(empty_dir), count=5)
        assert mandates == []

    def test_custom_max_turns(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=2,
            max_turns=12,
        )
        for m in mandates:
            assert m["max_turns"] == 12

    def test_custom_constraints(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        constraints = {"max_parameters": 3, "min_trades": 20}
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=2,
            constraints=constraints,
        )
        for m in mandates:
            assert m["constraints"]["max_parameters"] == 3
            assert m["constraints"]["min_trades"] == 20

    def test_single_ticker_no_secondary(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=1,
        )
        # With only 1 ticker, secondary list is empty
        assert mandates[0]["primary_ticker"] == "SPY"
        assert mandates[0]["tickers"][0] == "SPY"

    def test_tickers_includes_primary(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY", "AAPL", "NVDA"],
            count=1,
        )
        m = mandates[0]
        assert m["primary_ticker"] in m["tickers"]

    def test_mandate_has_description(self, mock_strategies_dir):
        from crabquant.refinement.mandate_generator import generate_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=1,
        )
        assert "description" in mandates[0]
        assert len(mandates[0]["description"]) > 0


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

    def test_creates_output_directory(self, mock_strategies_dir, tmp_path):
        from crabquant.refinement.mandate_generator import generate_mandates, save_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=1,
        )
        nested_dir = tmp_path / "a" / "b" / "c"
        paths = save_mandates(mandates, nested_dir)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_empty_list_creates_no_files(self, tmp_path):
        from crabquant.refinement.mandate_generator import save_mandates
        output_dir = tmp_path / "empty_mandates"
        paths = save_mandates([], output_dir)
        assert paths == []

    def test_file_names_are_lowercase(self, mock_strategies_dir, tmp_path):
        from crabquant.refinement.mandate_generator import generate_mandates, save_mandates
        mandates = generate_mandates(
            strategies_dir=mock_strategies_dir,
            tickers=["SPY"],
            count=1,
        )
        output_dir = tmp_path / "mandates"
        paths = save_mandates(mandates, output_dir)
        assert all(p.name == p.name.lower() for p in paths)
