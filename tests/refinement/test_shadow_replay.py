"""Tests for crabquant.refinement.shadow_replay — Shadow Replay module.

Covers:
- Strategy still works → passes shadow replay (not degraded)
- Strategy no longer works → fails shadow replay (degraded)
- Empty strategy code → error handling
- Syntax errors in strategy code → RuntimeError
- Missing callable in strategy code → RuntimeError
- Empty winners list → empty results
- None / missing fields → graceful defaults
- Custom min_sharpe threshold
- Custom engine injection
"""
from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from crabquant.refinement.shadow_replay import _compile_strategy, shadow_replay


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_bt_result(sharpe: float = 1.0, **overrides):
    """Build a minimal BacktestResult-like object."""
    defaults = dict(
        ticker="AAPL",
        strategy_name="test",
        iteration=0,
        sharpe=sharpe,
        total_return=0.15,
        max_drawdown=0.10,
        win_rate=0.55,
        num_trades=20,
        avg_trade_return=0.01,
        calmar_ratio=1.0,
        sortino_ratio=1.2,
        profit_factor=1.5,
        avg_holding_bars=5.0,
        best_trade=0.05,
        worst_trade=-0.02,
        passed=True,
        score=0.8,
        notes="",
        params={},
    )
    defaults.update(overrides)

    # Use a simple namespace / dataclass substitute
    @dataclasses.dataclass
    class _BT:
        pass

    obj = _BT()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def _make_dummy_df(n: int = 50) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame for testing."""
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [1_000_000] * n,
    })


VALID_STRATEGY_CODE = """
def simulate(df, params):
    import pandas as pd
    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    entries.iloc[0] = True
    exits.iloc[5] = True
    return entries, exits
"""

VALID_STRATEGY_CODE_ALT = """
def generate_signals(df, params):
    import pandas as pd
    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    entries.iloc[1] = True
    exits.iloc[6] = True
    return entries, exits
"""

SYNTAX_ERROR_CODE = "def simulate(df, params\n    return None, None\n"

NO_CALLABLE_CODE = """
x = 42
y = "hello"
"""


# ── _compile_strategy Tests ──────────────────────────────────────────────────

class TestCompileStrategy:
    """Tests for the _compile_strategy helper."""

    def test_compiles_simulate_function(self):
        fn = _compile_strategy(VALID_STRATEGY_CODE, "test_sim")
        assert callable(fn)

    def test_compiles_generate_signals_function(self):
        fn = _compile_strategy(VALID_STRATEGY_CODE_ALT, "test_gen")
        assert callable(fn)

    def test_compiled_function_is_callable(self):
        fn = _compile_strategy(VALID_STRATEGY_CODE, "test")
        df = _make_dummy_df()
        entries, exits = fn(df, {})
        assert len(entries) == len(df)
        assert len(exits) == len(df)

    def test_syntax_error_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Syntax error"):
            _compile_strategy(SYNTAX_ERROR_CODE, "bad_syntax")

    def test_no_callable_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="No 'simulate' or 'generate_signals'"):
            _compile_strategy(NO_CALLABLE_CODE, "no_fn")

    def test_empty_code_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="No 'simulate' or 'generate_signals'"):
            _compile_strategy("", "empty")


# ── shadow_replay Tests ──────────────────────────────────────────────────────

class TestShadowReplay:
    """Tests for the shadow_replay() main function."""

    def test_empty_winners_returns_empty(self):
        """Empty winners list should return empty results."""
        engine = MagicMock()
        result = shadow_replay([], _make_dummy_df(), engine=engine)
        assert result == []
        engine.run.assert_not_called()

    def test_strategy_still_works_not_degraded(self):
        """Strategy with good new Sharpe should NOT be flagged as degraded."""
        engine = MagicMock()
        engine.run.return_value = _make_bt_result(sharpe=1.2)

        winners = [{
            "name": "momentum_alpha",
            "ticker": "AAPL",
            "old_sharpe": 1.5,
            "params": {"window": 14},
            "strategy_code": VALID_STRATEGY_CODE,
        }]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        assert len(results) == 1
        r = results[0]
        assert r["name"] == "momentum_alpha"
        assert r["ticker"] == "AAPL"
        assert r["old_sharpe"] == 1.5
        assert r["new_sharpe"] == 1.2
        # 1.2 >= 0.5 * 1.5 = 0.75 and 1.2 >= 0.3 min_sharpe
        assert r["degraded"] is False
        assert r["error"] == ""

    def test_strategy_degraded_low_sharpe(self):
        """Strategy whose new Sharpe drops below 50% of old should be degraded."""
        engine = MagicMock()
        engine.run.return_value = _make_bt_result(sharpe=0.2)

        winners = [{
            "name": "fading_alpha",
            "ticker": "MSFT",
            "old_sharpe": 2.0,
            "params": {},
            "strategy_code": VALID_STRATEGY_CODE,
        }]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        r = results[0]
        assert r["new_sharpe"] == 0.2
        # 0.2 < 2.0 * 0.5 = 1.0  → degraded
        assert r["degraded"] is True

    def test_strategy_degraded_below_min_sharpe(self):
        """Strategy above 50% ratio but below min_sharpe should be degraded."""
        engine = MagicMock()
        engine.run.return_value = _make_bt_result(sharpe=0.2)

        winners = [{
            "name": "barely_alive",
            "ticker": "TSLA",
            "old_sharpe": 0.5,
            "params": {},
            "strategy_code": VALID_STRATEGY_CODE,
        }]

        # default min_sharpe=0.3, new sharpe=0.2 < 0.3
        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        r = results[0]
        assert r["degraded"] is True

    def test_strategy_zero_old_sharpe_uses_min_gate(self):
        """When old_sharpe is 0, only min_sharpe gate applies."""
        engine = MagicMock()
        engine.run.return_value = _make_bt_result(sharpe=0.5)

        winners = [{
            "name": "no_history",
            "ticker": "GOOG",
            "old_sharpe": 0.0,
            "params": {},
            "strategy_code": VALID_STRATEGY_CODE,
        }]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        r = results[0]
        # 0.5 >= 0.3 min_sharpe → not degraded
        assert r["degraded"] is False

    def test_strategy_zero_old_sharpe_below_min(self):
        """When old_sharpe is 0 and new is below min_sharpe → degraded."""
        engine = MagicMock()
        engine.run.return_value = _make_bt_result(sharpe=0.1)

        winners = [{
            "name": "no_history_bad",
            "ticker": "GOOG",
            "old_sharpe": 0.0,
            "params": {},
            "strategy_code": VALID_STRATEGY_CODE,
        }]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        r = results[0]
        assert r["degraded"] is True

    def test_missing_strategy_code_sets_error(self):
        """Winner with no strategy_code should get an error entry."""
        engine = MagicMock()
        winners = [{
            "name": "no_code",
            "ticker": "NVDA",
            "old_sharpe": 1.0,
            "params": {},
            "strategy_code": "",  # empty
        }]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        r = results[0]
        assert r["error"] == "No strategy_code provided"
        assert r["degraded"] is True
        assert r["new_sharpe"] == 0.0
        engine.run.assert_not_called()

    def test_whitespace_only_strategy_code_sets_error(self):
        """Winner with only whitespace strategy_code should get error."""
        engine = MagicMock()
        winners = [{
            "name": "whitespace",
            "ticker": "META",
            "old_sharpe": 1.0,
            "params": {},
            "strategy_code": "   \n  \t ",
        }]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        r = results[0]
        assert r["error"] == "No strategy_code provided"

    def test_syntax_error_in_code_sets_error(self):
        """Strategy code with syntax error should set error and degraded=True."""
        engine = MagicMock()
        winners = [{
            "name": "bad_syntax",
            "ticker": "AMZN",
            "old_sharpe": 1.0,
            "params": {},
            "strategy_code": SYNTAX_ERROR_CODE,
        }]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        r = results[0]
        assert "Syntax error" in r["error"]
        assert r["degraded"] is True

    def test_no_callable_in_code_sets_error(self):
        """Strategy code without simulate/generate_signals should set error."""
        engine = MagicMock()
        winners = [{
            "name": "no_fn",
            "ticker": "AMD",
            "old_sharpe": 1.0,
            "params": {},
            "strategy_code": NO_CALLABLE_CODE,
        }]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        r = results[0]
        assert "No 'simulate' or 'generate_signals'" in r["error"]
        assert r["degraded"] is True

    def test_runtime_exception_in_strategy_sets_error(self):
        """Strategy that raises during execution should be caught."""
        error_code = """
def simulate(df, params):
    raise ValueError("boom")
"""

        engine = MagicMock()
        winners = [{
            "name": "crasher",
            "ticker": "INTC",
            "old_sharpe": 1.0,
            "params": {},
            "strategy_code": error_code,
        }]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        r = results[0]
        assert "boom" in r["error"]
        assert r["degraded"] is True

    def test_engine_exception_sets_error(self):
        """Exception from engine.run() should be caught gracefully."""
        engine = MagicMock()
        engine.run.side_effect = RuntimeError("engine failure")

        winners = [{
            "name": "engine_fail",
            "ticker": "NFLX",
            "old_sharpe": 1.0,
            "params": {},
            "strategy_code": VALID_STRATEGY_CODE,
        }]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        r = results[0]
        assert "engine failure" in r["error"]
        assert r["degraded"] is True

    def test_custom_min_sharpe(self):
        """Custom min_sharpe threshold should be respected."""
        engine = MagicMock()
        engine.run.return_value = _make_bt_result(sharpe=0.4)

        winners = [{
            "name": "custom_threshold",
            "ticker": "SPY",
            "old_sharpe": 2.0,
            "params": {},
            "strategy_code": VALID_STRATEGY_CODE,
        }]

        # min_sharpe=0.5, new=0.4 < 0.5 → degraded even though > 50% of old
        results = shadow_replay(
            winners, _make_dummy_df(), min_sharpe=0.5, engine=engine,
        )
        r = results[0]
        assert r["degraded"] is True

    def test_multiple_winners(self):
        """Multiple winners processed in order, mixed results."""
        engine = MagicMock()
        # First call: good sharpe, second call: bad sharpe
        engine.run.side_effect = [
            _make_bt_result(sharpe=1.2),
            _make_bt_result(sharpe=0.1),
        ]

        winners = [
            {
                "name": "winner1",
                "ticker": "AAPL",
                "old_sharpe": 1.5,
                "params": {},
                "strategy_code": VALID_STRATEGY_CODE,
            },
            {
                "name": "winner2",
                "ticker": "MSFT",
                "old_sharpe": 1.0,
                "params": {},
                "strategy_code": VALID_STRATEGY_CODE,
            },
        ]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        assert len(results) == 2
        assert results[0]["degraded"] is False
        assert results[1]["degraded"] is True

    def test_missing_fields_use_defaults(self):
        """Winner missing optional fields should use sensible defaults."""
        engine = MagicMock()
        engine.run.return_value = _make_bt_result(sharpe=0.5)

        winners = [{}]  # Completely empty dict

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        r = results[0]
        assert r["name"] == "unnamed"
        assert r["ticker"] == "UNKNOWN"
        assert r["old_sharpe"] == 0.0
        assert r["error"] == "No strategy_code provided"
        assert r["degraded"] is True

    def test_strategy_name_fallback(self):
        """When 'name' is missing, falls back to 'strategy_name'."""
        engine = MagicMock()
        engine.run.return_value = _make_bt_result(sharpe=0.5)

        winners = [{
            "strategy_name": "my_strat",
            "ticker": "AAPL",
            "old_sharpe": 0.5,
            "params": {},
            "strategy_code": VALID_STRATEGY_CODE,
        }]

        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        assert results[0]["name"] == "my_strat"

    def test_engine_default_creation_when_none(self):
        """When engine=None, a BacktestEngine should be created via lazy import."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = _make_bt_result(sharpe=1.0)

        with patch("crabquant.engine.BacktestEngine", return_value=mock_engine):
            winners = [{
                "name": "test",
                "ticker": "AAPL",
                "old_sharpe": 1.0,
                "params": {},
                "strategy_code": VALID_STRATEGY_CODE,
            }]

            results = shadow_replay(winners, _make_dummy_df(), engine=None)
            assert len(results) == 1
            assert results[0]["new_sharpe"] == 1.0

    def test_exactly_50_percent_threshold(self):
        """Strategy at exactly 50% of old_sharpe should NOT be degraded."""
        engine = MagicMock()
        engine.run.return_value = _make_bt_result(sharpe=0.75)

        winners = [{
            "name": "boundary",
            "ticker": "AAPL",
            "old_sharpe": 1.5,
            "params": {},
            "strategy_code": VALID_STRATEGY_CODE,
        }]

        # 0.75 == 1.5 * 0.5 → not degraded (strict <)
        # and 0.75 >= 0.3 min_sharpe
        results = shadow_replay(winners, _make_dummy_df(), engine=engine)
        assert results[0]["degraded"] is False
