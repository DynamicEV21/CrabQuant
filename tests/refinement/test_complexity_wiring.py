"""Tests for complexity.py wiring into the refinement pipeline.

Verifies that:
1. High-complexity strategies get warnings injected into the LLM context
2. complexity_penalty() correctly adjusts promotion thresholds
3. Integration with promotion flow works end-to-end (mocked backtests)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ── Test fixtures ──────────────────────────────────────────────────────────────

SIMPLE_CODE = """
import pandas as pd
def generate_signals(df, params):
    fast = params.get("fast", 12)
    slow = params.get("slow", 26)
    df["ema_fast"] = df["close"].ewm(span=fast).mean()
    df["ema_slow"] = df["close"].ewm(span=slow).mean()
    entries = df["ema_fast"] > df["ema_slow"]
    exits = df["ema_fast"] < df["ema_slow"]
    return entries, exits
"""

COMPLEX_CODE = """
import pandas as pd
import numpy as np
from functools import lru_cache

def helper_a(df, p):
    if p.get("mode") == "aggressive":
        return df["close"] * 1.5
    elif p.get("mode") == "conservative":
        return df["close"] * 0.5
    else:
        if df["close"].mean() > 100:
            for i in range(len(df)):
                if df["close"].iloc[i] > df["high"].iloc[i]:
                    if df["volume"].iloc[i] > df["volume"].mean():
                        pass
        return df["close"]

def helper_b(df, p):
    result = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["open"].iloc[i]:
            result.iloc[i] = 1.0
        else:
            result.iloc[i] = -1.0
    return result

def helper_c(df, p):
    return df["close"].pct_change()

def helper_d(df, p):
    return df["high"] - df["low"]

def helper_e(df, p):
    return df["volume"].rolling(20).mean()

def helper_f(df, p):
    return df["close"].rolling(10).std()

def generate_signals(df, params):
    a = helper_a(df, params)
    b = helper_b(df, params)
    c = helper_c(df, params)
    d = helper_d(df, params)
    e = helper_e(df, params)
    f = helper_f(df, params)
    combined = a + b + c + d + e + f
    entries = combined > combined.mean()
    exits = combined < combined.mean()
    return entries, exits
"""

MANY_PARAMS = {
    "fast": 12, "slow": 26, "signal": 9, "vol_period": 20,
    "trend_period": 50, "momentum": 10, "threshold": 0.5,
    "stop_loss": 0.02, "take_profit": 0.05, "risk_factor": 1.5,
}

DEEPLY_NESTED_CODE = """
def generate_signals(df, params):
    entries = pd.Series(False, index=df.index)
    for i in range(len(df)):
        if df["close"].iloc[i] > 50:
            if df["volume"].iloc[i] > 1000:
                if df["high"].iloc[i] > df["close"].iloc[i]:
                    if df["close"].iloc[i] > df["open"].iloc[i]:
                        if df["low"].iloc[i] < df["close"].iloc[i]:
                            entries.iloc[i] = True
    return entries, entries
"""

MANY_BRANCHES_CODE = """
def generate_signals(df, params):
    entries = pd.Series(False, index=df.index)
    if df["close"].iloc[0] > 100:
        if df["volume"].iloc[0] > 1000:
            entries.iloc[0] = True
        elif df["volume"].iloc[0] > 500:
            entries.iloc[0] = True
        else:
            entries.iloc[0] = False
    else:
        if df["volume"].iloc[0] > 2000:
            entries.iloc[0] = True
        elif df["volume"].iloc[0] > 1500:
            entries.iloc[0] = True
        else:
            entries.iloc[0] = False
    if params.get("trend"):
        entries = entries | (df["close"] > df["close"].shift(5))
    elif params.get("mean_rev"):
        entries = entries | (df["close"] < df["close"].shift(5))
    if params.get("vol_filter"):
        entries = entries & (df["volume"] > df["volume"].mean())
    if params.get("trend2"):
        entries = entries & (df["close"] > df["close"].shift(10))
    if params.get("extra1"):
        entries = entries & True
    if params.get("extra2"):
        entries = entries & True
    return entries, entries
"""

INVALID_PYTHON = "def generate_signals(df, params):\n  return this is not valid python ))))\n"


# ── Test: complexity_score basics ─────────────────────────────────────────────


class TestComplexityScoreBasics:
    """Direct tests on complexity_score() function."""

    def test_simple_code_has_low_complexity(self):
        from crabquant.refinement.complexity import complexity_score
        result = complexity_score(SIMPLE_CODE, {"fast": 12, "slow": 26})
        assert result["complexity"] < 60
        assert result["complexity"] >= 0

    def test_complex_code_has_higher_complexity(self):
        from crabquant.refinement.complexity import complexity_score
        simple = complexity_score(SIMPLE_CODE, {})
        complex_ = complexity_score(COMPLEX_CODE, {})
        assert complex_["complexity"] > simple["complexity"]

    def test_many_params_triggers_flag(self):
        from crabquant.refinement.complexity import complexity_score
        result = complexity_score(SIMPLE_CODE, MANY_PARAMS)
        assert "too_many_params" in result["flags"]

    def test_deep_nesting_triggers_flag(self):
        from crabquant.refinement.complexity import complexity_score
        result = complexity_score(DEEPLY_NESTED_CODE, {})
        assert "deep_nesting" in result["flags"]

    def test_many_branches_triggers_flag(self):
        from crabquant.refinement.complexity import complexity_score
        result = complexity_score(MANY_BRANCHES_CODE, {})
        assert "too_many_branches" in result["flags"]

    def test_invalid_python_returns_max_penalty(self):
        from crabquant.refinement.complexity import complexity_score
        result = complexity_score(INVALID_PYTHON, {})
        assert result["complexity"] == 100.0
        assert "invalid_python" in result["flags"]

    def test_empty_code_returns_zero(self):
        from crabquant.refinement.complexity import complexity_score
        result = complexity_score("", {})
        assert result["complexity"] == 0.0
        assert result["flags"] == []

    def test_none_params_treated_as_zero(self):
        from crabquant.refinement.complexity import complexity_score
        result = complexity_score(SIMPLE_CODE, None)
        assert "too_many_params" not in result["flags"]

    def test_breakdown_has_all_keys(self):
        from crabquant.refinement.complexity import complexity_score
        result = complexity_score(SIMPLE_CODE, {"a": 1})
        bd = result["breakdown"]
        assert "n_params" in bd
        assert "n_nodes" in bd
        assert "n_functions" in bd
        assert "n_branches" in bd
        assert "max_nesting" in bd
        assert "n_features" in bd
        assert "total" in bd


# ── Test: complexity_penalty ──────────────────────────────────────────────────


class TestComplexityPenalty:
    """Tests for complexity_penalty() function."""

    def test_zero_complexity_no_penalty(self):
        from crabquant.refinement.complexity import complexity_penalty
        result = complexity_penalty(0.0, base_threshold=1.5)
        assert result == pytest.approx(1.5)

    def test_max_complexity_full_penalty(self):
        from crabquant.refinement.complexity import complexity_penalty
        result = complexity_penalty(100.0, base_threshold=1.5)
        # penalty factor = 1.0 * 0.5 = 0.5, threshold = 1.5 * 1.5 = 2.25
        assert result == pytest.approx(2.25)

    def test_medium_complexity_partial_penalty(self):
        from crabquant.refinement.complexity import complexity_penalty
        result = complexity_penalty(50.0, base_threshold=1.0)
        # penalty factor = 0.5 * 0.5 = 0.25, threshold = 1.0 * 1.25 = 1.25
        assert result == pytest.approx(1.25)

    def test_penalty_always_increases_threshold(self):
        from crabquant.refinement.complexity import complexity_penalty
        base = 1.5
        for score in range(0, 101, 10):
            adjusted = complexity_penalty(float(score), base_threshold=base)
            assert adjusted >= base

    def test_default_base_threshold(self):
        from crabquant.refinement.complexity import complexity_penalty
        result = complexity_penalty(0.0)
        assert result == pytest.approx(1.5)


# ── Test: Context builder integration ────────────────────────────────────────


class TestContextBuilderComplexity:
    """Tests that complexity warnings appear in build_llm_context() output."""

    def _make_mock_state(self, code="", params=None, history=None):
        state = MagicMock()
        state.current_turn = 3
        state.max_turns = 7
        state.sharpe_target = 1.5
        state.tickers = ["AAPL"]
        state.best_sharpe = 0.5
        state.best_composite_score = -999.0
        state.best_turn = 2
        state.code_quality_feedback = ""
        state.revert_notice = ""
        state.history = history or []
        return state

    def test_no_warning_for_simple_strategy(self):
        """Simple strategies should not get a complexity warning."""
        from crabquant.refinement.context_builder import build_llm_context

        history = [
            {"turn": 1, "sharpe": 0.3, "action": "novel", "code": SIMPLE_CODE,
             "params_used": {"fast": 12, "slow": 26}},
        ]
        state = self._make_mock_state(history=history)
        context = build_llm_context(state)

        prompt = context.get("prompt", "")
        assert "COMPLEXITY WARNING" not in prompt

    def test_warning_for_complex_strategy(self):
        """Complex strategies (score > 60 or with flags) get a warning."""
        from crabquant.refinement.context_builder import build_llm_context

        history = [
            {"turn": 1, "sharpe": 0.3, "action": "novel", "code": COMPLEX_CODE,
             "params_used": {}},
        ]
        state = self._make_mock_state(history=history)
        context = build_llm_context(state)

        prompt = context.get("prompt", "")
        assert "COMPLEXITY WARNING" in prompt

    def test_warning_includes_complexity_score(self):
        """Warning section includes the numeric complexity score."""
        from crabquant.refinement.context_builder import build_llm_context

        history = [
            {"turn": 1, "sharpe": 0.3, "action": "novel", "code": COMPLEX_CODE,
             "params_used": {}},
        ]
        state = self._make_mock_state(history=history)
        context = build_llm_context(state)

        prompt = context.get("prompt", "")
        assert "/100" in prompt

    def test_too_many_params_flag_listed(self):
        """When too_many_params flag is set, it appears in the warning."""
        from crabquant.refinement.context_builder import build_llm_context

        history = [
            {"turn": 1, "sharpe": 0.3, "action": "novel", "code": SIMPLE_CODE,
             "params_used": MANY_PARAMS},
        ]
        state = self._make_mock_state(history=history)
        context = build_llm_context(state)

        prompt = context.get("prompt", "")
        assert "COMPLEXITY WARNING" in prompt
        assert "too_many_params" in prompt

    def test_deep_nesting_flag_listed(self):
        """Deeply nested code triggers deep_nesting flag in warning."""
        from crabquant.refinement.context_builder import build_llm_context

        history = [
            {"turn": 1, "sharpe": 0.3, "action": "novel",
             "code": DEEPLY_NESTED_CODE, "params_used": {}},
        ]
        state = self._make_mock_state(history=history)
        context = build_llm_context(state)

        prompt = context.get("prompt", "")
        assert "COMPLEXITY WARNING" in prompt
        assert "deep_nesting" in prompt

    def test_invalid_python_flag_listed(self):
        """Invalid Python triggers invalid_python flag in warning."""
        from crabquant.refinement.context_builder import build_llm_context

        history = [
            {"turn": 1, "sharpe": 0.3, "action": "novel",
             "code": INVALID_PYTHON, "params_used": {}},
        ]
        state = self._make_mock_state(history=history)
        context = build_llm_context(state)

        prompt = context.get("prompt", "")
        assert "COMPLEXITY WARNING" in prompt
        assert "invalid_python" in prompt

    def test_no_warning_when_no_code_in_history(self):
        """No code available → no complexity warning (no crash)."""
        from crabquant.refinement.context_builder import build_llm_context

        state = self._make_mock_state(history=[])
        context = build_llm_context(state)

        prompt = context.get("prompt", "")
        assert "COMPLEXITY WARNING" not in prompt

    def test_no_warning_when_no_history(self):
        """Empty history → no complexity warning (no crash)."""
        from crabquant.refinement.context_builder import build_llm_context

        state = MagicMock()
        state.current_turn = 1
        state.max_turns = 7
        state.sharpe_target = 1.5
        state.tickers = ["SPY"]
        state.best_sharpe = 0.0
        state.best_composite_score = -999.0
        state.best_turn = 0
        state.code_quality_feedback = ""
        state.revert_notice = ""
        state.history = []

        context = build_llm_context(state)
        prompt = context.get("prompt", "")
        assert "COMPLEXITY WARNING" not in prompt


# ── Test: Promotion integration ───────────────────────────────────────────────


class TestPromotionComplexity:
    """Tests that complexity_penalty adjusts the walk-forward threshold."""

    def _mock_wf_result(self, avg_sharpe=1.5, robust=True):
        wf = MagicMock()
        wf.avg_test_sharpe = avg_sharpe
        wf.min_test_sharpe = avg_sharpe * 0.5
        wf.avg_degradation = 0.1
        wf.num_windows = 4
        wf.windows_passed = 4
        wf.robust = robust
        wf.notes = ""
        wf.window_results = []
        return wf

    def _mock_ct_result(self, avg_sharpe=1.0, robust=True):
        ct = MagicMock()
        ct.avg_sharpe = avg_sharpe
        ct.median_sharpe = avg_sharpe
        ct.robust = robust
        ct.tickers_profitable = 2
        ct.tickers_tested = 2
        ct.notes = ""
        return ct

    @patch("crabquant.validation.rolling_walk_forward")
    @patch("crabquant.validation.cross_ticker_validation")
    def test_complexity_score_in_result_when_code_provided(
        self, mock_ct, mock_wf
    ):
        """When strategy_code is provided, result includes complexity_score."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf.return_value = self._mock_wf_result()
        mock_ct.return_value = self._mock_ct_result()

        result = run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["SPY", "AAPL"],
            strategy_code=SIMPLE_CODE,
        )

        assert result["complexity_score"] is not None
        assert "complexity" in result["complexity_score"]
        assert "flags" in result["complexity_score"]
        assert "breakdown" in result["complexity_score"]

    @patch("crabquant.validation.rolling_walk_forward")
    @patch("crabquant.validation.cross_ticker_validation")
    def test_no_complexity_score_when_code_not_provided(
        self, mock_ct, mock_wf
    ):
        """When strategy_code is None, complexity_score is None."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf.return_value = self._mock_wf_result()
        mock_ct.return_value = self._mock_ct_result()

        result = run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["SPY"],
        )

        assert result["complexity_score"] is None

    @patch("crabquant.validation.rolling_walk_forward")
    @patch("crabquant.validation.cross_ticker_validation")
    def test_complex_strategy_faces_higher_threshold(
        self, mock_ct, mock_wf
    ):
        """Complex strategy code raises the walk-forward Sharpe bar."""
        from crabquant.refinement.promotion import run_full_validation_check
        from crabquant.refinement.complexity import complexity_score, complexity_penalty

        mock_wf.return_value = self._mock_wf_result(avg_sharpe=0.4)
        mock_ct.return_value = self._mock_ct_result()

        # Compute what the adjusted threshold should be
        cx = complexity_score(COMPLEX_CODE, {})
        base_threshold = 0.3  # default from VALIDATION_CONFIG
        adjusted = complexity_penalty(cx["complexity"], base_threshold=base_threshold)

        result = run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["SPY"],
            strategy_code=COMPLEX_CODE,
        )

        # The adjusted threshold is higher than base
        assert adjusted > base_threshold
        # Result should show complexity_score
        assert result["complexity_score"]["complexity"] == cx["complexity"]

    @patch("crabquant.validation.rolling_walk_forward")
    @patch("crabquant.validation.cross_ticker_validation")
    def test_simple_strategy_uses_near_base_threshold(
        self, mock_ct, mock_wf
    ):
        """Simple strategy code barely adjusts the threshold."""
        from crabquant.refinement.promotion import run_full_validation_check
        from crabquant.refinement.complexity import complexity_score, complexity_penalty

        mock_wf.return_value = self._mock_wf_result(avg_sharpe=1.5)
        mock_ct.return_value = self._mock_ct_result()

        cx = complexity_score(SIMPLE_CODE, {"fast": 12, "slow": 26})
        base_threshold = 0.3
        adjusted = complexity_penalty(cx["complexity"], base_threshold=base_threshold)

        # Simple code should have a low complexity → minimal penalty
        assert cx["complexity"] < 40
        assert adjusted < base_threshold * 1.1  # Less than 10% increase

    @patch("crabquant.validation.rolling_walk_forward")
    @patch("crabquant.validation.cross_ticker_validation")
    def test_complexity_does_not_affect_cross_ticker_threshold(
        self, mock_ct, mock_wf
    ):
        """Complexity penalty only adjusts walk-forward, not cross-ticker."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf.return_value = self._mock_wf_result()
        mock_ct.return_value = self._mock_ct_result(avg_sharpe=0.1, robust=False)

        result = run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["SPY", "AAPL"],
            strategy_code=COMPLEX_CODE,
            min_cross_ticker_sharpe=0.3,
        )

        # Cross-ticker should fail based on its own threshold, not complexity
        assert result["cross_ticker_robust"] is False

    @patch("crabquant.validation.rolling_walk_forward")
    @patch("crabquant.validation.cross_ticker_validation")
    def test_graceful_on_complexity_error(
        self, mock_ct, mock_wf
    ):
        """If complexity scoring fails, validation still runs normally."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf.return_value = self._mock_wf_result()
        mock_ct.return_value = self._mock_ct_result()

        # Patch complexity_score to raise an error
        with patch(
            "crabquant.refinement.complexity.complexity_score",
            side_effect=RuntimeError("AST parse error")
        ):
            result = run_full_validation_check(
                strategy_fn=MagicMock(),
                params={},
                discovery_ticker="SPY",
                validation_tickers=["SPY"],
                strategy_code=SIMPLE_CODE,
            )

        # Should still pass — complexity failure is non-fatal
        assert result["walk_forward_robust"] is True
        assert result["complexity_score"] is None

    @patch("crabquant.validation.rolling_walk_forward")
    @patch("crabquant.validation.cross_ticker_validation")
    def test_backward_compatible_without_strategy_code(
        self, mock_ct, mock_wf
    ):
        """Existing callers without strategy_code still work identically."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf.return_value = self._mock_wf_result()
        mock_ct.return_value = self._mock_ct_result()

        result = run_full_validation_check(
            strategy_fn=MagicMock(),
            params={"fast": 12, "slow": 26},
            discovery_ticker="SPY",
            validation_tickers=["SPY", "AAPL"],
            min_walk_forward_sharpe=0.5,
        )

        assert result["passed"] is True
        assert result["complexity_score"] is None

    @patch("crabquant.validation.rolling_walk_forward")
    @patch("crabquant.validation.cross_ticker_validation")
    def test_regime_specific_combined_with_complexity(
        self, mock_ct, mock_wf
    ):
        """Regime-specific relaxation and complexity penalty both apply."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf.return_value = self._mock_wf_result(avg_sharpe=0.3)
        mock_ct.return_value = self._mock_ct_result()

        result = run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["SPY"],
            is_regime_specific=True,
            strategy_code=COMPLEX_CODE,
        )

        # Both mechanisms should have applied — complexity_score should be present
        assert result["complexity_score"] is not None
        assert result["is_regime_specific"] is True
