"""Tests for crabquant.refinement.explainer — Explainer Agent module.

Covers:
- explain_strategy() with mock LLM call (via _llm_call injection)
- explain_strategy() with mocked real import path
- explain_strategy() fallback when LLM fails
- EXPLAINER_SYSTEM_PROMPT is well-formed
- EXPLAINER_USER_TEMPLATE formatting
- build_explainer_prompt() output formatting
- _cap_words() truncation logic
- _fallback_explanation() output
- Edge cases: empty metrics, missing data, None values
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crabquant.refinement.explainer import (
    EXPLAINER_SYSTEM_PROMPT,
    EXPLAINER_USER_TEMPLATE,
    _cap_words,
    _fallback_explanation,
    build_explainer_prompt,
    explain_strategy,
)


# ── Fixtures / Helpers ───────────────────────────────────────────────────────

MOCK_STRATEGY_CODE = """
def simulate(df, params):
    entries = df['close'].rolling(20).mean() > df['close'].rolling(50).mean()
    exits = ~entries
    return entries, exits
"""

FULL_METRICS = {
    "sharpe": 1.85,
    "total_return": 0.34,
    "max_drawdown": 0.12,
    "num_trades": 42,
    "win_rate": 0.62,
    "calmar_ratio": 2.1,
    "sortino_ratio": 2.4,
    "profit_factor": 1.8,
}

MINIMAL_METRICS = {
    "sharpe": 0.5,
}

MOCK_LLM_RESPONSE = (
    "This momentum strategy exploits the tendency of trending assets to continue "
    "in the same direction. The market inefficiency is the under-reaction to new "
    "information. Failure conditions include range-bound markets and sudden "
    "reversals. Key risks are overfitting to the backtest period and slippage."
)


# ── Constant / Prompt Tests ─────────────────────────────────────────────────

class TestConstants:
    """Test that prompt constants are well-formed."""

    def test_system_prompt_is_nonempty(self):
        assert isinstance(EXPLAINER_SYSTEM_PROMPT, str)
        assert len(EXPLAINER_SYSTEM_PROMPT) > 20
        assert "quantitative" in EXPLAINER_SYSTEM_PROMPT.lower()
        assert "explain" in EXPLAINER_SYSTEM_PROMPT.lower()

    def test_user_template_has_required_placeholders(self):
        """Template must contain all the metric placeholders."""
        required = (
            "{strategy_code}", "{ticker}", "{sharpe}", "{total_return}",
            "{max_drawdown}", "{num_trades}", "{win_rate}",
            "{calmar_ratio}", "{sortino_ratio}", "{profit_factor}",
        )
        for placeholder in required:
            assert placeholder in EXPLAINER_USER_TEMPLATE, (
                f"Missing placeholder {placeholder} in template"
            )

    def test_user_template_mentions_three_points(self):
        """Template should instruct the LLM on the 3 analysis points."""
        lower = EXPLAINER_USER_TEMPLATE.lower()
        assert "market inefficiency" in lower
        assert "failure conditions" in lower
        assert "key risk" in lower


# ── build_explainer_prompt Tests ─────────────────────────────────────────────

class TestBuildExplainerPrompt:
    """Tests for the prompt-building helper."""

    def test_formats_all_metrics(self):
        prompt = build_explainer_prompt(MOCK_STRATEGY_CODE, FULL_METRICS, "AAPL")
        assert "AAPL" in prompt
        assert "1.85" in prompt
        assert "0.34" in prompt
        assert "0.12" in prompt
        assert "42" in prompt
        assert "0.62" in prompt
        assert "2.1" in prompt
        assert "2.4" in prompt
        assert "1.8" in prompt
        assert "rolling" in prompt  # strategy code present

    def test_missing_metrics_use_na(self):
        """Metrics not provided should show as 'N/A'."""
        prompt = build_explainer_prompt(MOCK_STRATEGY_CODE, {}, "AAPL")
        # Count N/A occurrences — should be 8 metrics all showing N/A
        assert prompt.count("N/A") == 8

    def test_partial_metrics(self):
        """Some metrics provided, others should be N/A."""
        metrics = {"sharpe": 1.0, "total_return": 0.2}
        prompt = build_explainer_prompt(MOCK_STRATEGY_CODE, metrics, "MSFT")
        assert "1.0" in prompt
        assert "0.2" in prompt
        assert prompt.count("N/A") == 6  # 6 missing metrics


# ── _cap_words Tests ─────────────────────────────────────────────────────────

class TestCapWords:
    """Tests for the word-capping helper."""

    def test_short_text_unchanged(self):
        text = "Short explanation here."
        assert _cap_words(text, max_words=250) == text

    def test_exact_word_count_passes(self):
        words = ["word"] * 250
        text = " ".join(words)
        assert _cap_words(text, max_words=250) == text

    def test_over_word_count_truncated(self):
        words = ["word"] * 300
        text = " ".join(words)
        result = _cap_words(text, max_words=250)
        assert len(result.split()) <= 250

    def test_truncation_breaks_at_sentence(self):
        """Truncation should prefer sentence boundaries."""
        # 260 words, with a period at word 248
        words = ["word"] * 300
        words[247] = "sentence."  # word 248 (index 247)
        text = " ".join(words)
        result = _cap_words(text, max_words=250)
        # Should break at the period
        assert result.endswith("sentence.")

    def test_truncation_no_sentence_boundary(self):
        """If no sentence boundary, truncates at word limit."""
        words = ["word"] * 300
        text = " ".join(words)
        result = _cap_words(text, max_words=250)
        assert len(result.split()) == 250

    def test_exclamation_break(self):
        words = ["word"] * 300
        words[200] = "amazing!"
        text = " ".join(words)
        result = _cap_words(text, max_words=250)
        assert result.endswith("amazing!")

    def test_question_break(self):
        words = ["word"] * 300
        words[230] = "why?"
        text = " ".join(words)
        result = _cap_words(text, max_words=250)
        assert result.endswith("why?")

    def test_strips_whitespace(self):
        text = "  hello world  "
        assert _cap_words(text, max_words=250) == "hello world"


# ── _fallback_explanation Tests ──────────────────────────────────────────────

class TestFallbackExplanation:
    """Tests for the fallback explanation generator."""

    def test_basic_fallback(self):
        result = _fallback_explanation(FULL_METRICS, "AAPL")
        assert "AAPL" in result
        assert "1.85" in result
        assert "42" in result
        assert "LLM call failed" in result

    def test_missing_metrics_fallback(self):
        result = _fallback_explanation({}, "UNKNOWN")
        assert "UNKNOWN" in result
        assert "N/A" in result

    def test_partial_metrics_fallback(self):
        metrics = {"sharpe": 0.9}
        result = _fallback_explanation(metrics, "TSLA")
        assert "0.9" in result
        assert "N/A" in result  # num_trades missing


# ── explain_strategy Tests ───────────────────────────────────────────────────

class TestExplainStrategy:
    """Tests for the main explain_strategy function."""

    def test_successful_llm_call_via_injection(self):
        """Use _llm_call parameter to inject a mock LLM response."""
        mock_llm = MagicMock(return_value=MOCK_LLM_RESPONSE)
        result = explain_strategy(
            MOCK_STRATEGY_CODE,
            FULL_METRICS,
            "AAPL",
            _llm_call=mock_llm,
        )
        assert "momentum" in result.lower()
        mock_llm.assert_called_once()

    def test_llm_receives_correct_messages(self):
        """Verify the message structure passed to the LLM call."""
        captured_messages = []

        def capture_llm(messages):
            captured_messages.extend(messages)
            return MOCK_LLM_RESPONSE

        explain_strategy(
            MOCK_STRATEGY_CODE,
            FULL_METRICS,
            "MSFT",
            _llm_call=capture_llm,
        )

        assert len(captured_messages) == 2
        assert captured_messages[0]["role"] == "system"
        assert captured_messages[0]["content"] == EXPLAINER_SYSTEM_PROMPT
        assert captured_messages[1]["role"] == "user"
        assert "MSFT" in captured_messages[1]["content"]
        assert "rolling" in captured_messages[1]["content"]

    def test_llm_call_failure_returns_fallback(self):
        """When _llm_call raises, should return fallback explanation."""
        mock_llm = MagicMock(side_effect=RuntimeError("API timeout"))
        result = explain_strategy(
            MOCK_STRATEGY_CODE,
            FULL_METRICS,
            "AAPL",
            _llm_call=mock_llm,
        )
        assert "AAPL" in result
        assert "LLM call failed" in result

    def test_llm_return_value_capped(self):
        """Long LLM response should be capped at ~250 words."""
        long_response = " ".join(["word"] * 500)
        mock_llm = MagicMock(return_value=long_response)
        result = explain_strategy(
            MOCK_STRATEGY_CODE,
            FULL_METRICS,
            "AAPL",
            _llm_call=mock_llm,
        )
        assert len(result.split()) <= 250

    def test_empty_strategy_code(self):
        """Empty strategy code should still work (prompt will just be empty)."""
        mock_llm = MagicMock(return_value="Looks empty.")
        result = explain_strategy("", FULL_METRICS, "AAPL", _llm_call=mock_llm)
        assert result == "Looks empty."

    def test_empty_metrics(self):
        """Empty metrics dict should use N/A for all fields."""
        mock_llm = MagicMock(return_value="Analysis with N/A metrics.")
        result = explain_strategy(
            MOCK_STRATEGY_CODE, {}, "AAPL", _llm_call=mock_llm,
        )
        # Should not raise
        assert isinstance(result, str)
        # Verify the messages contain N/A
        call_args = mock_llm.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert user_msg.count("N/A") == 8

    def test_partial_metrics_fill_na(self):
        """Partial metrics should fill missing ones with N/A."""
        mock_llm = MagicMock(return_value="Partial analysis.")
        metrics = {"sharpe": 1.5}
        result = explain_strategy(
            MOCK_STRATEGY_CODE, metrics, "GOOG", _llm_call=mock_llm,
        )
        call_args = mock_llm.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "1.5" in user_msg
        assert user_msg.count("N/A") == 7  # 7 missing

    def test_real_llm_path_mocked(self):
        """Test the real import path (when _llm_call is None) with mocked module."""
        mock_llm_module = MagicMock()
        mock_llm_module.call_zai_llm.return_value = MOCK_LLM_RESPONSE

        with patch.dict("sys.modules", {"crabquant.refinement.llm_api": mock_llm_module}):
            result = explain_strategy(
                MOCK_STRATEGY_CODE,
                FULL_METRICS,
                "AAPL",
                _llm_call=None,
            )
            assert isinstance(result, str)
            assert len(result) > 0
            mock_llm_module.call_zai_llm.assert_called_once()

    def test_real_llm_path_failure_fallback(self):
        """When real LLM path raises, should return fallback."""
        mock_llm_module = MagicMock()
        mock_llm_module.call_zai_llm.side_effect = ConnectionError("No network")

        with patch.dict("sys.modules", {"crabquant.refinement.llm_api": mock_llm_module}):
            result = explain_strategy(
                MOCK_STRATEGY_CODE,
                FULL_METRICS,
                "AAPL",
                _llm_call=None,
            )
            assert "LLM call failed" in result
            assert "AAPL" in result

    def test_llm_call_receives_correct_params(self):
        """Verify call_zai_llm is called with correct kwargs."""
        mock_llm_module = MagicMock()
        mock_llm_module.call_zai_llm.return_value = "Short explanation."

        with patch.dict("sys.modules", {"crabquant.refinement.llm_api": mock_llm_module}):
            explain_strategy(
                MOCK_STRATEGY_CODE,
                FULL_METRICS,
                "NVDA",
                _llm_call=None,
            )

        call_kwargs = mock_llm_module.call_zai_llm.call_args
        assert call_kwargs.kwargs.get("model") == "glm-5-turbo"
        assert call_kwargs.kwargs.get("max_tokens") == 512
        assert call_kwargs.kwargs.get("temperature") == 0.4
        messages = call_kwargs.kwargs.get("messages")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"

    def test_none_llm_call_with_injection(self):
        """Passing _llm_call=None explicitly should use the import path."""
        mock_llm_module = MagicMock()
        mock_llm_module.call_zai_llm.return_value = "Result via import."

        with patch.dict("sys.modules", {"crabquant.refinement.llm_api": mock_llm_module}):
            result = explain_strategy(
                MOCK_STRATEGY_CODE,
                FULL_METRICS,
                "META",
                _llm_call=None,
            )
            assert result == "Result via import."
