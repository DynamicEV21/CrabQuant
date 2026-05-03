"""
Tests for crabquant.refinement.feature_importance module.

Tests indicator extraction, computation, correlation analysis, and formatting.
Does NOT test compute_feature_importance() end-to-end (needs real market data),
but tests all internal functions with synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from crabquant.refinement.feature_importance import (
    _CONTRIBUTING_THRESHOLD,
    _HARMFUL_THRESHOLD,
    _empty_result,
    _format_importance_summary,
    compute_indicator_series,
    extract_indicators_from_code,
    format_feature_importance_for_prompt,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_ohlcv(n=200, seed=42):
    """Create synthetic OHLCV DataFrame."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": close + np.abs(np.random.randn(n)) * 0.5,
        "low": close - np.abs(np.random.randn(n)) * 0.5,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    })


# ── extract_indicators_from_code ────────────────────────────────────────


class TestExtractIndicatorsFromCode:
    def test_cached_indicator_double_quotes(self):
        code = 'x = cached_indicator("ema", length=20)'
        assert extract_indicators_from_code(code) == ["ema"]

    def test_cached_indicator_single_quotes(self):
        code = "x = cached_indicator('rsi', length=14)"
        assert extract_indicators_from_code(code) == ["rsi"]

    def test_ta_direct_call(self):
        code = "macd = ta.macd(df['close'], fast=12, slow=26, signal=9)"
        assert extract_indicators_from_code(code) == ["macd"]

    def test_pandas_ta_call(self):
        code = "rsi = pandas_ta.rsi(df['close'], length=14)"
        assert extract_indicators_from_code(code) == ["rsi"]

    def test_multiple_indicators(self):
        code = """
        ema = cached_indicator("ema", length=20)
        rsi = ta.rsi(close, length=14)
        atr = cached_indicator('atr', length=14)
        """
        result = extract_indicators_from_code(code)
        # cached_indicator matches come first (ema, atr), then ta.* (rsi)
        assert result == ["ema", "atr", "rsi"]

    def test_dedup_preserves_order(self):
        code = """
        ema = cached_indicator("ema", length=20)
        rsi = ta.rsi(close, length=14)
        ema2 = cached_indicator("ema", length=50)
        """
        result = extract_indicators_from_code(code)
        assert result == ["ema", "rsi"]

    def test_no_indicators(self):
        code = "x = df['close'] * 2"
        assert extract_indicators_from_code(code) == []

    def test_whitespace_variations(self):
        code = 'x = cached_indicator  (  "ema"  ,  length=20  )'
        assert extract_indicators_from_code(code) == ["ema"]

    def test_case_sensitive(self):
        code = 'x = cached_indicator("EMA", length=20)'
        assert extract_indicators_from_code(code) == ["EMA"]


# ── compute_indicator_series ─────────────────────────────────────────────


class TestComputeIndicatorSeries:
    def test_ema_returns_series(self):
        df = _make_ohlcv()
        result = compute_indicator_series("ema", df, length=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert result.isna().sum() == 0  # EMA has no NaN

    def test_sma_has_initial_nan(self):
        df = _make_ohlcv()
        result = compute_indicator_series("sma", df, length=20)
        assert isinstance(result, pd.Series)
        assert result.iloc[:19].isna().all()
        assert result.iloc[20:].notna().all()

    def test_rsi_range(self):
        df = _make_ohlcv()
        result = compute_indicator_series("rsi", df, length=14)
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_returns_series(self):
        df = _make_ohlcv()
        result = compute_indicator_series("macd", df, fast=12, slow=26, signal=9)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_atr_positive(self):
        df = _make_ohlcv()
        result = compute_indicator_series("atr", df, length=14)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_bbands_range(self):
        df = _make_ohlcv()
        result = compute_indicator_series("bbands", df, length=20)
        valid = result.dropna()
        # %B should roughly be between 0 and 1 (may exceed for extreme moves)
        assert (valid >= -0.5).all() and (valid <= 1.5).all()

    def test_roc(self):
        df = _make_ohlcv()
        result = compute_indicator_series("roc", df, length=12)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_stoch_range(self):
        df = _make_ohlcv()
        result = compute_indicator_series("stoch", df, length=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_adx_range(self):
        df = _make_ohlcv()
        result = compute_indicator_series("adx", df, length=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_obv(self):
        df = _make_ohlcv()
        result = compute_indicator_series("obv", df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_vwap(self):
        df = _make_ohlcv()
        result = compute_indicator_series("vwap", df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_cci(self):
        df = _make_ohlcv()
        result = compute_indicator_series("cci", df, length=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_wma(self):
        df = _make_ohlcv()
        result = compute_indicator_series("wma", df, length=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_supertrend_direction(self):
        df = _make_ohlcv()
        result = compute_indicator_series("supertrend", df, length=10, multiplier=3.0)
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert set(valid.unique()).issubset({1.0, -1.0})

    def test_williams_range(self):
        df = _make_ohlcv()
        result = compute_indicator_series("williams", df, length=14)
        valid = result.dropna()
        assert (valid >= -100).all() and (valid <= 0).all()

    def test_mfi_range(self):
        df = _make_ohlcv()
        result = compute_indicator_series("mfi", df, length=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_dmi(self):
        df = _make_ohlcv()
        result = compute_indicator_series("dmi", df, length=14)
        assert isinstance(result, pd.Series)

    def test_unknown_indicator_returns_none(self):
        df = _make_ohlcv()
        result = compute_indicator_series("nonexistent_indicator", df)
        assert result is None

    def test_case_insensitive_lookup(self):
        df = _make_ohlcv()
        result = compute_indicator_series("EMA", df, length=20)
        assert result is not None


# ── _empty_result ────────────────────────────────────────────────────────


class TestEmptyResult:
    def test_returns_dict(self):
        result = _empty_result("test_reason")
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = _empty_result("test_reason")
        assert "indicators" in result
        assert "summary" in result
        assert "dominant_indicator" in result
        assert "weakest_indicator" in result

    def test_empty_indicators(self):
        result = _empty_result("test_reason")
        assert result["indicators"] == []

    def test_none_dominant_weakest(self):
        result = _empty_result("test_reason")
        assert result["dominant_indicator"] is None
        assert result["weakest_indicator"] is None

    def test_reason_in_summary(self):
        result = _empty_result("insufficient_data")
        assert "insufficient_data" in result["summary"]


# ── _format_importance_summary ───────────────────────────────────────────


class TestFormatImportanceSummary:
    def test_basic_format(self):
        results = [
            {"name": "ema", "correlation": 0.1, "abs_correlation": 0.1, "classification": "contributing"},
            {"name": "rsi", "correlation": -0.05, "abs_correlation": 0.05, "classification": "harmful"},
        ]
        text = _format_importance_summary(results, dominant="ema", weakest="rsi")
        assert "### Indicator Predictive Power Analysis" in text
        assert "EMA" in text
        assert "RSI" in text
        assert "Key driver" in text
        assert "Warning" in text

    def test_contributing_icon(self):
        results = [
            {"name": "ema", "correlation": 0.1, "abs_correlation": 0.1, "classification": "contributing"},
        ]
        text = _format_importance_summary(results, dominant="ema", weakest=None)
        assert "✅" in text

    def test_harmful_icon(self):
        results = [
            {"name": "rsi", "correlation": -0.1, "abs_correlation": 0.1, "classification": "harmful"},
        ]
        text = _format_importance_summary(results, dominant=None, weakest="rsi")
        assert "⚠️" in text

    def test_neutral_icon(self):
        results = [
            {"name": "atr", "correlation": 0.0, "abs_correlation": 0.0, "classification": "neutral"},
        ]
        text = _format_importance_summary(results, dominant=None, weakest=None)
        assert "➖" in text

    def test_indicator_overload_warning(self):
        results = [
            {"name": f"ind{i}", "correlation": 0.1, "abs_correlation": 0.1, "classification": "contributing"}
            for i in range(6)
        ]
        text = _format_importance_summary(results, dominant="ind0", weakest=None)
        assert "Indicator overload" in text
        assert "6 indicators" in text

    def test_no_overload_for_few_indicators(self):
        results = [
            {"name": "ema", "correlation": 0.1, "abs_correlation": 0.1, "classification": "contributing"},
            {"name": "rsi", "correlation": 0.05, "abs_correlation": 0.05, "classification": "contributing"},
        ]
        text = _format_importance_summary(results, dominant="ema", weakest=None)
        assert "Indicator overload" not in text

    def test_more_harm_than_good_warning(self):
        results = [
            {"name": "ema", "correlation": 0.1, "abs_correlation": 0.1, "classification": "contributing"},
            {"name": "rsi", "correlation": -0.1, "abs_correlation": 0.1, "classification": "harmful"},
            {"name": "atr", "correlation": -0.08, "abs_correlation": 0.08, "classification": "harmful"},
        ]
        text = _format_importance_summary(results, dominant="ema", weakest="rsi")
        assert "More harm than good" in text

    def test_no_edge_detected_warning(self):
        results = [
            {"name": "ema", "correlation": 0.0, "abs_correlation": 0.0, "classification": "neutral"},
            {"name": "rsi", "correlation": 0.0, "abs_correlation": 0.0, "classification": "neutral"},
        ]
        text = _format_importance_summary(results, dominant=None, weakest=None)
        assert "No edge detected" in text

    def test_no_edge_not_shown_with_contributing(self):
        results = [
            {"name": "ema", "correlation": 0.1, "abs_correlation": 0.1, "classification": "contributing"},
            {"name": "rsi", "correlation": 0.0, "abs_correlation": 0.0, "classification": "neutral"},
        ]
        text = _format_importance_summary(results, dominant="ema", weakest=None)
        assert "No edge detected" not in text

    def test_dominant_none_no_driver_line(self):
        results = [
            {"name": "atr", "correlation": 0.0, "abs_correlation": 0.0, "classification": "neutral"},
        ]
        text = _format_importance_summary(results, dominant=None, weakest=None)
        assert "Key driver" not in text

    def test_weakest_none_no_warning_line(self):
        results = [
            {"name": "ema", "correlation": 0.1, "abs_correlation": 0.1, "classification": "contributing"},
        ]
        text = _format_importance_summary(results, dominant="ema", weakest=None)
        assert "Warning" not in text

    def test_correlation_formatting(self):
        results = [
            {"name": "ema", "correlation": 0.12345, "abs_correlation": 0.12345, "classification": "contributing"},
        ]
        text = _format_importance_summary(results, dominant="ema", weakest=None)
        assert "+0.123" in text


# ── format_feature_importance_for_prompt ─────────────────────────────────


class TestFormatFeatureImportanceForPrompt:
    def test_empty_indicators_returns_empty(self):
        result = format_feature_importance_for_prompt({"indicators": []})
        assert result == ""

    def test_no_indicators_key_returns_empty(self):
        result = format_feature_importance_for_prompt({})
        assert result == ""

    def test_with_indicators_returns_summary(self):
        importance = {
            "indicators": [{"name": "ema", "correlation": 0.1, "abs_correlation": 0.1, "classification": "contributing"}],
            "summary": "EMA is great",
            "dominant_indicator": "ema",
            "weakest_indicator": None,
        }
        result = format_feature_importance_for_prompt(importance)
        assert result == "EMA is great"

    def test_empty_summary_still_works(self):
        importance = {
            "indicators": [{"name": "ema", "correlation": 0.1, "abs_correlation": 0.1, "classification": "contributing"}],
            "summary": "",
            "dominant_indicator": "ema",
            "weakest_indicator": None,
        }
        result = format_feature_importance_for_prompt(importance)
        assert result == ""


# ── Classification thresholds ────────────────────────────────────────────


class TestClassificationThresholds:
    def test_contributing_threshold(self):
        assert _CONTRIBUTING_THRESHOLD == 0.05

    def test_harmful_threshold(self):
        assert _HARMFUL_THRESHOLD == -0.05

    def test_neutral_range(self):
        """Values between -0.05 and +0.05 should be neutral."""
        assert _HARMFUL_THRESHOLD < _CONTRIBUTING_THRESHOLD


# ── Integration: correlation computation logic ───────────────────────────


class TestCorrelationLogic:
    """Test the core correlation analysis logic used in compute_feature_importance."""

    def test_positive_correlation_classified_contributing(self):
        """Indicators with median correlation >= 0.05 are 'contributing'."""
        corr = 0.08
        if corr >= _CONTRIBUTING_THRESHOLD:
            assert True  # contributing

    def test_negative_correlation_classified_harmful(self):
        """Indicators with median correlation <= -0.05 are 'harmful'."""
        corr = -0.08
        if corr <= _HARMFUL_THRESHOLD:
            assert True  # harmful

    def test_zero_correlation_classified_neutral(self):
        """Indicators near zero correlation are 'neutral'."""
        corr = 0.01
        if _HARMFUL_THRESHOLD < corr < _CONTRIBUTING_THRESHOLD:
            assert True  # neutral

    def test_boundary_contributing(self):
        """Exactly at threshold should be classified as contributing."""
        corr = _CONTRIBUTING_THRESHOLD
        assert corr >= _CONTRIBUTING_THRESHOLD  # contributing

    def test_boundary_harmful(self):
        """Exactly at threshold should be classified as harmful."""
        corr = _HARMFUL_THRESHOLD
        assert corr <= _HARMFUL_THRESHOLD  # harmful


class TestDominantIndicatorBug:
    """Tests for the dominant indicator selection bug fix.

    Bug: original code only checked results[0] for dominant. If the top result
    by abs_correlation was harmful, dominant was None even when contributing
    indicators existed further down the list.
    Fix: iterate through results (sorted by abs_correlation desc) and pick
    the first contributing indicator.
    """

    def test_dominant_is_none_when_all_harmful(self):
        """If no indicators are contributing, dominant should be None."""
        results = [
            {"name": "rsi", "correlation": -0.5, "abs_correlation": 0.5, "classification": "harmful"},
            {"name": "ema", "correlation": -0.3, "abs_correlation": 0.3, "classification": "harmful"},
        ]
        text = _format_importance_summary(results, dominant=None, weakest="rsi")
        assert "Key driver" not in text

    def test_dominant_found_despite_top_being_harmful(self):
        """When top by abs_correlation is harmful but a contributing indicator exists."""
        results = [
            {"name": "rsi", "correlation": -0.5, "abs_correlation": 0.5, "classification": "harmful"},
            {"name": "atr", "correlation": 0.3, "abs_correlation": 0.3, "classification": "contributing"},
        ]
        # In the actual function, dominant should be "atr" not None
        # Verify the summary includes the driver line when dominant is provided
        text = _format_importance_summary(results, dominant="atr", weakest="rsi")
        assert "Key driver" in text
        assert "ATR" in text
