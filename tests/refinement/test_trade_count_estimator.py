"""
Tests for the trade count estimator module.

Covers:
- Period parsing (various formats)
- Trade count estimation (all timeframe/strategy combinations)
- Recommended minimum always >= 20
- Formatting output
- Edge cases (unknown timeframe, unknown strategy_type)
- build_trade_count_guidance convenience function
"""

import math
import pytest

from crabquant.refinement.trade_count_estimator import (
    BARS_PER_YEAR,
    BARS_PER_TRADE,
    DEFAULT_BARS_PER_YEAR,
    MIN_TRADES_THRESHOLD,
    TIMEFRAME_LABELS,
    _normalize_timeframe,
    _parse_period_to_years,
    build_trade_count_guidance,
    estimate_trade_count,
    format_trade_count_guidance,
)


# ── Period Parsing Tests ─────────────────────────────────────────────────────

class TestParsePeriodToYears:
    """Test the _parse_period_to_years helper."""

    def test_years_suffix(self):
        assert _parse_period_to_years("2y") == 2.0
        assert _parse_period_to_years("5y") == 5.0
        assert _parse_period_to_years("1y") == 1.0

    def test_months_suffix(self):
        assert _parse_period_to_years("6mo") == pytest.approx(0.5)
        assert _parse_period_to_years("12mo") == pytest.approx(1.0)
        assert _parse_period_to_years("24mo") == pytest.approx(2.0)
        assert _parse_period_to_years("3mo") == pytest.approx(0.25)

    def test_days_suffix(self):
        assert _parse_period_to_years("365d") == pytest.approx(1.0)
        assert _parse_period_to_years("730d") == pytest.approx(2.0)
        assert _parse_period_to_years("90d") == pytest.approx(90 / 365)

    def test_weeks_suffix(self):
        assert _parse_period_to_years("52w") == pytest.approx(1.0)
        assert _parse_period_to_years("26w") == pytest.approx(0.5)

    def test_ambiguous_m_suffix_as_months(self):
        """When value <= 24, 'm' is treated as months."""
        assert _parse_period_to_years("6m") == pytest.approx(0.5)
        assert _parse_period_to_years("12m") == pytest.approx(1.0)

    def test_float_values(self):
        assert _parse_period_to_years("1.5y") == pytest.approx(1.5)
        assert _parse_period_to_years("0.5y") == pytest.approx(0.5)

    def test_whitespace_stripped(self):
        assert _parse_period_to_years(" 2y ") == 2.0
        assert _parse_period_to_years(" 6mo ") == pytest.approx(0.5)

    def test_case_insensitive(self):
        assert _parse_period_to_years("2Y") == 2.0
        assert _parse_period_to_years("6MO") == pytest.approx(0.5)
        assert _parse_period_to_years("6Mo") == pytest.approx(0.5)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            _parse_period_to_years("abc")
        with pytest.raises(ValueError):
            _parse_period_to_years("")
        with pytest.raises(ValueError):
            _parse_period_to_years("2")


# ── Timeframe Normalization Tests ────────────────────────────────────────────

class TestNormalizeTimeframe:
    """Test the _normalize_timeframe helper."""

    def test_daily_variants(self):
        assert _normalize_timeframe("daily") == "daily"
        assert _normalize_timeframe("Daily") == "daily"
        assert _normalize_timeframe("1d") == "1d"

    def test_hourly_variants(self):
        assert _normalize_timeframe("1h") == "1h"
        assert _normalize_timeframe("60m") == "1h"
        assert _normalize_timeframe("60min") == "1h"

    def test_4hour_variants(self):
        assert _normalize_timeframe("4h") == "4h"
        assert _normalize_timeframe("240m") == "4h"
        assert _normalize_timeframe("240min") == "4h"

    def test_whitespace_stripped(self):
        assert _normalize_timeframe(" daily ") == "daily"
        assert _normalize_timeframe(" 1h ") == "1h"


# ── Trade Count Estimation Tests ─────────────────────────────────────────────

class TestEstimateTradeCount:
    """Test the main estimate_trade_count function."""

    # --- Daily timeframe ---

    def test_daily_2y_momentum(self):
        result = estimate_trade_count("SPY", "2y", "daily", "momentum")
        # ~252 bars/year * 2 years = 504 bars * 0.9 = 453 min bars
        # momentum: 10-30 bars/trade
        # min_trades = 453/30 = 15, max_trades = 504/10 = 50
        assert result["min_bars"] == pytest.approx(453, abs=5)
        assert result["max_bars"] == pytest.approx(504, abs=5)
        assert result["min_trades"] >= 10
        assert result["max_trades"] <= 60
        assert result["strategy_type"] == "momentum"
        assert result["timeframe_label"] == "daily"

    def test_daily_5y_momentum(self):
        result = estimate_trade_count("SPY", "5y", "daily", "momentum")
        # ~252 * 5 = 1260 bars
        assert result["min_bars"] == pytest.approx(1134, abs=5)
        assert result["max_bars"] == pytest.approx(1260, abs=5)
        assert result["min_trades"] >= 30
        assert result["max_trades"] <= 140

    def test_daily_1y_momentum(self):
        result = estimate_trade_count("AAPL", "1y", "daily", "momentum")
        assert result["min_bars"] == pytest.approx(227, abs=5)
        assert result["max_bars"] == pytest.approx(252, abs=5)

    def test_daily_6mo_momentum(self):
        result = estimate_trade_count("SPY", "6mo", "daily", "momentum")
        # ~126 bars
        assert result["min_bars"] == pytest.approx(113, abs=5)
        assert result["max_bars"] == pytest.approx(126, abs=5)

    # --- 1h timeframe ---

    def test_1h_2y_momentum(self):
        result = estimate_trade_count("SPY", "2y", "1h", "momentum")
        # ~1638 bars/year * 2 = 3276 bars
        assert result["min_bars"] == pytest.approx(2948, abs=20)
        assert result["max_bars"] == pytest.approx(3276, abs=20)
        assert result["timeframe_label"] == "1-hour"

    def test_1h_2y_mean_reversion(self):
        result = estimate_trade_count("SPY", "2y", "1h", "mean_reversion")
        # mean_reversion: 5-15 bars/trade → more trades than momentum
        assert result["min_trades"] > 100
        assert result["strategy_type"] == "mean_reversion"

    # --- 4h timeframe ---

    def test_4h_2y_momentum(self):
        result = estimate_trade_count("SPY", "2y", "4h", "momentum")
        # ~504 bars/year * 2 = 1008 bars
        assert result["min_bars"] == pytest.approx(907, abs=10)
        assert result["max_bars"] == pytest.approx(1008, abs=10)
        assert result["timeframe_label"] == "4-hour"

    # --- Strategy types ---

    def test_mean_reversion_daily_2y(self):
        result = estimate_trade_count("SPY", "2y", "daily", "mean_reversion")
        # mean_reversion: 5-15 bars/trade → more trades
        assert result["bars_per_trade_range"] == (5, 15)
        assert result["min_trades"] >= 20

    def test_breakout_daily_2y(self):
        result = estimate_trade_count("SPY", "2y", "daily", "breakout")
        # breakout: 15-40 bars/trade → fewer trades
        assert result["bars_per_trade_range"] == (15, 40)

    def test_rsi_divergence_daily_2y(self):
        result = estimate_trade_count("SPY", "2y", "daily", "rsi_divergence")
        assert result["bars_per_trade_range"] == (8, 20)
        assert result["strategy_type"] == "rsi_divergence"

    # --- Edge cases ---

    def test_unknown_strategy_type_uses_default(self):
        result = estimate_trade_count("SPY", "2y", "daily", "unknown_fancy_strategy")
        assert result["bars_per_trade_range"] == BARS_PER_TRADE["default"]
        assert result["strategy_type"] == "unknown_fancy_strategy"

    def test_unknown_timeframe_uses_default(self):
        result = estimate_trade_count("SPY", "2y", "3m", "momentum")
        # Unknown timeframe should default to 252 bars/year
        assert result["bars_per_year"] == DEFAULT_BARS_PER_YEAR

    def test_invalid_period_defaults_to_2y(self):
        result = estimate_trade_count("SPY", "garbage", "daily", "momentum")
        # Should default to 2.0 years
        assert result["years"] == 2.0

    def test_60m_normalized_to_1h(self):
        result = estimate_trade_count("SPY", "2y", "60m", "momentum")
        result_1h = estimate_trade_count("SPY", "2y", "1h", "momentum")
        assert result["bars_per_year"] == result_1h["bars_per_year"]
        assert result["timeframe_label"] == result_1h["timeframe_label"]

    def test_240m_normalized_to_4h(self):
        result = estimate_trade_count("SPY", "2y", "240m", "momentum")
        result_4h = estimate_trade_count("SPY", "2y", "4h", "momentum")
        assert result["bars_per_year"] == result_4h["bars_per_year"]

    def test_result_dict_has_all_keys(self):
        result = estimate_trade_count("SPY", "2y", "daily", "momentum")
        expected_keys = {
            "min_bars", "max_bars", "min_trades", "max_trades",
            "recommended_min", "timeframe_label", "years", "bars_per_year",
            "bars_per_trade_range", "strategy_type",
        }
        assert set(result.keys()) == expected_keys

    def test_min_trades_at_least_1(self):
        """Even very short periods should produce min_trades >= 1."""
        result = estimate_trade_count("SPY", "1d", "daily", "breakout")
        assert result["min_trades"] >= 1

    def test_max_trades_at_least_2(self):
        """Even very short periods should produce max_trades >= 2."""
        result = estimate_trade_count("SPY", "1d", "daily", "breakout")
        assert result["max_trades"] >= 2

    def test_years_field_matches_period(self):
        result = estimate_trade_count("SPY", "3y", "daily", "momentum")
        assert result["years"] == 3.0


# ── Recommended Minimum Tests ────────────────────────────────────────────────

class TestRecommendedMinimum:
    """Test that recommended_min is always >= MIN_TRADES_THRESHOLD (20)."""

    def test_daily_2y_momentum_recommended_min(self):
        result = estimate_trade_count("SPY", "2y", "daily", "momentum")
        assert result["recommended_min"] >= MIN_TRADES_THRESHOLD

    def test_daily_2y_breakout_recommended_min(self):
        """Breakout has longest hold periods (15-40 bars), most likely to be low."""
        result = estimate_trade_count("SPY", "2y", "daily", "breakout")
        assert result["recommended_min"] >= MIN_TRADES_THRESHOLD

    def test_daily_1y_breakout_recommended_min(self):
        """Short period + breakout strategy — worst case for trade count."""
        result = estimate_trade_count("SPY", "1y", "daily", "breakout")
        assert result["recommended_min"] >= MIN_TRADES_THRESHOLD

    def test_daily_6mo_breakout_recommended_min(self):
        """Very short period + breakout — should still recommend >= 20."""
        result = estimate_trade_count("SPY", "6mo", "daily", "breakout")
        assert result["recommended_min"] >= MIN_TRADES_THRESHOLD

    def test_daily_3mo_breakout_recommended_min(self):
        """3 months + breakout — very tight but still >= 20."""
        result = estimate_trade_count("SPY", "3mo", "daily", "breakout")
        assert result["recommended_min"] >= MIN_TRADES_THRESHOLD

    def test_recommended_min_has_20_percent_margin(self):
        """When min_trades > 17, recommended_min should be ceil(min_trades * 1.2)."""
        # Use 5y daily momentum for high trade count
        result = estimate_trade_count("SPY", "5y", "daily", "momentum")
        expected = max(math.ceil(result["min_trades"] * 1.2), MIN_TRADES_THRESHOLD)
        assert result["recommended_min"] == expected

    def test_all_strategy_types_daily_2y_pass_threshold(self):
        """All strategy types on 2y daily should have recommended_min >= 20."""
        for strategy_type in BARS_PER_TRADE:
            if strategy_type == "default":
                continue
            result = estimate_trade_count("SPY", "2y", "daily", strategy_type)
            assert result["recommended_min"] >= MIN_TRADES_THRESHOLD, (
                f"Strategy {strategy_type} failed: recommended_min={result['recommended_min']}"
            )

    def test_all_timeframes_2y_momentum_pass_threshold(self):
        """All timeframes on 2y momentum should have recommended_min >= 20."""
        for tf in BARS_PER_YEAR:
            result = estimate_trade_count("SPY", "2y", tf, "momentum")
            assert result["recommended_min"] >= MIN_TRADES_THRESHOLD, (
                f"Timeframe {tf} failed: recommended_min={result['recommended_min']}"
            )


# ── Format Guidance Tests ────────────────────────────────────────────────────

class TestFormatTradeCountGuidance:
    """Test the format_trade_count_guidance function."""

    def test_basic_format(self):
        estimate = estimate_trade_count("SPY", "2y", "daily", "momentum")
        estimate["ticker"] = "SPY"
        estimate["period"] = "2y"
        text = format_trade_count_guidance(estimate)
        assert "### Trade Count Expectations" in text
        assert "SPY" in text
        assert "2y" in text
        assert "daily" in text
        assert "momentum" in text

    def test_includes_trade_range(self):
        estimate = estimate_trade_count("SPY", "2y", "daily", "momentum")
        estimate["ticker"] = "SPY"
        estimate["period"] = "2y"
        text = format_trade_count_guidance(estimate)
        assert f"{estimate['min_trades']}-{estimate['max_trades']}" in text

    def test_includes_recommended_min(self):
        estimate = estimate_trade_count("SPY", "2y", "daily", "momentum")
        estimate["ticker"] = "SPY"
        estimate["period"] = "2y"
        text = format_trade_count_guidance(estimate)
        assert str(estimate["recommended_min"]) in text
        assert "Target at least" in text

    def test_warning_for_low_expected_trades(self):
        """When min_trades < 20, a warning should be included."""
        estimate = estimate_trade_count("SPY", "1d", "daily", "breakout")
        estimate["ticker"] = "SPY"
        estimate["period"] = "1d"
        text = format_trade_count_guidance(estimate)
        if estimate["min_trades"] < MIN_TRADES_THRESHOLD:
            assert "WARNING" in text

    def test_no_warning_for_sufficient_trades(self):
        """When min_trades >= 20, no warning should be included."""
        estimate = estimate_trade_count("SPY", "5y", "daily", "mean_reversion")
        estimate["ticker"] = "SPY"
        estimate["period"] = "5y"
        text = format_trade_count_guidance(estimate)
        if estimate["min_trades"] >= MIN_TRADES_THRESHOLD:
            assert "WARNING" not in text

    def test_no_ticker_display(self):
        """When ticker is empty, no 'on TICKER' should appear."""
        estimate = estimate_trade_count("", "2y", "daily", "momentum")
        estimate["ticker"] = ""
        estimate["period"] = "2y"
        text = format_trade_count_guidance(estimate)
        assert " on " not in text

    def test_1h_timeframe_label(self):
        estimate = estimate_trade_count("SPY", "2y", "1h", "momentum")
        estimate["ticker"] = "SPY"
        estimate["period"] = "2y"
        text = format_trade_count_guidance(estimate)
        assert "1-hour" in text

    def test_4h_timeframe_label(self):
        estimate = estimate_trade_count("SPY", "2y", "4h", "momentum")
        estimate["ticker"] = "SPY"
        estimate["period"] = "2y"
        text = format_trade_count_guidance(estimate)
        assert "4-hour" in text

    def test_strategy_type_underscores_replaced(self):
        estimate = estimate_trade_count("SPY", "2y", "daily", "mean_reversion")
        estimate["ticker"] = "SPY"
        estimate["period"] = "2y"
        text = format_trade_count_guidance(estimate)
        assert "mean reversion" in text
        assert "mean_reversion" not in text


# ── Convenience Function Tests ───────────────────────────────────────────────

class TestBuildTradeCountGuidance:
    """Test the build_trade_count_guidance convenience function."""

    def test_returns_string(self):
        result = build_trade_count_guidance("SPY", "2y", "daily", "momentum")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_expected_sections(self):
        result = build_trade_count_guidance("SPY", "2y", "daily", "momentum")
        assert "### Trade Count Expectations" in result
        assert "SPY" in result

    def test_1h_timeframe(self):
        result = build_trade_count_guidance("AAPL", "1y", "1h", "mean_reversion")
        assert "1-hour" in result
        assert "AAPL" in result
        assert "mean reversion" in result

    def test_4h_timeframe(self):
        result = build_trade_count_guidance("TSLA", "2y", "4h", "breakout")
        assert "4-hour" in result
        assert "TSLA" in result

    def test_default_strategy_type(self):
        """Default strategy_type should be 'momentum'."""
        result = build_trade_count_guidance("SPY", "2y", "daily")
        assert "momentum" in result


# ── Constants Tests ──────────────────────────────────────────────────────────

class TestConstants:
    """Verify module constants are well-formed."""

    def test_bars_per_year_values_positive(self):
        for tf, bpy in BARS_PER_YEAR.items():
            assert bpy > 0, f"BARS_PER_YEAR['{tf}'] should be positive"

    def test_bars_per_trade_ranges_valid(self):
        for st, (min_bpt, max_bpt) in BARS_PER_TRADE.items():
            assert min_bpt >= 1, f"BARS_PER_TRADE['{st}'] min should be >= 1"
            assert max_bpt >= min_bpt, f"BARS_PER_TRADE['{st}'] max should be >= min"

    def test_min_trades_threshold(self):
        assert MIN_TRADES_THRESHOLD == 20

    def test_default_bars_per_year_matches_daily(self):
        assert DEFAULT_BARS_PER_YEAR == 252
