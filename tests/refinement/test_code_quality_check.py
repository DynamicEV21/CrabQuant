"""Tests for code_quality_check module — 30+ tests covering all anti-patterns."""

from crabquant.refinement.code_quality_check import (
    CodeQualityIssue,
    CodeQualityReport,
    check_code_quality,
    format_code_quality_for_prompt,
)

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────

GOOD_STRATEGY = '''\
from crabquant.indicator_cache import cached_indicator
import pandas as pd

DESCRIPTION = "Simple RSI strategy"
DEFAULT_PARAMS = {"period": 14, "threshold": 30}

def generate_signals(df, params):
    period = params.get("period", 14)
    threshold = params.get("threshold", 30)
    
    rsi = cached_indicator("rsi", close=df["close"], length=period)
    entries = (rsi < threshold) & (rsi > rsi.shift(1))
    exits = (rsi > 70) | (hold_periods > 10)
    return entries, exits
'''

GOOD_STRATEGY_WITH_TIME_STOP = '''\
from crabquant.indicator_cache import cached_indicator
import pandas as pd

DESCRIPTION = "Strategy with time stop"
DEFAULT_PARAMS = {"period": 14}

def generate_signals(df, params):
    period = params.get("period", 14)
    rsi = cached_indicator("rsi", close=df["close"], length=period)
    entries = rsi < 30
    exits = (rsi > 70) | (hold_periods > 10)
    return entries, exits
'''


# ═══════════════════════════════════════════════════════════════════════════
# 1. Over-complex entry detection
# ═══════════════════════════════════════════════════════════════════════════

class TestOverComplexEntry:
    """Test detection of stacked & operators in entry conditions."""

    def test_simple_entry_passes(self):
        report = check_code_quality(GOOD_STRATEGY)
        over_complex = [i for i in report.issues if i.category == "over_complex"]
        assert len(over_complex) == 0

    def test_four_ampersands_flagged_warning(self):
        code = GOOD_STRATEGY.replace(
            'entries = (rsi < threshold) & (rsi > rsi.shift(1))',
            'entries = (a > 1) & (b < 2) & (c > 3) & (d < 4) & (e > 5)',
        )
        report = check_code_quality(code)
        over_complex = [i for i in report.issues if i.category == "over_complex"]
        assert len(over_complex) == 1
        assert over_complex[0].severity == "warning"

    def test_six_ampersands_flagged_critical(self):
        code = GOOD_STRATEGY.replace(
            'entries = (rsi < threshold) & (rsi > rsi.shift(1))',
            'entries = (a > 1) & (b < 2) & (c > 3) & (d < 4) & (e > 5) & (f < 6)',
        )
        report = check_code_quality(code)
        over_complex = [i for i in report.issues if i.category == "over_complex"]
        assert len(over_complex) == 1
        assert over_complex[0].severity == "critical"

    def test_three_ampersands_ok(self):
        code = GOOD_STRATEGY.replace(
            'entries = (rsi < threshold) & (rsi > rsi.shift(1))',
            'entries = (a > 1) & (b < 2) & (c > 3)',
        )
        report = check_code_quality(code)
        over_complex = [i for i in report.issues if i.category == "over_complex"]
        assert len(over_complex) == 0

    def test_only_first_entry_checked(self):
        code = '''\
def generate_signals(df, params):
    entries = (a > 1) & (b < 2) & (c > 3) & (d < 4) & (e > 5) & (f < 6)
    exits = True
    return entries, exits
'''
        report = check_code_quality(code)
        over_complex = [i for i in report.issues if i.category == "over_complex"]
        assert len(over_complex) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. Contradictory conditions
# ═══════════════════════════════════════════════════════════════════════════

class TestContradictoryConditions:
    """Test detection of impossible conditions."""

    def test_contradictory_greater_lower(self):
        code = '''\
def generate_signals(df, params):
    entries = (rsi > 70) & (rsi < 30)
    exits = (rsi > 80)
    return entries, exits
'''
        report = check_code_quality(code)
        contradictions = [i for i in report.issues if i.category == "contradictory"]
        assert len(contradictions) == 1
        assert contradictions[0].severity == "critical"

    def test_contradictory_equal_values(self):
        code = '''\
def generate_signals(df, params):
    entries = (rsi > 50) & (rsi < 50)
    exits = True
    return entries, exits
'''
        report = check_code_quality(code)
        contradictions = [i for i in report.issues if i.category == "contradictory"]
        assert len(contradictions) == 1

    def test_non_contradictory_passes(self):
        code = '''\
def generate_signals(df, params):
    entries = (rsi > 30) & (rsi < 70)
    exits = True
    return entries, exits
'''
        report = check_code_quality(code)
        contradictions = [i for i in report.issues if i.category == "contradictory"]
        assert len(contradictions) == 0

    def test_contradictory_with_floats(self):
        code = '''\
def generate_signals(df, params):
    entries = (rsi > 80.5) & (rsi < 30.2)
    exits = True
    return entries, exits
'''
        report = check_code_quality(code)
        contradictions = [i for i in report.issues if i.category == "contradictory"]
        assert len(contradictions) == 1

    def test_reverse_order_contradiction(self):
        code = '''\
def generate_signals(df, params):
    entries = (rsi < 30) & (rsi > 70)
    exits = True
    return entries, exits
'''
        report = check_code_quality(code)
        contradictions = [i for i in report.issues if i.category == "contradictory"]
        assert len(contradictions) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. Long lookback periods
# ═══════════════════════════════════════════════════════════════════════════

class TestLongLookbackPeriods:
    """Test detection of very long indicator periods."""

    def test_long_sma_detected(self):
        code = '''\
def generate_signals(df, params):
    sma = cached_indicator("sma", close=df["close"], length=200)
    entries = close > sma
    exits = close < sma
    return entries, exits
'''
        report = check_code_quality(code)
        long_lb = [i for i in report.issues if i.category == "long_lookback"]
        assert len(long_lb) == 1
        assert long_lb[0].severity == "warning"

    def test_normal_period_passes(self):
        report = check_code_quality(GOOD_STRATEGY)
        long_lb = [i for i in report.issues if i.category == "long_lookback"]
        assert len(long_lb) == 0

    def test_period_exactly_100_passes(self):
        code = '''\
def generate_signals(df, params):
    ema = cached_indicator("ema", close=df["close"], length=100)
    entries = close > ema
    exits = close < ema
    return entries, exits
'''
        report = check_code_quality(code)
        long_lb = [i for i in report.issues if i.category == "long_lookback"]
        assert len(long_lb) == 0

    def test_period_101_detected(self):
        code = '''\
def generate_signals(df, params):
    ema = cached_indicator("ema", close=df["close"], length=101)
    entries = close > ema
    exits = close < ema
    return entries, exits
'''
        report = check_code_quality(code)
        long_lb = [i for i in report.issues if i.category == "long_lookback"]
        assert len(long_lb) == 1

    def test_multiple_long_lookbacks(self):
        code = '''\
def generate_signals(df, params):
    sma = cached_indicator("sma", close=df["close"], length=150)
    ema = cached_indicator("ema", close=df["close"], length=200)
    entries = close > sma
    exits = close < ema
    return entries, exits
'''
        report = check_code_quality(code)
        long_lb = [i for i in report.issues if i.category == "long_lookback"]
        assert len(long_lb) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. No exit logic
# ═══════════════════════════════════════════════════════════════════════════

class TestNoExitLogic:
    """Test detection of constant-false exits."""

    def test_constant_false_pd_series(self):
        code = '''\
def generate_signals(df, params):
    entries = close > sma
    exits = pd.Series(False, index=df.index)
    return entries, exits
'''
        report = check_code_quality(code)
        no_exit = [i for i in report.issues if i.category == "no_exit"]
        assert len(no_exit) == 1
        assert no_exit[0].severity == "critical"

    def test_constant_false_literal(self):
        code = '''\
def generate_signals(df, params):
    entries = close > sma
    exits = False
    return entries, exits
'''
        report = check_code_quality(code)
        no_exit = [i for i in report.issues if i.category == "no_exit"]
        assert len(no_exit) == 1

    def test_np_zeros_exit(self):
        code = '''\
def generate_signals(df, params):
    entries = close > sma
    exits = np.zeros(len(df), dtype=bool)
    return entries, exits
'''
        report = check_code_quality(code)
        no_exit = [i for i in report.issues if i.category == "no_exit"]
        assert len(no_exit) == 1

    def test_real_exit_passes(self):
        report = check_code_quality(GOOD_STRATEGY)
        no_exit = [i for i in report.issues if i.category == "no_exit"]
        assert len(no_exit) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Missing time stop
# ═══════════════════════════════════════════════════════════════════════════

class TestMissingTimeStop:
    """Test detection of missing time-based exit mechanism."""

    def test_no_time_stop_warning(self):
        code = '''\
def generate_signals(df, params):
    rsi = cached_indicator("rsi", close=df["close"], length=14)
    entries = rsi < 30
    exits = rsi > 70
    return entries, exits
'''
        report = check_code_quality(code)
        no_time = [i for i in report.issues if i.category == "no_time_stop"]
        assert len(no_time) == 1
        assert no_time[0].severity == "warning"

    def test_hold_periods_passes(self):
        report = check_code_quality(GOOD_STRATEGY)
        no_time = [i for i in report.issues if i.category == "no_time_stop"]
        assert len(no_time) == 0

    def test_bars_since_entry_passes(self):
        code = '''\
def generate_signals(df, params):
    entries = rsi < 30
    exits = (rsi > 70) | (bars_since_entry > 10)
    return entries, exits
'''
        report = check_code_quality(code)
        no_time = [i for i in report.issues if i.category == "no_time_stop"]
        assert len(no_time) == 0

    def test_max_hold_passes(self):
        code = '''\
def generate_signals(df, params):
    entries = rsi < 30
    exits = (rsi > 70) | (max_hold > 5)
    return entries, exits
'''
        report = check_code_quality(code)
        no_time = [i for i in report.issues if i.category == "no_time_stop"]
        assert len(no_time) == 0

    def test_no_time_stop_when_no_exit(self):
        """When exits are constant-false, time stop warning should not fire."""
        code = '''\
def generate_signals(df, params):
    entries = rsi < 30
    exits = pd.Series(False, index=df.index)
    return entries, exits
'''
        report = check_code_quality(code)
        no_time = [i for i in report.issues if i.category == "no_time_stop"]
        assert len(no_time) == 0  # no_exit fires instead


# ═══════════════════════════════════════════════════════════════════════════
# 6. Extreme thresholds
# ═══════════════════════════════════════════════════════════════════════════

class TestExtremeThresholds:
    """Test detection of hardcoded extreme thresholds."""

    def test_rsi_above_95(self):
        code = '''\
def generate_signals(df, params):
    rsi = cached_indicator("rsi", close=df["close"], length=14)
    entries = rsi > 96
    exits = rsi < 4
    return entries, exits
'''
        report = check_code_quality(code)
        extreme = [i for i in report.issues if i.category == "extreme_threshold"]
        assert len(extreme) == 2

    def test_rsi_below_5(self):
        code = '''\
def generate_signals(df, params):
    rsi = cached_indicator("rsi", close=df["close"], length=14)
    entries = rsi < 3
    exits = True
    return entries, exits
'''
        report = check_code_quality(code)
        extreme = [i for i in report.issues if i.category == "extreme_threshold"]
        assert len(extreme) == 1

    def test_normal_rsi_passes(self):
        report = check_code_quality(GOOD_STRATEGY)
        extreme = [i for i in report.issues if i.category == "extreme_threshold"]
        assert len(extreme) == 0

    def test_volume_above_5(self):
        code = '''\
def generate_signals(df, params):
    entries = volume > 10
    exits = True
    return entries, exits
'''
        report = check_code_quality(code)
        extreme = [i for i in report.issues if i.category == "extreme_threshold"]
        assert len(extreme) == 1

    def test_atr_multiplier_extreme(self):
        code = '''\
def generate_signals(df, params):
    atr = cached_indicator("atr", high=df["high"], low=df["low"], close=df["close"], length=14)
    entries = close > sma + atr * 5
    exits = True
    return entries, exits
'''
        report = check_code_quality(code)
        extreme = [i for i in report.issues if i.category == "extreme_threshold"]
        assert len(extreme) == 1

    def test_default_params_not_flagged(self):
        """Thresholds in DEFAULT_PARAMS should not be flagged."""
        code = '''\
DEFAULT_PARAMS = {"threshold": 99}

def generate_signals(df, params):
    threshold = params.get("threshold", 99)
    entries = rsi < threshold
    exits = True
    return entries, exits
'''
        report = check_code_quality(code)
        extreme = [i for i in report.issues if i.category == "extreme_threshold"]
        assert len(extreme) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases: empty code, missing functions, etc."""

    def test_empty_code(self):
        report = check_code_quality("")
        assert report.score == 0.0
        assert report.overall_verdict == "reject"
        assert len(report.issues) == 1

    def test_whitespace_only(self):
        report = check_code_quality("   \n\n  \t  ")
        assert report.score == 0.0
        assert report.overall_verdict == "reject"

    def test_no_generate_signals(self):
        code = '''\
import pandas as pd

DESCRIPTION = "No function"
DEFAULT_PARAMS = {}
'''
        report = check_code_quality(code)
        assert report.score == 0.0
        assert report.overall_verdict == "reject"
        assert any("generate_signals" in i.description for i in report.issues)

    def test_minimal_valid_strategy(self):
        code = '''\
def generate_signals(df, params):
    entries = close > sma
    exits = close < sma
    return entries, exits
'''
        report = check_code_quality(code)
        assert report.score > 0.0
        assert report.overall_verdict in ("good", "warning")

    def test_code_with_syntax_errors(self):
        """Code with syntax errors should still parse without crashing."""
        code = '''\
def generate_signals(df, params):
    entries = (rsi < 30 & (rsi > 20
    exits = True
    return entries, exits
'''
        # Should not raise
        report = check_code_quality(code)
        assert isinstance(report, CodeQualityReport)

    def test_multiline_entry_assignment(self):
        """Entries spread across multiple lines."""
        code = '''\
def generate_signals(df, params):
    entries = (
        rsi < 30
        & close > sma
        & volume > avg_vol
        & adx > 25
    )
    exits = (rsi > 70) | (hold_periods > 10)
    return entries, exits
'''
        report = check_code_quality(code)
        # The & count on the entries line itself: 0 on the assignment line
        # but let's verify it doesn't crash
        assert isinstance(report, CodeQualityReport)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Score calculation
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreCalculation:
    """Test score and verdict computation."""

    def test_clean_code_high_score(self):
        report = check_code_quality(GOOD_STRATEGY_WITH_TIME_STOP)
        assert report.score >= 0.90
        assert report.overall_verdict == "good"

    def test_single_warning_reduces_score(self):
        code = '''\
def generate_signals(df, params):
    entries = rsi < 30
    exits = rsi > 70
    return entries, exits
'''
        report = check_code_quality(code)
        assert report.score < 1.0
        assert report.overall_verdict == "good"  # single warning: 0.90 >= 0.75

    def test_multiple_criticals_reject(self):
        code = '''\
def generate_signals(df, params):
    entries = (rsi > 70) & (rsi < 30) & (a > 1) & (b < 2) & (c > 3) & (d < 4) & (e > 5) & (f < 6)
    exits = pd.Series(False, index=df.index)
    return entries, exits
'''
        report = check_code_quality(code)
        assert report.score < 0.5
        assert report.overall_verdict == "reject"

    def test_score_clamped_to_zero(self):
        code = '''\
def generate_signals(df, params):
    entries = (a > 1) & (b < 2) & (c > 3) & (d < 4) & (e > 5) & (f < 6)
    exits = pd.Series(False, index=df.index)
    return entries, exits
'''
        report = check_code_quality(code)
        assert 0.0 <= report.score <= 1.0

    def test_score_clamped_to_one(self):
        report = check_code_quality(GOOD_STRATEGY_WITH_TIME_STOP)
        assert report.score <= 1.0

    def test_verdict_thresholds(self):
        # good: >= 0.75
        good_report = check_code_quality(GOOD_STRATEGY_WITH_TIME_STOP)
        assert good_report.overall_verdict == "good"

        # warning: 0.50-0.74 — critical + warning = 0.70
        code = '''\
def generate_signals(df, params):
    sma = cached_indicator("sma", close=df["close"], length=150)
    entries = close > sma
    exits = pd.Series(False, index=df.index)
    return entries, exits
'''
        warn_report = check_code_quality(code)
        assert warn_report.overall_verdict == "warning"

        # reject: < 0.50 — multiple criticals
        code_bad = '''\
def generate_signals(df, params):
    entries = (rsi > 70) & (rsi < 30) & (a > 1) & (b < 2) & (c > 3) & (d < 4) & (e > 5) & (f < 6)
    exits = pd.Series(False, index=df.index)
    return entries, exits
'''
        bad_report = check_code_quality(code_bad)
        assert bad_report.overall_verdict == "reject"


# ═══════════════════════════════════════════════════════════════════════════
# 9. Formatting function
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatting:
    """Test format_code_quality_for_prompt."""

    def test_clean_report_format(self):
        report = check_code_quality(GOOD_STRATEGY_WITH_TIME_STOP)
        formatted = format_code_quality_for_prompt(report)
        assert "Code Quality Pre-Check" in formatted
        assert "GOOD" in formatted
        assert "No issues detected" in formatted

    def test_warning_report_format(self):
        code = '''\
def generate_signals(df, params):
    entries = rsi < 30
    exits = rsi > 70
    return entries, exits
'''
        report = check_code_quality(code)
        formatted = format_code_quality_for_prompt(report)
        assert "WARNING" in formatted or "WARNING" in report.overall_verdict.upper()
        assert "no_time_stop" in formatted

    def test_critical_report_format(self):
        code = '''\
def generate_signals(df, params):
    entries = (rsi > 70) & (rsi < 30)
    exits = pd.Series(False, index=df.index)
    return entries, exits
'''
        report = check_code_quality(code)
        formatted = format_code_quality_for_prompt(report)
        assert "CRITICAL" in formatted
        assert "Fix:" in formatted

    def test_format_includes_score(self):
        report = check_code_quality(GOOD_STRATEGY_WITH_TIME_STOP)
        formatted = format_code_quality_for_prompt(report)
        assert "Score:" in formatted
        assert str(round(report.score, 2)) in formatted

    def test_format_includes_summary(self):
        report = check_code_quality(GOOD_STRATEGY_WITH_TIME_STOP)
        formatted = format_code_quality_for_prompt(report)
        assert report.summary_for_llm in formatted

    def test_format_empty_code(self):
        report = check_code_quality("")
        formatted = format_code_quality_for_prompt(report)
        assert "REJECT" in formatted
        assert len(formatted) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 10. Real-world strategy patterns
# ═══════════════════════════════════════════════════════════════════════════

class TestRealWorldPatterns:
    """Test with realistic strategy code patterns."""

    def test_macd_bollinger_strategy(self):
        code = '''\
from crabquant.indicator_cache import cached_indicator
import pandas as pd

DESCRIPTION = "MACD Bollinger strategy"
DEFAULT_PARAMS = {"fast": 12, "slow": 26, "signal": 9, "bb_period": 20}

def generate_signals(df, params):
    macd = cached_indicator("macd", close=df["close"], fast_period=12, slow_period=26, signal_period=9)
    bb = cached_indicator("bbands", close=df["close"], length=20)
    entries = (macd["macd"] > macd["signal"]) & (df["close"] < bb["lower"])
    exits = (macd["macd"] < macd["signal"]) | (hold_periods > 15)
    return entries, exits
'''
        report = check_code_quality(code)
        assert report.score >= 0.70
        assert report.overall_verdict in ("good", "warning")

    def test_overfitted_strategy_rejected(self):
        code = '''\
from crabquant.indicator_cache import cached_indicator
import pandas as pd

DESCRIPTION = "Overfitted strategy"
DEFAULT_PARAMS = {}

def generate_signals(df, params):
    rsi = cached_indicator("rsi", close=df["close"], length=14)
    macd = cached_indicator("macd", close=df["close"], fast_period=12, slow_period=26)
    bb = cached_indicator("bbands", close=df["close"], length=20)
    atr = cached_indicator("atr", high=df["high"], low=df["low"], close=df["close"], length=14)
    adx = cached_indicator("adx", high=df["high"], low=df["low"], close=df["close"], length=14)
    stoch = cached_indicator("stoch", high=df["high"], low=df["low"], close=df["close"], length=14)
    entries = (
        (rsi > 70) & (rsi < 30) & (macd["macd"] > macd["signal"])
        & (df["close"] < bb["lower"]) & (atr > 1.5)
        & (adx > 25) & (stoch["k"] < 20) & (volume > 10)
    )
    exits = pd.Series(False, index=df.index)
    return entries, exits
'''
        report = check_code_quality(code)
        assert report.overall_verdict == "reject"
        assert any(i.category == "over_complex" for i in report.issues)
        assert any(i.category == "no_exit" for i in report.issues)
        assert any(i.category == "contradictory" for i in report.issues)

    def test_well_balanced_strategy_passes(self):
        code = '''\
from crabquant.indicator_cache import cached_indicator
import pandas as pd

DESCRIPTION = "Balanced momentum strategy"
DEFAULT_PARAMS = {"ema_fast": 12, "ema_slow": 26, "atr_mult": 2.0}

def generate_signals(df, params):
    ema_fast = cached_indicator("ema", close=df["close"], length=12)
    ema_slow = cached_indicator("ema", close=df["close"], length=26)
    atr = cached_indicator("atr", high=df["high"], low=df["low"], close=df["close"], length=14)
    entries = (ema_fast > ema_slow) & (df["close"] > ema_slow)
    exits = (ema_fast < ema_slow) | (hold_periods > 20)
    return entries, exits
'''
        report = check_code_quality(code)
        assert report.overall_verdict == "good"

    def test_strategy_with_extreme_lookback(self):
        code = '''\
from crabquant.indicator_cache import cached_indicator

def generate_signals(df, params):
    sma200 = cached_indicator("sma", close=df["close"], length=200)
    ema300 = cached_indicator("ema", close=df["close"], length=300)
    entries = df["close"] > sma200
    exits = (df["close"] < sma200) | (hold_periods > 30)
    return entries, exits
'''
        report = check_code_quality(code)
        long_lb = [i for i in report.issues if i.category == "long_lookback"]
        assert len(long_lb) == 2

    def test_strategy_with_negative_lookback(self):
        """Negative or very small lookbacks should not be flagged as long."""
        code = '''\
from crabquant.indicator_cache import cached_indicator

def generate_signals(df, params):
    rsi = cached_indicator("rsi", close=df["close"], length=5)
    entries = rsi < 20
    exits = (rsi > 80) | (hold_periods > 5)
    return entries, exits
'''
        report = check_code_quality(code)
        long_lb = [i for i in report.issues if i.category == "long_lookback"]
        assert len(long_lb) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 11. Dataclass types
# ═══════════════════════════════════════════════════════════════════════════

class TestDataclasses:
    """Test that dataclasses are properly constructed."""

    def test_issue_fields(self):
        issue = CodeQualityIssue(
            severity="warning",
            category="test",
            line_range="L1",
            description="test desc",
            fix_suggestion="test fix",
        )
        assert issue.severity == "warning"
        assert issue.category == "test"

    def test_report_fields(self):
        report = CodeQualityReport(
            score=0.85,
            issues=[],
            overall_verdict="good",
            summary_for_llm="All good.",
        )
        assert report.score == 0.85
        assert report.issues == []
        assert report.overall_verdict == "good"

    def test_report_issues_mutable(self):
        report = check_code_quality(GOOD_STRATEGY_WITH_TIME_STOP)
        report.issues.append(CodeQualityIssue(
            severity="info", category="test", line_range="L1",
            description="test", fix_suggestion="test",
        ))
        assert len(report.issues) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 12. Summary for LLM
# ═══════════════════════════════════════════════════════════════════════════

class TestSummaryForLLM:
    """Test the summary_for_llm field."""

    def test_clean_summary(self):
        report = check_code_quality(GOOD_STRATEGY_WITH_TIME_STOP)
        assert "passed" in report.summary_for_llm.lower()

    def test_warning_summary(self):
        code = '''\
def generate_signals(df, params):
    entries = rsi < 30
    exits = rsi > 70
    return entries, exits
'''
        report = check_code_quality(code)
        assert "warning" in report.summary_for_llm.lower()
        assert "no_time_stop" in report.summary_for_llm

    def test_reject_summary(self):
        code = '''\
def generate_signals(df, params):
    entries = (rsi > 70) & (rsi < 30) & (a > 1) & (b < 2) & (c > 3) & (d < 4) & (e > 5)
    exits = pd.Series(False, index=df.index)
    return entries, exits
'''
        report = check_code_quality(code)
        assert "reject" in report.summary_for_llm.lower()
