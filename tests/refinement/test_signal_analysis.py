"""
Tests for the signal_analysis module.

Covers:
- analyze_signal_density: entry/exit signal density analysis
- format_signal_analysis_for_prompt: formatting for LLM prompt
- check_signal_density_early_exit: quick pre-check before backtest

Edge cases: None inputs, non-Series, all-NaN, zero signals, very few,
many entries few exits, excessive entries, low trade estimate, clustered,
normal/regular, single entry, sparse entries, uneven gaps, OK severity.
"""

from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from crabquant.refinement.signal_analysis import (
    analyze_signal_density,
    check_signal_density_early_exit,
    format_signal_analysis_for_prompt,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_entries(length: int, true_indices: list[int] | None = None) -> pd.Series:
    """Create a boolean Series of given length with True at specified indices."""
    s = pd.Series(False, index=range(length))
    if true_indices:
        for i in true_indices:
            if 0 <= i < length:
                s.iloc[i] = True
    return s


# ── analyze_signal_density ──────────────────────────────────────────────────


class TestAnalyzeSignalDensityNoneInputs:
    """Tests where entries or exits are None."""

    def test_none_entries(self):
        result = analyze_signal_density(None, pd.Series(False, index=range(100)), 100)
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "signals_are_none"
        assert result["entry_count"] == 0

    def test_none_exits(self):
        result = analyze_signal_density(pd.Series(False, index=range(100)), None, 100)
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "signals_are_none"

    def test_both_none(self):
        result = analyze_signal_density(None, None, 100)
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "signals_are_none"
        assert result["fix_suggestion"] is not None


class TestAnalyzeSignalDensityNonSeries:
    """Tests where entries is not a pd.Series."""

    def test_list_entries(self):
        result = analyze_signal_density([True, False, True], pd.Series(False, index=range(3)), 3)
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "entries_not_series"
        assert "list" in result["fix_suggestion"]

    def test_numpy_array_entries(self):
        result = analyze_signal_density(
            np.array([True, False, True]),
            pd.Series(False, index=range(3)),
            3,
        )
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "entries_not_series"

    def test_dict_entries(self):
        result = analyze_signal_density(
            {0: True, 1: False},
            pd.Series(False, index=range(2)),
            2,
        )
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "entries_not_series"


class TestAnalyzeSignalDensityAllNaN:
    """Tests where entries are all NaN."""

    def test_all_nan_entries(self):
        entries = pd.Series([np.nan] * 100)
        exits = pd.Series([np.nan] * 100)
        result = analyze_signal_density(entries, exits, 100)
        # NaNs get filled with False -> zero entries
        assert result["entry_count"] == 0
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "zero_entries"

    def test_mixed_nan_and_true(self):
        entries = pd.Series([np.nan] * 50 + [True] * 10 + [False] * 40)
        exits = pd.Series([np.nan] * 50 + [True] * 10 + [False] * 40)
        result = analyze_signal_density(entries, exits, 100)
        # NaN filled -> 10 true entries, 10 true exits
        assert result["entry_count"] == 10
        assert result["exit_count"] == 10


class TestAnalyzeSignalDensityZeroSignals:
    """Tests with zero entries and/or zero exits."""

    def test_zero_entries_zero_exits(self):
        entries = _make_entries(500)
        exits = _make_entries(500)
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_count"] == 0
        assert result["exit_count"] == 0
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "zero_entries"
        assert result["signal_pattern"] == "none"
        assert result["entry_gaps"] == []

    def test_entries_zero_exits(self):
        entries = _make_entries(500, true_indices=[10, 50, 100, 200, 300, 400])
        exits = _make_entries(500)
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_count"] == 6
        assert result["exit_count"] == 0
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "zero_exits"
        assert result["estimated_trades"] == 0


class TestAnalyzeSignalDensityFewEntries:
    """Tests with very few entries (< 5)."""

    def test_one_entry(self):
        entries = _make_entries(500, true_indices=[100])
        exits = _make_entries(500, true_indices=[110])
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_count"] == 1
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "very_few_entries"
        assert result["signal_pattern"] == "single"
        assert result["entry_gaps"] == []

    def test_two_entries(self):
        entries = _make_entries(500, true_indices=[100, 200])
        exits = _make_entries(500, true_indices=[110, 210])
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_count"] == 2
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "very_few_entries"
        assert result["signal_pattern"] == "sparse"
        assert len(result["entry_gaps"]) == 1
        assert result["entry_gaps"][0] == 100

    def test_three_entries(self):
        entries = _make_entries(500, true_indices=[50, 150, 300])
        exits = _make_entries(500, true_indices=[60, 160, 310])
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_count"] == 3
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "very_few_entries"
        assert result["signal_pattern"] == "sparse"

    def test_four_entries(self):
        entries = _make_entries(500, true_indices=[10, 20, 30, 40])
        exits = _make_entries(500, true_indices=[15, 25, 35, 45])
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_count"] == 4
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "very_few_entries"


class TestAnalyzeSignalDensityManyEntriesFewExits:
    """Tests with many entries but very few exits."""

    def test_many_entries_few_exits(self):
        entries = _make_entries(500, true_indices=list(range(0, 100, 5)))  # 20 entries
        exits = _make_entries(500, true_indices=[50, 200])  # 2 exits
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_count"] == 20
        assert result["exit_count"] == 2
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "few_exits"
        assert result["estimated_trades"] == 2


class TestAnalyzeSignalDensityExcessiveEntries:
    """Tests with excessive entry rate (> 30%)."""

    def test_excessive_entry_rate(self):
        # 160 entries in 500 bars = 32% rate
        true_indices = list(range(0, 500, 3))[:160]
        entries = _make_entries(500, true_indices=true_indices)
        exits = _make_entries(500, true_indices=true_indices)
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_rate"] > 0.3
        assert result["severity"] == "warning"
        assert result["diagnosis"] == "excessive_entries"

    def test_just_over_threshold(self):
        # 152 entries in 500 bars = 30.4%
        entries = _make_entries(500, true_indices=list(range(0, 500))[:152])
        exits = _make_entries(500, true_indices=list(range(0, 500))[:152])
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_rate"] > 0.3
        assert result["diagnosis"] == "excessive_entries"


class TestAnalyzeSignalDensityLowTradeEstimate:
    """Tests where estimated trades < 10 but entries >= 5."""

    def test_low_trade_estimate_few_exits(self):
        # 15 entries, 7 exits -> estimated trades = 7
        entries = _make_entries(500, true_indices=list(range(20, 95, 5)))  # 15
        exits = _make_entries(500, true_indices=list(range(30, 65, 5)))  # 7
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_count"] == 15
        assert result["exit_count"] == 7
        assert result["estimated_trades"] == 7
        assert result["severity"] == "warning"
        assert result["diagnosis"] == "low_trade_estimate"

    def test_nine_trades(self):
        # 20 entries, 9 exits -> estimated trades = 9
        entries = _make_entries(500, true_indices=list(range(10, 110, 5)))  # 20
        exits = _make_entries(500, true_indices=list(range(15, 60, 5)))  # 9
        result = analyze_signal_density(entries, exits, 500)
        assert result["estimated_trades"] == 9
        assert result["diagnosis"] == "low_trade_estimate"


class TestAnalyzeSignalDensityClusteredEntries:
    """Tests where entries are clustered (avg gap > 100)."""

    def test_clustered_pattern(self):
        # 6 entries clustered in first 50 bars, then sparse rest of 500
        true_indices = [10, 12, 14, 16, 18, 450]
        entries = _make_entries(500, true_indices=true_indices)
        exits = _make_entries(500, true_indices=true_indices)
        result = analyze_signal_density(entries, exits, 500)
        # avg gap should be large: diffs are [2,2,2,2,432], avg=88
        # Actually 88 < 100 so let me adjust
        # Let's use fewer clustered entries
        pass

    def test_clustered_with_large_avg_gap(self):
        # Need >= 10 estimated trades and avg gap > 100 for clustered pattern.
        # Use 15 entries: 14 clustered at start, 1 far away.
        # gaps: [1,1,...,1, 800] -> avg = (13*1 + 800)/14 = 813/14 ≈ 58.1
        # Need avg > 100, so use fewer clustered entries relative to the big gap.
        # 5 entries at start: gaps [2,2,2,2,400] -> avg = 408/5 = 81.6
        # Need more extreme: 4 entries at start, 1 far: gaps [2,2,2,400] -> avg=101.5
        # But need >= 10 entries and >= 10 exits. Let me use two clusters.
        # Cluster 1: 5 entries close together, Cluster 2: 5 entries close together,
        # Big gap between clusters.
        # 10 entries at indices: 10,12,14,16,18,  500,502,504,506,508
        # gaps: [2,2,2,2, 482, 2,2,2,2] -> avg = (4*2 + 482 + 4*2)/9 = 498/9 ≈ 55.3
        # Still not > 100. Let me use fewer entries per cluster.
        # 10 entries at indices: 10,12, 900,902,904,906,908,910,912,914
        # gaps: [2, 888, 2,2,2,2,2,2,2] -> avg = (2+888+7*2)/9 = 904/9 ≈ 100.4
        true_indices = [10, 12, 900, 902, 904, 906, 908, 910, 912, 914]
        entries = _make_entries(1000, true_indices=true_indices)
        exits = _make_entries(1000, true_indices=true_indices)
        result = analyze_signal_density(entries, exits, 1000)
        assert result["entry_count"] == 10
        assert result["avg_entry_gap"] > 100
        assert result["signal_pattern"] == "clustered"
        assert result["severity"] == "warning"
        assert result["diagnosis"] == "clustered_entries"


class TestAnalyzeSignalDensityNormalRegular:
    """Tests with normal, regular entry pattern — should be OK."""

    def test_regular_entries_ok(self):
        # 25 entries, 25 exits, evenly spaced over 500 bars
        entry_indices = list(range(10, 500, 20))  # 25 entries
        exit_indices = [i + 5 for i in entry_indices]  # 25 exits
        entries = _make_entries(500, true_indices=entry_indices)
        exits = _make_entries(500, true_indices=exit_indices)
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_count"] == 25
        assert result["exit_count"] == 25
        assert result["severity"] == "ok"
        assert result["diagnosis"] is None
        assert result["fix_suggestion"] is None
        assert result["signal_pattern"] == "regular"

    def test_ok_severity_fields(self):
        entries = _make_entries(500, true_indices=list(range(10, 500, 18)))
        exits = _make_entries(500, true_indices=list(range(15, 500, 18)))
        result = analyze_signal_density(entries, exits, 500)
        assert result["severity"] == "ok"
        assert result["estimated_trades"] >= 10
        assert len(result["entry_gaps"]) > 0


class TestAnalyzeSignalDensitySingleEntry:
    """Tests with exactly one entry signal."""

    def test_single_entry(self):
        entries = _make_entries(500, true_indices=[250])
        exits = _make_entries(500, true_indices=[260])
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_count"] == 1
        assert result["exit_count"] == 1
        assert result["signal_pattern"] == "single"
        assert result["entry_gaps"] == []
        assert result["avg_entry_gap"] == 0.0
        assert result["max_entry_gap"] == 0
        assert result["severity"] == "critical"
        assert result["diagnosis"] == "very_few_entries"


class TestAnalyzeSignalDensitySparseEntries:
    """Tests with 2-3 sparse entries."""

    def test_two_sparse_entries(self):
        entries = _make_entries(1000, true_indices=[100, 900])
        exits = _make_entries(1000, true_indices=[200, 950])
        result = analyze_signal_density(entries, exits, 1000)
        assert result["entry_count"] == 2
        assert result["signal_pattern"] == "sparse"
        assert result["entry_gaps"] == [800]
        assert result["avg_entry_gap"] == 800.0
        assert result["max_entry_gap"] == 800

    def test_three_sparse_entries(self):
        entries = _make_entries(1000, true_indices=[50, 500, 950])
        exits = _make_entries(1000, true_indices=[100, 550, 980])
        result = analyze_signal_density(entries, exits, 1000)
        assert result["entry_count"] == 3
        assert result["signal_pattern"] == "sparse"
        assert result["entry_gaps"] == [450, 450]


class TestAnalyzeSignalDensityUnevenEntries:
    """Tests where max gap >> avg gap (uneven pattern)."""

    def test_uneven_pattern(self):
        # 10 entries: 9 clustered within first 100 bars, then 1 at bar 900
        # Gaps: ~10, ~10, ..., 800 -> avg ~ 89, max 800, 800 > 89*5=445
        true_indices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 890]
        entries = _make_entries(1000, true_indices=true_indices)
        exits = _make_entries(1000, true_indices=true_indices)
        result = analyze_signal_density(entries, exits, 1000)
        assert result["entry_count"] == 10
        assert result["signal_pattern"] == "uneven"
        assert result["max_entry_gap"] > result["avg_entry_gap"] * 5


class TestAnalyzeSignalDensityRatesAndCounts:
    """Tests for correct rate and count calculations."""

    def test_entry_rate_calculation(self):
        entries = _make_entries(200, true_indices=list(range(0, 200, 10)))  # 20
        exits = _make_entries(200, true_indices=list(range(5, 200, 10)))  # 20
        result = analyze_signal_density(entries, exits, 200)
        assert result["entry_count"] == 20
        assert result["exit_count"] == 20
        assert result["entry_rate"] == pytest.approx(20 / 200)
        assert result["exit_rate"] == pytest.approx(20 / 200)
        assert result["estimated_trades"] == 20

    def test_estimated_trades_is_min(self):
        entries = _make_entries(500, true_indices=list(range(0, 100, 5)))  # 20
        exits = _make_entries(500, true_indices=list(range(0, 60, 5)))  # 12
        result = analyze_signal_density(entries, exits, 500)
        assert result["estimated_trades"] == min(20, 12)

    def test_zero_df_length(self):
        entries = _make_entries(0)
        exits = _make_entries(0)
        result = analyze_signal_density(entries, exits, 0)
        assert result["entry_count"] == 0
        assert result["entry_rate"] == 0.0

    def test_gaps_between_consecutive_entries(self):
        entries = _make_entries(500, true_indices=[10, 25, 50, 100])
        exits = _make_entries(500, true_indices=[15, 30, 55, 105])
        result = analyze_signal_density(entries, exits, 500)
        assert result["entry_gaps"] == [15, 25, 50]
        assert result["avg_entry_gap"] == pytest.approx(30.0)
        assert result["max_entry_gap"] == 50


class TestAnalyzeSignalDensityReturnStructure:
    """Tests that the return dict always has all expected keys."""

    EXPECTED_KEYS = [
        "entry_count", "exit_count", "entry_rate", "exit_rate",
        "estimated_trades", "diagnosis", "fix_suggestion", "severity",
        "signal_pattern", "entry_gaps", "avg_entry_gap", "max_entry_gap",
    ]

    def test_keys_present_on_none(self):
        result = analyze_signal_density(None, None, 100)
        for key in self.EXPECTED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_keys_present_on_zero(self):
        entries = _make_entries(100)
        exits = _make_entries(100)
        result = analyze_signal_density(entries, exits, 100)
        for key in self.EXPECTED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_keys_present_on_ok(self):
        entries = _make_entries(500, true_indices=list(range(10, 500, 18)))
        exits = _make_entries(500, true_indices=list(range(15, 500, 18)))
        result = analyze_signal_density(entries, exits, 500)
        for key in self.EXPECTED_KEYS:
            assert key in result, f"Missing key: {key}"


# ── format_signal_analysis_for_prompt ────────────────────────────────────────


class TestFormatSignalAnalysisOkSeverity:
    """format returns empty string for ok severity."""

    def test_ok_returns_empty(self):
        analysis = {
            "severity": "ok",
            "diagnosis": None,
            "fix_suggestion": None,
            "entry_count": 25,
            "exit_count": 25,
            "estimated_trades": 25,
            "signal_pattern": "regular",
            "avg_entry_gap": 20.0,
        }
        result = format_signal_analysis_for_prompt(analysis)
        assert result == ""

    def test_ok_with_extra_fields(self):
        analysis = {
            "severity": "ok",
            "entry_count": 30,
            "exit_count": 30,
            "estimated_trades": 30,
            "signal_pattern": "regular",
        }
        result = format_signal_analysis_for_prompt(analysis)
        assert result == ""

    def test_missing_severity_defaults_ok(self):
        analysis = {}
        result = format_signal_analysis_for_prompt(analysis)
        assert result == ""


class TestFormatSignalAnalysisCriticalSeverity:
    """format returns non-empty string for critical severity."""

    def test_critical_returns_nonempty(self):
        analysis = {
            "severity": "critical",
            "diagnosis": "zero_entries",
            "fix_suggestion": "Widen your thresholds.",
            "entry_count": 0,
            "exit_count": 0,
            "estimated_trades": 0,
            "signal_pattern": "none",
            "avg_entry_gap": 0.0,
        }
        result = format_signal_analysis_for_prompt(analysis)
        assert result != ""
        assert "zero_entries" in result
        assert "Widen your thresholds" in result
        assert "Signal Analysis" in result

    def test_critical_includes_entry_exit_counts(self):
        analysis = {
            "severity": "critical",
            "diagnosis": "zero_exits",
            "fix_suggestion": "Add exit conditions.",
            "entry_count": 10,
            "exit_count": 0,
            "estimated_trades": 0,
            "signal_pattern": "regular",
            "avg_entry_gap": 50.0,
        }
        result = format_signal_analysis_for_prompt(analysis)
        assert "10" in result
        assert "0" in result

    def test_critical_includes_avg_gap(self):
        analysis = {
            "severity": "critical",
            "diagnosis": "very_few_entries",
            "fix_suggestion": "Widen thresholds.",
            "entry_count": 3,
            "exit_count": 3,
            "estimated_trades": 3,
            "signal_pattern": "sparse",
            "avg_entry_gap": 100.0,
        }
        result = format_signal_analysis_for_prompt(analysis)
        assert "100" in result

    def test_warning_severity(self):
        analysis = {
            "severity": "warning",
            "diagnosis": "excessive_entries",
            "fix_suggestion": "Tighten thresholds.",
            "entry_count": 200,
            "exit_count": 200,
            "estimated_trades": 200,
            "signal_pattern": "regular",
            "avg_entry_gap": 2.0,
        }
        result = format_signal_analysis_for_prompt(analysis)
        assert result != ""
        assert "excessive_entries" in result


# ── check_signal_density_early_exit ─────────────────────────────────────────


class TestCheckEarlyExitNone:
    """check returns (True, ...) for None inputs."""

    def test_none_entries(self):
        should_skip, reason = check_signal_density_early_exit(
            None, pd.Series(False, index=range(100)), 100
        )
        assert should_skip is True
        assert "None" in reason

    def test_none_exits(self):
        should_skip, reason = check_signal_density_early_exit(
            pd.Series(False, index=range(100)), None, 100
        )
        assert should_skip is True
        assert "None" in reason

    def test_both_none(self):
        should_skip, reason = check_signal_density_early_exit(None, None, 100)
        assert should_skip is True


class TestCheckEarlyExitNonSeries:
    """check returns (True, ...) for non-Series entries."""

    def test_list_entries(self):
        should_skip, reason = check_signal_density_early_exit(
            [True, False], pd.Series(False, index=range(2)), 2
        )
        assert should_skip is True
        assert "list" in reason

    def test_numpy_entries(self):
        should_skip, reason = check_signal_density_early_exit(
            np.array([True, False]),
            pd.Series(False, index=range(2)),
            2,
        )
        assert should_skip is True
        assert "ndarray" in reason

    def test_int_entries(self):
        should_skip, reason = check_signal_density_early_exit(
            42, pd.Series(False, index=range(2)), 2
        )
        assert should_skip is True
        assert "int" in reason


class TestCheckEarlyExitAllNaN:
    """check returns (True, ...) when most entries are NaN."""

    def test_majority_nan(self):
        # Need some True entries so we don't hit zero_entries first,
        # but majority NaN to trigger the NaN check.
        entries = pd.Series([True] * 5 + [np.nan] * 85 + [False] * 10)
        exits = pd.Series([True] * 5 + [False] * 95)
        should_skip, reason = check_signal_density_early_exit(entries, exits, 100)
        # 85% NaN > 50% threshold, but 5 entries exist so zero check doesn't trigger
        assert should_skip is True
        assert "NaN" in reason

    def test_exactly_half_nan(self):
        # 50% NaN — NOT > 50%, so should not trigger NaN check
        # But entries are all False after fill, so zero entries triggers
        entries = pd.Series([np.nan] * 50 + [False] * 50)
        exits = pd.Series([np.nan] * 50 + [False] * 50)
        should_skip, reason = check_signal_density_early_exit(entries, exits, 100)
        assert should_skip is True
        assert "zero" in reason.lower()


class TestCheckEarlyExitZeroEntries:
    """check returns (True, ...) for zero entry signals."""

    def test_all_false_entries(self):
        entries = _make_entries(200)
        exits = _make_entries(200)
        should_skip, reason = check_signal_density_early_exit(entries, exits, 200)
        assert should_skip is True
        assert "zero" in reason.lower()

    def test_all_nan_entries(self):
        entries = pd.Series([np.nan] * 100)
        exits = pd.Series([np.nan] * 100)
        should_skip, reason = check_signal_density_early_exit(entries, exits, 100)
        assert should_skip is True


class TestCheckEarlyExitEntriesButZeroExits:
    """check returns (True, ...) when entries exist but no exits."""

    def test_entries_no_exits(self):
        entries = _make_entries(500, true_indices=[10, 50, 100])
        exits = _make_entries(500)
        should_skip, reason = check_signal_density_early_exit(entries, exits, 500)
        assert should_skip is True
        assert "0 exits" in reason or "never close" in reason.lower()


class TestCheckEarlyExitNormalSignals:
    """check returns (False, "") for normal, healthy signals."""

    def test_normal_signals_pass(self):
        entries = _make_entries(500, true_indices=list(range(10, 500, 20)))
        exits = _make_entries(500, true_indices=list(range(15, 500, 20)))
        should_skip, reason = check_signal_density_early_exit(entries, exits, 500)
        assert should_skip is False
        assert reason == ""

    def test_few_but_valid_signals(self):
        # check_early_exit only catches critical issues; 5 entries + 5 exits is fine
        entries = _make_entries(500, true_indices=[10, 100, 200, 300, 400])
        exits = _make_entries(500, true_indices=[20, 110, 210, 310, 410])
        should_skip, reason = check_signal_density_early_exit(entries, exits, 500)
        assert should_skip is False
        assert reason == ""
