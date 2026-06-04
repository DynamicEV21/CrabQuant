"""Tests for crabquant.refinement.regime_diagnosis — Regime Diagnosis System.

Tests cover:
- Pattern classification for various Sharpe distributions
- Year-by-year breakdown formatting
- Specific actionable fixes for known year regimes
- Edge cases (empty data, single year, all same Sharpe)
- Integration with build_failure_guidance
- Integration with format_previous_attempts_section
"""

import pytest


# ─── Pattern Classification ─────────────────────────────────────────────


class TestClassifyRegimePattern:
    """Test _classify_regime_pattern returns correct pattern labels."""

    def test_empty_data(self):
        from crabquant.refinement.regime_diagnosis import _classify_regime_pattern
        assert _classify_regime_pattern({}) == "unknown"

    def test_single_year(self):
        from crabquant.refinement.regime_diagnosis import _classify_regime_pattern
        assert _classify_regime_pattern({"2023": 1.5}) == "unknown"

    def test_always_losing(self):
        from crabquant.refinement.regime_diagnosis import _classify_regime_pattern
        result = _classify_regime_pattern({
            "2021": -0.5, "2022": -1.2, "2023": -0.3
        })
        assert result == "always_losing"

    def test_single_year_fluke(self):
        from crabquant.refinement.regime_diagnosis import _classify_regime_pattern
        # Only 2021 is good (>= 1.0), others are bad (< 0.3)
        result = _classify_regime_pattern({
            "2020": 0.2, "2021": 2.5, "2022": -0.5, "2023": 0.1
        })
        assert result == "single_year_fluke"

    def test_volatile_adverse(self):
        from crabquant.refinement.regime_diagnosis import _classify_regime_pattern
        # Negative only in 2018, 2020, 2022 (volatile years) — 3/5 = 60%
        result = _classify_regime_pattern({
            "2018": -1.5, "2019": 2.0, "2020": -2.0, "2021": 1.8, "2022": -1.0
        })
        assert result == "volatile_adverse"

    def test_calm_adverse(self):
        from crabquant.refinement.regime_diagnosis import _classify_regime_pattern
        # Negative only in calm years (2019, 2021) — 2/4 = 50%, volatile years positive
        result = _classify_regime_pattern({
            "2018": 0.5, "2019": -0.5, "2020": 0.8, "2021": -0.3
        })
        assert result == "calm_adverse"

    def test_time_decay(self):
        from crabquant.refinement.regime_diagnosis import _classify_regime_pattern
        result = _classify_regime_pattern({
            "2019": 2.0, "2020": 1.5, "2021": 0.5, "2022": -0.5, "2023": -1.0
        })
        assert result == "time_decay"

    def test_time_improvement(self):
        from crabquant.refinement.regime_diagnosis import _classify_regime_pattern
        result = _classify_regime_pattern({
            "2019": -1.0, "2020": -0.5, "2021": 0.5, "2022": 1.5, "2023": 2.0
        })
        assert result == "time_improvement"

    def test_mostly_good_few_bad(self):
        from crabquant.refinement.regime_diagnosis import _classify_regime_pattern
        # Negative years are calm years (2021, 2023), not volatile years
        result = _classify_regime_pattern({
            "2019": 1.5, "2020": 1.2, "2021": -0.3, "2022": 1.8, "2023": -0.2
        })
        assert result == "mostly_good_few_bad"

    def test_mostly_bad(self):
        from crabquant.refinement.regime_diagnosis import _classify_regime_pattern
        # Only 1 positive out of 5 — pos_frac = 0.2 < 0.4
        result = _classify_regime_pattern({
            "2019": -0.5, "2020": 0.5, "2021": -0.3, "2022": -1.0, "2023": -0.4
        })
        assert result == "mostly_bad"

    def test_mixed(self):
        from crabquant.refinement.regime_diagnosis import _classify_regime_pattern
        # 50/50 positive/negative with mixed volatile/calm negative years
        result = _classify_regime_pattern({
            "2019": 0.5, "2020": -0.3, "2021": -0.2, "2022": 0.3
        })
        assert result == "mixed"


# ─── Main Diagnosis Function ────────────────────────────────────────────


class TestDiagnoseRegimeFragility:
    """Test the main diagnose_regime_fragility function."""

    def test_empty_input(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        assert diagnose_regime_fragility({}) == ""
        assert diagnose_regime_fragility(None) == ""
        assert diagnose_regime_fragility({"2023": 1.0}) == ""

    def test_volatile_adverse_diagnosis(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {
            "2019": 2.0, "2020": -2.5, "2021": 1.8, "2022": -1.5
        }
        result = diagnose_regime_fragility(sby)
        assert "Regime Fragility Diagnosis" in result
        assert "volatile" in result.lower()
        assert "2020" in result
        assert "2022" in result
        assert "VOLATILITY FILTER" in result
        assert "LOSING" in result
        assert "STRONG" in result

    def test_single_year_fluke_diagnosis(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {
            "2020": 0.2, "2021": 3.5, "2022": -0.3, "2023": 0.1
        }
        result = diagnose_regime_fragility(sby)
        assert "fluke" in result.lower()
        assert "2021" in result
        assert "curve" in result.lower()

    def test_always_losing_diagnosis(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {
            "2020": -0.5, "2021": -1.2, "2022": -0.8
        }
        result = diagnose_regime_fragility(sby)
        assert "ALL years" in result
        assert "fundamentally" in result.lower()

    def test_mostly_good_diagnosis(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {
            "2019": 1.5, "2020": 1.2, "2021": 1.8, "2022": -0.5, "2023": 1.0
        }
        result = diagnose_regime_fragility(sby)
        assert "Mostly good" in result
        assert "2022" in result

    def test_year_by_year_breakdown(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {
            "2020": -2.0, "2021": 1.5, "2022": -1.0
        }
        result = diagnose_regime_fragility(sby)
        # Check each year appears with its status
        assert "2020: Sharpe -2.00 [❌ LOSING]" in result
        assert "2021: Sharpe +1.50 [✅ STRONG]" in result
        assert "2022: Sharpe -1.00 [❌ LOSING]" in result

    def test_summary_stats(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {
            "2020": -1.0, "2021": 2.0, "2022": -0.5, "2023": 1.5
        }
        result = diagnose_regime_fragility(sby)
        assert "Positive years: 2/4" in result
        assert "Negative years: 2/4" in result
        assert "Worst year: 2020" in result
        assert "Best year: 2021" in result

    def test_sharpe_range_in_output(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {"2021": 3.0, "2022": -0.5}
        result = diagnose_regime_fragility(sby)
        assert "Sharpe range: 3.5" in result

    def test_custom_sharpe_range(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {"2021": 3.0, "2022": -0.5}
        result = diagnose_regime_fragility(sby, sharpe_range=99.9)
        assert "Sharpe range: 99.9" in result

    def test_weak_year_status(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {"2021": 0.3, "2022": 0.1}
        result = diagnose_regime_fragility(sby)
        assert "WEAK" in result

    def test_ok_year_status(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {"2021": 0.7, "2022": 0.5}
        result = diagnose_regime_fragility(sby)
        assert "🔶 OK" in result

    def test_known_year_regimes_appear(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {"2022": -1.5, "2023": 2.0}
        result = diagnose_regime_fragility(sby)
        assert "Rate hikes" in result or "bear" in result.lower()

    def test_unknown_year_graceful(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {"2015": -1.0, "2016": 2.0}
        result = diagnose_regime_fragility(sby)
        # Should not crash and should still produce output
        assert "Regime Fragility Diagnosis" in result

    def test_pattern_label_in_output(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {"2020": -2.0, "2021": 1.5}
        result = diagnose_regime_fragility(sby)
        assert "Pattern:" in result

    def test_specific_year_fixes_for_negative_years(self):
        from crabquant.refinement.regime_diagnosis import diagnose_regime_fragility
        sby = {"2022": -1.5, "2023": 2.0}
        result = diagnose_regime_fragility(sby)
        assert "Specific fixes for losing years" in result
        assert "2022" in result


# ─── Integration: build_failure_guidance ────────────────────────────────


class TestBuildFailureGuidanceRegimeIntegration:
    """Test that build_failure_guidance includes regime diagnosis."""

    def test_regime_fragility_includes_diagnosis(self):
        from crabquant.refinement.prompts import build_failure_guidance
        sby = {
            "2020": -2.0, "2021": 1.5, "2022": -1.5
        }
        result = build_failure_guidance(
            "regime_fragility", total_trades=40, sharpe_by_year=sby
        )
        assert "Regime Fragility Diagnosis" in result
        assert "2020" in result
        assert "2022" in result

    def test_regime_fragility_fallback_without_year_data(self):
        from crabquant.refinement.prompts import build_failure_guidance
        result = build_failure_guidance(
            "regime_fragility", total_trades=40, sharpe_by_year=None
        )
        assert "regime detection" in result.lower()

    def test_regime_fragility_fallback_empty_dict(self):
        from crabquant.refinement.prompts import build_failure_guidance
        result = build_failure_guidance(
            "regime_fragility", total_trades=40, sharpe_by_year={}
        )
        assert "regime detection" in result.lower()

    def test_other_failure_modes_unchanged(self):
        from crabquant.refinement.prompts import build_failure_guidance
        result = build_failure_guidance("low_sharpe", total_trades=50)
        assert "Regime Fragility Diagnosis" not in result
        assert "Improve Sharpe" in result

    def test_regime_fragility_with_volatile_pattern(self):
        from crabquant.refinement.prompts import build_failure_guidance
        sby = {
            "2018": -1.5, "2019": 2.0, "2020": -2.0, "2021": 1.8, "2022": -1.0
        }
        result = build_failure_guidance(
            "regime_fragility", total_trades=50, sharpe_by_year=sby
        )
        assert "VOLATILITY FILTER" in result
        assert "2018" in result
        assert "2020" in result
        assert "2022" in result


# ─── Integration: format_previous_attempts_section ─────────────────────


class TestFormatPreviousAttemptsRegimeIntegration:
    """Test that format_previous_attempts_section shows regime-specific notes."""

    def test_regime_note_without_year_data(self):
        from crabquant.refinement.prompts import format_previous_attempts_section
        attempts = [{
            "turn": 3, "sharpe": 0.8, "failure_mode": "regime_fragility",
            "action": "change_entry_logic", "hypothesis": "h3", "params_used": {},
            "delta_from_prev": "Changed entry",
        }]
        result = format_previous_attempts_section(attempts)
        assert "REGIME WARNING" in result
        assert "specific market conditions" in result
        assert "regime detection" in result

    def test_regime_note_with_negative_years(self):
        from crabquant.refinement.prompts import format_previous_attempts_section
        attempts = [{
            "turn": 3, "sharpe": 0.8, "failure_mode": "regime_fragility",
            "action": "change_entry_logic", "hypothesis": "h3", "params_used": {},
            "delta_from_prev": "Changed entry",
            "sharpe_by_year": {"2020": -1.5, "2021": 2.0, "2022": -0.8},
        }]
        result = format_previous_attempts_section(attempts)
        assert "REGIME WARNING" in result
        assert "2020" in result
        assert "2022" in result

    def test_regime_note_with_no_negative_years(self):
        from crabquant.refinement.prompts import format_previous_attempts_section
        attempts = [{
            "turn": 3, "sharpe": 0.8, "failure_mode": "regime_fragility",
            "action": "change_entry_logic", "hypothesis": "h3", "params_used": {},
            "delta_from_prev": "Changed entry",
            "sharpe_by_year": {"2020": 0.2, "2021": 1.5, "2022": 0.3},
        }]
        result = format_previous_attempts_section(attempts)
        assert "REGIME WARNING" in result
        assert "Weakest" in result

    def test_non_regime_failure_unchanged(self):
        from crabquant.refinement.prompts import format_previous_attempts_section
        attempts = [{
            "turn": 3, "sharpe": 0.8, "failure_mode": "low_sharpe",
            "action": "change_entry_logic", "hypothesis": "h3", "params_used": {},
            "delta_from_prev": "Changed entry",
        }]
        result = format_previous_attempts_section(attempts)
        assert "REGIME WARNING" not in result


# ─── Year Detail Builder ────────────────────────────────────────────────


class TestBuildYearDetail:
    """Test _build_year_detail formatting."""

    def test_negative_sharpe(self):
        from crabquant.refinement.regime_diagnosis import _build_year_detail
        result = _build_year_detail("2022", -1.5)
        assert "❌ LOSING" in result

    def test_strong_sharpe(self):
        from crabquant.refinement.regime_diagnosis import _build_year_detail
        result = _build_year_detail("2021", 2.0)
        assert "✅ STRONG" in result

    def test_weak_sharpe(self):
        from crabquant.refinement.regime_diagnosis import _build_year_detail
        result = _build_year_detail("2020", 0.2)
        assert "⚠️ WEAK" in result

    def test_ok_sharpe(self):
        from crabquant.refinement.regime_diagnosis import _build_year_detail
        result = _build_year_detail("2023", 0.7)
        assert "🔶 OK" in result

    def test_unknown_year(self):
        from crabquant.refinement.regime_diagnosis import _build_year_detail
        result = _build_year_detail("2015", 1.0)
        assert "2015" in result
        assert "unknown market conditions" in result

    def test_known_year_character(self):
        from crabquant.refinement.regime_diagnosis import _build_year_detail
        result = _build_year_detail("2022", -1.0)
        assert "Rate hikes" in result
