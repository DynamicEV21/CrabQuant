"""Tests for crabquant.refinement.classifier.classify_failure."""

import pytest
from crabquant.engine.backtest import BacktestResult
from crabquant.guardrails import GuardrailConfig, GuardrailReport
from crabquant.refinement.classifier import classify_failure


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_result(**overrides) -> BacktestResult:
    """Factory with sensible passing defaults."""
    defaults = dict(
        ticker="SPY",
        strategy_name="test",
        iteration=0,
        sharpe=1.8,
        total_return=0.20,
        max_drawdown=-0.10,
        win_rate=0.55,
        num_trades=50,
        avg_trade_return=0.02,
        calmar_ratio=1.5,
        sortino_ratio=2.0,
        profit_factor=1.5,
        avg_holding_bars=5.0,
        best_trade=500.0,
        worst_trade=-200.0,
        passed=True,
        score=1.2,
        notes="ok",
        params={},
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def make_guardrails(passed=True, violations=None) -> GuardrailReport:
    return GuardrailReport(
        passed=passed,
        violations=violations or [],
        warnings=[],
        score_adjustment=0.0,
    )


GOOD_SHARPE_BY_YEAR = {"2022": 1.5, "2023": 1.8, "2024": 2.0}


# ── 1. too_few_trades ────────────────────────────────────────────────────────

def test_too_few_trades_zero():
    result = make_result(num_trades=0, sharpe=0.0, total_return=0.0)
    mode, details = classify_failure(result, make_guardrails(), {})
    assert mode == "too_few_trades"
    assert "0" in details


def test_too_few_trades_four():
    result = make_result(num_trades=4)
    mode, details = classify_failure(result, make_guardrails(), {})
    assert mode == "too_few_trades"
    assert "4" in details


def test_too_few_trades_exactly_nineteen():
    """19 trades should trigger too_few_trades (threshold is 20)."""
    result = make_result(num_trades=19, sharpe=2.0)
    mode, details = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode == "too_few_trades"
    assert "19" in details


def test_too_few_trades_exactly_twenty_passes_through():
    """20 trades should NOT trigger too_few_trades."""
    result = make_result(num_trades=20, sharpe=2.0)
    mode, _ = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode != "too_few_trades"


# ── 2. flat_signal ───────────────────────────────────────────────────────────

def test_flat_signal_zero_return_and_zero_sharpe():
    """25+ trades but zero return and zero sharpe → flat_signal."""
    result = make_result(num_trades=25, total_return=0.0, sharpe=0.0)
    mode, details = classify_failure(result, make_guardrails(), {})
    assert mode == "flat_signal"


def test_flat_signal_not_triggered_with_nonzero_return():
    """If return is nonzero, flat_signal should not trigger."""
    result = make_result(num_trades=25, total_return=0.05, sharpe=0.5)
    mode, _ = classify_failure(result, make_guardrails(), {})
    assert mode != "flat_signal"


# ── 3. excessive_drawdown ────────────────────────────────────────────────────

def test_excessive_drawdown_below_threshold():
    result = make_result(max_drawdown=-0.35)
    mode, details = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode == "excessive_drawdown"
    assert "-35" in details or "35" in details


def test_excessive_drawdown_at_threshold_not_triggered():
    """-0.30 exactly should NOT trigger excessive_drawdown."""
    result = make_result(max_drawdown=-0.30, sharpe=0.5)
    mode, _ = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode != "excessive_drawdown"


def test_excessive_drawdown_just_above_threshold_not_triggered():
    """-0.29 should NOT trigger excessive_drawdown."""
    result = make_result(max_drawdown=-0.29, sharpe=0.5)
    mode, _ = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode != "excessive_drawdown"


# ── 4. regime_fragility ──────────────────────────────────────────────────────

def test_regime_fragility_high_range_with_negative():
    """Range > 2.5 AND has negative year → regime_fragility."""
    sharpe_by_year = {"2022": -0.5, "2023": 2.2}  # range = 2.7, has negative
    result = make_result()
    mode, details = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode == "regime_fragility"
    assert "2.7" in details or "regime" in details.lower()


def test_regime_fragility_very_high_range_no_negative_but_min_below_03():
    """Range > 3.0 AND min < 0.3 → regime_fragility (even without negative)."""
    sharpe_by_year = {"2022": 0.2, "2023": 3.5}  # range = 3.3, min = 0.2
    result = make_result()
    mode, details = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode == "regime_fragility"


def test_regime_fragility_not_triggered_small_range():
    """Range <= 2.5 → no regime_fragility."""
    sharpe_by_year = {"2022": 1.0, "2023": 3.0}  # range = 2.0, no negative
    result = make_result(sharpe=2.0)
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode != "regime_fragility"


def test_regime_fragility_not_triggered_high_range_all_positive():
    """Range > 2.5 but all positive and min >= 0.3 → no regime_fragility."""
    sharpe_by_year = {"2022": 0.5, "2023": 3.2}  # range = 2.7, no negative, min = 0.5 > 0.3
    result = make_result(sharpe=0.8)
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode != "regime_fragility"


def test_regime_fragility_requires_at_least_two_years():
    """Single year → no regime_fragility check."""
    sharpe_by_year = {"2022": -2.0}
    result = make_result(sharpe=0.8)
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode != "regime_fragility"


def test_regime_fragility_empty_dict():
    """Empty dict → no regime_fragility check."""
    result = make_result(sharpe=0.8)
    mode, _ = classify_failure(result, make_guardrails(), {})
    assert mode != "regime_fragility"


# ── 5. overtrading ───────────────────────────────────────────────────────────

def test_overtrading_more_than_half_of_bars():
    """Trades > 50% of bars → overtrading."""
    result = make_result(num_trades=300)
    mode, details = classify_failure(result, make_guardrails(), {}, data_length=500)
    assert mode == "overtrading"
    assert "300" in details


def test_overtrading_exactly_half_not_triggered():
    """Trades == 50% of bars → NOT overtrading."""
    result = make_result(num_trades=250, sharpe=0.5)
    mode, _ = classify_failure(result, make_guardrails(), {}, data_length=500)
    assert mode != "overtrading"


def test_overtrading_data_length_zero_skipped():
    """data_length=0 → overtrading check skipped."""
    result = make_result(num_trades=9999, sharpe=0.5)
    mode, _ = classify_failure(result, make_guardrails(), {}, data_length=0)
    assert mode != "overtrading"


# ── 6. low_sharpe ────────────────────────────────────────────────────────────

def test_low_sharpe_default_target():
    result = make_result(sharpe=1.2, total_return=0.15, max_drawdown=-0.10)
    mode, details = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode == "low_sharpe"
    assert "1.2" in details or "1.20" in details


def test_low_sharpe_custom_target():
    result = make_result(sharpe=0.8)
    mode, details = classify_failure(
        result, make_guardrails(), GOOD_SHARPE_BY_YEAR, sharpe_target=1.0
    )
    assert mode == "low_sharpe"


def test_low_sharpe_catchall_when_strategy_passes():
    """When no failure mode matches, returns low_sharpe catchall."""
    result = make_result(sharpe=2.0)
    mode, _ = classify_failure(
        result, make_guardrails(), GOOD_SHARPE_BY_YEAR, sharpe_target=1.5
    )
    # sharpe=2.0 >= target=1.5 → hits the safety catchall
    assert mode == "low_sharpe"


# ── 7. Priority ordering ─────────────────────────────────────────────────────

def test_priority_too_few_trades_before_flat_signal():
    """0 trades satisfies both too_few_trades and flat_signal, but too_few wins."""
    result = make_result(num_trades=0, total_return=0.0, sharpe=0.0)
    mode, _ = classify_failure(result, make_guardrails(), {})
    assert mode == "too_few_trades"


def test_priority_too_few_trades_before_excessive_drawdown():
    """3 trades AND severe drawdown → too_few_trades wins."""
    result = make_result(num_trades=3, max_drawdown=-0.50)
    mode, _ = classify_failure(result, make_guardrails(), {})
    assert mode == "too_few_trades"


def test_priority_flat_signal_before_excessive_drawdown():
    """25+ trades, zero return+sharpe, AND severe drawdown → flat_signal wins."""
    result = make_result(num_trades=25, total_return=0.0, sharpe=0.0, max_drawdown=-0.50)
    mode, _ = classify_failure(result, make_guardrails(), {})
    assert mode == "flat_signal"


def test_priority_excessive_drawdown_before_overtrading():
    """Severe drawdown AND overtrading → excessive_drawdown wins."""
    result = make_result(num_trades=300, max_drawdown=-0.40)
    mode, _ = classify_failure(
        result, make_guardrails(), GOOD_SHARPE_BY_YEAR, data_length=500
    )
    assert mode == "excessive_drawdown"


def test_priority_regime_fragility_before_overtrading():
    """Regime fragility AND overtrading → regime_fragility wins."""
    sharpe_by_year = {"2022": -1.0, "2023": 2.0}  # range=3.0, has_negative
    result = make_result(num_trades=300)
    mode, _ = classify_failure(
        result, make_guardrails(), sharpe_by_year, data_length=500
    )
    assert mode == "regime_fragility"


def test_priority_overtrading_before_low_sharpe():
    """Overtrading AND low sharpe → overtrading wins."""
    result = make_result(num_trades=300, sharpe=0.5)
    mode, _ = classify_failure(result, make_guardrails(), {}, data_length=500)
    assert mode == "overtrading"


# ── 8. Return type ───────────────────────────────────────────────────────────

def test_returns_tuple_of_two_strings():
    result = make_result(num_trades=3)
    output = classify_failure(result, make_guardrails(), {})
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], str)
    assert isinstance(output[1], str)


def test_failure_mode_is_valid_string():
    valid_modes = {
        "too_few_trades", "flat_signal", "excessive_drawdown",
        "regime_fragility", "overtrading", "low_sharpe",
    }
    result = make_result(num_trades=3)
    mode, _ = classify_failure(result, make_guardrails(), {})
    assert mode in valid_modes
