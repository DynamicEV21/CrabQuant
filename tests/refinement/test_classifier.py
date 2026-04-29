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


def test_too_few_trades_exactly_five_passes_through():
    """5 trades should NOT trigger too_few_trades."""
    result = make_result(num_trades=5, sharpe=2.0)
    mode, _ = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode != "too_few_trades"


def test_too_few_trades_single_trade():
    """1 trade → too_few_trades."""
    result = make_result(num_trades=1, sharpe=0.5)
    mode, details = classify_failure(result, make_guardrails(), {})
    assert mode == "too_few_trades"
    assert "1" in details


def test_too_few_trades_negative_count():
    """Negative num_trades is still < 5 → too_few_trades."""
    result = make_result(num_trades=-3, sharpe=0.5)
    mode, details = classify_failure(result, make_guardrails(), {})
    assert mode == "too_few_trades"
    assert "-3" in details


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_too_few_trades_all_values_below_five(n):
    """Every value 0-4 triggers too_few_trades."""
    result = make_result(num_trades=n, sharpe=0.5)
    mode, details = classify_failure(result, make_guardrails(), {})
    assert mode == "too_few_trades"
    assert str(n) in details


def test_too_few_trades_details_mentions_minimum():
    """Detail string should mention the minimum (5)."""
    result = make_result(num_trades=2)
    _, details = classify_failure(result, make_guardrails(), {})
    assert "min 5" in details


# ── 2. flat_signal ───────────────────────────────────────────────────────────

def test_flat_signal_zero_return_and_zero_sharpe():
    """5+ trades but zero return and zero sharpe → flat_signal."""
    result = make_result(num_trades=10, total_return=0.0, sharpe=0.0)
    mode, details = classify_failure(result, make_guardrails(), {})
    assert mode == "flat_signal"


def test_flat_signal_not_triggered_with_nonzero_return():
    """If return is nonzero, flat_signal should not trigger."""
    result = make_result(num_trades=10, total_return=0.05, sharpe=0.5)
    mode, _ = classify_failure(result, make_guardrails(), {})
    assert mode != "flat_signal"


def test_flat_signal_nonzero_return_zero_sharpe():
    """Nonzero return but zero sharpe alone → NOT flat_signal (need both zero)."""
    result = make_result(num_trades=10, total_return=0.05, sharpe=0.0)
    mode, _ = classify_failure(result, make_guardrails(), {})
    assert mode != "flat_signal"


def test_flat_signal_zero_return_nonzero_sharpe():
    """Zero return but nonzero sharpe → NOT flat_signal (need both zero)."""
    result = make_result(num_trades=10, total_return=0.0, sharpe=1.0)
    mode, _ = classify_failure(result, make_guardrails(), {})
    assert mode != "flat_signal"


def test_flat_signal_details_message():
    """Detail string should mention zero meaningful signals."""
    result = make_result(num_trades=10, total_return=0.0, sharpe=0.0)
    _, details = classify_failure(result, make_guardrails(), {})
    assert "zero" in details.lower()


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


def test_excessive_drawdown_very_extreme():
    """Very extreme drawdown (-0.99) still classified as excessive_drawdown."""
    result = make_result(max_drawdown=-0.99)
    mode, details = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode == "excessive_drawdown"


def test_excessive_drawdown_just_past_threshold():
    """-0.31 is just past the -0.30 threshold → excessive_drawdown."""
    result = make_result(max_drawdown=-0.31)
    mode, details = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode == "excessive_drawdown"


def test_excessive_drawdown_micro_above_threshold():
    """-0.30001 is microscopically past threshold → excessive_drawdown."""
    result = make_result(max_drawdown=-0.30001)
    mode, _ = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode == "excessive_drawdown"


def test_excessive_drawdown_micro_below_threshold():
    """-0.29999 is microscopically below threshold → NOT excessive_drawdown."""
    result = make_result(max_drawdown=-0.29999, sharpe=0.5)
    mode, _ = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode != "excessive_drawdown"


def test_excessive_drawdown_details_mentions_threshold():
    """Detail string should mention the 30% threshold."""
    result = make_result(max_drawdown=-0.35)
    _, details = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert "30%" in details or "30" in details


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


def test_regime_fragility_three_years_with_instability():
    """3 years with wide range → regime_fragility."""
    sharpe_by_year = {"2021": -1.0, "2022": 0.5, "2023": 2.5}  # range=3.5, has negative
    result = make_result()
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode == "regime_fragility"


def test_regime_fragility_boundary_range_exactly_2_5_with_negative():
    """Range exactly 2.5 with negative → NOT triggered (needs > 2.5)."""
    sharpe_by_year = {"2022": -0.5, "2023": 2.0}  # range = 2.5
    result = make_result(sharpe=0.8)
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode != "regime_fragility"


def test_regime_fragility_boundary_range_exactly_3_0_min_below_03():
    """Range exactly 3.0 with min < 0.3 → NOT triggered (needs > 3.0)."""
    sharpe_by_year = {"2022": 0.2, "2023": 3.2}  # range = 3.0
    result = make_result(sharpe=0.8)
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode != "regime_fragility"


def test_regime_fragility_all_years_negative():
    """All years negative → regime_fragility (has_negative + range > 2.5)."""
    sharpe_by_year = {"2022": -3.0, "2023": -0.5}  # range = 2.5, has negative
    result = make_result()
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    # range = 2.5 is NOT > 2.5, so this should NOT trigger
    assert mode != "regime_fragility"


def test_regime_fragility_all_years_negative_wide_range():
    """All years negative with wide range → regime_fragility."""
    sharpe_by_year = {"2022": -3.0, "2023": -0.2}  # range = 2.8, has negative
    result = make_result()
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode == "regime_fragility"


def test_regime_fragility_many_years():
    """5 years with instability → regime_fragility."""
    sharpe_by_year = {
        "2020": -1.0, "2021": 0.5, "2022": 1.0, "2023": 1.5, "2024": 2.5
    }
    result = make_result()
    mode, details = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode == "regime_fragility"


def test_regime_fragility_details_contains_year_keys():
    """Details string should contain the year labels from sharpe_by_year."""
    sharpe_by_year = {"2022": -0.5, "2023": 2.2}
    result = make_result()
    _, details = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert "2022" in details
    assert "2023" in details


def test_regime_fragility_both_conditions_simultaneously():
    """Both conditions (range>2.5+neg AND range>3.0+min<0.3) → still regime_fragility."""
    sharpe_by_year = {"2022": -0.2, "2023": 3.5}  # range=3.7, has_negative, min=-0.2
    result = make_result()
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode == "regime_fragility"


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


def test_overtrading_just_over_threshold():
    """251 trades on 500 bars → just over 50% → overtrading."""
    result = make_result(num_trades=251)
    mode, _ = classify_failure(result, make_guardrails(), {}, data_length=500)
    assert mode == "overtrading"


def test_overtrading_very_small_data_length():
    """5 trades on 9 bars → overtrading (55% > 50%)."""
    result = make_result(num_trades=5, sharpe=0.5)
    mode, _ = classify_failure(result, make_guardrails(), {}, data_length=9)
    assert mode == "overtrading"


def test_overtrading_negative_data_length():
    """Negative data_length → overtrading check skipped (data_length > 0 is false)."""
    result = make_result(num_trades=9999, sharpe=0.5)
    mode, _ = classify_failure(result, make_guardrails(), {}, data_length=-1)
    assert mode != "overtrading"


def test_overtrading_details_contains_transaction_costs():
    """Detail string should mention transaction costs."""
    result = make_result(num_trades=300)
    _, details = classify_failure(result, make_guardrails(), {}, data_length=500)
    assert "transaction cost" in details.lower()


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


def test_low_sharpe_exactly_at_target():
    """Sharpe exactly equal to target → catchall branch (not "below target")."""
    result = make_result(sharpe=1.5)
    mode, details = classify_failure(
        result, make_guardrails(), GOOD_SHARPE_BY_YEAR, sharpe_target=1.5
    )
    assert mode == "low_sharpe"
    # sharpe == target → should be the else branch
    assert "Below target but no specific" in details


def test_low_sharpe_negative_value():
    """Negative sharpe → low_sharpe with details."""
    result = make_result(sharpe=-1.5, num_trades=10, total_return=-0.20, max_drawdown=-0.15)
    mode, details = classify_failure(result, make_guardrails(), {})
    assert mode == "low_sharpe"


def test_low_sharpe_zero():
    """Sharpe of exactly 0.0 → low_sharpe."""
    result = make_result(sharpe=0.0, num_trades=10)
    mode, details = classify_failure(result, make_guardrails(), {})
    assert mode == "low_sharpe"


def test_low_sharpe_very_high():
    """Very high sharpe above target → still low_sharpe (catchall)."""
    result = make_result(sharpe=10.0)
    mode, details = classify_failure(
        result, make_guardrails(), GOOD_SHARPE_BY_YEAR, sharpe_target=1.5
    )
    assert mode == "low_sharpe"
    assert "Below target but no specific" in details


def test_low_sharpe_details_below_target_contains_metrics():
    """When sharpe < target, details should contain return and drawdown info."""
    result = make_result(sharpe=0.5, total_return=0.10, max_drawdown=-0.08)
    mode, details = classify_failure(result, make_guardrails(), {})
    assert mode == "low_sharpe"
    assert "0.50" in details
    assert "target" in details.lower()


def test_low_sharpe_custom_data_length():
    """Custom data_length doesn't affect low_sharpe classification."""
    result = make_result(sharpe=0.8, num_trades=10)
    mode, _ = classify_failure(
        result, make_guardrails(), {}, data_length=1000, sharpe_target=1.0
    )
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
    """5+ trades, zero return+sharpe, AND severe drawdown → flat_signal wins."""
    result = make_result(num_trades=10, total_return=0.0, sharpe=0.0, max_drawdown=-0.50)
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


def test_priority_too_few_trades_before_overtrading():
    """Few trades AND overtrading → too_few_trades wins."""
    result = make_result(num_trades=2)
    mode, _ = classify_failure(result, make_guardrails(), {}, data_length=3)
    assert mode == "too_few_trades"


def test_priority_too_few_trades_before_regime_fragility():
    """Few trades AND regime fragility → too_few_trades wins."""
    sharpe_by_year = {"2022": -1.0, "2023": 2.0}
    result = make_result(num_trades=3)
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode == "too_few_trades"


def test_priority_flat_signal_before_overtrading():
    """Flat signal AND overtrading → flat_signal wins."""
    result = make_result(num_trades=300, total_return=0.0, sharpe=0.0)
    mode, _ = classify_failure(result, make_guardrails(), {}, data_length=500)
    assert mode == "flat_signal"


def test_priority_flat_signal_before_regime_fragility():
    """Flat signal AND regime fragility → flat_signal wins."""
    sharpe_by_year = {"2022": -1.0, "2023": 2.0}
    result = make_result(num_trades=10, total_return=0.0, sharpe=0.0)
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode == "flat_signal"


def test_priority_excessive_drawdown_before_regime_fragility():
    """Excessive drawdown AND regime fragility → excessive_drawdown wins."""
    sharpe_by_year = {"2022": -1.0, "2023": 2.0}
    result = make_result(max_drawdown=-0.40)
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode == "excessive_drawdown"


def test_priority_excessive_drawdown_before_low_sharpe():
    """Excessive drawdown AND low sharpe → excessive_drawdown wins."""
    result = make_result(max_drawdown=-0.40, sharpe=0.3)
    mode, _ = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode == "excessive_drawdown"


def test_priority_regime_fragility_before_low_sharpe():
    """Regime fragility AND low sharpe → regime_fragility wins."""
    sharpe_by_year = {"2022": -1.0, "2023": 2.0}
    result = make_result(sharpe=0.3)
    mode, _ = classify_failure(result, make_guardrails(), sharpe_by_year)
    assert mode == "regime_fragility"


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


# ── 9. Integration / misc ────────────────────────────────────────────────────

def test_with_guardrails_violations():
    """Guardrails with violations should not affect classification logic."""
    gr = make_guardrails(passed=False, violations=["max_drawdown_exceeded"])
    result = make_result(num_trades=2)
    mode, _ = classify_failure(result, gr, {})
    # Classification doesn't inspect guardrails — still too_few_trades
    assert mode == "too_few_trades"


def test_with_guardrails_warnings():
    """Guardrails with warnings should not affect classification."""
    gr = make_guardrails(passed=True, violations=[])
    gr.warnings = ["high_turnover"]
    result = make_result(num_trades=2)
    mode, _ = classify_failure(result, gr, {})
    assert mode == "too_few_trades"


def test_all_failure_modes_are_reachable():
    """Parametrized check that every mode can be produced."""
    cases = [
        ("too_few_trades", dict(num_trades=3, sharpe=0.5), {}, {}),
        ("flat_signal", dict(num_trades=10, total_return=0.0, sharpe=0.0), {}, {}),
        ("excessive_drawdown", dict(max_drawdown=-0.35), {}, GOOD_SHARPE_BY_YEAR),
        ("regime_fragility", dict(), {}, {"2022": -0.5, "2023": 2.2}),
        ("overtrading", dict(num_trades=300), {"data_length": 500}, {}),
        ("low_sharpe", dict(sharpe=0.5, num_trades=10), {}, {}),
    ]
    for expected_mode, result_kw, classify_kw, sharpe_yr in cases:
        result = make_result(**result_kw)
        mode, details = classify_failure(result, make_guardrails(), sharpe_yr, **classify_kw)
        assert mode == expected_mode, f"Expected {expected_mode}, got {mode}: {details}"


def test_default_data_length_used():
    """Default data_length=500 should be used when not specified."""
    # 300 trades on default 500 bars → overtrading
    result = make_result(num_trades=300, sharpe=0.5)
    mode, _ = classify_failure(result, make_guardrails(), {})
    assert mode == "overtrading"


def test_default_sharpe_target_used():
    """Default sharpe_target=1.5 should be used when not specified."""
    result = make_result(sharpe=1.4, num_trades=10)
    mode, details = classify_failure(result, make_guardrails(), {})
    assert mode == "low_sharpe"
    assert "1.50" in details or "1.5" in details


def test_strategy_with_all_good_metrics():
    """A strategy that passes everything → low_sharpe catchall (no specific failure)."""
    result = make_result(sharpe=2.0, num_trades=50, total_return=0.25, max_drawdown=-0.08)
    mode, details = classify_failure(result, make_guardrails(), GOOD_SHARPE_BY_YEAR)
    assert mode == "low_sharpe"
    assert "Below target but no specific" in details
