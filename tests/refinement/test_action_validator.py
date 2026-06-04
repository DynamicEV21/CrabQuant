"""Tests for semantic action validation.

Covers all 7 override rules, pass-through cases, edge cases (negative trades,
zero trades with non-matching failure modes), and the total_trades predicate
boundary values.
"""

import pytest

from crabquant.refinement.action_validator import validate_action_semantically


# ── Helpers ──────────────────────────────────────────────────────────────────

def _check(action: str, failure_mode: str, total_trades: int):
    """Convenience: returns (validated_action, reason)."""
    return validate_action_semantically(action, failure_mode, total_trades)


# ── Rule 1: modify_params + flat_signal (0 trades) → change_entry_logic ─────

class TestModifyParamsFlatSignal:
    def test_zero_trades_overridden(self):
        action, reason = _check("modify_params", "flat_signal", 0)
        assert action == "change_entry_logic"
        assert "flat_signal" in reason

    def test_zero_trades_reason_nonempty(self):
        _, reason = _check("modify_params", "flat_signal", 0)
        assert reason  # non-empty


# ── Rule 2: modify_params + too_few_trades (1-4 trades) → change_entry_logic

class TestModifyParamsTooFewTrades:
    @pytest.mark.parametrize("trades", [1, 2, 3, 4])
    def test_low_trade_count_overridden(self, trades):
        action, reason = _check("modify_params", "too_few_trades", trades)
        assert action == "change_entry_logic"
        assert reason

    def test_five_trades_passes_through(self):
        """5 trades is the classifier boundary — semantic rule only fires for 1-4."""
        action, reason = _check("modify_params", "too_few_trades", 5)
        assert action == "modify_params"
        assert reason == ""

    def test_zero_trades_with_too_few_trades_mode_passes(self):
        """0 trades is classified as flat_signal, not too_few_trades — rule won't fire."""
        # Actually classifier returns flat_signal for 0 trades.  But even if called
        # with too_few_trades and 0 trades, the predicate (1<=t<=4) excludes it.
        action, reason = _check("modify_params", "too_few_trades", 0)
        assert action == "modify_params"
        assert reason == ""


# ── Rule 3: add_filter + flat_signal (0 trades) → change_entry_logic ────────

class TestAddFilterFlatSignal:
    def test_zero_trades_overridden(self):
        action, reason = _check("add_filter", "flat_signal", 0)
        assert action == "change_entry_logic"
        assert reason

    def test_nonzero_trades_passes_through(self):
        action, reason = _check("add_filter", "flat_signal", 5)
        assert action == "add_filter"
        assert reason == ""


# ── Rule 4: modify_params + module_load_failed → novel ──────────────────────

class TestModifyParamsModuleLoadFailed:
    def test_overridden(self):
        action, reason = _check("modify_params", "module_load_failed", 0)
        assert action == "novel"
        assert reason

    def test_overridden_any_trade_count(self):
        """Rule fires regardless of total_trades."""
        action, reason = _check("modify_params", "module_load_failed", 50)
        assert action == "novel"


# ── Rule 5: modify_params + backtest_crash → novel ─────────────────────────

class TestModifyParamsBacktestCrash:
    def test_overridden(self):
        action, reason = _check("modify_params", "backtest_crash", 0)
        assert action == "novel"
        assert reason

    def test_overridden_any_trade_count(self):
        action, reason = _check("modify_params", "backtest_crash", 100)
        assert action == "novel"


# ── Rule 6: add_filter + too_few_trades (1-4 trades) → change_entry_logic ──

class TestAddFilterTooFewTrades:
    @pytest.mark.parametrize("trades", [1, 2, 3, 4])
    def test_low_trade_count_overridden(self, trades):
        action, reason = _check("add_filter", "too_few_trades", trades)
        assert action == "change_entry_logic"
        assert reason

    def test_five_trades_passes_through(self):
        action, reason = _check("add_filter", "too_few_trades", 5)
        assert action == "add_filter"
        assert reason == ""


# ── Rule 7: change_exit_logic + flat_signal (0 trades) → change_entry_logic ─

class TestChangeExitLogicFlatSignal:
    def test_zero_trades_overridden(self):
        action, reason = _check("change_exit_logic", "flat_signal", 0)
        assert action == "change_entry_logic"
        assert reason

    def test_nonzero_trades_passes_through(self):
        action, reason = _check("change_exit_logic", "flat_signal", 10)
        assert action == "change_exit_logic"
        assert reason == ""


# ── Pass-through cases (no override) ────────────────────────────────────────

class TestPassThrough:
    def test_valid_combination_modify_params_low_sharpe(self):
        """modify_params + low_sharpe is a reasonable combination."""
        action, reason = _check("modify_params", "low_sharpe", 30)
        assert action == "modify_params"
        assert reason == ""

    def test_valid_combination_add_filter_overtrading(self):
        action, reason = _check("add_filter", "overtrading", 500)
        assert action == "add_filter"
        assert reason == ""

    def test_valid_combination_change_entry_logic_excessive_drawdown(self):
        action, reason = _check("change_entry_logic", "excessive_drawdown", 50)
        assert action == "change_entry_logic"
        assert reason == ""

    def test_valid_combination_novel_backtest_crash(self):
        """novel is already the right action for backtest_crash."""
        action, reason = _check("novel", "backtest_crash", 0)
        assert action == "novel"
        assert reason == ""

    def test_valid_combination_add_regime_filter_regime_fragility(self):
        action, reason = _check("add_regime_filter", "regime_fragility", 40)
        assert action == "add_regime_filter"
        assert reason == ""

    def test_change_exit_logic_too_few_trades_passes(self):
        """change_exit_logic + too_few_trades is not explicitly blocked."""
        action, reason = _check("change_exit_logic", "too_few_trades", 3)
        assert action == "change_exit_logic"
        assert reason == ""


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_negative_trades(self):
        """Negative trade counts are nonsensical but shouldn't crash."""
        action, reason = _check("modify_params", "flat_signal", -1)
        # The flat_signal rule checks total_trades == 0, so -1 won't match
        assert action == "modify_params"

    def test_unknown_failure_mode_passes_through(self):
        action, reason = _check("modify_params", "some_future_mode", 0)
        assert action == "modify_params"
        assert reason == ""

    def test_unknown_action_passes_through(self):
        action, reason = _check("some_future_action", "flat_signal", 0)
        assert action == "some_future_action"
        assert reason == ""

    def test_empty_action_passes_through(self):
        action, reason = _check("", "flat_signal", 0)
        assert action == ""
        assert reason == ""

    def test_empty_failure_mode_passes_through(self):
        action, reason = _check("modify_params", "", 0)
        assert action == "modify_params"
        assert reason == ""
