"""Tests for crabquant.refinement.schemas — TDD first pass + expanded coverage."""

import json
from dataclasses import fields

import pytest

from crabquant.refinement.schemas import (
    BacktestReport,
    RunState,
    StrategyModification,
    _fields,
)


# ─── RunState ────────────────────────────────────────────────────────────────

class TestRunState:
    def _minimal(self):
        return RunState(run_id="2026-04-26_140000_a3f2", mandate_name="momentum_spy_1", created_at="2026-04-26T14:00:00")

    def test_required_fields(self):
        rs = self._minimal()
        assert rs.run_id == "2026-04-26_140000_a3f2"
        assert rs.mandate_name == "momentum_spy_1"
        assert rs.created_at == "2026-04-26T14:00:00"

    def test_defaults(self):
        rs = self._minimal()
        assert rs.max_turns == 7
        assert rs.sharpe_target == 1.5
        assert rs.tickers == ["AAPL", "SPY"]
        assert rs.period == "2y"
        assert rs.current_turn == 0
        assert rs.status == "pending"
        assert rs.best_sharpe == -999.0
        assert rs.best_turn == 0
        assert rs.best_code_path == ""
        assert rs.history == []
        assert rs.lock_pid is None
        assert rs.lock_timestamp is None

    def test_tickers_default_is_independent(self):
        rs1 = self._minimal()
        rs2 = self._minimal()
        rs1.tickers.append("TSLA")
        assert "TSLA" not in rs2.tickers

    def test_history_default_is_independent(self):
        rs1 = self._minimal()
        rs2 = self._minimal()
        rs1.history.append({"turn": 1})
        assert rs2.history == []

    def test_valid_statuses(self):
        for status in ("pending", "running", "success", "max_turns", "failed", "stuck"):
            rs = self._minimal()
            rs.status = status
            assert rs.status == status

    def test_to_dict(self):
        rs = self._minimal()
        d = rs.to_dict()
        assert isinstance(d, dict)
        assert d["run_id"] == "2026-04-26_140000_a3f2"
        assert d["max_turns"] == 7
        assert d["tickers"] == ["AAPL", "SPY"]
        assert d["lock_pid"] is None

    def test_from_dict_roundtrip(self):
        rs = self._minimal()
        rs.current_turn = 3
        rs.status = "running"
        rs.best_sharpe = 1.23
        rs.history = [{"turn": 1, "sharpe": 0.45}]
        rs.lock_pid = 12345

        d = rs.to_dict()
        rs2 = RunState.from_dict(d)

        assert rs2.run_id == rs.run_id
        assert rs2.mandate_name == rs.mandate_name
        assert rs2.current_turn == 3
        assert rs2.status == "running"
        assert rs2.best_sharpe == 1.23
        assert rs2.history == [{"turn": 1, "sharpe": 0.45}]
        assert rs2.lock_pid == 12345

    def test_json_roundtrip(self):
        rs = self._minimal()
        rs.tickers = ["AAPL", "SPY", "NVDA"]
        rs.current_turn = 2

        blob = rs.to_json()
        assert isinstance(blob, str)
        rs2 = RunState.from_json(blob)
        assert rs2.tickers == ["AAPL", "SPY", "NVDA"]
        assert rs2.current_turn == 2

    def test_json_is_valid_json(self):
        rs = self._minimal()
        blob = rs.to_json()
        parsed = json.loads(blob)
        assert parsed["run_id"] == rs.run_id

    def test_from_dict_none_lock_fields(self):
        rs = self._minimal()
        d = rs.to_dict()
        d["lock_pid"] = None
        d["lock_timestamp"] = None
        rs2 = RunState.from_dict(d)
        assert rs2.lock_pid is None
        assert rs2.lock_timestamp is None

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_from_dict_ignores_extra_keys(self):
        """from_dict should silently ignore keys that aren't dataclass fields."""
        rs = self._minimal()
        d = rs.to_dict()
        d["bogus_key"] = "should be ignored"
        d["another_fake"] = 42
        rs2 = RunState.from_dict(d)
        assert rs2.run_id == rs.run_id
        assert not hasattr(rs2, "bogus_key")

    def test_from_dict_empty_dict_raises(self):
        """from_dict with empty dict should fail because required fields are missing."""
        with pytest.raises(TypeError):
            RunState.from_dict({})

    def test_from_dict_missing_required_field_raises(self):
        """from_dict missing a required field should raise TypeError."""
        with pytest.raises(TypeError):
            RunState.from_dict({"run_id": "test"})

    def test_to_dict_has_all_field_names(self):
        """to_dict should contain every dataclass field name."""
        rs = self._minimal()
        d = rs.to_dict()
        field_names = {f.name for f in fields(RunState)}
        assert set(d.keys()) == field_names

    def test_custom_tickers_and_period(self):
        """RunState accepts custom tickers and period values."""
        rs = RunState(
            run_id="r1", mandate_name="m1", created_at="2026-01-01",
            tickers=["TSLA", "NVDA", "AMD"], period="5y",
        )
        assert rs.tickers == ["TSLA", "NVDA", "AMD"]
        assert rs.period == "5y"

    def test_max_turns_custom(self):
        rs = RunState(run_id="r1", mandate_name="m1", created_at="2026-01-01", max_turns=15)
        assert rs.max_turns == 15

    def test_sharpe_target_custom(self):
        rs = RunState(run_id="r1", mandate_name="m1", created_at="2026-01-01", sharpe_target=2.0)
        assert rs.sharpe_target == 2.0

    def test_best_composite_score_default(self):
        rs = self._minimal()
        assert rs.best_composite_score == -999.0

    def test_best_composite_score_custom(self):
        rs = self._minimal()
        rs.best_composite_score = 0.85
        assert rs.best_composite_score == 0.85

    def test_best_composite_score_in_roundtrip(self):
        rs = self._minimal()
        rs.best_composite_score = 1.23
        rs2 = RunState.from_dict(rs.to_dict())
        assert rs2.best_composite_score == pytest.approx(1.23)

    def test_lock_pid_and_timestamp_set(self):
        rs = self._minimal()
        rs.lock_pid = 9999
        rs.lock_timestamp = "2026-04-26T14:05:00"
        d = rs.to_dict()
        rs2 = RunState.from_dict(d)
        assert rs2.lock_pid == 9999
        assert rs2.lock_timestamp == "2026-04-26T14:05:00"

    def test_history_complex_entries(self):
        """History can contain complex nested dicts."""
        rs = self._minimal()
        rs.history = [
            {"turn": 1, "sharpe": 0.5, "action": "modify_params",
             "params_used": {"fast": 12, "slow": 26}, "delta": {"fast": 2}},
        ]
        rs2 = RunState.from_dict(rs.to_dict())
        assert rs2.history[0]["params_used"]["fast"] == 12

    def test_json_pretty_print(self):
        """to_json supports indent kwarg."""
        rs = self._minimal()
        blob = rs.to_json(indent=2)
        assert "\n" in blob

    def test_json_with_sort_keys(self):
        """to_json supports sort_keys kwarg."""
        rs = self._minimal()
        blob = rs.to_json(sort_keys=True)
        parsed = json.loads(blob)
        assert list(parsed.keys()) == sorted(parsed.keys())

    def test_equality(self):
        rs1 = self._minimal()
        rs2 = self._minimal()
        assert rs1 == rs2

    def test_inequality(self):
        rs1 = self._minimal()
        rs2 = self._minimal()
        rs2.status = "running"
        assert rs1 != rs2

    def test_single_ticker_list(self):
        rs = RunState(run_id="r1", mandate_name="m1", created_at="2026-01-01", tickers=["SPY"])
        assert rs.tickers == ["SPY"]

    def test_empty_tickers_list(self):
        rs = RunState(run_id="r1", mandate_name="m1", created_at="2026-01-01", tickers=[])
        assert rs.tickers == []


# ─── BacktestReport ──────────────────────────────────────────────────────────

class TestBacktestReport:
    def _minimal(self):
        return BacktestReport(
            strategy_id="test_strat_v1",
            iteration=1,
            sharpe_ratio=0.85,
            total_return_pct=0.12,
            max_drawdown_pct=-0.18,
            win_rate=0.52,
            total_trades=42,
            profit_factor=1.3,
            calmar_ratio=0.9,
            sortino_ratio=1.1,
            composite_score=0.76,
            failure_mode="low_sharpe",
            failure_details="Sharpe 0.85 < target 1.5.",
            sharpe_by_year={"2024": 1.1, "2025": 0.6},
            stagnation_score=0.2,
            stagnation_trend="improving",
            previous_sharpes=[0.4, 0.6],
            previous_actions=["modify_params"],
            guardrail_violations=[],
            guardrail_warnings=["Low trade count"],
            regime_sharpe=None,
            regime_regime_shift=None,
            top_drawdowns=None,
            portfolio_correlation=None,
            benchmark_return_pct=None,
            market_regime=None,
            current_strategy_code="def generate_signals(df, **p): ...",
            current_params={"fast": 12, "slow": 26},
            previous_attempts=[],
        )

    def test_required_tier1_fields_present(self):
        br = self._minimal()
        assert br.strategy_id == "test_strat_v1"
        assert br.iteration == 1
        assert br.sharpe_ratio == 0.85
        assert br.total_return_pct == 0.12
        assert br.max_drawdown_pct == -0.18
        assert br.total_trades == 42
        assert br.failure_mode == "low_sharpe"

    def test_tier2_fields_default_none(self):
        br = self._minimal()
        assert br.regime_sharpe is None
        assert br.regime_regime_shift is None
        assert br.top_drawdowns is None
        assert br.portfolio_correlation is None
        assert br.benchmark_return_pct is None
        assert br.market_regime is None

    def test_to_dict(self):
        br = self._minimal()
        d = br.to_dict()
        assert isinstance(d, dict)
        assert d["strategy_id"] == "test_strat_v1"
        assert d["sharpe_by_year"] == {"2024": 1.1, "2025": 0.6}
        assert d["regime_sharpe"] is None

    def test_from_dict_roundtrip(self):
        br = self._minimal()
        br.regime_sharpe = {"uptrend": 2.1, "downtrend": -0.5}
        br.top_drawdowns = [{"start": "2024-01-01", "depth_pct": -0.15}]

        d = br.to_dict()
        br2 = BacktestReport.from_dict(d)

        assert br2.strategy_id == br.strategy_id
        assert br2.sharpe_ratio == br.sharpe_ratio
        assert br2.regime_sharpe == {"uptrend": 2.1, "downtrend": -0.5}
        assert br2.top_drawdowns == [{"start": "2024-01-01", "depth_pct": -0.15}]
        assert br2.previous_actions == ["modify_params"]

    def test_json_roundtrip(self):
        br = self._minimal()
        blob = br.to_json()
        assert isinstance(blob, str)
        br2 = BacktestReport.from_json(blob)
        assert br2.iteration == 1
        assert br2.failure_mode == "low_sharpe"
        assert br2.sharpe_by_year == {"2024": 1.1, "2025": 0.6}

    def test_json_is_valid_json(self):
        br = self._minimal()
        blob = br.to_json()
        parsed = json.loads(blob)
        assert parsed["composite_score"] == br.composite_score

    def test_from_backtest_result(self):
        """from_backtest_result classmethod maps BacktestResult fields correctly."""
        from crabquant.engine.backtest import BacktestResult

        result = BacktestResult(
            ticker="AAPL",
            strategy_name="test_v1",
            iteration=2,
            sharpe=1.2,
            total_return=0.25,
            max_drawdown=-0.18,
            win_rate=0.55,
            num_trades=30,
            avg_trade_return=0.02,
            calmar_ratio=1.1,
            sortino_ratio=1.4,
            profit_factor=1.6,
            avg_holding_bars=5.0,
            best_trade=500.0,
            worst_trade=-200.0,
            passed=False,
            score=0.88,
            notes="Sharpe 1.2 < 1.5",
            params={"fast": 10},
        )

        br = BacktestReport.from_backtest_result(
            result=result,
            failure_mode="low_sharpe",
            failure_details="Sharpe 1.2 < target 1.5.",
            sharpe_by_year={"2024": 1.5, "2025": 0.9},
            stagnation_score=0.1,
            stagnation_trend="improving",
            previous_sharpes=[],
            previous_actions=[],
            guardrail_violations=[],
            guardrail_warnings=[],
            current_strategy_code="code here",
            current_params={"fast": 10},
            previous_attempts=[],
        )

        assert br.strategy_id == "test_v1"
        assert br.iteration == 2
        assert br.sharpe_ratio == 1.2
        assert br.total_return_pct == 0.25
        assert br.max_drawdown_pct == -0.18
        assert br.win_rate == 0.55
        assert br.total_trades == 30
        assert br.profit_factor == 1.6
        assert br.calmar_ratio == 1.1
        assert br.sortino_ratio == 1.4
        assert br.composite_score == 0.88

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_from_dict_ignores_extra_keys(self):
        """from_dict should ignore unknown keys."""
        br = self._minimal()
        d = br.to_dict()
        d["extra_key"] = "ignored"
        br2 = BacktestReport.from_dict(d)
        assert br2.strategy_id == br.strategy_id

    def test_to_dict_has_all_fields(self):
        br = self._minimal()
        d = br.to_dict()
        field_names = {f.name for f in fields(BacktestReport)}
        assert set(d.keys()) == field_names

    def test_tier2_fields_with_values(self):
        """Tier 2 optional fields can be populated."""
        br = self._minimal()
        br.regime_sharpe = {"bull": 2.0, "bear": -0.3}
        br.regime_regime_shift = True
        br.top_drawdowns = [{"period": "2024-Q1", "depth": -0.20}]
        br.portfolio_correlation = 0.85
        br.benchmark_return_pct = 0.10
        br.market_regime = "high_volatility"

        d = br.to_dict()
        assert d["regime_regime_shift"] is True
        assert d["portfolio_correlation"] == 0.85
        assert d["market_regime"] == "high_volatility"

    def test_tier2_roundtrip_with_all_values(self):
        """All tier 2 fields survive roundtrip."""
        br = self._minimal()
        br.regime_sharpe = {"bull": 2.0}
        br.regime_regime_shift = False
        br.top_drawdowns = []
        br.portfolio_correlation = 0.5
        br.benchmark_return_pct = 0.08
        br.market_regime = "low_volatility"

        br2 = BacktestReport.from_dict(br.to_dict())
        assert br2.regime_sharpe == {"bull": 2.0}
        assert br2.regime_regime_shift is False
        assert br2.top_drawdowns == []
        assert br2.portfolio_correlation == 0.5
        assert br2.benchmark_return_pct == 0.08
        assert br2.market_regime == "low_volatility"

    def test_multi_ticker_results_default_none(self):
        br = self._minimal()
        assert br.multi_ticker_results is None

    def test_multi_ticker_results_with_data(self):
        br = self._minimal()
        br.multi_ticker_results = {"AAPL": {"sharpe": 1.2}, "SPY": {"sharpe": 0.8}}
        br2 = BacktestReport.from_dict(br.to_dict())
        assert br2.multi_ticker_results == {"AAPL": {"sharpe": 1.2}, "SPY": {"sharpe": 0.8}}

    def test_feature_importance_default_none(self):
        br = self._minimal()
        assert br.feature_importance is None

    def test_feature_importance_with_data(self):
        br = self._minimal()
        br.feature_importance = {"rsi": 0.4, "macd": 0.3, "volume": 0.3}
        br2 = BacktestReport.from_dict(br.to_dict())
        assert br2.feature_importance == {"rsi": 0.4, "macd": 0.3, "volume": 0.3}

    def test_stagnation_trend_values(self):
        """Stagnation trend accepts documented values."""
        for trend in ("improving", "flat", "declining"):
            br = self._minimal()
            br.stagnation_trend = trend
            assert br.stagnation_trend == trend

    def test_stagnation_score_boundary(self):
        br = self._minimal()
        br.stagnation_score = 0.0
        assert br.stagnation_score == 0.0
        br.stagnation_score = 1.0
        assert br.stagnation_score == 1.0

    def test_guardrail_violations_and_warnings(self):
        br = self._minimal()
        br.guardrail_violations = ["max_dd_exceeded", "min_trades"]
        br.guardrail_warnings = ["low_win_rate"]
        br2 = BacktestReport.from_dict(br.to_dict())
        assert br2.guardrail_violations == ["max_dd_exceeded", "min_trades"]
        assert br2.guardrail_warnings == ["low_win_rate"]

    def test_previous_attempts_roundtrip(self):
        br = self._minimal()
        br.previous_attempts = [
            {"turn": 1, "sharpe": 0.5, "action": "modify_params"},
            {"turn": 2, "sharpe": 0.8, "action": "change_entry_logic"},
        ]
        br2 = BacktestReport.from_dict(br.to_dict())
        assert len(br2.previous_attempts) == 2
        assert br2.previous_attempts[1]["action"] == "change_entry_logic"

    def test_current_strategy_code_multiline(self):
        br = self._minimal()
        code = "import pandas as pd\n\ndef generate_signals(df, **p):\n    return df\n"
        br.current_strategy_code = code
        br2 = BacktestReport.from_json(br.to_json())
        assert br2.current_strategy_code == code

    def test_json_with_tier2_data(self):
        br = self._minimal()
        br.multi_ticker_results = {"AAPL": {"sharpe": 1.5}}
        br.feature_importance = {"macd": 0.6}
        blob = br.to_json()
        parsed = json.loads(blob)
        assert parsed["multi_ticker_results"]["AAPL"]["sharpe"] == 1.5
        assert parsed["feature_importance"]["macd"] == 0.6

    def test_equality(self):
        br1 = self._minimal()
        br2 = self._minimal()
        assert br1 == br2

    def test_inequality(self):
        br1 = self._minimal()
        br2 = self._minimal()
        br2.sharpe_ratio = 2.0
        assert br1 != br2

    def test_from_backtest_result_with_tier2(self):
        """from_backtest_result passes through optional tier 2 kwargs."""
        from crabquant.engine.backtest import BacktestResult

        result = BacktestResult(
            ticker="AAPL", strategy_name="s1", iteration=1,
            sharpe=1.5, total_return=0.3, max_drawdown=-0.1,
            win_rate=0.6, num_trades=50, avg_trade_return=0.01,
            calmar_ratio=2.0, sortino_ratio=2.5, profit_factor=1.8,
            avg_holding_bars=5.0, best_trade=300.0, worst_trade=-100.0,
            passed=True, score=0.95, notes="", params={},
        )
        br = BacktestReport.from_backtest_result(
            result=result,
            failure_mode="", failure_details="",
            sharpe_by_year={}, stagnation_score=0.0, stagnation_trend="improving",
            previous_sharpes=[], previous_actions=[],
            guardrail_violations=[], guardrail_warnings=[],
            current_strategy_code="", current_params={},
            previous_attempts=[],
            multi_ticker_results={"AAPL": {"sharpe": 1.5}},
            feature_importance={"rsi": 0.7},
            regime_sharpe={"bull": 2.0},
            regime_regime_shift=True,
            top_drawdowns=[{"depth": -0.1}],
            portfolio_correlation=0.9,
            benchmark_return_pct=0.15,
            market_regime="bull",
        )
        assert br.multi_ticker_results == {"AAPL": {"sharpe": 1.5}}
        assert br.feature_importance == {"rsi": 0.7}
        assert br.regime_sharpe == {"bull": 2.0}
        assert br.regime_regime_shift is True
        assert br.market_regime == "bull"


# ─── StrategyModification ────────────────────────────────────────────────────

VALID_ACTIONS = (
    "replace_indicator",
    "add_filter",
    "modify_params",
    "change_entry_logic",
    "change_exit_logic",
    "add_regime_filter",
    "full_rewrite",
    "novel",
)


class TestStrategyModification:
    def _minimal(self):
        return StrategyModification(
            addresses_failure="low_sharpe",
            hypothesis="Adding histogram crossover will improve entries.",
            action="change_entry_logic",
            new_strategy_code="def generate_signals(df, **p): ...",
            reasoning="Entry timing is the primary issue.",
            expected_impact="moderate",
        )

    def test_required_fields(self):
        sm = self._minimal()
        assert sm.addresses_failure == "low_sharpe"
        assert sm.hypothesis == "Adding histogram crossover will improve entries."
        assert sm.action == "change_entry_logic"
        assert sm.new_strategy_code == "def generate_signals(df, **p): ..."
        assert sm.reasoning == "Entry timing is the primary issue."
        assert sm.expected_impact == "moderate"

    @pytest.mark.parametrize("action", VALID_ACTIONS)
    def test_valid_action_types(self, action):
        sm = self._minimal()
        sm.action = action
        assert sm.action == action

    def test_to_dict(self):
        sm = self._minimal()
        d = sm.to_dict()
        assert isinstance(d, dict)
        assert d["action"] == "change_entry_logic"
        assert d["expected_impact"] == "moderate"
        assert "new_strategy_code" in d

    def test_from_dict_roundtrip(self):
        sm = self._minimal()
        d = sm.to_dict()
        sm2 = StrategyModification.from_dict(d)

        assert sm2.addresses_failure == sm.addresses_failure
        assert sm2.hypothesis == sm.hypothesis
        assert sm2.action == sm.action
        assert sm2.new_strategy_code == sm.new_strategy_code
        assert sm2.reasoning == sm.reasoning
        assert sm2.expected_impact == sm.expected_impact

    def test_json_roundtrip(self):
        sm = self._minimal()
        blob = sm.to_json()
        assert isinstance(blob, str)
        sm2 = StrategyModification.from_json(blob)
        assert sm2.action == "change_entry_logic"
        assert sm2.hypothesis == sm.hypothesis

    def test_json_is_valid_json(self):
        sm = self._minimal()
        blob = sm.to_json()
        parsed = json.loads(blob)
        assert parsed["addresses_failure"] == "low_sharpe"

    def test_multiline_code_roundtrip(self):
        code = "import pandas as pd\n\ndef generate_signals(df, **p):\n    return df\n"
        sm = self._minimal()
        sm.new_strategy_code = code
        sm2 = StrategyModification.from_json(sm.to_json())
        assert sm2.new_strategy_code == code

    def test_expected_impact_values(self):
        for impact in ("minor", "moderate", "major"):
            sm = self._minimal()
            sm.expected_impact = impact
            blob = sm.to_json()
            sm2 = StrategyModification.from_json(blob)
            assert sm2.expected_impact == impact

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_from_dict_ignores_extra_keys(self):
        sm = self._minimal()
        d = sm.to_dict()
        d["bogus"] = True
        sm2 = StrategyModification.from_dict(d)
        assert sm2.action == sm.action

    def test_to_dict_has_all_fields(self):
        sm = self._minimal()
        d = sm.to_dict()
        field_names = {f.name for f in fields(StrategyModification)}
        assert set(d.keys()) == field_names

    def test_empty_strings_allowed(self):
        """Empty strings are technically valid for text fields."""
        sm = StrategyModification(
            addresses_failure="",
            hypothesis="",
            action="novel",
            new_strategy_code="",
            reasoning="",
            expected_impact="minor",
        )
        assert sm.hypothesis == ""
        assert sm.new_strategy_code == ""

    def test_hypothesis_with_special_chars(self):
        sm = self._minimal()
        sm.hypothesis = "If RSI > 70 & MACD < 0, then entries are filtered — see §3.2"
        sm2 = StrategyModification.from_json(sm.to_json())
        assert "§3.2" in sm2.hypothesis

    def test_reasoning_with_unicode(self):
        sm = self._minimal()
        sm.reasoning = "Strategy underperforms during bear regimes 📉"
        sm2 = StrategyModification.from_json(sm.to_json())
        assert "📉" in sm2.reasoning

    def test_addresses_failure_various_modes(self):
        for mode in ("low_sharpe", "too_few_trades", "high_drawdown",
                      "overfitting", "stagnation", "max_turns", "crash"):
            sm = self._minimal()
            sm.addresses_failure = mode
            sm2 = StrategyModification.from_dict(sm.to_dict())
            assert sm2.addresses_failure == mode

    def test_json_pretty_print(self):
        sm = self._minimal()
        blob = sm.to_json(indent=4)
        assert "\n" in blob

    def test_equality(self):
        sm1 = self._minimal()
        sm2 = self._minimal()
        assert sm1 == sm2

    def test_inequality(self):
        sm1 = self._minimal()
        sm2 = self._minimal()
        sm2.action = "full_rewrite"
        assert sm1 != sm2

    def test_all_action_types_roundtrip(self):
        """Every documented action type survives JSON roundtrip."""
        for action in VALID_ACTIONS:
            sm = self._minimal()
            sm.action = action
            sm2 = StrategyModification.from_json(sm.to_json())
            assert sm2.action == action

    def test_new_strategy_code_with_backticks(self):
        """Code with backticks (f-strings) survives JSON."""
        sm = self._minimal()
        sm.new_strategy_code = "def go(df):\n    x = f'{1+1}'\n    return x\n"
        sm2 = StrategyModification.from_json(sm.to_json())
        assert "f'" in sm2.new_strategy_code


# ─── _fields helper ─────────────────────────────────────────────────────────

class TestFieldsHelper:

    def test_returns_field_descriptors(self):
        flds = _fields(RunState)
        names = {f.name for f in flds}
        assert "run_id" in names
        assert "mandate_name" in names
        assert "history" in names

    def test_works_for_all_three_schemas(self):
        for cls in (RunState, BacktestReport, StrategyModification):
            flds = _fields(cls)
            assert len(flds) > 0
