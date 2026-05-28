"""Tests for crabquant.refinement.schemas — TDD first pass."""

import json
from dataclasses import fields

import pytest

from crabquant.refinement.schemas import BacktestReport, RunState, StrategyModification


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
