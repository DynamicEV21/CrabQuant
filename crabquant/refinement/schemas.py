"""
Refinement pipeline dataclasses: RunState, BacktestReport, StrategyModification.
All support JSON roundtrip via to_dict/from_dict/to_json/from_json.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunState:
    """Persistent loop state serialised to state.json between turns."""

    # Identity (required)
    run_id: str
    mandate_name: str
    created_at: str

    # Configuration
    max_turns: int = 7
    sharpe_target: float = 1.5
    tickers: list = field(default_factory=lambda: ["AAPL", "SPY"])
    period: str = "2y"

    # Progress
    current_turn: int = 0
    status: str = "pending"  # pending|running|success|max_turns|failed|stuck
    best_sharpe: float = -999.0
    best_composite_score: float = -999.0  # Composite score for best-strategy tracking
    best_turn: int = 0
    best_code_path: str = ""

    # History (append-only)
    history: list = field(default_factory=list)

    # Concurrency lock
    lock_pid: int | None = None
    lock_timestamp: str | None = None

    # ── serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunState":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in _fields(cls)}})

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, blob: str) -> "RunState":
        return cls.from_dict(json.loads(blob))


@dataclass
class BacktestReport:
    """Rich diagnostic payload produced by Python and consumed by the LLM."""

    # Identity
    strategy_id: str
    iteration: int

    # ── Tier 1: always computed ─────────────────────────────────────────────

    # Core metrics (field names intentionally differ from BacktestResult)
    sharpe_ratio: float        # BacktestResult.sharpe
    total_return_pct: float    # BacktestResult.total_return  (raw, e.g. 0.15)
    max_drawdown_pct: float    # BacktestResult.max_drawdown  (raw, e.g. -0.25)
    win_rate: float
    total_trades: int          # BacktestResult.num_trades
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    composite_score: float     # BacktestResult.score

    # Failure classification
    failure_mode: str
    failure_details: str

    # Temporal resolution
    sharpe_by_year: dict

    # Stagnation context
    stagnation_score: float
    stagnation_trend: str      # "improving" | "flat" | "declining"
    previous_sharpes: list
    previous_actions: list

    # Guardrail results
    guardrail_violations: list
    guardrail_warnings: list

    # ── Tier 2: conditional ─────────────────────────────────────────────────

    regime_sharpe: dict | None
    regime_regime_shift: bool | None
    top_drawdowns: list | None
    portfolio_correlation: float | None
    benchmark_return_pct: float | None
    market_regime: str | None

    # Strategy context (always included)
    current_strategy_code: str
    current_params: dict
    previous_attempts: list

    # ── serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BacktestReport":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in _fields(cls)}})

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, blob: str) -> "BacktestReport":
        return cls.from_dict(json.loads(blob))

    @classmethod
    def from_backtest_result(
        cls,
        result: Any,  # crabquant.engine.backtest.BacktestResult
        *,
        failure_mode: str,
        failure_details: str,
        sharpe_by_year: dict,
        stagnation_score: float,
        stagnation_trend: str,
        previous_sharpes: list,
        previous_actions: list,
        guardrail_violations: list,
        guardrail_warnings: list,
        current_strategy_code: str,
        current_params: dict,
        previous_attempts: list,
        regime_sharpe: dict | None = None,
        regime_regime_shift: bool | None = None,
        top_drawdowns: list | None = None,
        portfolio_correlation: float | None = None,
        benchmark_return_pct: float | None = None,
        market_regime: str | None = None,
    ) -> "BacktestReport":
        """Build a BacktestReport from a BacktestResult, mapping renamed fields."""
        return cls(
            strategy_id=result.strategy_name,
            iteration=result.iteration,
            sharpe_ratio=result.sharpe,
            total_return_pct=result.total_return,
            max_drawdown_pct=result.max_drawdown,
            win_rate=result.win_rate,
            total_trades=result.num_trades,
            profit_factor=result.profit_factor,
            calmar_ratio=result.calmar_ratio,
            sortino_ratio=result.sortino_ratio,
            composite_score=result.score,
            failure_mode=failure_mode,
            failure_details=failure_details,
            sharpe_by_year=sharpe_by_year,
            stagnation_score=stagnation_score,
            stagnation_trend=stagnation_trend,
            previous_sharpes=previous_sharpes,
            previous_actions=previous_actions,
            guardrail_violations=guardrail_violations,
            guardrail_warnings=guardrail_warnings,
            regime_sharpe=regime_sharpe,
            regime_regime_shift=regime_regime_shift,
            top_drawdowns=top_drawdowns,
            portfolio_correlation=portfolio_correlation,
            benchmark_return_pct=benchmark_return_pct,
            market_regime=market_regime,
            current_strategy_code=current_strategy_code,
            current_params=current_params,
            previous_attempts=previous_attempts,
        )


@dataclass
class StrategyModification:
    """LLM output: what change to make and why."""

    addresses_failure: str   # must match a known failure_mode
    hypothesis: str          # causal hypothesis (required)
    action: str              # one of 8 action types below
    new_strategy_code: str   # full Python file content
    reasoning: str
    expected_impact: str     # "minor" | "moderate" | "major"

    # Action types:
    # replace_indicator | add_filter | modify_params | change_entry_logic
    # change_exit_logic | add_regime_filter | full_rewrite | novel

    # ── serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyModification":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in _fields(cls)}})

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, blob: str) -> "StrategyModification":
        return cls.from_dict(json.loads(blob))


# ── helpers ───────────────────────────────────────────────────────────────────

def _fields(cls):
    """Return dataclass field descriptors for cls."""
    import dataclasses
    return dataclasses.fields(cls)
