"""
RefinementConfig — pipeline-wide configuration with mandate JSON load/save.
"""

import json
from dataclasses import asdict, dataclass, field, fields as dc_fields
from pathlib import Path


@dataclass
class RefinementConfig:
    # Loop
    max_turns: int = 7
    sharpe_target: float = 1.5
    max_drawdown_pct: float = 25.0
    min_trades: int = 5

    # LLM (z.ai API — OpenAI-compatible)
    llm_model: str = "glm-5-turbo"
    llm_base_url: str = "https://api.z.ai/api/coding/paas/v4"
    llm_api_key_env: str = ""
    llm_timeout_seconds: int = 120
    max_code_repair_attempts: int = 3
    max_llm_output_tokens: int = 4096

    # Validation
    smoke_backtest_timeout: int = 10
    signal_sanity_ticker: str = "AAPL"
    signal_sanity_period: str = "1y"

    # Stagnation
    stagnation_abandon_threshold: float = 0.8
    stagnation_nuclear_threshold: float = 0.6
    stagnation_pivot_threshold: float = 0.7
    stagnation_broaden_threshold: float = 0.5

    # Tier 2
    tier2_stagnation_trigger: float = 0.4
    tier2_iteration_trigger: int = 3

    # Circuit breaker
    circuit_breaker_window: int = 20
    circuit_breaker_min_pass_rate: float = 0.3

    # Timeouts
    per_strategy_timeout_minutes: int = 15
    backtest_timeout_seconds: int = 60
    lock_staleness_minutes: int = 20

    # Paths
    mandates_dir: str = "refinement/mandates"
    runs_dir: str = "refinement/runs"
    winners_file: str = "results/winners/winners.json"

    # ── Walk-forward validation thresholds ──────────────────────────────

    # Default values consumed by ``walk_forward_test()`` keyword args.
    # train_months and test_months are handled separately via the
    # period strings passed to the validation function.

    # ── serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RefinementConfig":
        known = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, blob: str) -> "RefinementConfig":
        return cls.from_dict(json.loads(blob))

    # ── file I/O ───────────────────────────────────────────────────────────

    def save(self, path) -> None:
        Path(path).write_text(self.to_json(indent=2))

    @classmethod
    def load(cls, path) -> "RefinementConfig":
        return cls.from_json(Path(path).read_text())

    # ── mandate integration ────────────────────────────────────────────────

    @classmethod
    def from_mandate(cls, mandate: dict, **overrides) -> "RefinementConfig":
        """Build config from a mandate dict, pulling loop-level fields."""
        kwargs: dict = {}

        if "max_turns" in mandate:
            kwargs["max_turns"] = mandate["max_turns"]
        if "sharpe_target" in mandate:
            kwargs["sharpe_target"] = mandate["sharpe_target"]

        constraints = mandate.get("constraints", {})
        if "min_trades" in constraints:
            kwargs["min_trades"] = constraints["min_trades"]
        if "max_drawdown_pct" in constraints:
            kwargs["max_drawdown_pct"] = float(constraints["max_drawdown_pct"])

        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def from_mandate_file(cls, path, **overrides) -> "RefinementConfig":
        """Load a mandate JSON file and build config from it."""
        mandate = json.loads(Path(path).read_text())
        return cls.from_mandate(mandate, **overrides)


# ── Standalone validation config ────────────────────────────────────────
# These defaults mirror the keyword arguments of ``walk_forward_test()``
# and ``rolling_walk_forward()`` in ``crabquant.validation``.

# ── Mandate diversity scoring ─────────────────────────────────────────────
# Controls how the mandate generator penalises already-explored
# (strategy_type, ticker) combinations to ensure broad coverage.

DIVERSITY_CONFIG: dict = {
    "max_winners_per_combo": 5,
    "min_ticker_diversity": 3,
    "min_archetype_diversity": 3,
    "winners_file": "results/winners/winners.json",
}

VALIDATION_CONFIG: dict = {
    # walk_forward_test() thresholds
    "train_pct": 0.75,            # 18mo / 24mo (≈ 3y of data)
    "min_train_bars": 252,        # ~1 year of trading days
    "min_test_sharpe": 0.3,       # relaxed from old hardcoded 0.5
    "min_test_trades": 10,        # enough trades for statistical significance
    "max_degradation": 0.7,       # test_sharpe >= train_sharpe * (1 - 0.7)
    # rolling_walk_forward() thresholds
    "train_window": "18mo",
    "test_window": "6mo",
    "step": "6mo",
    "min_avg_test_sharpe": 0.5,
    "min_windows_passed": 2,
}
