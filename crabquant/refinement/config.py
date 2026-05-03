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

    # ── Invention accelerators (Phase 5.6) ──────────────────────────────────
    # These toggles can be set individually or via preset modes.
    # Modes: "conservative" (all off), "fast" (cross-run only), "explorer" (all on)

    cross_run_learning: bool = True      # Feed proven winners into LLM context
    parallel_invention: bool = False     # Spawn N strategies in parallel on turn 1
    parallel_invention_count: int = 3    # How many parallel strategies to spawn
    soft_promote: bool = False           # Promote "good enough" strategies to candidates pool
    soft_promote_sharpe: float = 0.5     # Min avg test Sharpe for soft promote
    soft_promote_min_windows: int = 2    # Min windows passing for soft promote

    # ── Multi-ticker backtest (Phase 5.6) ───────────────────────────────
    # Run strategy on multiple tickers during refinement, not just primary.
    # Helps catch overfitting to a single ticker early.
    multi_ticker_backtest: bool = False  # Enable multi-ticker backtesting
    multi_ticker_min_pass: int = 2       # Min tickers that must pass (Sharpe >= target)
    # Additional tickers beyond primary — if empty, uses mandate tickers minus primary
    multi_ticker_extra: list = field(default_factory=list)

    # Feature importance analysis during refinement.
    # Analyzes which indicators contribute to returns and feeds this
    # back to the LLM so it can make data-driven code changes.
    feature_importance: bool = True

    # Parameter optimization during refinement.
    # After backtest, sweeps nearby parameter combinations to find better settings.
    # Frees LLM turns from parameter tuning so it can focus on structural changes.
    param_optimization: bool = True

    # ── Adaptive Sharpe targeting (Phase 6) ────────────────────────────
    # Starts with a lower target on early turns and ramps up to the
    # full target. Makes early turns productive instead of always wasted.
    adaptive_sharpe_target: bool = False   # Enable/disable adaptive targeting
    adaptive_start_factor: float = 0.5     # Initial target = sharpe_target * this
    adaptive_ramp_turns: int = 3           # Turns to ramp from start_factor to 1.0

    # Timeouts
    per_strategy_timeout_minutes: int = 15
    backtest_timeout_seconds: int = 60
    lock_staleness_minutes: int = 20

    # Paths
    mandates_dir: str = "refinement/mandates"
    runs_dir: str = "refinement/runs"
    winners_file: str = "results/winners/winners.json"

    # ── Walk-forward validation thresholds ──────────────────────────────

    # Per-window thresholds for rolling_walk_forward().
    # These control what counts as a "passed" individual window.
    min_window_test_sharpe: float = 0.0   # per-window test Sharpe floor
    max_window_degradation: float = 1.0   # per-window max degradation

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

    # ── preset modes ────────────────────────────────────────────────────────

    def apply_mode(self, mode: str) -> "RefinementConfig":
        """Apply a preset mode that flips the invention accelerator toggles.

        Args:
            mode: One of "conservative", "fast", "explorer", or "custom".

        Returns:
            self (for chaining).
        """
        mode = mode.lower().strip()
        if mode == "conservative":
            self.cross_run_learning = False
            self.parallel_invention = False
            self.soft_promote = False
        elif mode == "fast":
            self.cross_run_learning = True
            self.parallel_invention = False
            self.soft_promote = False
        elif mode == "explorer":
            self.cross_run_learning = True
            self.parallel_invention = True
            self.soft_promote = True
        elif mode == "balanced":
            self.cross_run_learning = True
            self.parallel_invention = True
            self.soft_promote = False
        else:
            # "custom" — leave toggles as-is
            pass
        return self

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


def compute_effective_target(
    sharpe_target: float,
    turn: int,
    adaptive_sharpe_target: bool,
    adaptive_start_factor: float,
    adaptive_ramp_turns: int,
) -> float:
    """Compute the effective Sharpe target for a given turn.

    When adaptive targeting is enabled, the target starts at
    ``sharpe_target * adaptive_start_factor`` on turn 1 and linearly
    ramps up to ``sharpe_target`` over ``adaptive_ramp_turns`` turns.
    After the ramp period (or when disabled), returns the original target.

    Args:
        sharpe_target: The original/final Sharpe target from the mandate.
        turn: The current turn number (1-indexed).
        adaptive_sharpe_target: Whether adaptive targeting is enabled.
        adaptive_start_factor: Multiplier for the initial target (e.g. 0.5).
        adaptive_ramp_turns: Number of turns to ramp from start to full.

    Returns:
        The effective Sharpe target for this turn.
    """
    if not adaptive_sharpe_target:
        return sharpe_target

    if turn <= adaptive_ramp_turns:
        progress = (turn - 1) / adaptive_ramp_turns
        return sharpe_target * (
            adaptive_start_factor
            + (1.0 - adaptive_start_factor) * progress
        )

    return sharpe_target


# ── Standalone validation config ────────────────────────────────────────
# These defaults mirror the keyword arguments of ``rolling_walk_forward()``
# in ``crabquant.validation``.  Rolling walk-forward is the default validation
# method (replaced single-split walk_forward_test in Phase 5).

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
    # rolling_walk_forward() — default validation method
    "train_window": "18mo",
    "test_window": "6mo",
    "step": "6mo",
    "min_avg_test_sharpe": 0.4,          # was 0.3 — raised floor to filter flukes
    "min_windows_passed": 3,              # was 1 — require majority (3/6 windows)
    # rolling sub-config (tightened to match above)
    "rolling": {
        "min_avg_test_sharpe": 0.4,
        "min_windows_passed": 3,
        "min_window_test_sharpe": 0.1,
        "max_window_degradation": 0.8,
    },
    "use_rolling_wf": True,
    # legacy walk_forward_test() thresholds (still available, not default)
    "train_pct": 0.75,
    "min_train_bars": 252,
    "min_test_sharpe": 0.3,
    "min_test_trades": 10,
    "max_degradation": 0.7,
    # per-window thresholds for rolling_walk_forward()
    "min_window_test_sharpe": 0.1,        # was 0.0 (disabled) — each window must be slightly positive
    "max_window_degradation": 0.8,        # was 1.0 (disabled) — cap train→test drop at 80%
    # cross-ticker validation
    "min_cross_ticker_sharpe": 0.3,
    "min_ct_profitable_pct": 0.3,  # min fraction of tickers profitable for robust=True
    # regime-specific strategy thresholds (lower bar — they excel in their regime)
    "regime_specific_wf_sharpe_factor": 0.5,   # multiply min_walk_forward_sharpe by this
    "regime_specific_ct_sharpe_factor": 0.6,   # multiply min_cross_ticker_sharpe by this
    "soft_promote_test_sharpe": 0.3,            # below this, never promote even for regime-specific
}

# ── Regime tagging config ────────────────────────────────────────────────
REGIME_TAG_CONFIG: dict = {
    "sharpe_good_threshold": 0.8,      # Sharpe above this → "preferred" regime
    "sharpe_acceptable_threshold": 0.3, # Sharpe above this → "acceptable"
    "min_bars_per_regime": 20,          # Minimum bars to trust a regime Sharpe
}
