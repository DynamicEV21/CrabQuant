"""
CrabQuant Refinement Pipeline — Automated Parameter Optimization

After the LLM generates a strategy, this module automatically sweeps nearby
parameter combinations to find the best-performing set. This turns many
"low_sharpe" failures into successes without wasting LLM turns on
parameter tuning.

Design principles:
- Lightweight: max ~20 param combinations tested (grid expansion)
- Fast: reuses the same OHLCV data already loaded for the backtest
- Non-destructive: always reports both default and optimized results
- Configurable: sweep can be disabled or tuned per-mandate

The optimization runs AFTER the initial backtest. If the optimized Sharpe
is significantly better (above a threshold), the optimized params replace
the defaults for classification and history recording. The LLM context
includes both results so it understands what changed.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a parameter optimization sweep.

    Attributes:
        default_sharpe: Sharpe ratio with original DEFAULT_PARAMS.
        optimized_sharpe: Best Sharpe ratio found during sweep.
        default_params: Original parameter dict.
        optimized_params: Best parameter dict found.
        default_trades: Trade count with original params.
        optimized_trades: Trade count with optimized params.
        combinations_tested: Number of param combinations evaluated.
        improvement_pct: Percentage improvement from default to optimized.
        was_optimized: Whether optimization found a better result.
        sweep_time_seconds: Time taken for the sweep.
    """
    default_sharpe: float = 0.0
    optimized_sharpe: float = 0.0
    default_params: dict = field(default_factory=dict)
    optimized_params: dict = field(default_factory=dict)
    default_trades: int = 0
    optimized_trades: int = 0
    combinations_tested: int = 0
    improvement_pct: float = 0.0
    was_optimized: bool = False
    sweep_time_seconds: float = 0.0

    def summary(self) -> str:
        """Human-readable summary for logging."""
        if not self.was_optimized:
            return (
                f"Param optimization: no improvement "
                f"(default Sharpe={self.default_sharpe:.3f}, "
                f"{self.combinations_tested} combos tested in "
                f"{self.sweep_time_seconds:.1f}s)"
            )
        return (
            f"Param optimization: IMPROVED "
            f"Sharpe {self.default_sharpe:.3f} → {self.optimized_sharpe:.3f} "
            f"(+{self.improvement_pct:.1f}%), "
            f"trades {self.default_trades} → {self.optimized_trades}, "
            f"{self.combinations_tested} combos in {self.sweep_time_seconds:.1f}s"
        )


def _generate_param_variants(
    base_params: dict,
    sweep_factor: float = 0.5,
    max_combinations: int = 20,
) -> List[dict]:
    """Generate nearby parameter variants from a base parameter dict.

    For each numeric parameter, generates 3 values: base*factor, base, base/factor.
    Then creates all combinations, capped at max_combinations.

    Non-numeric parameters (strings, bools) are kept as-is.

    Args:
        base_params: The original DEFAULT_PARAMS dict.
        sweep_factor: Multiplier for generating nearby values (default 0.5).
            E.g., base=20, factor=0.5 → [10, 20, 40].
        max_combinations: Maximum number of combinations to return.

    Returns:
        List of parameter dicts to test. Always includes the original.
    """
    if not base_params:
        return [base_params]

    # Separate numeric from non-numeric params
    numeric_keys = []
    numeric_values = []
    non_numeric = {}

    for k, v in base_params.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric_keys.append(k)
            # Generate 3 values: base*(1-factor), base, base*(1+factor)
            lo = v * (1 - sweep_factor)
            hi = v * (1 + sweep_factor)
            # Ensure positive for period-type params
            if v > 0:
                # Use float floor to preserve float type for float params
                floor = 1.0 if isinstance(v, float) else 1
                lo = max(floor, lo)
            variants = sorted(set([lo, v, hi]))
            numeric_values.append(variants)
        else:
            non_numeric[k] = v

    if not numeric_keys:
        # All params are non-numeric, nothing to sweep
        return [base_params]

    # Generate all combinations
    all_combos = list(itertools.product(*numeric_values))

    # Cap combinations to avoid exponential blowup
    if len(all_combos) > max_combinations:
        # Sample evenly from the full grid
        step = len(all_combos) / max_combinations
        indices = [int(i * step) for i in range(max_combinations)]
        all_combos = [all_combos[i] for i in indices]

    # Build param dicts
    variants = []
    for combo in all_combos:
        params = dict(non_numeric)
        for k, v in zip(numeric_keys, combo):
            # Preserve int type for int params
            if isinstance(base_params[k], int):
                params[k] = int(round(v))
            else:
                params[k] = v
        variants.append(params)

    # Ensure original is included
    original_present = any(
        all(params.get(k) == base_params.get(k) for k in base_params)
        for params in variants
    )
    if not original_present:
        variants.insert(0, dict(base_params))

    return variants


def _run_param_backtest(
    df: pd.DataFrame,
    strategy_fn,
    params: dict,
    initial_cash: float = 100_000,
    commission: float = 0.001,
) -> Optional[dict]:
    """Run a single backtest with given params and return metrics.

    Args:
        df: OHLCV DataFrame.
        strategy_fn: generate_signals function from strategy module.
        params: Parameter dict to use.
        initial_cash: Starting capital.
        commission: Per-trade commission.

    Returns:
        Dict with sharpe, num_trades, max_drawdown, total_return, or None on error.
    """
    try:
        entries, exits = strategy_fn(df, params)

        # Quick sanity: skip if no signals at all
        if entries is None or exits is None:
            return None
        if entries.sum() == 0 and exits.sum() == 0:
            return None

        import vectorbt as vbt

        pf = vbt.Portfolio.from_signals(
            close=df["close"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            entries=entries,
            exits=exits,
            init_cash=initial_cash,
            fees=commission,
            freq="1D",
            accumulate=False,
            call_seq="auto",
        )

        stats = pf.stats()
        sharpe = float(stats.get("Sharpe Ratio", 0))
        total_return = float(stats.get("Total Return [%]", 0)) / 100
        max_dd = -float(stats.get("Max Drawdown [%]", 0)) / 100
        num_trades = int(stats.get("Total Trades", 0))
        win_rate = float(stats.get("Win Rate [%]", 0)) / 100

        return {
            "sharpe": sharpe,
            "num_trades": num_trades,
            "max_drawdown": max_dd,
            "total_return": total_return,
            "win_rate": win_rate,
            "params": params,
        }
    except Exception:
        return None


def optimize_parameters(
    df: pd.DataFrame,
    strategy_fn,
    base_params: dict,
    sweep_factor: float = 0.5,
    max_combinations: int = 20,
    min_improvement: float = 0.1,
    min_trades: int = 5,
) -> OptimizationResult:
    """Run parameter optimization sweep on a strategy.

    Tests nearby parameter combinations and returns the best one found.
    Only reports improvement if the optimized Sharpe is at least
    min_improvement better than the default, AND meets the min_trades threshold.

    Args:
        df: OHLCV DataFrame (already loaded).
        strategy_fn: generate_signals function from strategy module.
        base_params: The DEFAULT_PARAMS dict from the strategy module.
        sweep_factor: How far to sweep around base values (default 0.5).
        max_combinations: Maximum param combinations to test.
        min_improvement: Minimum Sharpe improvement to count as optimized.
        min_trades: Minimum trades required for a valid result.

    Returns:
        OptimizationResult with default and optimized metrics.
    """
    t0 = time.time()

    # Run default params first
    default_result = _run_param_backtest(df, strategy_fn, base_params)
    if default_result is None:
        return OptimizationResult(
            default_params=base_params,
            optimized_params=base_params,
        )

    default_sharpe = default_result["sharpe"]
    default_trades = default_result["num_trades"]

    # Generate variants
    variants = _generate_param_variants(base_params, sweep_factor, max_combinations)

    best_sharpe = default_sharpe
    best_params = dict(base_params)
    best_trades = default_trades
    best_result = default_result

    # Sweep through variants (skip first if it's the original)
    for params in variants:
        result = _run_param_backtest(df, strategy_fn, params)
        if result is None:
            continue

        # Must meet minimum trade count
        if result["num_trades"] < min_trades:
            continue

        # Prefer higher Sharpe, with trade count as tiebreaker
        if (result["sharpe"] > best_sharpe or
                (result["sharpe"] == best_sharpe and result["num_trades"] > best_trades)):
            best_sharpe = result["sharpe"]
            best_params = result["params"]
            best_trades = result["num_trades"]
            best_result = result

    elapsed = time.time() - t0

    # Determine if optimization found something meaningfully better
    improvement = best_sharpe - default_sharpe
    improvement_pct = (improvement / abs(default_sharpe) * 100) if default_sharpe != 0 else 0
    was_optimized = (
        improvement >= min_improvement
        and best_trades >= min_trades
        and best_params != base_params
    )

    return OptimizationResult(
        default_sharpe=default_sharpe,
        optimized_sharpe=best_sharpe,
        default_params=base_params,
        optimized_params=best_params if was_optimized else dict(base_params),
        default_trades=default_trades,
        optimized_trades=best_trades,
        combinations_tested=len(variants),
        improvement_pct=improvement_pct,
        was_optimized=was_optimized,
        sweep_time_seconds=elapsed,
    )


def format_optimization_for_prompt(opt_result: OptimizationResult) -> str:
    """Format optimization results for injection into LLM context.

    This tells the LLM that parameter optimization was applied and
    what the results were, so it can focus on structural changes
    rather than parameter tuning in future turns.

    Args:
        opt_result: The OptimizationResult from optimize_parameters().

    Returns:
        Formatted string for prompt injection, or empty string if not optimized.
    """
    if not opt_result.was_optimized:
        return ""

    lines = [
        "## ⚡ Parameter Optimization Applied",
        "",
        f"Default params gave Sharpe={opt_result.default_sharpe:.3f} "
        f"({opt_result.default_trades} trades).",
        f"After automatic parameter sweep ({opt_result.combinations_tested} "
        f"combos, {opt_result.sweep_time_seconds:.1f}s):",
        f"  → Optimized Sharpe={opt_result.optimized_sharpe:.3f} "
        f"({opt_result.optimized_trades} trades, "
        f"+{opt_result.improvement_pct:.1f}%).",
        "",
        "Key param changes:",
    ]

    for key in opt_result.default_params:
        default_val = opt_result.default_params[key]
        optimized_val = opt_result.optimized_params.get(key, default_val)
        if default_val != optimized_val:
            lines.append(f"  - {key}: {default_val} → {optimized_val}")

    lines.append("")
    lines.append(
        "💡 Focus on STRUCTURAL changes (indicator replacement, entry/exit logic) "
        "in your next iteration. Parameter tuning is handled automatically."
    )

    return "\n".join(lines)


def format_optimization_for_context(opt_result: OptimizationResult) -> dict:
    """Format optimization results as a dict for context injection.

    Args:
        opt_result: The OptimizationResult from optimize_parameters().

    Returns:
        Dict with optimization metadata for context_builder.
    """
    return {
        "param_optimization_applied": opt_result.was_optimized,
        "default_sharpe": opt_result.default_sharpe,
        "optimized_sharpe": opt_result.optimized_sharpe,
        "improvement_pct": opt_result.improvement_pct,
        "combinations_tested": opt_result.combinations_tested,
        "sweep_time_seconds": opt_result.sweep_time_seconds,
    }
