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
from scipy.optimize import differential_evolution

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


def _compute_param_ranges(
    base_params: dict,
    sweep_factor: float = 0.5,
) -> Dict[str, Tuple[float, float]]:
    """Compute DE-compatible param ranges from base params.

    For each numeric param, generates (lo, hi) bounds using the same
    logic as _generate_param_variants: base*(1-factor) to base*(1+factor).
    Non-numeric params are excluded (DE only handles continuous params).

    Args:
        base_params: The original DEFAULT_PARAMS dict.
        sweep_factor: How far to sweep around base values (default 0.5).

    Returns:
        Dict mapping param names to (min, max) bounds for DE.
    """
    param_ranges = {}
    for k, v in base_params.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lo = v * (1 - sweep_factor)
            hi = v * (1 + sweep_factor)
            if v > 0:
                floor = 1.0 if isinstance(v, float) else 1
                lo = max(floor, lo)
            param_ranges[k] = (float(lo), float(hi))
    return param_ranges


def optimize_parameters(
    df: pd.DataFrame,
    strategy_fn,
    base_params: dict,
    sweep_factor: float = 0.5,
    max_combinations: int = 20,
    min_improvement: float = 0.1,
    min_trades: int = 5,
    sharpe_target: Optional[float] = None,
    use_de: bool = True,
) -> OptimizationResult:
    """Run parameter optimization sweep on a strategy.

    By default, uses scipy differential_evolution (DE) as the primary
    optimization method for strategies with >= 2 numeric params. DE explores
    the parameter space much more thoroughly than grid search and reliably
    finds better optima. Falls back to grid search if DE fails or if there
    are fewer than 2 numeric params.

    Only reports improvement if the optimized Sharpe is at least
    min_improvement better than the default, AND meets the min_trades threshold.

    When sharpe_target is set, optimization also counts as successful if
    the best Sharpe meets or exceeds the target, even if the improvement
    is below min_improvement. This enables "gap rescue" mode where the
    optimizer specifically tries to push a strategy above a threshold.

    Args:
        df: OHLCV DataFrame (already loaded).
        strategy_fn: generate_signals function from strategy module.
        base_params: The DEFAULT_PARAMS dict from the strategy module.
        sweep_factor: How far to sweep around base values (default 0.5).
        max_combinations: Maximum param combinations for grid fallback.
        min_improvement: Minimum Sharpe improvement to count as optimized.
        min_trades: Minimum trades required for a valid result.
        sharpe_target: If set, optimization also succeeds when best_sharpe >= target.
        use_de: Whether to use differential evolution (default True).

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

    # ── Phase 1: Try scipy DE (if enabled and enough numeric params) ──
    de_used = False
    de_n_evals = 0
    if use_de:
        param_ranges = _compute_param_ranges(base_params, sweep_factor)
        if len(param_ranges) >= 2:
            try:
                de_config = {
                    "maxiter": 15,
                    "popsize": 10,
                    "workers": 1,  # serial to avoid multiprocessing issues
                    "seed": 42,
                    "tol": 1e-4,
                }
                de_result = optimize_with_de(
                    strategy_fn, param_ranges, df, de_config,
                )
                de_n_evals = de_result["n_evals"]
                de_used = True

                if de_result["best_sharpe"] > default_sharpe:
                    # Verify the DE result meets min_trades
                    de_params = de_result["best_params"]
                    # Preserve int types for int params
                    for k, v in de_params.items():
                        if isinstance(base_params.get(k), int):
                            de_params[k] = int(round(v))
                    de_verify = _run_param_backtest(df, strategy_fn, de_params)

                    if (de_verify is not None
                            and de_verify["num_trades"] >= min_trades):
                        elapsed = time.time() - t0
                        improvement = de_result["best_sharpe"] - default_sharpe
                        improvement_pct = (
                            improvement / abs(default_sharpe) * 100
                            if default_sharpe != 0 else 0
                        )
                        target_reached = (
                            sharpe_target is not None
                            and de_result["best_sharpe"] >= sharpe_target
                        )
                        was_optimized = (
                            (improvement >= min_improvement or target_reached)
                            and de_verify["num_trades"] >= min_trades
                            and de_params != base_params
                        )
                        logger.info(
                            "DE optimization: Sharpe %.3f → %.3f (+%.1f%%), "
                            "%d evals in %.1fs, %d trades",
                            default_sharpe, de_result["best_sharpe"],
                            improvement_pct, de_n_evals, elapsed,
                            de_verify["num_trades"],
                        )
                        return OptimizationResult(
                            default_sharpe=default_sharpe,
                            optimized_sharpe=de_result["best_sharpe"],
                            default_params=base_params,
                            optimized_params=(
                                de_params if was_optimized
                                else dict(base_params)
                            ),
                            default_trades=default_trades,
                            optimized_trades=de_verify["num_trades"],
                            combinations_tested=de_n_evals,
                            improvement_pct=improvement_pct,
                            was_optimized=was_optimized,
                            sweep_time_seconds=elapsed,
                        )

                # DE ran but didn't improve — return early, skip grid
                elapsed = time.time() - t0
                logger.info(
                    "DE optimization: no improvement (Sharpe %.3f, %d evals "
                    "in %.1fs), skipping grid fallback",
                    default_sharpe, de_n_evals, elapsed,
                )
                return OptimizationResult(
                    default_sharpe=default_sharpe,
                    optimized_sharpe=default_sharpe,
                    default_params=base_params,
                    optimized_params=dict(base_params),
                    default_trades=default_trades,
                    optimized_trades=default_trades,
                    combinations_tested=de_n_evals,
                    improvement_pct=0.0,
                    was_optimized=False,
                    sweep_time_seconds=elapsed,
                )
            except Exception:
                logger.warning(
                    "DE optimization failed, falling back to grid search",
                    exc_info=True,
                )

    # ── Phase 2: Grid search fallback (DE disabled/failed/<2 params) ──
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

        # Prefer higher Sharpe, with trade count as near-Sharpe tiebreaker.
        # When Sharpe is within 5% of best, prefer the variant with more trades.
        sharpe_epsilon = best_sharpe * 0.05 if best_sharpe > 0 else 0.05
        if (result["sharpe"] > best_sharpe or
                (abs(result["sharpe"] - best_sharpe) <= sharpe_epsilon
                 and result["num_trades"] > best_trades) or
                (result["sharpe"] == best_sharpe and result["num_trades"] > best_trades)):
            best_sharpe = result["sharpe"]
            best_params = result["params"]
            best_trades = result["num_trades"]
            best_result = result

    elapsed = time.time() - t0

    # Determine if optimization found something meaningfully better
    improvement = best_sharpe - default_sharpe
    improvement_pct = (improvement / abs(default_sharpe) * 100) if default_sharpe != 0 else 0
    target_reached = sharpe_target is not None and best_sharpe >= sharpe_target
    was_optimized = (
        (improvement >= min_improvement or target_reached)
        and best_trades >= min_trades
        and best_params != base_params
    )

    method = "differential_evolution" if de_used and was_optimized else "grid"
    logger.info(
        "%s optimization: Sharpe %.3f → %.3f (%s), %d evals in %.1fs",
        method, default_sharpe, best_sharpe,
        "improved" if was_optimized else "no improvement",
        de_n_evals + len(variants) if de_used else len(variants),
        elapsed,
    )

    return OptimizationResult(
        default_sharpe=default_sharpe,
        optimized_sharpe=best_sharpe,
        default_params=base_params,
        optimized_params=best_params if was_optimized else dict(base_params),
        default_trades=default_trades,
        optimized_trades=best_trades,
        combinations_tested=de_n_evals + len(variants) if de_used else len(variants),
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


def optimize_for_trade_count(
    df: pd.DataFrame,
    strategy_fn,
    base_params: dict,
    current_sharpe: float,
    target_trades: int = 20,
    max_sharpe_penalty: float = 0.3,
    sweep_factor: float = 0.6,
    max_combinations: int = 15,
) -> OptimizationResult:
    """Optimize parameters specifically to increase trade count.

    When a strategy has good Sharpe but too few trades for validation,
    this sweeps wider parameter ranges and accepts a Sharpe penalty
    in exchange for reaching the target trade count.

    The optimization criterion is:
    1. Primary: trades >= target_trades
    2. Secondary: maximize Sharpe among variants that hit the trade target
    3. Fallback: if no variant hits target_trades, pick the one with most trades
       (as long as Sharpe > current_sharpe * (1 - max_sharpe_penalty))

    Args:
        df: OHLCV DataFrame (already loaded).
        strategy_fn: generate_signals function from strategy module.
        base_params: The DEFAULT_PARAMS dict from the strategy module.
        current_sharpe: The Sharpe ratio achieved with base params.
        target_trades: Minimum trades needed (default 20 for validation).
        max_sharpe_penalty: Max acceptable Sharpe drop as fraction (0.3 = 30%).
        sweep_factor: How far to sweep around base values (wider than default).
        max_combinations: Maximum param combinations to test.

    Returns:
        OptimizationResult indicating whether trade count was rescued.
    """
    t0 = time.time()

    min_sharpe = current_sharpe * (1 - max_sharpe_penalty)

    # Run default params first
    default_result = _run_param_backtest(df, strategy_fn, base_params)
    if default_result is None:
        return OptimizationResult(
            default_params=base_params,
            optimized_params=base_params,
        )

    default_sharpe = default_result["sharpe"]
    default_trades = default_result["num_trades"]

    # If default already has enough trades, no rescue needed
    if default_trades >= target_trades:
        return OptimizationResult(
            default_sharpe=default_sharpe,
            optimized_sharpe=default_sharpe,
            default_params=base_params,
            optimized_params=dict(base_params),
            default_trades=default_trades,
            optimized_trades=default_trades,
            combinations_tested=0,
            was_optimized=False,
            sweep_time_seconds=0.0,
        )

    # Generate variants with wider sweep
    variants = _generate_param_variants(base_params, sweep_factor, max_combinations)

    # Collect all valid results
    valid_results = []
    for params in variants:
        result = _run_param_backtest(df, strategy_fn, params)
        if result is None:
            continue
        # Must not tank Sharpe below minimum threshold
        if result["sharpe"] < min_sharpe:
            continue
        valid_results.append(result)

    elapsed = time.time() - t0

    if not valid_results:
        return OptimizationResult(
            default_sharpe=default_sharpe,
            optimized_sharpe=default_sharpe,
            default_params=base_params,
            optimized_params=dict(base_params),
            default_trades=default_trades,
            optimized_trades=default_trades,
            combinations_tested=len(variants),
            was_optimized=False,
            sweep_time_seconds=elapsed,
        )

    # Phase 1: Find variants that hit the trade target
    target_hitters = [r for r in valid_results if r["num_trades"] >= target_trades]

    if target_hitters:
        # Pick the one with best Sharpe among those that hit the target
        best = max(target_hitters, key=lambda r: r["sharpe"])
        improvement = best["sharpe"] - default_sharpe
        improvement_pct = (improvement / abs(default_sharpe) * 100) if default_sharpe != 0 else 0
        return OptimizationResult(
            default_sharpe=default_sharpe,
            optimized_sharpe=best["sharpe"],
            default_params=base_params,
            optimized_params=best["params"],
            default_trades=default_trades,
            optimized_trades=best["num_trades"],
            combinations_tested=len(variants),
            improvement_pct=improvement_pct,
            was_optimized=True,
            sweep_time_seconds=elapsed,
        )

    # Phase 2: Fallback — pick the variant with most trades
    best_by_trades = max(valid_results, key=lambda r: (r["num_trades"], r["sharpe"]))

    # Only report as optimized if we got meaningfully more trades
    if best_by_trades["num_trades"] > default_trades * 1.5:
        improvement = best_by_trades["sharpe"] - default_sharpe
        improvement_pct = (improvement / abs(default_sharpe) * 100) if default_sharpe != 0 else 0
        return OptimizationResult(
            default_sharpe=default_sharpe,
            optimized_sharpe=best_by_trades["sharpe"],
            default_params=base_params,
            optimized_params=best_by_trades["params"],
            default_trades=default_trades,
            optimized_trades=best_by_trades["num_trades"],
            combinations_tested=len(variants),
            improvement_pct=improvement_pct,
            was_optimized=True,
            sweep_time_seconds=elapsed,
        )

    return OptimizationResult(
        default_sharpe=default_sharpe,
        optimized_sharpe=default_sharpe,
        default_params=base_params,
        optimized_params=dict(base_params),
        default_trades=default_trades,
        optimized_trades=default_trades,
        combinations_tested=len(variants),
        was_optimized=False,
        sweep_time_seconds=elapsed,
    )


def format_trade_count_rescue_for_prompt(
    opt_result: OptimizationResult,
    target_trades: int = 20,
) -> str:
    """Format trade count rescue results for LLM prompt injection.

    Args:
        opt_result: The OptimizationResult from optimize_for_trade_count().
        target_trades: The trade count target we were trying to reach.

    Returns:
        Formatted string for prompt injection, or empty string if not optimized.
    """
    if not opt_result.was_optimized:
        return ""

    hit_target = opt_result.optimized_trades >= target_trades
    sharpe_drop = opt_result.default_sharpe - opt_result.optimized_sharpe

    if hit_target:
        header = f"## 🎯 Trade Count Rescue — TARGET REACHED ({opt_result.optimized_trades} trades)"
    else:
        header = f"## 🎯 Trade Count Rescue — Partial ({opt_result.optimized_trades}/{target_trades} trades)"

    lines = [
        header,
        "",
        f"Original: Sharpe={opt_result.default_sharpe:.3f}, {opt_result.default_trades} trades",
        f"After sweep ({opt_result.combinations_tested} combos, {opt_result.sweep_time_seconds:.1f}s):",
        f"  → Sharpe={opt_result.optimized_sharpe:.3f}, {opt_result.optimized_trades} trades",
    ]

    if sharpe_drop > 0:
        lines.append(f"  → Sharpe penalty: -{sharpe_drop:.3f} (acceptable to gain trades)")

    if hit_target:
        lines.append("")
        lines.append("✅ The strategy now has enough trades for walk-forward validation!")
        lines.append("💡 If validation fails, the LLM may still need to simplify the logic.")
    else:
        lines.append("")
        lines.append(f"⚠️ Still {target_trades - opt_result.optimized_trades} trades short of validation threshold.")
        lines.append("💡 The LLM should further loosen entry conditions or shorten indicator periods.")

    return "\\\\n".join(lines)


def optimize_with_de(
    strategy_fn,
    param_ranges: Dict[str, Tuple[float, float]],
    data: pd.DataFrame,
    config: Optional[dict] = None,
) -> dict:
    """Optimize strategy parameters using scipy differential evolution.

    Uses a population-based global search to find parameters that maximize
    the Sharpe ratio. Much more effective than grid search for continuous
    parameter spaces with many dimensions.

    Args:
        strategy_fn: generate_signals(df, params) -> (entries, exits) function.
            Must be compatible with _run_param_backtest (returns entries/exits).
        param_ranges: Dict mapping param names to (min, max) bounds.
            E.g. {"rsi_period": (5, 30), "bb_std": (1.0, 3.0)}.
        data: OHLCV DataFrame for backtesting.
        config: Optional dict with DE settings. Supported keys:
            - maxiter (int, default 50): Max generations.
            - popsize (int, default 15): Population size multiplier.
            - tol (float, default 1e-4): Convergence tolerance.
            - workers (int, default -1): Parallel workers (-1 = all CPUs).
            - polish (bool, default True): Local refinement after DE.
            - seed (int, default 42): Random seed for reproducibility.

    Returns:
        Dict with keys:
            - best_params (dict): Best parameter values found.
            - best_sharpe (float): Sharpe ratio at best params.
            - n_evals (int): Number of function evaluations.
            - success (bool): Whether optimization converged.
            - method (str): Always "differential_evolution".
    """
    cfg = {
        "maxiter": 50,
        "popsize": 15,
        "tol": 1e-4,
        "workers": -1,
        "polish": True,
        "seed": 42,
    }
    if config:
        cfg.update(config)

    param_names = list(param_ranges.keys())
    bounds = list(param_ranges.values())

    n_evals = [0]  # mutable counter for closure

    def objective(x: np.ndarray) -> float:
        """Objective function: negative Sharpe ratio (DE minimizes)."""
        n_evals[0] += 1
        params = dict(zip(param_names, x.tolist()))

        try:
            result = _run_param_backtest(data, strategy_fn, params)
            if result is None:
                return 100.0  # penalty for failed backtest
            sharpe = result.get("sharpe", 0.0)
            # DE minimizes, so negate Sharpe
            return -sharpe
        except Exception:
            logger.debug(
                "DE objective failed for params %s", params, exc_info=True
            )
            return 100.0  # penalty for exceptions

    try:
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=cfg["maxiter"],
            popsize=cfg["popsize"],
            tol=cfg["tol"],
            workers=cfg["workers"],
            polish=cfg["polish"],
            seed=cfg["seed"],
            updating="deferred" if cfg["workers"] != 1 else "immediate",
        )

        best_params = dict(zip(param_names, de_result.x.tolist()))
        best_sharpe = -de_result.fun

        return {
            "best_params": best_params,
            "best_sharpe": best_sharpe,
            "n_evals": n_evals[0],
            "success": de_result.success,
            "method": "differential_evolution",
        }

    except Exception as e:
        logger.warning("differential_evolution failed: %s", e)
        # Return a graceful failure with midpoint params
        return {
            "best_params": {k: (lo + hi) / 2 for k, (lo, hi) in param_ranges.items()},
            "best_sharpe": 0.0,
            "n_evals": n_evals[0],
            "success": False,
            "method": "differential_evolution",
            "error": str(e),
        }
