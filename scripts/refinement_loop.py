#!/usr/bin/env python3
"""
CrabQuant Refinement Pipeline — Main Orchestrator Loop

This script implements the core refinement loop that:
1. Loads a mandate (strategy requirements)
2. Runs up to 7 turns of LLM → validate → backtest → classify
3. Handles all error paths and stagnation detection
4. Promotes successful strategies to the winner registry

Usage:
  python scripts/refinement_loop.py --mandate mandates/momentum_spy.json
  python scripts/refinement_loop.py --mandate mandates/momentum_spy.json --max-turns 5
  python scripts/refinement_loop.py --mandate mandates/momentum_spy.json --sharpe-target 2.0
"""

import json
import logging
import os
import tempfile
import time
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys

logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import CrabQuant modules
from crabquant.refinement.schemas import RunState, BacktestReport, StrategyModification
from crabquant.refinement.config import RefinementConfig, compute_effective_target
from crabquant.refinement.module_loader import load_strategy_module
from crabquant.refinement.validation_gates import run_validation_gates
from crabquant.refinement.diagnostics import (
    run_backtest_safely, compute_sharpe_by_year, compute_strategy_hash,
    run_multi_ticker_backtest,
)
from crabquant.refinement.classifier import classify_failure
from crabquant.refinement.context_builder import build_llm_context
from crabquant.refinement.llm_api import call_zai_llm, call_llm_inventor, load_api_config
from crabquant.guardrails import check_guardrails, GuardrailReport, GuardrailConfig

# Phase 3 module imports — replace all inline implementations
from crabquant.refinement.circuit_breaker import CircuitBreaker
from crabquant.refinement.cosmetic_guard import check_cosmetic_guard, CosmeticGuardState
from crabquant.refinement.action_analytics import (
    track_action_result, generate_llm_context as generate_analytics_context,
    load_run_history,
)
from crabquant.refinement.promotion import auto_promote, run_full_validation_check, soft_promote
from crabquant.refinement.stagnation import compute_stagnation, get_stagnation_response
from crabquant.refinement.wave_dashboard import generate_dashboard, snapshot_to_json
from crabquant.refinement.prompts import (
    get_parallel_prompt_variants,
    compute_composite_score,
    VALID_ACTIONS,
)


# ── Action Type Validation ──────────────────────────────────────────────────
# The LLM frequently returns action types that aren't in VALID_ACTIONS.
# This mapping auto-corrects common invalid actions to valid ones.

ACTION_ALIASES: Dict[str, str] = {
    "new_strategy": "novel",
    "create": "novel",
    "propose_strategy": "novel",
    "propose": "novel",
    "iterate": "modify_params",
    "simplify": "replace_indicator",
    "modify": "modify_params",
    "refine": "replace_indicator",
    "refine_params": "modify_params",
}

DEFAULT_FALLBACK_ACTION = "novel"


def validate_action(raw_action: str) -> Tuple[str, bool]:
    """Validate and normalise an LLM-returned action type.

    If the action is already in VALID_ACTIONS, return it unchanged.
    If it matches a known alias, map it to the canonical action.
    Otherwise, fall back to ``DEFAULT_FALLBACK_ACTION`` ("novel") and log a warning.

    Args:
        raw_action: The action string from the LLM response.

    Returns:
        Tuple of (validated_action, was_remapped).
        ``was_remapped`` is True if the action was not already valid.
    """
    # Strip whitespace first so "  novel  " still matches
    stripped = raw_action.strip() if raw_action else ""
    if stripped in VALID_ACTIONS:
        return stripped, False

    # Try alias lookup (case-insensitive)
    normalised = stripped.lower().replace("-", "_")
    if normalised in ACTION_ALIASES:
        mapped = ACTION_ALIASES[normalised]
        logger.info(
            "Action validation: mapped '%s' → '%s'",
            raw_action, mapped,
        )
        return mapped, True

    # Unknown action — default to "novel" with a warning
    logger.warning(
        "Action validation: unknown action '%s' — defaulting to '%s'. "
        "Consider adding an alias for this action.",
        raw_action, DEFAULT_FALLBACK_ACTION,
    )
    return DEFAULT_FALLBACK_ACTION, True


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON from file."""
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON to file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def create_run_directory(mandate: Dict[str, Any]) -> Path:
    """Create timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mandate_name = mandate.get("name", "unknown").replace(" ", "_").lower()
    run_dir = project_root / "refinement_runs" / f"{mandate_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def acquire_lock(run_dir: Path) -> bool:
    """Acquire lock directory to prevent concurrent runs."""
    lock_file = run_dir / "lock.json"
    if lock_file.exists():
        try:
            lock_data = load_json(str(lock_file))
            lock_time = datetime.fromisoformat(lock_data["timestamp"])
            # Stale lock if older than 1 hour
            if (datetime.now(timezone.utc) - lock_time).total_seconds() < 3600:
                return False
        except:
            pass
    
    lock_data = {
        "pid": os.getpid(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    write_json(lock_file, lock_data)
    return True


def release_lock(run_dir: Path) -> None:
    """Release lock."""
    lock_file = run_dir / "lock.json"
    if lock_file.exists():
        lock_file.unlink()


def save_state(run_dir: Path, state: RunState) -> None:
    """Save RunState to file."""
    write_json(run_dir / "state.json", state.__dict__)


def load_state(run_dir: Path) -> Optional[RunState]:
    """Load RunState from file."""
    state_file = run_dir / "state.json"
    if not state_file.exists():
        return None
    
    try:
        data = load_json(str(state_file))
        return RunState(**data)
    except:
        return None


def save_report(run_dir: Path, turn: int, report: BacktestReport) -> None:
    """Save BacktestReport to file."""
    write_json(run_dir / f"report_v{turn}.json", report.__dict__)


def load_report(run_dir: Path, turn: int) -> Optional[BacktestReport]:
    """Load BacktestReport from file."""
    report_file = run_dir / f"report_v{turn}.json"
    if not report_file.exists():
        return None
    
    try:
        data = load_json(str(report_file))
        return BacktestReport(**data)
    except:
        return None


# Backwards-compatible aliases: keep these in the module namespace so that
# existing test patches (e.g. @patch("refinement_loop.promote_to_winner"))
# continue to resolve correctly.
promote_to_winner = None  # Will be set below
run_full_validation = None  # Will be set below

# Import the promotion module's promote_to_winner for backwards compat
from crabquant.refinement.promotion import promote_to_winner as _module_promote_to_winner
promote_to_winner = _module_promote_to_winner


def run_full_validation(state: RunState, run_dir: Path) -> Dict[str, Any]:
    """Run full validation on the best strategy (walk-forward + cross-ticker).
    
    This is a lightweight wrapper that returns a simple result dict.
    The heavy lifting is done by the promotion module's run_full_validation_check.
    """
    try:
        from crabquant.validation import full_validation
        best_code = Path(state.best_code_path).read_text()
        # Placeholder: actual validation would load strategy and run
        return {"status": "ok", "message": "Validation passed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def clear_cache() -> None:
    """Clear indicator cache between turns."""
    # Implementation depends on your caching system
    pass


def _promote_post_loop(
    strategy_code: str,
    strategy_module: Any,
    state: RunState,
    run_dir: Path,
    mandate: Dict[str, Any],
    report: Optional[BacktestReport] = None,
) -> None:
    """Attempt to promote the best strategy after the loop ends.

    This runs full validation (walk-forward + cross-ticker).  If validation
    passes, uses ``auto_promote``; otherwise falls back to legacy
    ``promote_to_winner`` so the strategy is at least recorded in the
    winners file.
    """
    # Try full validation first
    validation: Dict[str, Any] = {"status": "ok", "passed": False}
    is_regime_specific = False
    try:
        strategy_fn = strategy_module.generate_signals
        params = (strategy_module.DEFAULT_PARAMS.copy()
                  if hasattr(strategy_module, 'DEFAULT_PARAMS') else {})
        primary_ticker = mandate.get("primary_ticker", state.tickers[0])
        validation_tickers = mandate.get("tickers", [primary_ticker])

        # Quick regime-specificity pre-check (uses backtest on primary ticker)
        try:
            from crabquant.refinement.regime_tagger import compute_strategy_regime_tags
            tags = compute_strategy_regime_tags(strategy_fn, params, ticker=primary_ticker)
            is_regime_specific = tags.get("is_regime_specific", False)
        except Exception:
            pass  # Non-critical — proceed without regime info

        validation = run_full_validation_check(
            strategy_fn=strategy_fn,
            params=params,
            discovery_ticker=primary_ticker,
            validation_tickers=validation_tickers,
            is_regime_specific=is_regime_specific,
            rolling_config={
                "min_window_test_sharpe": config.min_window_test_sharpe,
                "max_window_degradation": config.max_window_degradation,
            },
        )
    except Exception as e:
        print(f"  ⚠️ Post-loop validation error: {e}")
        validation = {"status": "error", "message": str(e), "passed": False}

    if validation.get("passed", False):
        print(f"  📋 Post-loop validation passed — auto-promoting…")
        # Build a lightweight result-like object for auto_promote
        result_proxy = _make_result_proxy(state, report)
        try:
            promo_result = auto_promote(
                strategy_code=strategy_code,
                strategy_module=strategy_module,
                result=result_proxy,
                validation=validation,
                state=state,
            )
            if promo_result.get("registered"):
                print(f"  ✅ Post-loop strategy registered: {promo_result['strategy_name']}")
                state.status = "success"  # upgrade status
                save_state(run_dir, state)
            else:
                print(f"  ⚠️ Post-loop auto-promote skipped: {promo_result.get('error', 'unknown')}")
        except Exception as e:
            print(f"  ⚠️ Post-loop auto-promote error: {e}")
    else:
        # Validation didn't pass but Sharpe is excellent — still record in winners
        print(f"  📋 Post-loop validation not passed — using legacy promotion…")
        result_proxy = _make_result_proxy(state, report)
        try:
            promote_to_winner(strategy_code, result_proxy, validation, state,
                              strategy_module=strategy_module)
            state.status = "success"  # upgrade status
            save_state(run_dir, state)
            print(f"  ✅ Post-loop legacy promotion done")
        except Exception as e:
            print(f"  ⚠️ Post-loop legacy promotion error: {e}")


def _make_result_proxy(state: RunState, report: Optional[BacktestReport] = None) -> Any:
    """Build a lightweight mock of BacktestResult from RunState + report."""
    proxy = type('BacktestResultProxy', (), {})()
    proxy.sharpe = state.best_sharpe
    proxy.total_return = report.total_return_pct if report else 0.0
    proxy.max_drawdown = report.max_drawdown_pct if report else 0.0
    proxy.num_trades = report.total_trades if report else 0
    proxy.win_rate = report.win_rate if report else 0.0
    proxy.profit_factor = report.profit_factor if report else 0.0
    proxy.calmar_ratio = report.calmar_ratio if report else 0.0
    proxy.sortino_ratio = report.sortino_ratio if report else 0.0
    proxy.score = report.composite_score if report else 0.0
    proxy.passed = True  # Sharpe already verified >= target
    proxy.params = report.current_params if report else {}
    proxy.ticker = state.tickers[0] if state.tickers else "SPY"
    proxy.strategy_name = f"refined_{state.mandate_name}"
    proxy.iteration = state.best_turn
    return proxy


def _build_retry_feedback(gate_errors: list[str]) -> str:
    """Build specific, actionable feedback from gate failure errors.

    Categorizes errors by type (syntax/import, signal sanity, smoke backtest)
    and provides targeted guidance for the LLM to fix them.

    Args:
        gate_errors: List of error strings from run_validation_gates.

    Returns:
        A feedback string to inject into the next LLM prompt.
    """
    if not gate_errors:
        return "Your previous code failed validation but no specific error was reported. Please try again."

    parts: list[str] = []
    parts.append(f"Your previous code failed validation with {len(gate_errors)} error(s). Fix them:")

    for err in gate_errors:
        err_lower = err.lower()
        if "syntaxerror" in err_lower or "importerror" in err_lower or "missing required" in err_lower:
            parts.append(f"  - {err}. Fix the syntax/import error or add the missing definitions.")
        elif any(kw in err_lower for kw in ("zero entry", "signal", "entries must be", "exits must be", "overtrading", "runtime error in generate_signals", "index does not match", "contain nan")):
            parts.append(f"  - {err}. Your entry/exit conditions may be too restrictive or produce wrong types. Loosen them or fix the logic.")
        elif "backtest" in err_lower or "smoke" in err_lower:
            parts.append(f"  - {err}. Common causes: wrong column names, incompatible data types.")
        else:
            parts.append(f"  - {err}.")

    return "\n".join(parts)


def _run_parallel_invention(
    context: Dict[str, Any],
    context_path: Path,
    run_dir: Path,
    mandate: Dict[str, Any],
    config: RefinementConfig,
    cb: Any,
    turn: int,
) -> Optional[Tuple[str, Any, bool, list]]:
    """Run parallel strategy invention on turn 1.

    Spawns N strategies via N LLM calls, each with a different prompt variant.
    Backtests all N, ranks by composite score, returns the best.

    Args:
        context: Base LLM context dict.
        context_path: Path to save context JSON.
        run_dir: Run directory for saving artifacts.
        mandate: Mandate dict.
        config: RefinementConfig (uses parallel_invention_count).
        cb: CircuitBreaker instance.
        turn: Current turn number (should be 1).

    Returns:
        Tuple of (strategy_code, strategy_module, gates_ok, gate_errors) for the
        best parallel strategy, or None if all variants failed.
    """
    count = config.parallel_invention_count
    print(f"  🔄 Parallel invention: spawning {count} strategy variants...", flush=True)

    # Build base prompt from context (reuse the LLM inventor's prompt construction)
    from crabquant.refinement.llm_api import call_llm_inventor

    # Generate prompt variants
    # The base prompt is what call_llm_inventor would build internally.
    # We need to pass variant info through context so call_llm_inventor picks it up.
    parallel_results = []

    for i in range(count):
        variant_label = f"variant_{i}"
        print(f"    Variant {i+1}/{count}...", flush=True)

        # Clone context and add variant bias
        variant_context = dict(context)
        variant_context["parallel_variant_index"] = i
        variant_context["parallel_variant_count"] = count

        # Save variant context
        variant_context_path = run_dir / f"context_v{turn}_variant{i}.json"
        write_json(variant_context_path, variant_context)

        # Call LLM with variant context
        modification = None
        strategy_code = None
        gates_ok = False
        gate_errors = []

        for attempt in range(2):  # Fewer retries per variant (2 instead of 3)
            try:
                modification = call_llm_inventor(
                    context=variant_context,
                    context_path=str(variant_context_path),
                )
                if modification is None:
                    cb.record(False, turn=turn, mandate=variant_label)
                    continue

                strategy_code = modification.get("new_strategy_code", "")
                if not strategy_code:
                    cb.record(False, turn=turn, mandate=variant_label)
                    continue

                variant_ticker = mandate.get("primary_ticker", mandate.get("tickers", ["SPY"])[0])
                variant_period = mandate.get("period", "1y")
                gates_ok, gate_errors = run_validation_gates(strategy_code, ticker=variant_ticker, period=variant_period)
                cb.record(gates_ok, turn=turn, mandate=variant_label)
                if gates_ok:
                    break
            except Exception as e:
                print(f"      Variant {i+1} error (attempt {attempt+1}): {e}")
                cb.record(False, turn=turn, mandate=variant_label)
                continue

        if gates_ok and strategy_code:
            # Save variant strategy
            variant_path = run_dir / f"strategy_v{turn}_variant{i}.py"
            variant_path.write_text(strategy_code)

            # Load and backtest
            load_result = load_strategy_module(variant_path)
            # Handle both old (ModuleType|None) and new (ModuleType | tuple) return types
            if isinstance(load_result, tuple):
                variant_module = load_result[0]
            else:
                variant_module = load_result
            if variant_module is not None:
                primary_ticker = mandate.get("primary_ticker", mandate.get("tickers", ["SPY"])[0])
                period = mandate.get("period", "1y")
                backtest_output = run_backtest_safely(
                    variant_module, primary_ticker, period,
                    return_portfolio=True,
                )
                if backtest_output is not None:
                    result, df, portfolio, _ = backtest_output
                    composite = compute_composite_score(
                        sharpe=result.sharpe,
                        trades=result.num_trades,
                        max_drawdown=result.max_drawdown,
                    )
                    parallel_results.append({
                        "variant_index": i,
                        "strategy_code": strategy_code,
                        "strategy_module": variant_module,
                        "strategy_path": variant_path,
                        "gates_ok": True,
                        "gate_errors": [],
                        "sharpe": result.sharpe,
                        "trades": result.num_trades,
                        "max_drawdown": result.max_drawdown,
                        "composite_score": composite,
                        "modification": modification,
                    })
                    print(f"      Variant {i+1}: Sharpe={result.sharpe:.2f}, "
                          f"Trades={result.num_trades}, Composite={composite:.2f}", flush=True)
                else:
                    parallel_results.append({
                        "variant_index": i,
                        "strategy_code": strategy_code,
                        "strategy_module": variant_module,
                        "strategy_path": variant_path,
                        "gates_ok": True,
                        "gate_errors": [],
                        "sharpe": 0.0, "trades": 0, "max_drawdown": 0.0,
                        "composite_score": 0.0,
                        "modification": modification,
                    })
                    print(f"      Variant {i+1}: backtest failed", flush=True)
            else:
                print(f"      Variant {i+1}: module load failed", flush=True)
        else:
            print(f"      Variant {i+1}: validation failed: {gate_errors}", flush=True)

    if not parallel_results:
        print(f"  ⚠️ All {count} parallel variants failed. Falling back to sequential.", flush=True)
        return None

    # Rank by composite score, pick the best
    parallel_results.sort(key=lambda x: x["composite_score"], reverse=True)
    best = parallel_results[0]

    # Log all parallel results
    parallel_log = {
        "turn": turn,
        "variant_count": count,
        "variants": [
            {
                "variant_index": r["variant_index"],
                "sharpe": r["sharpe"],
                "trades": r["trades"],
                "max_drawdown": r["max_drawdown"],
                "composite_score": r["composite_score"],
                "selected": r["variant_index"] == best["variant_index"],
            }
            for r in parallel_results
        ],
    }
    write_json(run_dir / f"parallel_results_v{turn}.json", parallel_log)

    print(f"  ✅ Best variant: #{best['variant_index']+1} "
          f"(Sharpe={best['sharpe']:.2f}, Composite={best['composite_score']:.2f})", flush=True)

    return (
        best["strategy_code"],
        best["strategy_module"],
        best["gates_ok"],
        best["gate_errors"],
    )


def refinement_loop(mandate_path: str, max_turns: int = 7, 
                    sharpe_target: float = 1.5,
                    config: Optional[RefinementConfig] = None) -> RunState:
    """
    Main refinement loop. One strategy, up to 7 turns.
    
    Args:
        mandate_path: Path to mandate JSON file
        max_turns: Maximum number of refinement turns
        sharpe_target: Target Sharpe ratio for success
        config: Optional RefinementConfig. If None, builds from mandate.
    
    Returns:
        Final RunState with results
    """
    mandate = load_json(mandate_path)
    
    # Build config from mandate if not provided
    if config is None:
        config = RefinementConfig.from_mandate(mandate)
    
    # Apply mode from mandate if present
    mode = mandate.get("mode")
    if mode:
        config.apply_mode(mode)
    
    # Apply individual toggles from mandate (override mode)
    toggles = mandate.get("toggles", {})
    if toggles:
        if "cross_run_learning" in toggles:
            config.cross_run_learning = toggles["cross_run_learning"]
        if "parallel_invention" in toggles:
            config.parallel_invention = toggles["parallel_invention"]
        if "parallel_count" in toggles:
            config.parallel_invention_count = toggles["parallel_count"]
        if "soft_promote" in toggles:
            config.soft_promote = toggles["soft_promote"]
        if "soft_promote_min_sharpe" in toggles:
            config.soft_promote_sharpe = toggles["soft_promote_min_sharpe"]
        if "soft_promote_min_windows" in toggles:
            config.soft_promote_min_windows = toggles["soft_promote_min_windows"]
        if "multi_ticker_backtest" in toggles:
            config.multi_ticker_backtest = toggles["multi_ticker_backtest"]
        if "multi_ticker_min_pass" in toggles:
            config.multi_ticker_min_pass = toggles["multi_ticker_min_pass"]
    
    # Also pass toggle values through to mandate for context_builder
    mandate.setdefault("cross_run_learning", config.cross_run_learning)
    run_dir = create_run_directory(mandate)
    state = RunState(
        run_id=run_dir.name,
        mandate_name=mandate["name"],
        created_at=datetime.now(timezone.utc).isoformat(),
        max_turns=max_turns,
        sharpe_target=sharpe_target,
        tickers=mandate.get("tickers", ["AAPL", "SPY"]),
        period=mandate.get("period", "1y")
    )
    
    # Acquire lock
    if not acquire_lock(run_dir):
        print(f"Run {run_dir.name} is locked. Skipping.")
        return state
    
    state.status = "running"
    save_state(run_dir, state)
    
    # Phase 3: Initialize circuit breaker and cosmetic guard
    cb = CircuitBreaker(
        window=20,
        min_pass_rate=0.2,
        min_attempts=5,
        grace_turns=2,
    )
    cosmetic_state = CosmeticGuardState(threshold=3)
    
    # Results directory for dashboard and analytics
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for turn in range(1, max_turns + 1):
        turn_start = time.time()
        state.current_turn = turn
        print(f"\n{'='*60}")
        print(f"Turn {turn}/{max_turns} | Best Sharpe: {state.best_sharpe:.2f}")
        
        # ── Adaptive Sharpe targeting ────────────────────────────────
        # Compute the effective target for this turn (lower on early turns).
        effective_target = compute_effective_target(
            sharpe_target=sharpe_target,
            turn=turn,
            adaptive_sharpe_target=config.adaptive_sharpe_target,
            adaptive_start_factor=config.adaptive_start_factor,
            adaptive_ramp_turns=config.adaptive_ramp_turns,
        )
        if effective_target != sharpe_target:
            print(f"  📈 Adaptive target: {effective_target:.2f} "
                  f"(ramping from {sharpe_target * config.adaptive_start_factor:.2f} "
                  f"→ {sharpe_target:.2f})")
        
        # 0. Clear indicator cache between turns
        clear_cache()
        
        # Phase 3: Check circuit breaker BEFORE calling LLM
        if cb.is_open():
            print(f"  🛑 Circuit breaker OPEN — pass rate too low ({cb.pass_rate:.0%}). Halting.")
            state.status = "abandoned"
            state.history.append({
                "turn": turn, "status": "circuit_breaker_open",
                "circuit_breaker_summary": cb.summary(),
            })
            save_state(run_dir, state)
            _write_dashboard(run_dir)
            release_lock(run_dir)
            return state
        
        # 1. Build context
        prev_report = load_report(run_dir, turn - 1) if turn > 1 else None
        context = build_llm_context(state, prev_report, mandate,
                                   effective_target=effective_target)
        
        # Phase 3: Append action analytics context to LLM prompt
        try:
            run_history = load_run_history(str(results_dir / "run_history.jsonl"))
            analytics_text = generate_analytics_context(run_history)
            # Inject analytics into the context as an additional section
            context["action_analytics"] = analytics_text
        except Exception:
            context["action_analytics"] = "No historical action data available."
        
        context_path = run_dir / f"context_v{turn}.json"
        write_json(context_path, context)
        
        # 2. Call LLM (with retry on code generation)
        modification = None
        strategy_code = None
        gates_ok = False
        gate_errors = []
        retry_feedback = ""  # Accumulated error feedback for retries

        # ── Parallel Invention (Phase 5.6.2) ──────────────────────────────
        # On turn 1, if parallel_invention is enabled, spawn N strategies
        # in parallel, backtest all, and keep the best by composite score.
        parallel_used = False
        if turn == 1 and config.parallel_invention and config.parallel_invention_count > 1:
            parallel_result = _run_parallel_invention(
                context=context,
                context_path=context_path,
                run_dir=run_dir,
                mandate=mandate,
                config=config,
                cb=cb,
                turn=turn,
            )
            if parallel_result is not None:
                strategy_code, strategy_module, gates_ok, gate_errors = parallel_result
                modification = {}  # Will be reconstructed below from strategy_code
                parallel_used = True
                print(f"  🔄 Parallel invention succeeded — using best variant", flush=True)
            else:
                print(f"  🔄 Parallel invention failed — falling back to sequential", flush=True)

        # ── Sequential LLM call (default or parallel fallback) ───────────
        if not parallel_used:
            for attempt in range(3):  # up to 3 code-repair attempts
                try:
                    # If this is a retry, inject specific error feedback into context
                    if retry_feedback and attempt > 0:
                        context["retry_feedback"] = retry_feedback

                    # Use call_llm_inventor for structured JSON output
                    modification = call_llm_inventor(
                        context=context,
                        context_path=str(context_path),
                    )
                    
                    if modification is None:
                        print(f"  LLM call failed (attempt {attempt+1})")
                        # Phase 3: Record validation failure for circuit breaker
                        cb.record(False, turn=turn, mandate=state.mandate_name)
                        continue
                    
                    strategy_code = modification.get("new_strategy_code", "")
                    if not strategy_code:
                        print(f"  No strategy code in LLM response (attempt {attempt+1})")
                        cb.record(False, turn=turn, mandate=state.mandate_name)
                        continue
                    
                    # 3. Validate through gates
                    primary_ticker = mandate.get("primary_ticker", state.tickers[0])
                    gates_ok, gate_errors = run_validation_gates(strategy_code, ticker=primary_ticker, period=state.period)
                    
                    # Phase 3: Record pass/fail for circuit breaker
                    cb.record(gates_ok, turn=turn, mandate=state.mandate_name)
                    
                    if gates_ok:
                        break
                    
                    # Build specific feedback for the next retry (Fix 3)
                    retry_feedback = _build_retry_feedback(gate_errors)
                    print(f"  Gate failed (attempt {attempt+1}): {gate_errors}")
                        
                except Exception as e:
                    print(f"  LLM call error (attempt {attempt+1}): {e}")
                    cb.record(False, turn=turn, mandate=state.mandate_name)
                    continue
        
        if not gates_ok or strategy_code is None:
            print(f"  All 3 validation attempts failed. Advancing turn.")
            state.history.append({
                "turn": turn, "status": "code_generation_failed",
                "errors": gate_errors
            })
            save_state(run_dir, state)
            _write_dashboard(run_dir)
            continue
        
        # Create modification object from LLM response
        raw_action = modification.get("action", "novel") if isinstance(modification, dict) else "novel"
        action, action_was_remapped = validate_action(raw_action)
        if action_was_remapped and action != raw_action:
            print(f"  🔀 Action validation: '{raw_action}' → '{action}'")
        hypothesis = modification.get("hypothesis", "LLM-generated strategy") if isinstance(modification, dict) else "LLM-generated strategy"
        expected_impact = modification.get("expected_impact", "higher") if isinstance(modification, dict) else "higher"
        
        modification = StrategyModification(
            action=action,
            hypothesis=hypothesis,
            new_strategy_code=strategy_code,
            reasoning=modification.get("reasoning", "") if isinstance(modification, dict) else "",
            addresses_failure=modification.get("addresses_failure", "") if isinstance(modification, dict) else "",
            expected_impact=expected_impact
        )
        
        # Phase 3: Cosmetic guard — check if LLM is stuck doing consecutive modify_params
        temp_history_for_cosmetic = list(state.history)
        temp_history_for_cosmetic.append({"turn": turn, "action": action})
        cosmetic_state, cosmetic_result = check_cosmetic_guard(temp_history_for_cosmetic, cosmetic_state)
        
        if cosmetic_result.forced:
            print(f"  ⚠️ Cosmetic guard: {cosmetic_result.warning}")
            print(f"  🔧 Overriding action: {action} → {cosmetic_result.forced_action}")
            action = cosmetic_result.forced_action
            modification.action = action
        
        # Save strategy code
        strategy_path = run_dir / f"strategy_v{turn}.py"
        strategy_path.write_text(strategy_code)
        
        # 3b. Load strategy module
        load_result = load_strategy_module(strategy_path)
        # Handle both old (ModuleType|None) and new (ModuleType | tuple) return types
        if isinstance(load_result, tuple):
            strategy_module, load_error = load_result
        else:
            strategy_module = load_result
            load_error = None

        if strategy_module is None:
            error_detail = load_error or {}
            print(f"  Failed to load strategy module: {error_detail.get('error_type', 'unknown')} - {error_detail.get('error_message', 'no details')}")
            # Phase 6: Track action analytics for failed turn
            track_action_result(
                mandate=state.mandate_name,
                turn=turn,
                action=action,
                sharpe=0.0,
                success=False,
                failure_mode="module_load_failed",
                path=str(results_dir / "run_history.jsonl"),
                error_info=error_detail,
            )
            state.history.append({
                "turn": turn,
                "status": "module_load_failed",
                "error": error_detail,
                "action": action,
                "hypothesis": getattr(modification, "hypothesis", ""),
            })
            save_state(run_dir, state)
            _write_dashboard(run_dir)
            continue
        
        # 4. Backtest on primary ticker
        primary_ticker = mandate.get("primary_ticker", state.tickers[0])
        print(f"  Running backtest...", flush=True)
        bt_start = time.time()
        backtest_output = run_backtest_safely(
            strategy_module, primary_ticker, state.period,
            return_portfolio=True,
        )
        bt_elapsed = time.time() - bt_start
        print(f"  Backtest completed in {bt_elapsed:.1f}s", flush=True)
        if bt_elapsed > 30:
            print(f"  ⚠️ Backtest slow ({bt_elapsed:.1f}s > 30s)", flush=True)
        
        # Phase 6: Extract error info from 4th return element
        backtest_error = backtest_output[3] if backtest_output and len(backtest_output) > 3 else {}
        if backtest_output is None or backtest_output[0] is None:
            error_detail = backtest_error or {}
            print(f"  Backtest crashed: {error_detail.get('error_type', 'unknown')} - {error_detail.get('error_message', 'no details')}")
            # Phase 3: Track action analytics for failed turn
            track_action_result(
                mandate=state.mandate_name,
                turn=turn,
                action=action,
                sharpe=0.0,
                success=False,
                failure_mode="backtest_crash",
                path=str(results_dir / "run_history.jsonl"),
                error_info=error_detail,
            )
            state.history.append({
                "turn": turn, "status": "backtest_crash",
                "error": error_detail,
                "action": action,
                "hypothesis": getattr(modification, "hypothesis", ""),
            })
            save_state(run_dir, state)
            _write_dashboard(run_dir)
            continue
        
        result, df, portfolio, _ = backtest_output
        
        # ── Multi-ticker backtest (Phase 5.6) ────────────────────────────
        # If enabled, backtest on secondary tickers to detect single-ticker overfit.
        multi_ticker_results = None
        if config.multi_ticker_backtest:
            # Determine which tickers to test (exclude primary, already tested)
            secondary_tickers = list(
                set(state.tickers) - {primary_ticker}
            )
            if config.multi_ticker_extra:
                # Explicit extra tickers take precedence
                secondary_tickers = [
                    t for t in config.multi_ticker_extra if t != primary_ticker
                ]
            if secondary_tickers:
                mt_start = time.time()
                multi_ticker_results = run_multi_ticker_backtest(
                    strategy_module, secondary_tickers, state.period,
                    sharpe_target=effective_target,
                )
                mt_elapsed = time.time() - mt_start
                print(f"  Multi-ticker backtest ({len(secondary_tickers)} tickers, "
                      f"{mt_elapsed:.1f}s): "
                      f"{multi_ticker_results['tickers_passed']}/"
                      f"{multi_ticker_results['tickers_tested']} passed, "
                      f"avg Sharpe={multi_ticker_results['avg_sharpe']:.2f}",
                      flush=True)
        
        # ── Feature importance analysis (Phase 5.6) ────────────────────────
        # Analyze which indicators actually contribute to returns.
        # This gives the LLM concrete data about indicator quality.
        feature_importance = None
        if config.feature_importance and strategy_code:
            try:
                from crabquant.refinement.feature_importance import compute_feature_importance
                fi_start = time.time()
                feature_importance = compute_feature_importance(
                    strategy_code, primary_ticker, state.period
                )
                fi_elapsed = time.time() - fi_start
                if feature_importance.get("indicators"):
                    dominant = feature_importance.get("dominant_indicator", "")
                    weakest = feature_importance.get("weakest_indicator", "")
                    print(f"  Feature importance ({fi_elapsed:.1f}s): "
                          f"{len(feature_importance['indicators'])} indicators, "
                          f"driver={dominant or 'none'}, "
                          f"weakest={weakest or 'none'}",
                          flush=True)
            except Exception as e:
                print(f"  ⚠️ Feature importance error: {e}", flush=True)
        
        # 5. Compute diagnostics + classify failure
        if portfolio is not None:
            sharpe_by_year = compute_sharpe_by_year(portfolio)
        else:
            sharpe_by_year = {}
        
        # Run guardrails
        guardrail_config = GuardrailConfig()
        guardrail_report = check_guardrails(result, guardrail_config)
        guardrail_violations = getattr(guardrail_report, 'violations', [])
        guardrail_warnings = getattr(guardrail_report, 'warnings', [])
        
        failure_mode, failure_details = classify_failure(
            result, guardrail_report, sharpe_by_year,
            data_length=len(df),
            sharpe_target=effective_target
        )
        
        # 6. Compute stagnation (using Phase 3 module)
        stagnation_score, stagnation_trend = compute_stagnation(state.history)
        
        # 7. Build report
        current_code = strategy_path.read_text()
        
        report = BacktestReport(
            strategy_id=state.run_id,
            iteration=turn,
            sharpe_ratio=result.sharpe,
            total_return_pct=result.total_return,
            max_drawdown_pct=result.max_drawdown,
            win_rate=result.win_rate,
            total_trades=result.num_trades,
            profit_factor=result.profit_factor,
            calmar_ratio=result.calmar_ratio,
            sortino_ratio=getattr(result, 'sortino_ratio', 0.0),
            composite_score=result.score,
            failure_mode=failure_mode,
            failure_details=failure_details,
            sharpe_by_year=sharpe_by_year,
            stagnation_score=stagnation_score,
            stagnation_trend=stagnation_trend,
            previous_sharpes=[h.get("sharpe", 0) for h in state.history],
            previous_actions=[h.get("action", "") for h in state.history],
            guardrail_violations=guardrail_violations,
            guardrail_warnings=guardrail_warnings,
            regime_sharpe=None,
            regime_regime_shift=None,
            top_drawdowns=None,
            portfolio_correlation=None,
            benchmark_return_pct=None,
            market_regime=None,
            current_strategy_code=current_code,
            current_params=result.params if result.params else strategy_module.DEFAULT_PARAMS.copy(),
            previous_attempts=state.history[-3:],
            multi_ticker_results=multi_ticker_results,
            feature_importance=feature_importance,
        )
        
        save_report(run_dir, turn, report)
        
        # Phase 3: Track action analytics for this turn
        turn_success = result.sharpe >= effective_target
        track_action_result(
            mandate=state.mandate_name,
            turn=turn,
            action=action,
            sharpe=result.sharpe,
            success=turn_success,
            failure_mode="" if turn_success else failure_mode,
            path=str(results_dir / "run_history.jsonl"),
        )
        
        # 8. Check success — primary gate is Sharpe >= effective_target;
        #    secondary guardrails (trade count, drawdown, etc.) are logged
        #    as warnings but do NOT block promotion.
        if result.sharpe >= effective_target:
            # ── Hard pre-validation gate: skip expensive WF on sparse strategies ──
            # A strategy with too few trades has no statistical power.
            # No point running 6-window walk-forward on a curve-fit.
            MIN_TRADES_FOR_VALIDATION = 20
            if result.num_trades < MIN_TRADES_FOR_VALIDATION:
                print(f"  🏆 Sharpe {result.sharpe:.2f} >= {effective_target}, "
                      f"but only {result.num_trades} trades "
                      f"(need >= {MIN_TRADES_FOR_VALIDATION}) — skipping validation")
                failure_mode = "too_few_trades_for_validation"
                # Update state to reflect best Sharpe found but not validated
                composite = compute_composite_score(
                    result.sharpe, result.num_trades, result.max_drawdown
                )
                if composite > state.best_composite_score:
                    state.best_sharpe = result.sharpe
                    state.best_composite_score = composite
                    state.best_turn = turn
                    state.best_code_path = str(strategy_path)
                # Record in history so LLM gets feedback about trade count
                state.history.append({
                    "turn": turn, "sharpe": result.sharpe,
                    "failure_mode": failure_mode,
                    "action": modification.action,
                    "hypothesis": modification.hypothesis,
                    "code_path": str(strategy_path),
                    "num_trades": result.num_trades,
                    "composite_score": composite,
                    "params_used": result.params if result.params else strategy_module.DEFAULT_PARAMS.copy(),
                    "strategy_hash": compute_strategy_hash(strategy_code),
                    "delta_from_prev": "Sharpe hit but too few trades for validation",
                })
                save_state(run_dir, state)
                continue  # Keep iterating — LLM might find a strategy with more trades

            print(f"  🏆 SUCCESS! Sharpe {result.sharpe:.2f} >= {effective_target} "
                  f"({result.num_trades} trades)")

            # Log secondary guardrail issues as warnings (non-blocking)
            if guardrail_violations:
                print(f"  ⚠️ Guardrail warnings (non-blocking):")
                for v in guardrail_violations:
                    print(f"    - {v}")
            if guardrail_warnings:
                for w in guardrail_warnings:
                    print(f"    - {w}")

            # ── Multi-ticker gate (Phase 5.6) ────────────────────────────
            # If multi-ticker backtest ran, check that enough tickers passed.
            # This catches single-ticker overfit before expensive walk-forward.
            if multi_ticker_results is not None:
                min_pass = config.multi_ticker_min_pass
                if multi_ticker_results["tickers_passed"] < min_pass:
                    print(f"  🏆 Sharpe hit on primary, but multi-ticker gate FAILED: "
                          f"only {multi_ticker_results['tickers_passed']}/"
                          f"{multi_ticker_results['tickers_tested']} tickers passed "
                          f"(need >= {min_pass})")
                    failure_mode = "multi_ticker_gate_failed"
                    # Feed back to LLM — record in history with multi-ticker context
                    state.history.append({
                        "turn": turn, "sharpe": result.sharpe,
                        "failure_mode": failure_mode,
                        "action": modification.action,
                        "hypothesis": modification.hypothesis,
                        "code_path": str(strategy_path),
                        "num_trades": result.num_trades,
                        "composite_score": compute_composite_score(
                            result.sharpe, result.num_trades, result.max_drawdown
                        ),
                        "params_used": result.params if result.params else strategy_module.DEFAULT_PARAMS.copy(),
                        "strategy_hash": compute_strategy_hash(strategy_code),
                        "delta_from_prev": (f"Sharpe hit but multi-ticker gate failed: "
                                            f"{multi_ticker_results['tickers_passed']}/"
                                            f"{multi_ticker_results['tickers_tested']} tickers passed"),
                        "multi_ticker_results": multi_ticker_results,
                    })
                    save_state(run_dir, state)
                    continue  # LLM gets feedback, tries again
                else:
                    print(f"  ✅ Multi-ticker gate passed: "
                          f"{multi_ticker_results['tickers_passed']}/"
                          f"{multi_ticker_results['tickers_tested']} tickers >= "
                          f"{min_pass} required")

            # Phase 3: Run full validation check (walk-forward + cross-ticker)
            validation = {"status": "ok", "passed": False}
            _is_regime_specific = False
            try:
                strategy_fn = strategy_module.generate_signals
                params = result.params if result.params else strategy_module.DEFAULT_PARAMS.copy()
                validation_tickers = mandate.get("tickers", [primary_ticker])

                # Quick regime-specificity pre-check
                try:
                    from crabquant.refinement.regime_tagger import compute_strategy_regime_tags
                    _tags = compute_strategy_regime_tags(strategy_fn, params, ticker=primary_ticker)
                    _is_regime_specific = _tags.get("is_regime_specific", False)
                except Exception:
                    pass

                validation = run_full_validation_check(
                    strategy_fn=strategy_fn,
                    params=params,
                    discovery_ticker=primary_ticker,
                    validation_tickers=validation_tickers,
                    is_regime_specific=_is_regime_specific,
                    rolling_config={
                        "min_window_test_sharpe": config.min_window_test_sharpe,
                        "max_window_degradation": config.max_window_degradation,
                    },
                )
            except Exception as e:
                print(f"  ⚠️ Full validation error: {e}")
                validation = {"status": "error", "message": str(e), "passed": False}
            
            state.status = "success"
            state.best_sharpe = result.sharpe
            state.best_composite_score = compute_composite_score(
                result.sharpe, result.num_trades, result.max_drawdown
            )
            state.best_turn = turn
            state.best_code_path = str(strategy_path)
            state.history.append({
                "turn": turn, "sharpe": result.sharpe,
                "composite_score": state.best_composite_score,
                "failure_mode": failure_mode,
                "action": modification.action,
                "hypothesis": modification.hypothesis,
                "validation": validation,
                "code_path": str(strategy_path),
                "params_used": result.params if result.params else strategy_module.DEFAULT_PARAMS.copy(),
                "strategy_hash": compute_strategy_hash(strategy_code),
                "delta_from_prev": "Initial strategy (no prior version)",
            })
            save_state(run_dir, state)
            
            # Phase 3: Auto-promote if validation passed
            if validation.get("passed", False):
                print(f"  📋 Validation passed — auto-promoting to registry...")
                try:
                    promo_result = auto_promote(
                        strategy_code=strategy_code,
                        strategy_module=strategy_module,
                        result=result,
                        validation=validation,
                        state=state,
                    )
                    if promo_result.get("registered"):
                        print(f"  ✅ Strategy registered: {promo_result['strategy_name']}")
                    else:
                        print(f"  ⚠️ Promotion skipped: {promo_result.get('error', 'unknown')}")
                except Exception as e:
                    print(f"  ⚠️ Auto-promote error: {e}")
            else:
                # Validation did not pass — update history failure_mode
                # so the LLM gets actionable feedback on the next turn.
                wf_result = (validation.get("walk_forward") or {})
                state.history[-1]["failure_mode"] = "validation_failed"
                state.history[-1]["num_trades"] = result.num_trades
                state.history[-1]["validation"] = {
                    "avg_test_sharpe": wf_result.get("avg_test_sharpe"),
                    "windows_passed": wf_result.get("windows_passed"),
                    "num_windows": wf_result.get("num_windows"),
                    "window_results": wf_result.get("window_results", []),
                }

                # Phase 5.6.3: Soft-promote near-miss strategies to candidates pool
                if config.soft_promote:
                    try:
                        sp_result = soft_promote(
                            strategy_code=strategy_code,
                            strategy_module=strategy_module,
                            result=result,
                            validation=validation,
                            state=state,
                            min_sharpe=config.soft_promote_sharpe,
                            min_windows=config.soft_promote_min_windows,
                            is_regime_specific=_is_regime_specific,
                        )
                        if sp_result.get("promoted"):
                            print(f"  📋 Soft-promoted to candidates: {sp_result['candidate_file']}")
                            print(f"     Avg test Sharpe: {sp_result['avg_test_sharpe']:.3f}, "
                                  f"Windows passed: {sp_result['windows_passed']}")
                        else:
                            print(f"  📋 Soft-promote skipped: {sp_result.get('reason', 'unknown')}")
                    except Exception as e:
                        print(f"  ⚠️ Soft-promote error: {e}")

                # Fall back to legacy promote_to_winner
                print(f"  📋 Validation not passed — using legacy promotion...")
                try:
                    promote_to_winner(strategy_code, result, validation, state, strategy_module=strategy_module)
                except Exception as e:
                    print(f"  ⚠️ Legacy promotion error: {e}")
            
            _write_dashboard(run_dir)
            turn_elapsed = time.time() - turn_start
            print(f"  Turn {turn} completed in {turn_elapsed:.1f}s", flush=True)
            release_lock(run_dir)
            return state
        
        # Track best attempt (by composite score, not raw Sharpe)
        composite = compute_composite_score(
            result.sharpe, result.num_trades, result.max_drawdown
        )
        if composite > state.best_composite_score:
            state.best_sharpe = result.sharpe
            state.best_composite_score = composite
            state.best_turn = turn
            state.best_code_path = str(strategy_path)
        
        # 9. Check stagnation-based early exit
        stag_response = get_stagnation_response(turn, stagnation_score)
        if stag_response["constraint"] == "abandon":
            print(f"  🛑 Abandoning: stagnation score {stagnation_score:.2f}")
            state.status = "abandoned"
            save_state(run_dir, state)
            _write_dashboard(run_dir)
            turn_elapsed = time.time() - turn_start
            print(f"  Turn {turn} completed in {turn_elapsed:.1f}s", flush=True)
            release_lock(run_dir)
            return state
        
        # Append to history
        state.history.append({
            "turn": turn, "sharpe": result.sharpe,
            "composite_score": composite,
            "failure_mode": failure_mode,
            "action": modification.action,
            "hypothesis": modification.hypothesis,
            "code_path": str(strategy_path),
            "params_used": result.params if result.params else strategy_module.DEFAULT_PARAMS.copy(),
            "strategy_hash": compute_strategy_hash(strategy_code),
            "delta_from_prev": "Initial strategy (no prior version)",
        })
        save_state(run_dir, state)
        
        # Phase 3: Write dashboard snapshot after each turn
        _write_dashboard(run_dir)
        
        turn_elapsed = time.time() - turn_start
        print(f"  Turn {turn} completed in {turn_elapsed:.1f}s", flush=True)
    
    # ── Exhausted all turns ─────────────────────────────────────────
    state.status = "max_turns_exhausted"
    save_state(run_dir, state)
    print(f"  Max turns exhausted. Best Sharpe: {state.best_sharpe:.2f} (composite: {state.best_composite_score:.2f}) at turn {state.best_turn}")

    # Post-loop promotion: if best Sharpe meets target, promote regardless
    # of exit reason (max_turns_exhausted, stagnation, etc.).  A high-Sharpe
    # strategy should never be silently discarded.
    if (state.best_sharpe >= sharpe_target
            and state.best_turn > 0
            and state.best_code_path):
        print(f"  🔄 Post-loop: best Sharpe {state.best_sharpe:.2f} >= target {sharpe_target} — attempting promotion…")
        try:
            best_strategy_file = Path(state.best_code_path)
            if best_strategy_file.exists():
                strategy_code = best_strategy_file.read_text()
                best_module_load = load_strategy_module(best_strategy_file)
                # Handle both old (ModuleType|None) and new (ModuleType | tuple) return types
                if isinstance(best_module_load, tuple):
                    best_module = best_module_load[0]
                else:
                    best_module = best_module_load

                if best_module is not None:
                    # Build a synthetic result-like object for promote_to_winner
                    # from the last saved report.
                    best_report = load_report(run_dir, state.best_turn)
                    _promote_post_loop(
                        strategy_code=strategy_code,
                        strategy_module=best_module,
                        state=state,
                        run_dir=run_dir,
                        mandate=mandate,
                        report=best_report,
                    )
                else:
                    print(f"  ⚠️ Post-loop promotion skipped: could not load strategy module")
            else:
                print(f"  ⚠️ Post-loop promotion skipped: strategy file not found at {state.best_code_path}")
        except Exception as e:
            print(f"  ⚠️ Post-loop promotion error: {e}")

    _write_dashboard(run_dir)
    release_lock(run_dir)
    return state


def _write_dashboard(run_dir: Path) -> None:
    """Write dashboard snapshot to results/dashboard.json."""
    try:
        runs_dir = project_root / "refinement_runs"
        snapshot = generate_dashboard(runs_dir)
        dashboard_path = project_root / "results" / "dashboard.json"
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text(snapshot_to_json(snapshot, indent=2))
    except Exception:
        pass  # Dashboard is non-critical


def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CrabQuant Refinement Loop")
    parser.add_argument("--mandate", required=True, help="Path to mandate JSON file")
    parser.add_argument("--max-turns", type=int, default=7, help="Maximum number of turns")
    parser.add_argument("--sharpe-target", type=float, default=1.5, help="Target Sharpe ratio")
    
    args = parser.parse_args()
    
    if not Path(args.mandate).exists():
        print(f"Error: Mandate file not found: {args.mandate}")
        return
    
    print(f"Starting refinement loop for: {args.mandate}")
    state = refinement_loop(args.mandate, args.max_turns, args.sharpe_target)
    print(f"\nFinal status: {state.status}")
    print(f"Best Sharpe: {state.best_sharpe:.2f}")


if __name__ == "__main__":
    main()
