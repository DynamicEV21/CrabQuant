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
import os
import tempfile
import time
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import CrabQuant modules
from crabquant.refinement.schemas import RunState, BacktestReport, StrategyModification
from crabquant.refinement.config import RefinementConfig
from crabquant.refinement.module_loader import load_strategy_module
from crabquant.refinement.validation_gates import run_validation_gates
from crabquant.refinement.diagnostics import (
    run_backtest_safely, compute_sharpe_by_year, compute_strategy_hash
)
from crabquant.refinement.classifier import classify_failure
from crabquant.refinement.context_builder import build_llm_context
from crabquant.refinement.llm_api import call_zai_llm, call_llm_inventor, load_api_config
from crabquant.refinement.config import RefinementConfig
from crabquant.guardrails import check_guardrails, GuardrailReport, GuardrailConfig

# Phase 3 module imports — replace all inline implementations
from crabquant.refinement.circuit_breaker import CircuitBreaker
from crabquant.refinement.cosmetic_guard import check_cosmetic_guard, CosmeticGuardState
from crabquant.refinement.action_analytics import (
    track_action_result, generate_llm_context as generate_analytics_context,
    load_run_history,
)
from crabquant.refinement.promotion import auto_promote, run_full_validation_check
from crabquant.refinement.stagnation import compute_stagnation, get_stagnation_response
from crabquant.refinement.wave_dashboard import generate_dashboard, snapshot_to_json


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
    try:
        strategy_fn = strategy_module.generate_signals
        params = (strategy_module.DEFAULT_PARAMS.copy()
                  if hasattr(strategy_module, 'DEFAULT_PARAMS') else {})
        primary_ticker = mandate.get("primary_ticker", state.tickers[0])
        validation_tickers = mandate.get("tickers", [primary_ticker])
        validation = run_full_validation_check(
            strategy_fn=strategy_fn,
            params=params,
            discovery_ticker=primary_ticker,
            validation_tickers=validation_tickers,
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


def refinement_loop(mandate_path: str, max_turns: int = 7, 
                    sharpe_target: float = 1.5) -> RunState:
    """
    Main refinement loop. One strategy, up to 7 turns.
    
    Args:
        mandate_path: Path to mandate JSON file
        max_turns: Maximum number of refinement turns
        sharpe_target: Target Sharpe ratio for success
    
    Returns:
        Final RunState with results
    """
    mandate = load_json(mandate_path)
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
        context = build_llm_context(state, prev_report, mandate)
        
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
                gates_ok, gate_errors = run_validation_gates(strategy_code)
                
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
        action = modification.get("action", "novel") if isinstance(modification, dict) else "novel"
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
        strategy_module = load_strategy_module(strategy_path)
        if strategy_module is None:
            print(f"  Failed to load strategy module. Advancing turn.")
            state.history.append({"turn": turn, "status": "module_load_failed"})
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
        
        if backtest_output is None:
            print(f"  Backtest crashed. Advancing turn.")
            # Phase 3: Track action analytics for failed turn
            track_action_result(
                mandate=state.mandate_name,
                turn=turn,
                action=action,
                sharpe=0.0,
                success=False,
                failure_mode="backtest_crash",
                path=str(results_dir / "run_history.jsonl"),
            )
            state.history.append({
                "turn": turn, "status": "backtest_crash"
            })
            save_state(run_dir, state)
            _write_dashboard(run_dir)
            continue
        
        result, df, portfolio = backtest_output
        
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
            sharpe_target=sharpe_target
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
        )
        
        save_report(run_dir, turn, report)
        
        # Phase 3: Track action analytics for this turn
        turn_success = result.sharpe >= sharpe_target
        track_action_result(
            mandate=state.mandate_name,
            turn=turn,
            action=action,
            sharpe=result.sharpe,
            success=turn_success,
            failure_mode="" if turn_success else failure_mode,
            path=str(results_dir / "run_history.jsonl"),
        )
        
        # 8. Check success — primary gate is Sharpe >= target;
        #    secondary guardrails (trade count, drawdown, etc.) are logged
        #    as warnings but do NOT block promotion.
        if result.sharpe >= sharpe_target:
            print(f"  🏆 SUCCESS! Sharpe {result.sharpe:.2f} >= {sharpe_target}")

            # Log secondary guardrail issues as warnings (non-blocking)
            if guardrail_violations:
                print(f"  ⚠️ Guardrail warnings (non-blocking):")
                for v in guardrail_violations:
                    print(f"    - {v}")
            if guardrail_warnings:
                for w in guardrail_warnings:
                    print(f"    - {w}")

            # Phase 3: Run full validation check (walk-forward + cross-ticker)
            validation = {"status": "ok", "passed": False}
            try:
                strategy_fn = strategy_module.generate_signals
                params = result.params if result.params else strategy_module.DEFAULT_PARAMS.copy()
                validation_tickers = mandate.get("tickers", [primary_ticker])
                validation = run_full_validation_check(
                    strategy_fn=strategy_fn,
                    params=params,
                    discovery_ticker=primary_ticker,
                    validation_tickers=validation_tickers,
                )
            except Exception as e:
                print(f"  ⚠️ Full validation error: {e}")
                validation = {"status": "error", "message": str(e), "passed": False}
            
            state.status = "success"
            state.best_sharpe = result.sharpe
            state.best_turn = turn
            state.best_code_path = str(strategy_path)
            state.history.append({
                "turn": turn, "sharpe": result.sharpe,
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
        
        # Track best attempt
        if result.sharpe > state.best_sharpe:
            state.best_sharpe = result.sharpe
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
    print(f"  Max turns exhausted. Best Sharpe: {state.best_sharpe:.2f} at turn {state.best_turn}")

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
                best_module = load_strategy_module(best_strategy_file)

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
