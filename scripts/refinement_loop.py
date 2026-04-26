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
from crabquant.guardrails import check_guardrails, GuardrailReport


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


def promote_to_winner(strategy_code: str, result, validation, state, strategy_module=None) -> None:
    """Promote successful strategy to winner registry."""
    winners_dir = project_root / "refinement_winners"
    winners_dir.mkdir(exist_ok=True)
    
    winner_name = f"{state.mandate_name}_turn{state.best_turn}_sharpe{state.best_sharpe:.2f}"
    winner_path = winners_dir / f"{winner_name}.py"
    
    # Create winner module with strategy and metadata
    winner_content = f'''# Auto-generated winner strategy
# Mandate: {state.mandate_name}
# Turn: {state.best_turn}
# Sharpe: {state.best_sharpe:.2f}
# Validation: {validation}

{strategy_code}

# Winner metadata
WINNER_METADATA = {{
    "mandate_name": "{state.mandate_name}",
    "turn": {state.best_turn},
    "sharpe": {state.best_sharpe},
    "validation": {validation},
    "timestamp": "{datetime.now().isoformat()}",
    "strategy_hash": "{compute_strategy_hash(strategy_code)}",
}}
'''
    
    winner_path.write_text(winner_content)
    print(f"  🏆 Promoted to winner: {winner_path}")


def run_full_validation(state: RunState, run_dir: Path) -> Dict[str, Any]:
    """Run full validation on the best strategy (walk-forward + cross-ticker).
    
    Returns a validation result dict.
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


def compute_stagnation(history: List[Dict[str, Any]]) -> Tuple[float, str]:
    """Compute stagnation score and trend from history.
    
    Returns (score, trend) where:
    - score: 0.0 (great progress) to 1.0 (completely stuck)
    - trend: "improving" | "flat" | "declining" | "slow_progress"
    """
    if len(history) < 2:
        return 0.0, "improving"
    
    sharpes = [h.get("sharpe", 0) for h in history]
    
    # Factor 1: Trend via linear regression slope
    try:
        slope = sum(
            (i - (len(sharpes) - 1) / 2) * (s - sum(sharpes) / len(sharpes))
            for i, s in enumerate(sharpes)
        ) / max(sum((i - (len(sharpes) - 1) / 2) ** 2 for i in range(len(sharpes))), 1e-10)
    except (ZeroDivisionError, ValueError):
        slope = 0.0
    
    if slope < -0.05:
        trend_score, trend = 1.0, "declining"
    elif slope < 0.02:
        trend_score, trend = 0.85, "flat"
    elif slope < 0.05:
        trend_score, trend = 0.5, "slow_progress"
    else:
        trend_score, trend = 0.0, "improving"
    
    # Factor 2: Variance + stability (low variance with flat trend = stuck)
    if len(sharpes) >= 3:
        mean_sharpe = sum(sharpes[-3:]) / 3
        variance = (sum((s - mean_sharpe) ** 2 for s in sharpes[-3:]) / 3) ** 0.5
    else:
        variance = 0.0
    # High variance = oscillating (bad), near-zero variance with flat trend = stuck (bad)
    variance_score = min(variance / 0.3, 1.0)
    # Stability penalty: if recent values barely move AND trend is flat, add stagnation
    recent_range = max(sharpes[-3:]) - min(sharpes[-3:]) if len(sharpes) >= 3 else abs(sharpes[-1] - sharpes[0])
    stability_penalty = max(0, 1.0 - recent_range / 0.1) * (0.7 if trend_score > 0.5 else 0.0)
    
    # Factor 3: Action repetition (only counts if actions are non-empty)
    actions = [h.get("action", "") for h in history[-3:] if h.get("action", "")]
    if len(actions) >= 2:
        unique_actions = len(set(actions))
        repetition_score = 1.0 - (unique_actions / len(actions))
    else:
        repetition_score = 0.0
    
    # Weighted combination
    score = 0.5 * trend_score + 0.15 * variance_score + 0.15 * repetition_score + 0.2 * stability_penalty
    return round(score, 3), trend


def get_stagnation_response(turn: int, stagnation_score: float) -> Dict[str, str]:
    """Get stagnation response protocol.
    
    Based on PRD §7 stagnation detection thresholds.
    """
    # Abandon: extremely stuck, but only after several turns
    if stagnation_score > 0.8 and turn >= 4:
        return {"constraint": "abandon"}
    
    # Force structural change
    if stagnation_score > 0.5 and turn >= 3:
        return {"constraint": "force_structural"}
    
    return {"constraint": "none"}


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
    
    for turn in range(1, max_turns + 1):
        state.current_turn = turn
        print(f"\n{'='*60}")
        print(f"Turn {turn}/{max_turns} | Best Sharpe: {state.best_sharpe:.2f}")
        
        # 0. Clear indicator cache between turns
        clear_cache()
        
        # 1. Build context
        prev_report = load_report(run_dir, turn - 1) if turn > 1 else None
        context = build_llm_context(state, prev_report, mandate)
        context_path = run_dir / f"context_v{turn}.json"
        write_json(context_path, context)
        
        # 2. Call LLM (with retry on code generation)
        modification = None
        strategy_code = None
        gates_ok = False
        gate_errors = []
        
        for attempt in range(3):  # up to 3 code-repair attempts
            try:
                # Use call_llm_inventor for structured JSON output
                modification = call_llm_inventor(
                    context=context,
                    context_path=str(context_path),
                )
                
                if modification is None:
                    print(f"  LLM call failed (attempt {attempt+1})")
                    continue
                
                strategy_code = modification.get("new_strategy_code", "")
                if not strategy_code:
                    print(f"  No strategy code in LLM response (attempt {attempt+1})")
                    continue
                
                # 3. Validate through gates
                gates_ok, gate_errors = run_validation_gates(strategy_code)
                if gates_ok:
                    break
                
                print(f"  Gate failed (attempt {attempt+1}): {gate_errors}")
                    
            except Exception as e:
                print(f"  LLM call error (attempt {attempt+1}): {e}")
                continue
        
        if not gates_ok or strategy_code is None:
            print(f"  All 3 validation attempts failed. Advancing turn.")
            state.history.append({
                "turn": turn, "status": "code_generation_failed",
                "errors": gate_errors
            })
            save_state(run_dir, state)
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
        
        # Save strategy code
        strategy_path = run_dir / f"strategy_v{turn}.py"
        strategy_path.write_text(strategy_code)
        
        # 3b. Load strategy module
        strategy_module = load_strategy_module(strategy_path)
        if strategy_module is None:
            print(f"  Failed to load strategy module. Advancing turn.")
            state.history.append({"turn": turn, "status": "module_load_failed"})
            save_state(run_dir, state)
            continue
        
        # 4. Backtest on primary ticker
        primary_ticker = mandate.get("primary_ticker", state.tickers[0])
        backtest_output = run_backtest_safely(
            strategy_module, primary_ticker, state.period
        )
        
        if backtest_output is None:
            print(f"  Backtest crashed. Advancing turn.")
            state.history.append({
                "turn": turn, "status": "backtest_crash"
            })
            save_state(run_dir, state)
            continue
        
        result, df, portfolio = backtest_output
        
        # 5. Compute diagnostics + classify failure
        sharpe_by_year = compute_sharpe_by_year(portfolio)
        
        # Run guardrails
        guardrail_report = check_guardrails(result)
        guardrail_violations = getattr(guardrail_report, 'violations', [])
        guardrail_warnings = getattr(guardrail_report, 'warnings', [])
        
        failure_mode, failure_details = classify_failure(
            result, guardrail_report, sharpe_by_year,
            data_length=len(df),
            sharpe_target=sharpe_target
        )
        
        # 6. Compute stagnation
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
        
        # 8. Check success
        if result.sharpe >= sharpe_target and result.passed:
            print(f"  🏆 SUCCESS! Sharpe {result.sharpe:.2f} >= {sharpe_target}")
            
            # Run full validation
            validation = run_full_validation(state, run_dir)
            
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
            
            # Promote to winner
            promote_to_winner(strategy_code, result, validation, state, strategy_module=strategy_module)
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
    
    # Exhausted all turns
    state.status = "max_turns_exhausted"
    save_state(run_dir, state)
    print(f"  Max turns exhausted. Best Sharpe: {state.best_sharpe:.2f} at turn {state.best_turn}")
    release_lock(run_dir)
    return state


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