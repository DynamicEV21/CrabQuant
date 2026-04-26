"""
CrabQuant Refinement Pipeline — Main Orchestrator Loop

The core loop: load mandate → LLM invents → validate gates → backtest →
classify failure → build report → repeat until Sharpe target hit or max turns.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure crabquant is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from crabquant.refinement.schemas import RunState, BacktestReport, StrategyModification
from crabquant.refinement.classifier import classify_failure
from crabquant.refinement.validation_gates import run_validation_gates
from crabquant.refinement.diagnostics import (
    run_backtest_safely,
    compute_sharpe_by_year,
    compute_strategy_hash,
)
from crabquant.refinement.module_loader import load_strategy_module
from crabquant.refinement.llm_api import call_llm_inventor
from crabquant.refinement.context_builder import build_llm_context, compute_delta

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    """Load JSON file."""
    return json.loads(Path(path).read_text())


def write_json(path, data):
    """Write JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str))


def save_state(run_dir: Path, state: RunState):
    """Save RunState to state.json."""
    write_json(run_dir / "state.json", state.to_dict())


def load_state(run_dir: Path) -> Optional[RunState]:
    """Load RunState from state.json."""
    path = run_dir / "state.json"
    if path.exists():
        return RunState.from_dict(json.loads(path.read_text()))
    return None


def save_report(run_dir: Path, turn: int, report: BacktestReport):
    """Save BacktestReport for a specific turn."""
    write_json(run_dir / f"report_v{turn}.json", report.to_dict())


def load_report(run_dir: Path, turn: int) -> Optional[BacktestReport]:
    """Load BacktestReport for a specific turn."""
    path = run_dir / f"report_v{turn}.json"
    if path.exists():
        return BacktestReport.from_dict(json.loads(path.read_text()))
    return None


def create_run_directory(mandate: dict) -> Path:
    """Create a unique run directory."""
    runs_dir = Path("refinement/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{mandate.get('name', 'unknown')}_{timestamp}"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def acquire_lock(run_dir: Path) -> bool:
    """Acquire a lock for this run. Returns False if already locked."""
    lock_file = run_dir / "lock.json"
    if lock_file.exists():
        lock_data = json.loads(lock_file.read_text())
        # Check if lock is stale (>1 hour old)
        lock_time = datetime.fromisoformat(lock_data.get("timestamp", ""))
        age = (datetime.now(timezone.utc) - lock_time).total_seconds()
        if age < 3600:
            return False
        logger.warning("Stale lock detected (%.0f min old), overriding", age / 60)
    
    lock_file = run_dir / "lock.json"
    lock_file.write_text(json.dumps({
        "pid": __import__("os").getpid(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }))
    return True


def release_lock(run_dir: Path):
    """Release the run lock."""
    lock_file = run_dir / "lock.json"
    if lock_file.exists():
        lock_file.unlink()


def check_guardrails(result, config=None):
    """Check backtest result against guardrails. Returns violations/warnings."""
    try:
        from crabquant.guardrails import GuardrailConfig, check_guardrails as _check
        config = config or GuardrailConfig()
        return _check(result, config)
    except ImportError:
        # Guardrails not available — return empty result
        return MagicMock(violations=[], warnings=[])


def compute_stagnation(history: list) -> tuple:
    """Compute stagnation score (0=progress, 1=stuck) and trend.
    
    Simple heuristic: compare recent Sharpe values.
    """
    if len(history) < 2:
        return 0.0, "insufficient_data"
    
    recent_sharpes = [h.get("sharpe", 0.0) for h in history[-3:]]
    if len(recent_sharpes) < 2:
        return 0.0, "insufficient_data"
    
    # Check if sharpes are stagnating (all within 0.1 of each other)
    max_s = max(recent_sharpes)
    min_s = min(recent_sharpes)
    spread = max_s - min_s
    
    if spread < 0.05:
        return 0.9, "flat"
    elif spread < 0.1:
        return 0.5, "slow_progress"
    else:
        return 0.1, "improving"


def get_stagnation_response(turn: int, score: float) -> dict:
    """Determine response to stagnation. Returns constraint dict."""
    if score >= 0.9 and turn >= 4:
        return {"constraint": "abandon", "reason": f"Score {score:.2f} at turn {turn}"}
    elif score >= 0.7:
        return {"constraint": "force_structural", "reason": "Stagnation detected"}
    return {"constraint": "none", "reason": ""}


def run_full_validation(strategy_fn, params, discovery_ticker, validation_tickers) -> dict:
    """Run full walk-forward + cross-ticker validation.
    
    Args:
        strategy_fn: generate_signals function
        params: DEFAULT_PARAMS dict
        discovery_ticker: ticker used for discovery
        validation_tickers: list of tickers to validate on
    
    Returns:
        Dict with validation results.
    """
    try:
        from crabquant.validation import full_validation
        return full_validation(strategy_fn, params, discovery_ticker, validation_tickers)
    except ImportError:
        logger.warning("full_validation not available, skipping")
        return {"status": "skipped", "reason": "validation module not available"}
    except Exception as e:
        logger.warning("Validation failed: %s", e)
        return {"status": "error", "reason": str(e)}


def promote_to_winner(strategy_code, result, validation, state, strategy_module=None):
    """Promote a successful strategy to winners.json.
    
    In a full implementation, this would also register in STRATEGY_REGISTRY.
    For Phase 1, we just save to winners.json.
    """
    winners_path = Path("results/winners/winners.json")
    winners_path.parent.mkdir(parents=True, exist_ok=True)
    
    winners = []
    if winners_path.exists():
        try:
            winners = json.loads(winners_path.read_text())
        except json.JSONDecodeError:
            winners = []
    
    winners.append({
        "strategy": f"refined_{state.mandate_name}",
        "ticker": result.ticker if hasattr(result, "ticker") else "unknown",
        "sharpe": result.sharpe if hasattr(result, "sharpe") else 0,
        "return": result.total_return if hasattr(result, "total_return") else 0,
        "max_drawdown": result.max_drawdown if hasattr(result, "max_drawdown") else 0,
        "trades": result.num_trades if hasattr(result, "num_trades") else 0,
        "params": result.params if hasattr(result, "params") else {},
        "refinement_run": state.run_id,
        "refinement_turns": state.current_turn,
        "validation": validation,
    })
    
    winners_path.write_text(json.dumps(winners, indent=2, default=str))
    logger.info("Promoted to winner: refined_%s", state.mandate_name)


# ── Main Loop ───────────────────────────────────────────────────────

def refinement_loop(mandate_path: str, max_turns: int = 7,
                    sharpe_target: float = 1.5) -> RunState:
    """Main refinement loop. One strategy, up to max_turns iterations.
    
    Args:
        mandate_path: Path to mandate JSON file.
        max_turns: Maximum refinement turns.
        sharpe_target: Target Sharpe ratio for success.
    
    Returns:
        Final RunState.
    """
    mandate = load_json(mandate_path)
    run_dir = create_run_directory(mandate)
    
    state = RunState(
        run_id=run_dir.name,
        mandate_name=mandate.get("name", "unknown"),
        created_at=datetime.now(timezone.utc).isoformat(),
        max_turns=max_turns,
        sharpe_target=sharpe_target,
        tickers=mandate.get("tickers", ["AAPL", "SPY"]),
    )
    
    if not acquire_lock(run_dir):
        logger.error("Run %s is locked. Skipping.", run_dir.name)
        return state
    
    state.status = "running"
    save_state(run_dir, state)
    
    try:
        for turn in range(1, max_turns + 1):
            state.current_turn = turn
            print(f"\n{'='*60}")
            print(f"Turn {turn}/{max_turns} | Best Sharpe: {state.best_sharpe:.2f}")
            
            # Clear indicator cache between turns
            try:
                from crabquant.indicator_cache import clear_cache
                clear_cache()
            except ImportError:
                pass
            
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
            
            for attempt in range(3):
                modification_dict = call_llm_inventor(context, str(context_path))
                if modification_dict is None:
                    logger.warning("LLM call failed (attempt %d)", attempt + 1)
                    continue
                
                strategy_code = modification_dict.get("new_strategy_code", "")
                if not strategy_code:
                    logger.warning("LLM returned empty code (attempt %d)", attempt + 1)
                    continue
                
                # 3. Validate through gates
                gates_ok, gate_errors = run_validation_gates(strategy_code)
                if gates_ok:
                    break
                
                # Feed errors back for next attempt
                context["validation_errors"] = gate_errors
                logger.warning("Gate failed (attempt %d): %s", attempt + 1, gate_errors)
            
            if not gates_ok or strategy_code is None:
                logger.warning("All validation attempts failed. Advancing turn.")
                state.history.append({
                    "turn": turn, "status": "code_generation_failed",
                    "errors": gate_errors,
                })
                save_state(run_dir, state)
                continue
            
            # Save strategy code
            strategy_path = run_dir / f"strategy_v{turn}.py"
            strategy_path.write_text(strategy_code)
            
            # Load strategy module
            strategy_module = load_strategy_module(strategy_path)
            if strategy_module is None:
                logger.warning("Failed to load strategy module. Advancing turn.")
                state.history.append({
                    "turn": turn, "status": "module_load_failed",
                })
                save_state(run_dir, state)
                continue
            
            # 4. Backtest
            primary_ticker = mandate.get("primary_ticker", state.tickers[0])
            period = mandate.get("period", "2y")
            backtest_output = run_backtest_safely(strategy_module, primary_ticker, period)
            
            if backtest_output is None or backtest_output[0] is None:
                logger.warning("Backtest crashed. Advancing turn.")
                state.history.append({
                    "turn": turn, "status": "backtest_crash",
                })
                save_state(run_dir, state)
                continue
            
            result, df, portfolio = backtest_output
            
            # 5. Diagnostics
            sharpe_by_year = compute_sharpe_by_year(portfolio)
            guardrail_result = check_guardrails(result)
            
            if hasattr(guardrail_result, 'violations'):
                guardrails = guardrail_result
            else:
                guardrails = MagicMock(violations=[], warnings=[])
            
            failure_mode, failure_details = classify_failure(
                result, guardrails, sharpe_by_year,
                data_length=len(df),
                sharpe_target=sharpe_target,
            )
            
            # 5b. Delta from previous
            prev_code_path = state.history[-1].get("code_path") if state.history else None
            action = modification_dict.get("action", "unknown")
            hypothesis = modification_dict.get("hypothesis", "")
            delta_from_prev = compute_delta(
                strategy_code, action, hypothesis, prev_code_path
            )
            
            # 6. Stagnation
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
                profit_factor=getattr(result, 'profit_factor', 0.0),
                calmar_ratio=getattr(result, 'calmar_ratio', 0.0),
                sortino_ratio=getattr(result, 'sortino_ratio', 0.0),
                composite_score=getattr(result, 'score', 0.0),
                failure_mode=failure_mode,
                failure_details=failure_details,
                sharpe_by_year=sharpe_by_year,
                stagnation_score=stagnation_score,
                stagnation_trend=stagnation_trend,
                previous_sharpes=[h.get("sharpe", 0) for h in state.history],
                previous_actions=[h.get("action", "") for h in state.history],
                guardrail_violations=getattr(guardrails, 'violations', []),
                guardrail_warnings=getattr(guardrails, 'warnings', []),
                # Tier 2 (computed conditionally later)
                regime_sharpe=None,
                regime_regime_shift=None,
                top_drawdowns=None,
                portfolio_correlation=None,
                benchmark_return_pct=None,
                market_regime=None,
                # Strategy context
                current_strategy_code=current_code,
                current_params=result.params if result.params else strategy_module.DEFAULT_PARAMS.copy(),
                previous_attempts=state.history[-3:],
            )
            
            save_report(run_dir, turn, report)
            
            # 8. Check success
            if result.sharpe >= sharpe_target and result.passed:
                print(f"  🏆 SUCCESS! Sharpe {result.sharpe:.2f} >= {sharpe_target}")
                
                validation = run_full_validation(
                    strategy_module.generate_signals,
                    strategy_module.DEFAULT_PARAMS,
                    primary_ticker,
                    state.tickers,
                )
                
                state.status = "success"
                state.best_sharpe = result.sharpe
                state.best_turn = turn
                state.best_code_path = str(strategy_path)
                state.history.append({
                    "turn": turn, "sharpe": result.sharpe,
                    "failure_mode": failure_mode,
                    "action": action,
                    "hypothesis": hypothesis,
                    "validation": validation,
                    "code_path": str(strategy_path),
                    "params_used": result.params if result.params else strategy_module.DEFAULT_PARAMS.copy(),
                    "strategy_hash": compute_strategy_hash(strategy_code),
                    "delta_from_prev": delta_from_prev,
                })
                save_state(run_dir, state)
                
                promote_to_winner(strategy_code, result, validation, state,
                                  strategy_module=strategy_module)
                return state
            
            # Track best attempt
            if result.sharpe > state.best_sharpe:
                state.best_sharpe = result.sharpe
                state.best_turn = turn
                state.best_code_path = str(strategy_path)
            
            # 9. Check stagnation
            stag_response = get_stagnation_response(turn, stagnation_score)
            if stag_response["constraint"] == "abandon":
                print(f"  🛑 Abandoning: stagnation {stagnation_score:.2f}")
                state.status = "abandoned"
                save_state(run_dir, state)
                return state
            
            # Append to history
            state.history.append({
                "turn": turn, "sharpe": result.sharpe,
                "failure_mode": failure_mode,
                "action": action,
                "hypothesis": hypothesis,
                "code_path": str(strategy_path),
                "params_used": result.params if result.params else strategy_module.DEFAULT_PARAMS.copy(),
                "strategy_hash": compute_strategy_hash(strategy_code),
                "delta_from_prev": delta_from_prev,
            })
            save_state(run_dir, state)
        
        # Exhausted all turns
        state.status = "max_turns_exhausted"
        save_state(run_dir, state)
        print(f"  Max turns exhausted. Best Sharpe: {state.best_sharpe:.2f} at turn {state.best_turn}")
        return state
    
    finally:
        release_lock(run_dir)


def main():
    parser = argparse.ArgumentParser(description="CrabQuant Refinement Loop")
    parser.add_argument("--mandate", required=True, help="Path to mandate JSON")
    parser.add_argument("--max-turns", type=int, default=7, help="Max refinement turns")
    parser.add_argument("--sharpe-target", type=float, default=1.5, help="Target Sharpe ratio")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    state = refinement_loop(args.mandate, args.max_turns, args.sharpe_target)
    print(f"\nFinal status: {state.status}")
    print(f"Best Sharpe: {state.best_sharpe:.2f} at turn {state.best_turn}")


if __name__ == "__main__":
    main()
