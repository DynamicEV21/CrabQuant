"""CrabQuant Refinement Pipeline — LLM-driven strategy research."""

from crabquant.refinement.schemas import RunState, BacktestReport, StrategyModification
from crabquant.refinement.config import RefinementConfig
from crabquant.refinement.classifier import classify_failure
from crabquant.refinement.validation_gates import run_validation_gates
from crabquant.refinement.module_loader import load_strategy_module
from crabquant.refinement.circuit_breaker import CircuitBreaker, CircuitBreakerState
from crabquant.refinement.cosmetic_guard import check_cosmetic_guard, CosmeticGuardState, CosmeticGuardResult
from crabquant.refinement.action_analytics import (
    track_action_result, load_run_history, aggregate_action_stats,
    compute_action_success_rates, generate_llm_context,
)
from crabquant.refinement.promotion import (
    run_full_validation_check, promote_to_winner,
    auto_promote, is_already_registered, register_strategy,
)
from crabquant.refinement.stagnation import compute_stagnation, get_stagnation_response, check_hypothesis_failure_alignment
from crabquant.refinement.mandate_generator import (
    scan_strategy_catalog, detect_archetype,
    generate_mandates, save_mandates,
)
from crabquant.refinement.wave_dashboard import (
    generate_dashboard, DashboardSnapshot, snapshot_to_json,
)
from crabquant.refinement.wave_manager import run_waves
from crabquant.refinement.wave_scaling import WaveStatusTracker
from crabquant.refinement.state import DaemonState
from crabquant.refinement.context_builder import build_llm_context
from crabquant.refinement.llm_api import call_zai_llm, call_llm_inventor
from crabquant.refinement.diagnostics import run_backtest_safely, compute_sharpe_by_year, compute_strategy_hash, compute_tier2_diagnostics
from crabquant.refinement.deflated_sharpe import deflated_sharpe, deflated_sharpe_ratio
