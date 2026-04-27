"""
CrabQuant Refinement Pipeline — Context Builder

Assembles the context payload for LLM strategy generation/refinement.
Includes strategy examples, catalogs, and delta computation.
"""

import inspect
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from crabquant.strategies import STRATEGY_REGISTRY


def get_strategy_catalog() -> list[dict]:
    """Return one-liner descriptions of all registered strategies.
    
    Returns:
        List of dicts with 'name', 'description' keys.
    """
    catalog = []
    for name, entry in STRATEGY_REGISTRY.items():
        # entry = (fn, defaults, grid, desc, matrix_fn)
        desc = entry[3] if len(entry) > 3 else "No description"
        catalog.append({"name": name, "description": desc})
    return catalog


def _strip_advanced_functions(source: str) -> str:
    """Strip PARAM_GRID and generate_signals_matrix from strategy source code.
    
    These are not needed by the refinement loop and their inclusion in examples
    causes the LLM to generate unnecessarily large strategy code, often exceeding
    max_tokens and producing truncated/incomplete JSON responses.
    """
    lines = source.split('\n')
    result_lines = []
    skip_mode = None  # None, 'braces', 'function'
    brace_depth = 0
    func_indent = 0
    
    for line in lines:
        stripped = line.strip()
        leading = len(line) - len(line.lstrip())
        
        # Skip blank lines during any skip mode
        if not stripped and skip_mode:
            continue
        
        if skip_mode == 'braces':
            brace_depth += stripped.count('{') - stripped.count('}')
            if brace_depth <= 0:
                skip_mode = None
                brace_depth = 0
            continue
        
        if skip_mode == 'function':
            # Stop when we see another top-level definition at the same indent
            if (stripped.startswith('def ') or stripped.startswith('class ') or
                    stripped.startswith('PARAM_GRID') or stripped.startswith('DESCRIPTION') or
                    stripped.startswith('DEFAULT_PARAMS')) and leading <= func_indent:
                skip_mode = None
                # Don't skip this line — it's the next top-level block
                result_lines.append(line)
                continue
            # Skip everything else (including signature continuations at indent 0)
            continue
        
        # Check if this line starts a block we want to skip
        if stripped.startswith('PARAM_GRID'):
            if '{' in stripped:
                brace_depth = stripped.count('{') - stripped.count('}')
                skip_mode = 'braces' if brace_depth > 0 else None
            continue
        
        if stripped.startswith('def generate_signals_matrix'):
            func_indent = leading
            skip_mode = 'function'
            continue
        
        result_lines.append(line)
    
    return '\n'.join(result_lines).strip()


def get_strategy_examples(archetype: str = "any") -> list[dict]:
    """Return strategy source as examples for the LLM.
    
    Picks 2 strategies relevant to the mandate archetype.
    Falls back to macd_momentum + rsi_crossover for unknown archetypes.
    
    PARAM_GRID and generate_signals_matrix are stripped from examples
    to reduce token usage and prevent LLM from generating unnecessary code.
    
    Args:
        archetype: Strategy archetype string (momentum, mean_reversion, etc.)
    
    Returns:
        List of dicts with 'name', 'description', 'default_params', 'source_code'.
    """
    examples_map = {
        "momentum": ["macd_momentum", "ema_crossover"],
        "mean_reversion": ["rsi_crossover", "bollinger_squeeze"],
        "breakout": ["atr_channel_breakout", "volume_breakout"],
        "trend": ["ichimoku_trend", "ema_ribbon_reversal"],
    }
    names = examples_map.get(archetype, ["macd_momentum", "rsi_crossover"])
    
    examples = []
    for name in names:
        if name in STRATEGY_REGISTRY:
            entry = STRATEGY_REGISTRY[name]
            fn = entry[0]
            defaults = entry[1]
            desc = entry[3] if len(entry) > 3 else name
            
            try:
                module = inspect.getmodule(fn)
                source = inspect.getsource(module) if module else inspect.getsource(fn)
                # Strip PARAM_GRID and generate_signals_matrix from examples
                # to reduce token usage and prevent LLM from generating them
                source = _strip_advanced_functions(source)
            except (TypeError, OSError):
                source = f"# Source unavailable for {name}"
            
            examples.append({
                "name": name,
                "description": desc,
                "default_params": defaults,
                "source_code": source,
            })
    
    return examples


def compute_delta(
    current_code: str,
    action: str,
    hypothesis: str,
    prev_code_path: Optional[str] = None,
) -> str:
    """Compute human-readable summary of what changed vs the previous turn.
    
    Detects indicator additions/removals and parameter changes.
    
    Args:
        current_code: The new strategy code.
        action: The modification action taken.
        hypothesis: The hypothesis for the modification.
        prev_code_path: Path to the previous strategy code file.
    
    Returns:
        String summary of changes.
    """
    parts = [f"Action: {action}", f"Hypothesis: {hypothesis}"]
    
    if not prev_code_path:
        return "Initial strategy (no prior version)"
    
    prev_path = Path(prev_code_path)
    if not prev_path.exists():
        return "; ".join(parts) + "; (previous code file not found)"
    
    try:
        prev_code = prev_path.read_text()
        
        # Detect indicator changes via cached_indicator calls
        old_indicators = set(re.findall(r'cached_indicator\(["\'](\w+)', prev_code))
        new_indicators = set(re.findall(r'cached_indicator\(["\'](\w+)', current_code))
        
        added = new_indicators - old_indicators
        removed = old_indicators - new_indicators
        
        if added:
            parts.append(f"Added indicators: {', '.join(sorted(added))}")
        if removed:
            parts.append(f"Removed indicators: {', '.join(sorted(removed))}")
        if not added and not removed:
            parts.append("Same indicators, logic/params changed")
    except Exception:
        parts.append("(could not compare with previous code)")
    
    return "; ".join(parts)


def build_llm_context(
    state,
    report: Optional[object] = None,
    mandate: Optional[dict] = None,
) -> dict:
    """Build the context payload for the LLM inventor agent.
    
    Args:
        state: RunState dataclass instance with current loop state.
        report: BacktestReport from previous turn (None on turn 1).
        mandate: Dict with mandate configuration.
    
    Returns:
        Context dict to be passed to the LLM.
    """
    mandate = mandate or {}
    
    context = {
        "mandate": mandate,
        "current_turn": getattr(state, "current_turn", 0) + 1,
        "max_turns": getattr(state, "max_turns", 7),
        "sharpe_target": getattr(state, "sharpe_target", 1.5),
        "tickers": getattr(state, "tickers", ["AAPL", "SPY"]),
        
        # Previous attempts (last 3, with params and deltas)
        "previous_attempts": getattr(state, "history", [])[-3:],
        
        # Current best
        "best_sharpe_so_far": getattr(state, "best_sharpe", 0.0),
        "best_turn": getattr(state, "best_turn", 0),
        
        # Strategy examples — FULL code for 2 representative strategies
        "strategy_examples": get_strategy_examples(
            mandate.get("strategy_archetype", "any")
        ),
        
        # Strategy catalog — one-liner descriptions of all strategies
        "strategy_catalog": get_strategy_catalog(),
    }
    
    if report is not None:
        # Convert report to dict if it's a dataclass
        if hasattr(report, "to_dict"):
            context["backtest_report"] = report.to_dict()
        elif hasattr(report, "__dataclass_fields__"):
            context["backtest_report"] = asdict(report)
        else:
            context["backtest_report"] = dict(report)
        
        # Include current strategy code so LLM can modify it
        context["current_strategy_code"] = getattr(report, "current_strategy_code", None)
        context["current_params"] = getattr(report, "current_params", None)
    
    return context
