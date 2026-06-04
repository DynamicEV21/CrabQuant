"""
CrabQuant Refinement Pipeline — Context Builder

Assembles the context payload for LLM strategy generation/refinement.
Includes strategy examples, catalogs, and delta computation.
"""

import inspect
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from crabquant.strategies import STRATEGY_REGISTRY
from crabquant.strategies._registry_compat import (
    get_fn as _get_fn,
    get_defaults as _get_defaults,
    get_description as _get_desc,
)
from crabquant.refinement.prompts import (
    load_indicator_reference,
    extract_quick_reference,
    build_turn1_prompt,
    build_refinement_prompt,
    format_stagnation_suffix,
)
from crabquant.refinement.trade_count_estimator import build_trade_count_guidance

# Path to winners database
_WINNERS_PATH = Path(__file__).parent.parent.parent / "results" / "winners" / "winners.json"
_RUNS_DIR = Path(__file__).parent.parent.parent / "refinement_runs"


def get_strategy_catalog() -> list[dict]:
    """Return one-liner descriptions of all registered strategies.
    
    Returns:
        List of dicts with 'name', 'description' keys.
    """
    catalog = []
    for name, entry in STRATEGY_REGISTRY.items():
        desc = _get_desc(entry)
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
            fn = _get_fn(entry)
            defaults = _get_defaults(entry)
            desc = _get_desc(entry)
            
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


def get_winner_examples(
    ticker: str = "",
    archetype: str = "any",
    max_examples: int = 2,
    runs_dir: Optional[Path] = None,
) -> list[dict]:
    """Load top-performing winner strategies as cross-run learning examples.

    Loads from results/winners/winners.json, ranks by composite score
    (Sharpe × √trades) to favor robust strategies over curve-fits, and
    attempts to load actual strategy code from refinement run directories.

    Args:
        ticker: Ticker to prefer (e.g. "SPY"). Falls back to any ticker.
        archetype: Strategy archetype to prefer. Falls back to any.
        max_examples: Maximum number of examples to return.
        runs_dir: Override path to refinement_runs/ directory.

    Returns:
        List of dicts with 'name', 'sharpe', 'trades', 'ticker', 'source_code'.
        Empty list if no winners found or no code available.
    """
    runs_dir = runs_dir or _RUNS_DIR

    # Load winners
    if not _WINNERS_PATH.exists():
        return []

    try:
        with open(_WINNERS_PATH) as f:
            winners = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    if not winners:
        return []

    # Compute composite score and deduplicate by strategy name
    scored = []
    seen_names = set()
    for w in winners:
        name = w.get("strategy", "")
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        sharpe = w.get("sharpe", 0) or 0
        trades = w.get("trades", 0) or 0
        if trades < 1:
            continue

        composite = sharpe * (trades ** 0.5)
        w_ticker = w.get("ticker", "")
        w_run = w.get("refinement_run", "")

        # Bonus for matching ticker
        if ticker and w_ticker == ticker:
            composite *= 1.5

        scored.append({
            "name": name,
            "sharpe": sharpe,
            "trades": trades,
            "ticker": w_ticker,
            "composite": composite,
            "refinement_run": w_run,
            "params": w.get("params", {}),
        })

    if not scored:
        return []

    # Sort by composite score, take top N
    scored.sort(key=lambda x: x["composite"], reverse=True)
    candidates = scored[:max_examples * 5]  # Oversample — some won't have code

    # ── Archetype-aware diversity selection ──
    # Classify each candidate by its strategy name pattern to ensure
    # we don't feed the LLM examples all from the same archetype.
    # This prevents the LLM from converging on a single strategy family.
    def _classify_archetype(name: str) -> str:
        """Classify strategy name into an archetype bucket."""
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["momentum", "roc", "trend", "ema_cross", "ichimoku"]):
            return "momentum"
        elif any(kw in name_lower for kw in ["reversion", "mean_rev", "rsi", "bollinger", "bb_"]):
            return "mean_reversion"
        elif any(kw in name_lower for kw in ["breakout", "channel", "squeeze", "atr"]):
            return "breakout"
        elif any(kw in name_lower for kw in ["volume", "vol_", "obv", "vwap"]):
            return "volume"
        elif any(kw in name_lower for kw in ["adx", "pullback", "macd", "stoch"]):
            return "trend_following"
        else:
            return "other"

    # Select diverse candidates: pick best from each archetype first,
    # then fill remaining slots from highest composite score.
    archetype_buckets: dict[str, list] = {}
    for cand in candidates:
        arch = _classify_archetype(cand["name"])
        archetype_buckets.setdefault(arch, []).append(cand)

    diverse_candidates = []
    seen_archetypes = set()

    # First pass: take best from each archetype (diversity-first)
    for arch in sorted(archetype_buckets.keys()):
        if len(diverse_candidates) >= max_examples * 3:
            break
        diverse_candidates.append(archetype_buckets[arch][0])
        seen_archetypes.add(arch)

    # Second pass: fill with remaining high-scoring candidates
    for cand in candidates:
        if len(diverse_candidates) >= max_examples * 3:
            break
        if cand not in diverse_candidates:
            diverse_candidates.append(cand)

    examples = []
    for cand in diverse_candidates:
        if len(examples) >= max_examples:
            break

        source_code = None

        # Try loading from refinement run directory
        if cand["refinement_run"]:
            run_path = runs_dir / cand["refinement_run"]
            # Try the best turn's strategy file
            state_path = run_path / "state.json"
            if state_path.exists():
                try:
                    with open(state_path) as f:
                        state = json.load(f)
                    best_turn = state.get("best_turn", 1)
                    code_path = run_path / f"strategy_v{best_turn}.py"
                    if code_path.exists():
                        source_code = code_path.read_text()
                except (json.JSONDecodeError, OSError):
                    pass

            # Fallback: try latest strategy file
            if not source_code:
                try:
                    latest = sorted(run_path.glob("strategy_v*.py"))[-1]
                    source_code = latest.read_text()
                except (IndexError, OSError):
                    pass

        # Try loading from STRATEGY_REGISTRY for sweep winners
        if not source_code and cand["name"] in STRATEGY_REGISTRY:
            try:
                entry = STRATEGY_REGISTRY[cand["name"]]
                fn = _get_fn(entry)
                module = inspect.getmodule(fn)
                source_code = inspect.getsource(module) if module else inspect.getsource(fn)
                source_code = _strip_advanced_functions(source_code)
            except (TypeError, OSError):
                pass

        if source_code:
            # Truncate very long strategies to avoid token bloat
            if len(source_code) > 3000:
                # Keep imports, generate_signals, DEFAULT_PARAMS, DESCRIPTION
                lines = source_code.split("\n")
                kept = []
                in_target = False
                for line in lines:
                    stripped = line.strip()
                    if any(stripped.startswith(kw) for kw in [
                        "import ", "from ", "def generate_signals",
                        "DEFAULT_PARAMS", "DESCRIPTION",
                    ]):
                        in_target = True
                    if in_target:
                        kept.append(line)
                        # End of function
                        if stripped.startswith("def ") and ":" in stripped and "generate_signals" not in stripped:
                            in_target = False
                source_code = "\n".join(kept) if kept else source_code[:3000]

            examples.append({
                "name": cand["name"],
                "sharpe": cand["sharpe"],
                "trades": cand["trades"],
                "ticker": cand["ticker"],
                "source_code": source_code,
                "params": cand["params"],
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


def _enrich_context_with_report(
    context: dict,
    report: object,
    state: object,
    sharpe_target: float,
) -> None:
    """Extract report data and enrich context with analysis sections.

    Handles report conversion, revert logic, multi-ticker feedback,
    feature importance, parameter optimization, and positive feedback.
    """
    # Convert report to dict
    if hasattr(report, "to_dict"):
        context["backtest_report"] = report.to_dict()
    elif hasattr(report, "__dataclass_fields__"):
        context["backtest_report"] = asdict(report)
    else:
        context["backtest_report"] = dict(report)

    context["current_strategy_code"] = getattr(report, "current_strategy_code", None)
    context["current_params"] = getattr(report, "current_params", None)

    # Auto-revert: inject best code instead of regressed code
    revert_notice = getattr(state, "revert_notice", "")
    if revert_notice:
        best_code = getattr(state, "best_strategy_code", "")
        if best_code:
            context["current_strategy_code"] = best_code

    # Multi-ticker backtest feedback
    mt_results = getattr(report, "multi_ticker_results", None)
    if mt_results is not None:
        context["multi_ticker_feedback"] = _format_multi_ticker_feedback(mt_results)

    # Feature importance feedback
    fi = getattr(report, "feature_importance", None)
    if fi is not None and fi.get("indicators"):
        from crabquant.refinement.feature_importance import format_feature_importance_for_prompt
        context["feature_importance_section"] = format_feature_importance_for_prompt(fi)

    # Parameter optimization feedback
    po = getattr(report, "param_optimization", None)
    if po is not None and po.get("param_optimization_applied"):
        from crabquant.refinement.param_optimizer import format_optimization_for_prompt
        opt = type("obj", (object,), {
            "was_optimized": po["param_optimization_applied"],
            "default_sharpe": po.get("default_sharpe", 0),
            "optimized_sharpe": po.get("optimized_sharpe", 0),
            "default_trades": 0,
            "optimized_trades": 0,
            "combinations_tested": po.get("combinations_tested", 0),
            "improvement_pct": po.get("improvement_pct", 0),
            "sweep_time_seconds": po.get("sweep_time_seconds", 0),
            "default_params": {},
            "optimized_params": {},
        })()
        context["param_optimization_section"] = format_optimization_for_prompt(opt)

    # Positive feedback analysis
    try:
        from crabquant.refinement.positive_feedback import (
            analyze_positive_feedback,
            format_positive_feedback_for_prompt,
        )
        report_dict = context.get("backtest_report", {})
        pos_feedback = analyze_positive_feedback(
            sharpe_ratio=report_dict.get("sharpe_ratio", 0.0),
            sharpe_target=sharpe_target,
            total_return_pct=report_dict.get("total_return_pct", 0.0),
            max_drawdown_pct=report_dict.get("max_drawdown_pct", 0.0),
            win_rate=report_dict.get("win_rate", 0.0),
            profit_factor=report_dict.get("profit_factor", 0.0),
            sortino_ratio=report_dict.get("sortino_ratio", 0.0),
            calmar_ratio=report_dict.get("calmar_ratio", 0.0),
            total_trades=report_dict.get("total_trades", 0),
            avg_holding_bars=report_dict.get("avg_holding_bars"),
            sharpe_by_year=report_dict.get("sharpe_by_year"),
            failure_mode=report_dict.get("failure_mode", ""),
        )
        section = format_positive_feedback_for_prompt(pos_feedback)

        # Surface previous successful turns
        history = getattr(state, "history", [])
        successful_turns = [
            h for h in history
            if h.get("sharpe") is not None and h["sharpe"] >= sharpe_target
        ]
        if successful_turns:
            if section:
                section += "\n\n"
            section += "### 🏆 Previous Successful Turns\n\n"
            section += "These turns hit the Sharpe target — study what made them work:\n"
            for st in successful_turns[-3:]:
                turn_num = st.get("turn", "?")
                sharpe = st.get("sharpe", 0)
                action = st.get("action", "unknown")
                trades = st.get("num_trades", "?")
                section += (
                    f"- **Turn {turn_num}**: Sharpe={sharpe:.2f}, "
                    f"Action={action}, Trades={trades}\n"
                )
        if section:
            context["positive_feedback_section"] = section
    except Exception:
        logger.debug("Positive feedback analysis failed, skipping", exc_info=True)


def _enrich_context_with_analyses(
    context: dict, state: object, mandate: dict
) -> None:
    """Enrich context with stagnation recovery, plateau detection, crash feedback,
    and action analytics."""
    # Stagnation recovery
    stagnation_recovery = _build_stagnation_recovery_section(state)
    if stagnation_recovery:
        context["stagnation_recovery"] = stagnation_recovery

    # Indicator family plateau detection
    try:
        from crabquant.refinement.stagnation import check_family_plateau
        history = getattr(state, "history", [])
        should_pivot, pivot_type, plateau_message = check_family_plateau(
            history, mandate
        )
        if should_pivot and plateau_message:
            context["family_plateau_section"] = (
                f"## 🔄 INDICATOR FAMILY PLATEAU DETECTED\n\n"
                f"**Pivot type:** {pivot_type}\n\n"
                f"{plateau_message}"
            )
    except Exception:
        logger.debug("Family plateau check failed, skipping", exc_info=True)

    # Crash error feedback
    crash_feedback = _build_crash_error_feedback(state)
    if crash_feedback:
        context["crash_error_feedback"] = crash_feedback

    # Action analytics
    try:
        from crabquant.refinement.action_analytics import (
            generate_llm_context,
            load_run_history,
        )
        from crabquant.refinement.action_analytics import RUN_HISTORY_FILE
        run_history = load_run_history(RUN_HISTORY_FILE)
        context["action_analytics"] = generate_llm_context(run_history)
    except Exception:
        context["action_analytics"] = "No historical action data available."


def build_llm_context(
    state,
    report: Optional[object] = None,
    mandate: Optional[dict] = None,
    effective_target: Optional[float] = None,
) -> dict:
    """Build the context payload for the LLM inventor agent.
    
    Args:
        state: RunState dataclass instance with current loop state.
        report: BacktestReport from previous turn (None on turn 1).
        mandate: Dict with mandate configuration.
        effective_target: Adaptive Sharpe target for this turn. If None,
            uses the original sharpe_target from state.
    
    Returns:
        Context dict to be passed to the LLM.
    """
    mandate = mandate or {}
    
    # If no effective_target provided, fall back to original sharpe_target
    original_sharpe_target = getattr(state, "sharpe_target", 1.5)
    if effective_target is None:
        effective_target = original_sharpe_target
    
    context = {
        "mandate": mandate,
        "current_turn": getattr(state, "current_turn", 0),
        "max_turns": getattr(state, "max_turns", 7),
        "sharpe_target": original_sharpe_target,
        "effective_target": effective_target,
        "tickers": getattr(state, "tickers", ["AAPL", "SPY"]),
        
        # Previous attempts (last 3, with params and deltas)
        "previous_attempts": getattr(state, "history", [])[-3:],
        
        # Current best
        "best_sharpe_so_far": getattr(state, "best_sharpe", 0.0),
        "best_composite_score": getattr(state, "best_composite_score", -999.0),
        "best_turn": getattr(state, "best_turn", 0),
        
        # Strategy examples — FULL code for 2 representative strategies
        "strategy_examples": get_strategy_examples(
            mandate.get("strategy_archetype", "any")
        ),

        # Winner examples — proven strategies from past runs (cross-run learning)
        # Respects cross_run_learning toggle (default: True)
        "winner_examples": (
            get_winner_examples(
                ticker=(mandate.get("tickers") or [""])[0],
                archetype=mandate.get("strategy_archetype", "any"),
                max_examples=2,
            )
            if mandate.get("cross_run_learning", True)
            else []
        ),
        
        # Strategy catalog — one-liner descriptions of all strategies
        "strategy_catalog": get_strategy_catalog(),

        # Indicator API reference — loaded once and passed to LLM prompts
        "indicator_reference": load_indicator_reference(),
        "indicator_quick_ref": extract_quick_reference(load_indicator_reference()),
    }
    
    # Inject archetype template if mandate specifies one
    archetype_name = mandate.get("strategy_archetype", "any")
    if archetype_name != "any":
        from crabquant.refinement.archetypes import get_archetype, format_archetype_for_prompt
        archetype = get_archetype(archetype_name)
        if archetype:
            context["archetype_section"] = format_archetype_for_prompt(archetype)
    
    if report is not None:
        _enrich_context_with_report(context, report, state, original_sharpe_target)

    # Enrich context with side-effect analyses
    _enrich_context_with_analyses(context, state, mandate)

    # ── CRITICAL: Build the formatted prompt for call_llm_inventor ──────
    current_turn_num = getattr(state, "current_turn", 0)
    indicator_ref = context.get("indicator_reference", "")
    indicator_qr = context.get("indicator_quick_ref", "")

    trade_count_guidance, failure_pattern_section, dominant_failure_mode = (
        _compute_prompt_guidance(mandate)
    )

    # Phase 6.1: Inject too_few_trades hint
    if dominant_failure_mode == "too_few_trades":
        trade_count_guidance += _TOO_FEW_TRADES_HINT

    # Build prompt (turn 1 invention vs turn 2+ refinement)
    if report is None:
        _build_turn1_prompt(
            context, state, mandate, current_turn_num,
            indicator_ref, indicator_qr, effective_target,
            trade_count_guidance, failure_pattern_section,
        )
    else:
        _build_turn2plus_prompt(
            context, state, report, mandate, current_turn_num,
            indicator_ref, indicator_qr, effective_target,
            trade_count_guidance, failure_pattern_section,
        )

    # Append extra sections to the built prompt
    _append_prompt_sections(context, state, mandate)

    return context


_TOO_FEW_TRADES_HINT = (
    "\n\n### ⚠️ CRITICAL: too_few_trades is the #1 failure mode right now\n"
    "Recent strategies are generating too few signals. You MUST:\n"
    "1. Use SHORT lookback windows (5-12 periods) — long lookbacks filter out too many signals.\n"
    "2. Keep entry logic to 1-2 conditions MAX — each additional condition cuts trade count.\n"
    "3. Add a RE-ENTRY mechanism after exits (3-5 bar cooldown), so you can catch repeated opportunities.\n"
    "4. Target 10-15+ trades minimum. A strategy with 3 trades is NOT acceptable regardless of Sharpe.\n"
    "5. Prefer EMA over SMA (faster response), shorter RSI periods (7-10 instead of 14), "
    "and wider threshold bands (RSI < 35 instead of < 20).\n"
)


def _compute_prompt_guidance(mandate: dict) -> tuple[str, str, str | None]:
    """Compute trade count guidance and failure pattern analysis.

    Returns:
        (trade_count_guidance, failure_pattern_section, dominant_failure_mode)
    """
    trade_count_guidance = ""
    try:
        tc_ticker = (mandate.get("tickers") or ["SPY"])[0]
        tc_period = mandate.get("period", "2y")
        tc_timeframe = mandate.get("timeframe", "daily")
        tc_strategy_type = mandate.get("strategy_archetype", "momentum")
        trade_count_guidance = build_trade_count_guidance(
            ticker=tc_ticker,
            period=tc_period,
            timeframe=tc_timeframe,
            strategy_type=tc_strategy_type,
        )
    except Exception:
        pass

    failure_pattern_section = ""
    dominant_failure_mode = None
    try:
        from crabquant.refinement.failure_patterns import (
            analyze_failure_patterns,
            format_failure_patterns_for_prompt,
        )
        from crabquant.refinement.action_analytics import (
            RUN_HISTORY_FILE,
            load_run_history,
        )
        history = load_run_history(RUN_HISTORY_FILE)
        pattern_data = analyze_failure_patterns(history)
        if pattern_data.get("total_failures", 0) > 0:
            failure_pattern_section = format_failure_patterns_for_prompt(pattern_data)
            dominant_failure_mode = pattern_data.get("dominant_mode")
    except Exception:
        pass

    return trade_count_guidance, failure_pattern_section, dominant_failure_mode


def _build_turn1_prompt(
    context: dict, state, mandate: dict, current_turn_num: int,
    indicator_ref: str, indicator_qr: str, effective_target: float,
    trade_count_guidance: str, failure_pattern_section: str,
) -> None:
    """Build the Turn 1 invention prompt."""
    try:
        base_prompt = build_turn1_prompt(
            mandate=mandate,
            current_turn=current_turn_num,
            max_turns=context.get("max_turns", 7),
            strategy_examples=context.get("strategy_examples"),
            winner_examples=context.get("winner_examples"),
            indicator_reference=indicator_ref,
            indicator_quick_ref=indicator_qr,
            archetype_section=context.get("archetype_section"),
            effective_target=effective_target,
            trade_count_guidance=trade_count_guidance,
        )
        # Apply adaptive prompt modifications (regime hints, portfolio gaps)
        try:
            from crabquant.refinement.adaptive_prompts import (
                build_adaptive_invention_prompt,
            )
            from crabquant.regime import detect_regime as _detect_regime
            from crabquant.data import load_data as _load_data

            regime = _detect_current_regime(mandate)
            portfolio_gaps = _compute_portfolio_gaps()
            context["prompt"] = build_adaptive_invention_prompt(
                base_prompt=base_prompt,
                regime=regime,
                portfolio_gaps=portfolio_gaps,
                adaptation_rate=mandate.get("adaptation_rate", 0.80),
            )
        except Exception:
            context["prompt"] = base_prompt

        if failure_pattern_section:
            context["failure_pattern_section"] = failure_pattern_section
    except Exception:
        logger.warning("Turn 1 prompt build failed, falling back to per-field prompt")


def _detect_current_regime(mandate: dict) -> str:
    """Detect current market regime from the primary ticker."""
    try:
        from crabquant.regime import detect_regime as _detect_regime
        from crabquant.data import load_data as _load_data
        ticker = (mandate.get("tickers") or ["SPY"])[0]
        period = mandate.get("period", "6mo")
        df = _load_data(ticker, period=period)
        if df is not None and len(df) >= 20:
            regime_enum, _ = _detect_regime(df)
            regime = regime_enum.name if hasattr(regime_enum, "name") else str(regime_enum)
            if regime == "MEAN_REVERSION":
                regime = "RANGING"
            return regime
    except Exception:
        pass
    return "UNKNOWN"


def _compute_portfolio_gaps() -> dict:
    """Compute portfolio coverage gaps from strategy registry."""
    try:
        from crabquant.refinement.archetypes import list_archetypes
        from crabquant.strategies import STRATEGY_REGISTRY
        archetypes = list_archetypes()
        gaps = {}
        for arch in archetypes:
            arch_strategies = [
                name for name in STRATEGY_REGISTRY
                if arch.lower() in name.lower()
            ]
            gaps[arch] = min(1.0, len(arch_strategies) / 5.0)
        return gaps
    except Exception:
        return {}


def _build_turn2plus_prompt(
    context: dict, state, report, mandate: dict, current_turn_num: int,
    indicator_ref: str, indicator_qr: str, effective_target: float,
    trade_count_guidance: str, failure_pattern_section: str,
) -> None:
    """Build the Turn 2+ refinement prompt."""
    try:
        tier1 = _report_to_tier1(report, context)

        # Compute action effectiveness for the current failure mode
        action_effectiveness_section = _compute_action_effectiveness(tier1)

        # Build stagnation suffix
        stag_suffix = _build_stagnation_suffix(state, current_turn_num)

        context["prompt"] = build_refinement_prompt(
            tier1_report=tier1,
            current_turn=current_turn_num,
            max_turns=context.get("max_turns", 7),
            sharpe_target=context.get("sharpe_target", 1.5),
            effective_target=effective_target,
            best_sharpe=context.get("best_sharpe_so_far", 0.0),
            best_turn=context.get("best_turn", 0),
            stagnation_suffix=stag_suffix,
            strategy_examples=context.get("strategy_examples"),
            winner_examples=context.get("winner_examples"),
            archetype_section=context.get("archetype_section"),
            indicator_reference=indicator_ref,
            indicator_quick_ref=indicator_qr,
            action_effectiveness_section=action_effectiveness_section,
            failure_pattern_section=failure_pattern_section,
            trade_count_guidance=trade_count_guidance,
        )
    except Exception:
        logger.warning("Turn 2+ prompt build failed, falling back to per-field prompt")


def _report_to_tier1(report: object, context: dict) -> dict:
    """Convert a BacktestReport to a tier1 dict for build_refinement_prompt."""
    if hasattr(report, "to_dict"):
        tier1 = report.to_dict()
    elif hasattr(report, "__dataclass_fields__"):
        tier1 = asdict(report)
    else:
        tier1 = dict(report)

    tier1.setdefault("sharpe_target", context.get("sharpe_target", 1.5))
    tier1.setdefault("total_return_pct", 0.0)
    tier1.setdefault("max_drawdown_pct", 0.0)
    tier1.setdefault("win_rate", 0.0)
    tier1.setdefault("profit_factor", 0.0)
    tier1.setdefault("sortino_ratio", 0.0)
    tier1.setdefault("calmar_ratio", 0.0)
    tier1.setdefault("avg_holding_bars", None)
    tier1.setdefault("sharpe_by_year", {})

    if context.get("feature_importance_section"):
        tier1["feature_importance_section"] = context["feature_importance_section"]
    return tier1


def _compute_action_effectiveness(tier1: dict) -> str:
    """Compute action effectiveness for the current failure mode."""
    try:
        from crabquant.refinement.action_effectiveness import (
            analyze_action_effectiveness,
            format_action_effectiveness_for_prompt,
        )
        from crabquant.refinement.action_analytics import RUN_HISTORY_FILE
        eff_data = analyze_action_effectiveness(RUN_HISTORY_FILE)
        fm = tier1.get("failure_mode", "")
        if eff_data.get("by_failure_mode") and fm:
            return format_action_effectiveness_for_prompt(eff_data, fm)
    except Exception:
        pass
    return ""


def _build_stagnation_suffix(state: object, current_turn_num: int) -> str:
    """Build stagnation suffix from state history."""
    try:
        from crabquant.refinement.stagnation import (
            compute_stagnation,
            get_stagnation_response,
        )
        history = getattr(state, "history", [])
        if len(history) >= 2:
            stag_score, _ = compute_stagnation(history)
            stag_response = get_stagnation_response(current_turn_num, stag_score)
            return format_stagnation_suffix(
                stag_response.get("constraint"),
                stag_response.get("prompt_suffix"),
            )
    except Exception:
        pass
    return ""


_APPEND_KEYS = [
    "multi_ticker_feedback",
    "crash_error_feedback",
    "action_analytics",
    "stagnation_recovery",
    "family_plateau_section",
    "failure_pattern_section",
    "param_optimization_section",
    "positive_feedback_section",
]


def _append_prompt_sections(context: dict, state: object, mandate: dict) -> None:
    """Append extra analysis sections to the built prompt."""
    prompt = context.get("prompt", "")
    if not prompt:
        return

    append_sections = []
    for key in _APPEND_KEYS:
        if context.get(key):
            append_sections.append(context[key])

    # Gate validation retry feedback
    if context.get("retry_feedback"):
        append_sections.append(
            f"\n## ⛔ GATE VALIDATION FAILED — FIX THESE ERRORS\n\n"
            f"{context['retry_feedback']}\n\n"
            f"Your previous code was REJECTED by validation gates. You MUST fix the "
            f"above errors in your new_strategy_code. Do NOT repeat the same mistake."
        )

    # Code quality pre-check feedback
    code_quality_fb = getattr(state, "code_quality_feedback", "")
    if code_quality_fb:
        append_sections.append(
            f"\n## ⛔ CODE QUALITY PRE-CHECK FAILED\n\n"
            f"{code_quality_fb}\n\n"
            f"Your previous code was REJECTED by the code quality pre-check. "
            f"You MUST fix the above anti-patterns in your new_strategy_code. "
            f"Do NOT repeat the same mistakes."
        )

    # Complexity analysis
    try:
        from crabquant.refinement.complexity import complexity_score
        history = getattr(state, "history", [])
        latest_turn = history[-1] if history else {}
        strategy_code = (
            latest_turn.get("code")
            or context.get("current_strategy_code")
            or ""
        )
        params = latest_turn.get("params_used") or context.get("current_params") or {}
        if strategy_code:
            cx = complexity_score(strategy_code, params)
            if cx["complexity"] > 60 or cx["flags"]:
                append_sections.append(_format_complexity_warning(cx))
    except Exception:
        logger.debug("Complexity analysis failed, skipping", exc_info=True)

    # Auto-revert notice
    revert_notice = getattr(state, "revert_notice", "")
    if revert_notice:
        append_sections.append(f"\n## ⚠️ STRATEGY REVERTED\n\n{revert_notice}\n")

    # Per-ticker alpha decomposition
    try:
        ticker_alpha = build_ticker_alpha_context(
            ticker=(mandate.get("tickers") or ["SPY"])[0],
            turn_history=getattr(state, "history", []),
        )
        if not ticker_alpha.startswith("No prior data"):
            append_sections.append(
                f"\n## 📊 Per-Ticker Alpha Decomposition\n\n{ticker_alpha}\n"
            )
    except Exception:
        pass

    if append_sections:
        context["prompt"] = prompt.rstrip() + "\n\n" + "\n\n".join(append_sections)


def _format_complexity_warning(cx: dict) -> str:
    """Format a complexity warning section for the LLM prompt."""
    warn_lines = [
        "## ⚠️ STRATEGY COMPLEXITY WARNING",
        "",
        f"Complexity score: **{cx['complexity']:.1f}**/100",
    ]
    if cx["complexity"] > 60:
        warn_lines.append(
            "Your strategy is overly complex and at high risk of "
            "overfitting. Simpler strategies generalize better."
        )
    _FLAG_SUGGESTIONS = {
        "high_complexity": (
            "Reduce overall code size. Remove unnecessary helper "
            "functions and consolidate logic."
        ),
        "too_many_params": (
            "Reduce the number of tunable parameters. Each extra "
            "parameter increases overfitting risk. Target ≤8 params."
        ),
        "deep_nesting": (
            "Flatten deeply nested if/for blocks. Deep nesting "
            "makes the strategy fragile and hard to reason about."
        ),
        "too_many_branches": (
            "Reduce the number of conditional branches. Excessive "
            "branching leads to sparse data per path."
        ),
        "too_many_functions": (
            "Consolidate helper functions. Too many functions "
            "increase code surface area without proportional benefit."
        ),
        "invalid_python": (
            "Your strategy code has syntax errors. Fix them before "
            "proceeding."
        ),
    }
    for flag in cx["flags"]:
        suggestion = _FLAG_SUGGESTIONS.get(flag, f"Address the `{flag}` issue.")
        warn_lines.append(f"- **{flag}**: {suggestion}")
    return "\n\n".join(warn_lines)


def build_ticker_alpha_context(ticker: str, turn_history: list) -> str:
    """Build per-ticker alpha decomposition summary for LLM prompts.

    Scans turn_history for all turns involving *ticker*, groups by
    archetype/indicator family, and returns a concise summary so the LLM
    knows what approaches have already been tried on this ticker and
    which showed promise.

    Only archetypes with 2+ attempts are included (one-offs are noise).

    Args:
        ticker: Uppercase ticker symbol (e.g. "AAPL").
        turn_history: List of history-entry dicts from RunState.history.

    Returns:
        Formatted string for prompt injection, or a "no data" message.
    """
    ticker_upper = ticker.upper()
    # Collect relevant turns
    relevant = [
        h for h in turn_history
        if h.get("ticker", "").upper() == ticker_upper
        and h.get("sharpe") is not None
    ]
    if not relevant:
        return f"No prior data for {ticker_upper}."

    # Group by archetype
    groups: dict[str, list[dict]] = {}
    for entry in relevant:
        archetype = entry.get("archetype", "unknown")
        groups.setdefault(archetype, []).append(entry)

    # Build per-archetype lines (skip families with < 2 attempts)
    lines = [f"{ticker_upper} past attempts ({len(relevant)} turns):"]
    family_stats: list[tuple[str, float, float, int, int]] = []

    for archetype, entries in sorted(groups.items()):
        if len(entries) < 2:
            continue
        sharpes = [e["sharpe"] for e in entries]
        avg_sharpe = sum(sharpes) / len(sharpes)
        best_sharpe = max(sharpes)
        total_trades = sum(e.get("num_trades", 0) for e in entries)
        # "passed" means sharpe >= 1.0 or status indicates success
        passed = sum(
            1 for e in entries
            if e["sharpe"] >= 1.0 or e.get("status") == "promoted"
        )
        # Extract primary indicator from action or indicators field
        indicators = entry.get("indicators", [])
        if isinstance(indicators, list) and indicators:
            ind_label = indicators[0]
        else:
            ind_label = archetype
        lines.append(
            f"  - {archetype} ({ind_label}): avg Sharpe {avg_sharpe:.2f}, "
            f"best {best_sharpe:.2f} ({passed}/{len(entries)} passed initial screen)"
        )
        family_stats.append((archetype, avg_sharpe, best_sharpe, passed, len(entries)))

    if len(lines) == 1:
        # All archetypes were one-offs
        return f"No prior data for {ticker_upper}."

    # Cap at 10 lines total (header + up to 9 archetype lines)
    if len(lines) > 10:
        lines = lines[:10]

    # Add suggestion: highlight best-performing family and worst
    if family_stats:
        best = max(family_stats, key=lambda s: s[2])  # best best_sharpe
        worst = min(family_stats, key=lambda s: s[1])  # worst avg_sharpe
        suggestion_parts = []
        if worst[1] < 0:
            suggestion_parts.append(f"Avoid {worst[0]} on {ticker_upper}")
        if best[2] > 0:
            suggestion_parts.append(f"{best[0]} shows promise")
        if suggestion_parts:
            lines.append(f"Suggestion: {', '.join(suggestion_parts)}.")

    return "\n".join(lines)


def _format_multi_ticker_feedback(mt_results: dict) -> str:
    """Format multi-ticker backtest results for LLM context injection.
    
    Produces a concise summary that tells the LLM which tickers the strategy
    passed/failed on, so it can generalize better.
    
    Args:
        mt_results: Dict from run_multi_ticker_backtest() with keys:
            tickers_tested, tickers_passed, avg_sharpe, min_sharpe,
            pass_rate, per_ticker (per-ticker list).
    
    Returns:
        Formatted string for prompt injection.
    """
    lines = [
        "## Multi-Ticker Validation Results",
        f"Tested on {mt_results['tickers_tested']} secondary tickers: "
        f"{mt_results['tickers_passed']} passed (Sharpe >= target).",
        f"Average Sharpe: {mt_results['avg_sharpe']:.2f}, "
        f"Range: [{mt_results['min_sharpe']:.2f}], "
        f"Pass rate: {mt_results['pass_rate']:.0%}",
        "",
        "Per-ticker breakdown:",
    ]
    for r in mt_results.get("per_ticker", []):
        status = "✅ PASS" if r.get("passed") else "❌ FAIL"
        lines.append(
            f"  {status} {r['ticker']}: Sharpe={r['sharpe']:.2f}, "
            f"Trades={r['trades']}, MaxDD={r.get('max_drawdown', 0):.1%}"
        )
    
    # Add actionable guidance
    failed = [r for r in mt_results.get("per_ticker", []) if not r.get("passed")]
    if failed:
        lines.append("")
        lines.append("The strategy is overfit to the primary ticker. Consider:")
        lines.append("- Simplifying logic (fewer conditions = more generalizable)")
        lines.append("- Using more universal signals (trend, momentum, volatility)")
        lines.append("- Avoiding ticker-specific parameter tuning")
        lines.append("- Reducing the number of combined indicators")
    
    return "\n".join(lines)


def _build_stagnation_recovery_section(state) -> str:
    """Build the stagnation recovery section for LLM context injection.

    Analyzes turn history to detect specific stagnation trap types and
    generates targeted recovery instructions. Only returns content when
    a meaningful trap is detected (medium severity or above).

    Args:
        state: RunState dataclass instance.

    Returns:
        Formatted recovery string for prompt injection, or empty string.
    """
    from crabquant.refinement.stagnation import (
        detect_stagnation_trap,
        build_stagnation_recovery,
    )

    history = getattr(state, "history", [])
    best_sharpe = getattr(state, "best_sharpe", 0.0)
    sharpe_target = getattr(state, "sharpe_target", 1.5)
    current_turn = getattr(state, "current_turn", 0)

    # Need at least 2 turns with Sharpe data to diagnose
    sharpes = [h.get("sharpe", None) for h in history if "sharpe" in h]
    if len(sharpes) < 2:
        return ""

    trap_info = detect_stagnation_trap(history, best_sharpe, sharpe_target)

    # Only inject for medium+ severity (not "low" or "no_trap")
    if trap_info.get("severity") in ("low",) or trap_info.get("trap") == "no_trap":
        return ""

    recovery = build_stagnation_recovery(trap_info)
    if not recovery:
        return ""

    # Add metadata header
    header = (
        f"<!-- Stagnation detected at turn {current_turn}: "
        f"{trap_info['trap']} (severity: {trap_info['severity']}, "
        f"{trap_info['turns_in_trap']} turns in trap) -->\n\n"
    )

    return header + recovery


def _build_crash_error_feedback(state) -> str:
    """Build crash error feedback section for LLM context injection.

    Analyzes turn history to find recent backtest_crash and module_load_failed
    entries with error details. Formats the last 2-3 crash errors as actionable
    feedback so the LLM can learn from its mistakes.

    Args:
        state: RunState dataclass instance.

    Returns:
        Formatted feedback string for prompt injection, or empty string if
        no crash errors found.
    """
    history = getattr(state, "history", [])

    # Find recent crash errors (with error detail)
    crash_entries = [
        h for h in history
        if h.get("status") in ("backtest_crash", "module_load_failed")
        and h.get("error")
    ]

    if not crash_entries:
        return ""

    # Take the last 3 most recent crashes
    recent_crashes = crash_entries[-3:]

    from crabquant.refinement.prompts import get_crash_recovery_hints

    lines = [
        "## Recent Code Crashes — FIX THESE ERRORS",
        "",
        "Your previous strategy code crashed during backtesting. Below are the ",
        "exact error messages. Read them carefully and fix the root cause.",
        "",
    ]

    for entry in recent_crashes:
        turn = entry.get("turn", "?")
        status = entry.get("status", "unknown")
        error = entry.get("error", {})
        error_type = error.get("error_type", "UnknownError")
        error_msg = error.get("error_message", "No details available")
        tb = error.get("error_traceback", "")

        lines.append(f"### Turn {turn} — {status}")
        lines.append(f"**Error:** `{error_type}: {error_msg}`")

        # Show traceback (truncated to last 5 lines for readability)
        if tb:
            tb_lines = tb.strip().split("\n")
            relevant = tb_lines[-5:] if len(tb_lines) > 5 else tb_lines
            lines.append("**Traceback (last frames):**")
            for tb_line in relevant:
                lines.append(f"```\n{tb_line}\n```")

        # Add recovery hints based on error type
        hints = get_crash_recovery_hints(error_type, error_msg)
        if hints:
            lines.append(f"**How to fix:** {hints}")

        lines.append("")

    # Add common patterns summary
    lines.append("---")
    lines.append("**Common crash patterns and fixes:**")
    lines.append("- `KeyError` on column name → Use lowercase: 'open', 'high', 'low', 'close', 'volume'")
    lines.append("- `NameError` for indicator → Import from `crabquant.indicators` or use `cached_indicator()`")
    lines.append("- `TypeError` in `generate_signals` → Must accept exactly `(df, params)` and return `(entries, exits)`")
    lines.append("- `AttributeError` on DataFrame → Check column names, use `df['close']` not `df.Close`")
    lines.append("- `ImportError` → Only use standard lib + pandas + pandas_ta + crabquant modules")

    return "\n".join(lines)
