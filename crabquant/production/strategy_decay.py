"""
Strategy Decay Detection

Monitors promoted strategies for performance degradation by periodically
backtesting on recent data and comparing current Sharpe ratios to the
promotion-time Sharpe.  Strategies that consistently underperform are
flagged for retirement.

Phase 6 — autonomous strategy lifecycle management.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DecayCheckResult:
    """Result of a single strategy decay check."""

    strategy_name: str
    promotion_sharpe: float
    current_sharpe: float
    sharpe_decline_pct: float
    current_regime: str = ""
    is_decayed: bool = False
    consecutive_decayed_checks: int = 0
    should_retire: bool = False


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_decay_state(decay_state_file: str = "results/decay_state.json") -> dict:
    """Load decay state from a JSON file.

    The state file maps strategy keys to their consecutive decay count
    and last-check metadata.  Returns an empty dict if the file does not
    exist or is malformed.

    Args:
        decay_state_file: Path to the JSON state file.

    Returns:
        Dict mapping strategy name -> ``{consecutive_decayed: int, ...}``
    """
    path = Path(decay_state_file)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load decay state from %s: %s", decay_state_file, exc)
        return {}


def save_decay_state(decay_state_file: str, state: dict) -> None:
    """Persist decay state to a JSON file.

    Creates parent directories if they do not exist.

    Args:
        decay_state_file: Path to the JSON state file.
        state: Dict mapping strategy name -> state info.
    """
    path = Path(decay_state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str))


# ---------------------------------------------------------------------------
# Regime detection helper
# ---------------------------------------------------------------------------

def _detect_regime(ticker: str = "SPY", period: str = "6mo") -> str:
    """Detect the current market regime.

    Returns the regime name as a string, or ``"unknown"`` on failure.
    """
    try:
        from crabquant.regime import detect_regime
        from crabquant.data import load_data
        df = load_data(ticker, period=period)
        regime, _ = detect_regime(df)
        return regime.name if hasattr(regime, "name") else str(regime)
    except Exception as exc:
        logger.debug("Regime detection failed: %s", exc)
        return "unknown"


# ---------------------------------------------------------------------------
# Single-strategy decay check
# ---------------------------------------------------------------------------

def check_strategy_decay(
    strategy_code: str,
    promotion_sharpe: float,
    strategy_params: dict | None = None,
    ticker: str = "SPY",
    period: str = "6mo",
    decay_threshold: float = 0.30,
    consecutive_required: int = 3,
) -> DecayCheckResult:
    """Run a backtest on recent data and compare to promotion-time Sharpe.

    Loads recent OHLCV data, generates signals using the strategy function
    from ``STRATEGY_REGISTRY``, runs the backtest engine, and computes the
    Sharpe decline percentage.

    If the backtest fails for any reason (data fetch error, signal
    generation error, etc.) the result is returned with ``is_decayed=False``
    so that transient failures do not trigger false retirement.

    Args:
        strategy_code: Strategy name as it appears in ``STRATEGY_REGISTRY``.
        promotion_sharpe: Sharpe ratio at time of promotion.
        strategy_params: Override parameters.  If *None*, the strategy's
            ``DEFAULT_PARAMS`` from the registry are used.
        ticker: Ticker to backtest on.
        period: Data period for the recent backtest window.
        decay_threshold: Fractional decline (e.g. 0.30 = 30 %) that
            qualifies as *decayed* for a single check.
        consecutive_required: Number of consecutive decayed checks before
            ``should_retire`` flips to ``True``.  This value is informational
            here; the caller (``check_all_strategies_decay``) manages the
            counter.

    Returns:
        ``DecayCheckResult`` with all fields populated.
    """
    strategy_params = strategy_params or {}
    current_regime = "unknown"

    try:
        # --- Load recent data ---
        from crabquant.data import load_data
        df = load_data(ticker, period=period)

        # --- Resolve strategy function ---
        from crabquant.strategies import STRATEGY_REGISTRY
        if strategy_code not in STRATEGY_REGISTRY:
            logger.warning("Strategy %s not found in STRATEGY_REGISTRY", strategy_code)
            return DecayCheckResult(
                strategy_name=strategy_code,
                promotion_sharpe=promotion_sharpe,
                current_sharpe=0.0,
                sharpe_decline_pct=0.0,
                current_regime=current_regime,
                is_decayed=False,
                consecutive_decayed_checks=0,
                should_retire=False,
            )

        signal_fn, defaults, _grid, _desc, _matrix_fn = STRATEGY_REGISTRY[strategy_code]
        params = {**defaults, **strategy_params}

        entries, exits = signal_fn(df, params)

        # --- Run backtest ---
        from crabquant.engine.backtest import BacktestEngine
        engine = BacktestEngine()
        result = engine.run(
            df=df,
            entries=entries,
            exits=exits,
            strategy_name=strategy_code,
            ticker=ticker,
            params=params,
        )

        current_sharpe = result.sharpe

        # --- Detect regime ---
        current_regime = _detect_regime(ticker, period)

    except Exception as exc:
        logger.warning(
            "Decay check failed for %s/%s: %s — returning not-decayed",
            strategy_code, ticker, exc,
        )
        return DecayCheckResult(
            strategy_name=strategy_code,
            promotion_sharpe=promotion_sharpe,
            current_sharpe=0.0,
            sharpe_decline_pct=0.0,
            current_regime="unknown",
            is_decayed=False,
            consecutive_decayed_checks=0,
            should_retire=False,
        )

    # --- Compute decline ---
    if promotion_sharpe > 0:
        sharpe_decline_pct = (promotion_sharpe - current_sharpe) / promotion_sharpe
    elif current_sharpe < 0:
        # Promotion Sharpe was 0 or negative — any negative current is decayed
        sharpe_decline_pct = 1.0 if current_sharpe < 0 else 0.0
    else:
        sharpe_decline_pct = 0.0

    is_decayed = sharpe_decline_pct >= decay_threshold

    return DecayCheckResult(
        strategy_name=strategy_code,
        promotion_sharpe=promotion_sharpe,
        current_sharpe=current_sharpe,
        sharpe_decline_pct=sharpe_decline_pct,
        current_regime=current_regime,
        is_decayed=is_decayed,
        consecutive_decayed_checks=0,  # managed by caller
        should_retire=False,            # managed by caller
    )


# ---------------------------------------------------------------------------
# Batch check all promoted strategies
# ---------------------------------------------------------------------------

def _load_registry() -> list[dict]:
    """Load promoted strategies from available registry sources.

    Checks, in order:
    1. ``results/STRATEGY_REGISTRY.json`` — a flat list of promoted entries.
    2. ``strategies/production/registry.json`` — the promoter's registry.
    3. Falls back to all entries in ``crabquant.strategies.STRATEGY_REGISTRY``.

    Returns:
        List of dicts with at least ``strategy_name`` and ``promotion_sharpe``.
    """
    # Option 1: results/STRATEGY_REGISTRY.json
    results_reg = BASE_DIR / "results" / "STRATEGY_REGISTRY.json"
    if results_reg.exists():
        try:
            data = json.loads(results_reg.read_text())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass

    # Option 2: strategies/production/registry.json (promoter format)
    prod_reg = BASE_DIR / "strategies" / "production" / "registry.json"
    if prod_reg.exists():
        try:
            data = json.loads(prod_reg.read_text())
            if isinstance(data, list):
                return [
                    {
                        "strategy_name": entry.get("strategy_name", ""),
                        "promotion_sharpe": entry.get("promotion_sharpe", 1.5),
                        "ticker": entry.get("ticker", "SPY"),
                        "params": entry.get("params", {}),
                        "status": entry.get("status", "active"),
                    }
                    for entry in data
                ]
        except (json.JSONDecodeError, OSError):
            pass

    # Option 3: fallback to in-code STRATEGY_REGISTRY
    try:
        from crabquant.strategies import STRATEGY_REGISTRY
        return [
            {
                "strategy_name": name,
                "promotion_sharpe": 1.5,  # default assumption
                "ticker": "SPY",
                "params": {},
                "status": "active",
            }
            for name in STRATEGY_REGISTRY
        ]
    except ImportError:
        return []


def check_all_strategies_decay(
    strategies_dir: str | None = None,
    decay_state_file: str = "results/decay_state.json",
    decay_threshold: float = 0.30,
    consecutive_required: int = 3,
) -> list[DecayCheckResult]:
    """Check all promoted strategies for decay.

    Reads the strategy registry, loads persisted decay state, runs a decay
    check for each active strategy, updates consecutive decay counts, saves
    the updated state, and returns results sorted by severity (most decayed
    first).

    Args:
        strategies_dir: Ignored; kept for API compatibility.  Registry is
            loaded from the standard locations.
        decay_state_file: Path to the JSON file that persists consecutive
            decay counts across runs.
        decay_threshold: Fractional Sharpe decline that qualifies as decayed.
        consecutive_required: Consecutive decayed checks before a strategy
            is marked ``should_retire=True``.

    Returns:
        List of ``DecayCheckResult`` sorted by ``sharpe_decline_pct``
        descending (most decayed first).
    """
    strategies = _load_registry()
    state = load_decay_state(decay_state_file)
    results: list[DecayCheckResult] = []

    for entry in strategies:
        name = entry.get("strategy_name", "")
        status = entry.get("status", "active")

        # Skip already-retired strategies
        if status == "retired" or status == "inactive":
            # Preserve existing state but don't re-check
            continue

        promo_sharpe = float(entry.get("promotion_sharpe", 1.5))
        ticker = entry.get("ticker", "SPY")
        params = entry.get("params", {})

        # Run the single-strategy check
        check = check_strategy_decay(
            strategy_code=name,
            promotion_sharpe=promo_sharpe,
            strategy_params=params,
            ticker=ticker,
            decay_threshold=decay_threshold,
            consecutive_required=consecutive_required,
        )

        # --- Update consecutive counter ---
        prev = state.get(name, {})
        prev_count = int(prev.get("consecutive_decayed", 0))

        if check.is_decayed:
            new_count = prev_count + 1
        else:
            new_count = 0

        check.consecutive_decayed_checks = new_count
        check.should_retire = new_count >= consecutive_required

        # Persist
        state[name] = {
            "consecutive_decayed": new_count,
            "last_check": datetime.now().isoformat(),
            "current_sharpe": check.current_sharpe,
            "promotion_sharpe": check.promotion_sharpe,
            "is_decayed": check.is_decayed,
            "should_retire": check.should_retire,
        }

        results.append(check)

    # Mark retired strategies in state if needed
    for check in results:
        if check.should_retire:
            state[check.strategy_name]["status"] = "retired"

    # Save updated state
    save_decay_state(decay_state_file, state)

    # Sort by severity: most decayed first
    results.sort(key=lambda r: r.sharpe_decline_pct, reverse=True)

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_decay_report(results: list[DecayCheckResult]) -> str:
    """Format decay check results into a human-readable report.

    Args:
        results: List of ``DecayCheckResult`` instances.

    Returns:
        Multi-line string suitable for logging or messaging.
    """
    if not results:
        return "No strategies checked for decay."

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("STRATEGY DECAY REPORT")
    lines.append(f"Checked: {len(results)} strategies")
    lines.append("=" * 60)

    decayed = [r for r in results if r.is_decayed]
    retired = [r for r in results if r.should_retire]
    healthy = [r for r in results if not r.is_decayed]

    if decayed:
        lines.append("")
        lines.append(f"⚠️  DECAYED ({len(decayed)}):")
        for r in decayed:
            marker = "🔴 RETIRE" if r.should_retire else "🟡 WATCH"
            lines.append(
                f"  {marker} {r.strategy_name}: "
                f"Sharpe {r.promotion_sharpe:.2f} → {r.current_sharpe:.2f} "
                f"({r.sharpe_decline_pct:.1%} decline) "
                f"[{r.consecutive_decayed_checks}/{consecutive_required_from(r)} consecutive]"
            )

    if healthy:
        lines.append("")
        lines.append(f"✅ HEALTHY ({len(healthy)}):")
        for r in healthy:
            lines.append(
                f"  {r.strategy_name}: "
                f"Sharpe {r.promotion_sharpe:.2f} → {r.current_sharpe:.2f} "
                f"({r.sharpe_decline_pct:.1%} decline)"
            )

    lines.append("")
    lines.append("=" * 60)
    if retired:
        lines.append(f"ACTION REQUIRED: {len(retired)} strategy(ies) recommended for retirement.")
    elif decayed:
        lines.append(f"ATTENTION: {len(decayed)} strategy(ies) showing decay — monitoring.")
    else:
        lines.append("All strategies healthy. No action required.")
    lines.append("=" * 60)

    return "\n".join(lines)


def consecutive_required_from(result: DecayCheckResult) -> int:
    """Infer the consecutive_required value from a result's state.

    This is a convenience helper for formatting.  It returns the value
    such that ``should_retire`` would flip at the boundary.
    """
    if result.should_retire and result.consecutive_decayed_checks > 0:
        return result.consecutive_decayed_checks
    # Best guess: use the default
    return 3
