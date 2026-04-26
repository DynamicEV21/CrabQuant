"""
Production Strategy Promoter

Handles promoting confirmed ROBUST strategies to the production registry.
Creates markdown reports and maintains registry.json.
"""

import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from crabquant.production.report import (
    StrategyReport,
    SlippageResult,
    PeriodResult,
    RegimeInfo,
)

logger = logging.getLogger(__name__)

PRODUCTION_DIR = Path(__file__).resolve().parent.parent.parent / "strategies" / "production"
REGISTRY_FILE = PRODUCTION_DIR / "registry.json"


def _params_hash(params: dict) -> str:
    """Deterministic hash of params dict for dedup."""
    serialized = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def _make_key(strategy_name: str, ticker: str, params: dict) -> str:
    """Create the unique key for a strategy+ticker+params combo."""
    return f"{strategy_name}|{ticker}|{_params_hash(params)}"


def _load_registry() -> list[dict]:
    """Load the production registry."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return []


def _save_registry(registry: list[dict]):
    """Save the production registry."""
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def _infer_regime(strategy_name: str) -> RegimeInfo:
    """
    Infer regime affinity from the strategy registry's regime affinity scores.
    
    Looks up the strategy across all regimes and returns the top regime as best,
    regimes with score >= 0.6 as works_in, and regimes with score < 0.5 as avoid_in.
    """
    try:
        from crabquant.regime import REGIME_STRATEGY_AFFINITY, MarketRegime

        # REGIME_STRATEGY_AFFINITY is keyed by MarketRegime, each value is a
        # dict of {strategy_name: score}. Build reverse lookup: strategy -> [(regime, score)]
        strategy_regimes = []
        for regime, strategies in REGIME_STRATEGY_AFFINITY.items():
            if strategy_name in strategies:
                strategy_regimes.append((regime, strategies[strategy_name]))

        if not strategy_regimes:
            return RegimeInfo()

        ranked = sorted(strategy_regimes, key=lambda x: x[1], reverse=True)

        best_regime = ranked[0][0].value.upper() if ranked else ""
        works_in = [r[0].value.upper() for r in ranked if r[1] >= 0.6]
        avoid_in = [r[0].value.upper() for r in ranked if r[1] < 0.5]

        return RegimeInfo(
            best_regime=best_regime,
            works_in=works_in,
            avoid_in=avoid_in,
        )
    except ImportError:
        logger.warning("Could not import regime module for affinity lookup")
        return RegimeInfo()


def promote_strategy(
    strategy_name: str,
    ticker: str,
    params: dict,
    vbt_result: dict,
    confirm_result: object,
) -> StrategyReport:
    """
    Promote a strategy to the production registry.

    Args:
        strategy_name: Strategy name
        ticker: Stock ticker
        params: Strategy parameters
        vbt_result: Dict from winners.json (sharpe, return, max_dd, trades, score, win_rate, etc.)
        confirm_result: ConfirmationResult from batch_confirm, or a dict with confirm_* keys
                        from confirmed.json.

    Returns:
        StrategyReport for the promoted strategy.

    Raises:
        ValueError: If strategy is not ROBUST or already promoted with same params.
    """
    # Normalize confirm_result
    if hasattr(confirm_result, "verdict"):
        verdict = confirm_result.verdict
        c_sharpe = confirm_result.sharpe
        c_return = confirm_result.total_return
        c_max_dd = confirm_result.max_dd
        c_trades = confirm_result.trades
        c_win_rate = confirm_result.win_rate
        c_pf = confirm_result.profit_factor
        c_exp = confirm_result.expectancy
    else:
        verdict = confirm_result.get("verdict", "FAILED")
        c_sharpe = confirm_result.get("confirm_sharpe", 0)
        c_return = confirm_result.get("confirm_return", 0)
        c_max_dd = confirm_result.get("confirm_max_dd", 0)
        c_trades = confirm_result.get("confirm_trades", 0)
        c_win_rate = confirm_result.get("confirm_win_rate", 0)
        c_pf = confirm_result.get("confirm_profit_factor", 0)
        c_exp = confirm_result.get("confirm_expectancy", 0)

    if verdict != "ROBUST":
        raise ValueError(
            f"Cannot promote {strategy_name}/{ticker}: verdict is {verdict}, not ROBUST"
        )

    key = _make_key(strategy_name, ticker, params)

    # Check for duplicates
    registry = _load_registry()
    for entry in registry:
        if entry["key"] == key:
            raise ValueError(
                f"{strategy_name}/{ticker} already promoted with these params "
                f"(on {entry.get('promoted_at', 'unknown date')})"
            )

    # Build slippage results from confirm_result notes if available
    slippage_results = _extract_slippage_results(confirm_result)

    # Build period results from confirm_result notes if available
    period_results = _extract_period_results(confirm_result)

    # Build regime info
    regime_info = _infer_regime(strategy_name)

    # Extract regime tags from winner and confirm data
    discovery_regime = vbt_result.get("regime", "")
    validation_regime = ""
    if hasattr(confirm_result, "verdict"):
        # Object-style: won't have regime tags directly
        validation_regime = ""
    else:
        validation_regime = confirm_result.get("validation_regime", "")

    # Build report
    report = StrategyReport(
        strategy_name=strategy_name,
        ticker=ticker,
        params=params,
        date_promoted=date.today().isoformat(),
        verdict=verdict,
        vbt_sharpe=vbt_result.get("sharpe", 0),
        vbt_total_return=vbt_result.get("return", 0),
        vbt_max_drawdown=vbt_result.get("max_dd", 0),
        vbt_num_trades=vbt_result.get("trades", 0),
        vbt_win_rate=vbt_result.get("win_rate", 0),
        vbt_score=vbt_result.get("score", 0),
        confirm_sharpe=c_sharpe,
        confirm_total_return=c_return,
        confirm_max_drawdown=c_max_dd,
        confirm_num_trades=c_trades,
        confirm_win_rate=c_win_rate,
        confirm_profit_factor=c_pf,
        confirm_expectancy=c_exp,
        slippage_results=slippage_results,
        period_results=period_results,
        regime_info=regime_info,
        discovery_regime=discovery_regime,
        validation_regime=validation_regime,
        key=key,
    )

    # Write markdown report
    report_filename = f"{strategy_name}_{ticker}.md"
    report_path = PRODUCTION_DIR / report_filename
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.to_markdown())

    # Update registry
    registry.append({
        "key": key,
        "strategy_name": strategy_name,
        "ticker": ticker,
        "params_hash": _params_hash(params),
        "params": params,
        "promoted_at": date.today().isoformat(),
        "verdict": verdict,
        "report_file": report_filename,
    })
    _save_registry(registry)

    logger.info(
        f"Promoted {strategy_name}/{ticker} to production "
        f"(confirm sharpe={c_sharpe:.2f}, return={c_return:.2%})"
    )

    return report


def _extract_slippage_results(confirm_result) -> list[SlippageResult]:
    """
    Extract slippage test results from batch_confirm notes.

    Batch confirm runs 3 slippage levels (0%, 0.1%, 0.2%) per period.
    We extract the primary period (2y) slippage results.
    """
    results = []
    notes = getattr(confirm_result, "notes", []) or confirm_result.get("notes", [])

    # Pattern: "2y @ 0.0% slip: ..." followed by check results
    slippage_map = {0.0: None, 0.001: None, 0.002: None}

    for note in notes:
        # Look for patterns like "2y @ 0.0% slip"
        for slip in slippage_map:
            marker = f"2y @ {slip*100:.1f}% slip"
            if marker in note and ":" in note:
                # Extract the check info
                # Notes format from batch_confirm: "2y @ X% slip | ...: PASS/FAIL"
                parts = note.split(":")
                if len(parts) >= 2:
                    status = "PASS" in parts[-1].upper()
                    slippage_map[slip] = SlippageResult(
                        slippage_pct=slip,
                        sharpe=0.0,  # Not directly available from notes
                        total_return=0.0,
                        max_drawdown=0.0,
                        num_trades=0,
                        win_rate=0.0,
                        passed=status,
                    )

    # If we got slippage results from notes, return them
    found = [v for v in slippage_map.values() if v is not None]
    if found:
        return found

    # Fallback: if we have a ROBUST verdict, assume all passed
    verdict = getattr(confirm_result, "verdict", confirm_result.get("verdict", ""))
    if verdict == "ROBUST":
        return [
            SlippageResult(slippage_pct=0.0, passed=True, sharpe=0, total_return=0, max_drawdown=0, num_trades=0, win_rate=0),
            SlippageResult(slippage_pct=0.001, passed=True, sharpe=0, total_return=0, max_drawdown=0, num_trades=0, win_rate=0),
            SlippageResult(slippage_pct=0.002, passed=True, sharpe=0, total_return=0, max_drawdown=0, num_trades=0, win_rate=0),
        ]

    return []


def _extract_period_results(confirm_result) -> list[PeriodResult]:
    """
    Extract period test results from batch_confirm notes.

    Batch confirm runs 3 periods (2y, 1y, 6mo). We extract 0% slippage results per period.
    """
    notes = getattr(confirm_result, "notes", []) or confirm_result.get("notes", [])

    # For now, the primary result is the confirm_result itself (2y period)
    c_sharpe = getattr(confirm_result, "sharpe", 0)
    c_return = getattr(confirm_result, "total_return", 0)
    c_max_dd = getattr(confirm_result, "max_dd", 0)
    c_trades = getattr(confirm_result, "trades", 0)
    c_win_rate = getattr(confirm_result, "win_rate", 0)
    # Handle dict-style access
    if hasattr(confirm_result, "get"):
        c_sharpe = confirm_result.get("confirm_sharpe", c_sharpe)
        c_return = confirm_result.get("confirm_return", c_return)
        c_max_dd = confirm_result.get("confirm_max_dd", c_max_dd)
        c_trades = confirm_result.get("confirm_trades", c_trades)
        c_win_rate = confirm_result.get("confirm_win_rate", c_win_rate)

    # Check if we have multi-period data in notes
    periods = ["2y", "1y", "6mo"]
    period_map = {}

    for note in notes:
        for period in periods:
            marker = f"{period} @ 0.0% slip"
            if marker in note and ":" in note:
                parts = note.split(":")
                if len(parts) >= 2:
                    status = "PASS" in parts[-1].upper()
                    period_map[period] = PeriodResult(
                        period=period,
                        sharpe=0.0,
                        total_return=0.0,
                        max_drawdown=0.0,
                        num_trades=0,
                        win_rate=0.0,
                        passed=status,
                    )

    # Always include the primary (2y) period with actual values
    if "2y" not in period_map:
        period_map["2y"] = PeriodResult(
            period="2y",
            sharpe=c_sharpe,
            total_return=c_return,
            max_drawdown=c_max_dd,
            num_trades=c_trades,
            win_rate=c_win_rate,
            passed=True,
        )

    # Return in order
    return [period_map.get(p) for p in periods if p in period_map]
