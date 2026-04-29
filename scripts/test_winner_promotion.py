#!/usr/bin/env python3
"""
Diagnostic script: Test whether existing winner strategies can pass the
relaxed rolling walk-forward validation and get promoted.

Purely diagnostic — does NOT modify any source code or winners.json.
"""

import importlib
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("winner_promotion_test")

# ── Config ────────────────────────────────────────────────────────────────
WINNERS_PATH = PROJECT_ROOT / "results" / "winners" / "winners.json"
TOP_N = 10

# Relaxed thresholds (matching what was described)
RELAXED_THRESHOLDS = {
    "min_avg_test_sharpe": 0.3,
    "min_windows_passed": 1,
    "min_window_test_sharpe": 0.0,
    "max_window_degradation": 1.0,
    "min_cross_ticker_sharpe": 0.3,
}

# Strategy name → module path mapping
STRATEGY_MODULES = {
    "rsi_crossover": "crabquant.strategies.rsi_crossover",
    "roc_ema_volume": "crabquant.strategies.roc_ema_volume",
    "informed_simple_adaptive": "crabquant.strategies.informed_simple_adaptive",
    "informed_adaptive_trend_reversion": "crabquant.strategies.informed_adaptive_trend_reversion",
    "ema_crossover": "crabquant.strategies.ema_crossover",
    "ema_ribbon_reversal": "crabquant.strategies.ema_ribbon_reversal",
    "macd_momentum": "crabquant.strategies.macd_momentum",
    "adx_pullback": "crabquant.strategies.adx_pullback",
    "atr_channel_breakout": "crabquant.strategies.atr_channel_breakout",
    "volume_breakout": "crabquant.strategies.volume_breakout",
    "multi_rsi_confluence": "crabquant.strategies.multi_rsi_confluence",
    "bollinger_squeeze": "crabquant.strategies.bollinger_squeeze",
    "ichimoku_trend": "crabquant.strategies.ichimoku_trend",
    "bb_stoch_macd": "crabquant.strategies.bb_stoch_macd",
    "rsi_regime_dip": "crabquant.strategies.rsi_regime_dip",
    "vpt_crossover": "crabquant.strategies.vpt_crossover",
    "injected_momentum_atr_volume": "crabquant.strategies.injected_momentum_atr_volume",
    "invented_momentum_rsi_atr": "crabquant.strategies.invented_momentum_rsi_atr",
    "invented_momentum_rsi_stoch": "crabquant.strategies.invented_momentum_rsi_stoch",
    "invented_vpt_roc_ema": "crabquant.strategies.invented_vpt_roc_ema",
    "invented_volume_momentum_trend": "crabquant.strategies.invented_volume_momentum_trend",
    "invented_volume_adx_ema": "crabquant.strategies.invented_volume_adx_ema",
    "invented_momentum_confluence": "crabquant.strategies.invented_momentum_confluence",
    "invented_volume_breakout_adx": "crabquant.strategies.invented_volume_breakout_adx",
    "invented_volume_roc_atr_trend": "crabquant.strategies.invented_volume_roc_atr_trend",
    "invented_rsi_volume_atr": "crabquant.strategies.invented_rsi_volume_atr",
    "invented_volume_roc_rsi_ema": "crabquant.strategies.invented_volume_roc_rsi_ema",
    "invented_volatility_rsi_breakout": "crabquant.strategies.invented_volatility_rsi_breakout",
    "invented_volume_roc_atr_trend": "crabquant.strategies.invented_volume_roc_atr_trend",
    "invented_momentum_rsi_volume": "crabquant.strategies.invented_momentum_rsi_volume",
    "roc_ema_volume_googl": "crabquant.strategies.roc_ema_volume_googl",
    "refined_e2e_test_momentum": "crabquant.strategies.refined_e2e_test_momentum",
}


def load_strategy_module(strategy_name: str):
    """Dynamically load a strategy module by name."""
    module_path = STRATEGY_MODULES.get(strategy_name)
    if not module_path:
        # Try direct import as a fallback
        module_path = f"crabquant.strategies.{strategy_name}"
    try:
        return importlib.import_module(module_path)
    except ImportError:
        # Try with the strategy name as-is (for special names)
        for candidate in STRATEGY_MODULES:
            if candidate in strategy_name or strategy_name in candidate:
                return importlib.import_module(STRATEGY_MODULES[candidate])
        return None


def main():
    # Load winners
    if not WINNERS_PATH.exists():
        logger.error("Winners file not found: %s", WINNERS_PATH)
        return

    winners = json.loads(WINNERS_PATH.read_text())
    logger.info("Loaded %d winners from winners.json", len(winners))

    # Sort by sharpe and take top N
    winners.sort(key=lambda w: w.get("sharpe", 0), reverse=True)
    top_winners = winners[:TOP_N]
    logger.info("Testing top %d winners by Sharpe ratio", TOP_N)
    print("\n" + "=" * 90)
    print(f"{'#':>2} | {'Strategy':<35} | {'Ticker':<6} | {'Sharpe':>6} | {'Status':<12}")
    print("-" * 90)

    for i, w in enumerate(top_winners, 1):
        print(f"{i:>2} | {w['strategy']:<35} | {w['ticker']:<6} | {w['sharpe']:>6.2f} | backtest_only")

    # Import validation infrastructure
    from crabquant.refinement.promotion import run_full_validation_check
    from crabquant.strategies import DEFAULT_TICKERS, STRATEGY_REGISTRY

    logger.info("Validation tickers (%d): %s", len(DEFAULT_TICKERS), DEFAULT_TICKERS)
    logger.info("STRATEGY_REGISTRY has %d entries", len(STRATEGY_REGISTRY))
    print(f"\nSTRATEGY_REGISTRY currently has {len(STRATEGY_REGISTRY)} entries:")
    for name in sorted(STRATEGY_REGISTRY.keys()):
        print(f"  - {name}")

    # Test each top winner
    results = []
    for i, winner in enumerate(top_winners, 1):
        strategy_name = winner["strategy"]
        params = winner.get("params", {})
        ticker = winner["ticker"]
        bt_sharpe = winner.get("sharpe", 0)
        is_regime_specific = bool(winner.get("regime"))

        print(f"\n{'=' * 90}")
        print(f"[{i}/{TOP_N}] Testing: {strategy_name} on {ticker} (backtest Sharpe={bt_sharpe:.2f})")
        print(f"    Params: {params}")
        if is_regime_specific:
            print(f"    Regime-specific: {winner.get('regime')}")

        # Load strategy module
        mod = load_strategy_module(strategy_name)
        if mod is None:
            print(f"    ❌ Could not load strategy module for '{strategy_name}'")
            results.append({
                "strategy": strategy_name, "ticker": ticker, "bt_sharpe": bt_sharpe,
                "status": "MODULE_NOT_FOUND", "error": f"No module for {strategy_name}",
            })
            continue

        strategy_fn = getattr(mod, "generate_signals", None)
        if strategy_fn is None:
            print(f"    ❌ Module has no generate_signals function")
            results.append({
                "strategy": strategy_name, "ticker": ticker, "bt_sharpe": bt_sharpe,
                "status": "NO_GENERATE_SIGNALS", "error": "No generate_signals",
            })
            continue

        # Run validation
        t0 = time.time()
        try:
            validation = run_full_validation_check(
                strategy_fn=strategy_fn,
                params=params,
                discovery_ticker=ticker,
                validation_tickers=DEFAULT_TICKERS,
                min_walk_forward_sharpe=RELAXED_THRESHOLDS["min_avg_test_sharpe"],
                min_cross_ticker_sharpe=RELAXED_THRESHOLDS["min_cross_ticker_sharpe"],
                use_rolling=True,
                rolling_config={
                    "min_avg_test_sharpe": RELAXED_THRESHOLDS["min_avg_test_sharpe"],
                    "min_windows_passed": RELAXED_THRESHOLDS["min_windows_passed"],
                    "min_window_test_sharpe": RELAXED_THRESHOLDS["min_window_test_sharpe"],
                    "max_window_degradation": RELAXED_THRESHOLDS["max_window_degradation"],
                },
                is_regime_specific=is_regime_specific,
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    ❌ Validation error ({elapsed:.1f}s): {e}")
            results.append({
                "strategy": strategy_name, "ticker": ticker, "bt_sharpe": bt_sharpe,
                "status": "VALIDATION_ERROR", "error": str(e), "elapsed": elapsed,
            })
            continue

        elapsed = time.time() - t0
        passed = validation.get("passed", False)
        wf_robust = validation.get("walk_forward_robust", False)
        ct_robust = validation.get("cross_ticker_robust", False)
        error = validation.get("error")

        # Extract detailed metrics
        wf = validation.get("walk_forward") or {}
        ct = validation.get("cross_ticker") or {}

        avg_test_sharpe = wf.get("avg_test_sharpe", "N/A")
        min_test_sharpe = wf.get("min_test_sharpe", "N/A")
        avg_degradation = wf.get("avg_degradation", "N/A")
        windows_passed = wf.get("windows_passed", "N/A")
        num_windows = wf.get("num_windows", "N/A")
        wf_robust_flag = wf.get("robust", False)

        ct_avg_sharpe = ct.get("avg_sharpe", "N/A")
        ct_median_sharpe = ct.get("median_sharpe", "N/A")
        ct_robust_flag = ct.get("robust", False)
        ct_tickers_profitable = ct.get("tickers_profitable", "N/A")
        ct_tickers_tested = ct.get("tickers_tested", "N/A")

        # Print detailed results
        status_icon = "✅" if passed else "❌"
        print(f"    {status_icon} PASSED={passed} | WF_robust={wf_robust} | CT_robust={ct_robust} | ({elapsed:.1f}s)")

        if error:
            print(f"    ⚠️  Error: {error}")

        if isinstance(avg_test_sharpe, float):
            print(f"    Walk-Forward:")
            print(f"      avg_test_sharpe={avg_test_sharpe:.3f} (threshold: {RELAXED_THRESHOLDS['min_avg_test_sharpe']})")
            print(f"      min_test_sharpe={min_test_sharpe:.3f}")
            print(f"      avg_degradation={avg_degradation:.1%}")
            print(f"      windows_passed={windows_passed}/{num_windows}")
            print(f"      robust={wf_robust_flag}")

            # Per-window breakdown
            window_results = wf.get("window_results", [])
            if window_results:
                print(f"      Per-window:")
                for wr in window_results:
                    wstatus = "✓" if wr.get("passed") else "✗"
                    print(f"        W{wr.get('window', '?')}: train={wr.get('train_sharpe', 0):.2f} "
                          f"test={wr.get('test_sharpe', 0):.2f} "
                          f"degrad={wr.get('degradation', 0):.1%} {wstatus}")

        if isinstance(ct_avg_sharpe, float):
            print(f"    Cross-Ticker:")
            print(f"      avg_sharpe={ct_avg_sharpe:.3f} (threshold: {RELAXED_THRESHOLDS['min_cross_ticker_sharpe']})")
            print(f"      median_sharpe={ct_median_sharpe:.3f}")
            print(f"      tickers_profitable={ct_tickers_profitable}/{ct_tickers_tested}")
            print(f"      robust={ct_robust_flag}")

        results.append({
            "strategy": strategy_name,
            "ticker": ticker,
            "bt_sharpe": bt_sharpe,
            "passed": passed,
            "wf_robust": wf_robust,
            "ct_robust": ct_robust,
            "avg_test_sharpe": avg_test_sharpe if isinstance(avg_test_sharpe, float) else None,
            "avg_degradation": avg_degradation if isinstance(avg_degradation, float) else None,
            "windows_passed": windows_passed if isinstance(windows_passed, int) else None,
            "num_windows": num_windows if isinstance(num_windows, int) else None,
            "ct_avg_sharpe": ct_avg_sharpe if isinstance(ct_avg_sharpe, float) else None,
            "ct_robust_flag": ct_robust_flag,
            "is_regime_specific": is_regime_specific,
            "error": error,
            "elapsed": elapsed,
        })

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("SUMMARY")
    print("=" * 90)

    passed_strategies = [r for r in results if r.get("passed")]
    failed_strategies = [r for r in results if not r.get("passed") and r.get("status") is None]
    error_strategies = [r for r in results if r.get("status") is not None]

    print(f"\nTested: {len(results)} strategies")
    print(f"  ✅ PASSED validation: {len(passed_strategies)}")
    print(f"  ❌ FAILED validation: {len(failed_strategies)}")
    print(f"  ⚠️  ERRORS (module/data): {len(error_strategies)}")

    if passed_strategies:
        print(f"\n🎉 PROMOTABLE STRATEGIES:")
        for r in passed_strategies:
            print(f"  ✅ {r['strategy']} ({r['ticker']}) - "
                  f"BT Sharpe={r['bt_sharpe']:.2f}, "
                  f"WF avg={r['avg_test_sharpe']:.3f}, "
                  f"CT avg={r['ct_avg_sharpe']:.3f}")
    else:
        print(f"\n❌ NO strategies passed validation.")
        print(f"\n🔍 BOTTLENECK ANALYSIS:")
        
        wf_bottleneck = 0
        ct_bottleneck = 0
        both_bottleneck = 0
        low_wf_sharpes = []
        high_degradations = []
        low_ct_sharpes = []

        for r in failed_strategies:
            wf_ok = r.get("wf_robust", False)
            ct_ok = r.get("ct_robust", False)
            if not wf_ok and not ct_ok:
                both_bottleneck += 1
            elif not wf_ok:
                wf_bottleneck += 1
            elif not ct_ok:
                ct_bottleneck += 1
            
            if r.get("avg_test_sharpe") is not None:
                low_wf_sharpes.append((r["strategy"], r["avg_test_sharpe"], r.get("avg_degradation")))
            if r.get("ct_avg_sharpe") is not None:
                low_ct_sharpes.append((r["strategy"], r["ct_avg_sharpe"]))

        print(f"  Walk-forward bottleneck only: {wf_bottleneck}")
        print(f"  Cross-ticker bottleneck only: {ct_bottleneck}")
        print(f"  Both failed: {both_bottleneck}")

        if low_wf_sharpes:
            print(f"\n  Walk-Forward Avg Test Sharpe (threshold: {RELAXED_THRESHOLDS['min_avg_test_sharpe']}):")
            for name, sharpe, degrad in sorted(low_wf_sharpes, key=lambda x: x[1], reverse=True):
                marker = "✓" if sharpe >= RELAXED_THRESHOLDS["min_avg_test_sharpe"] else "✗"
                degrad_str = f"{degrad:.1%}" if degrad is not None else "N/A"
                print(f"    {marker} {name:<35} sharpe={sharpe:>7.3f}  degrad={degrad_str}")

        if low_ct_sharpes:
            print(f"\n  Cross-Ticker Avg Sharpe (threshold: {RELAXED_THRESHOLDS['min_cross_ticker_sharpe']}):")
            for name, sharpe in sorted(low_ct_sharpes, key=lambda x: x[1], reverse=True):
                marker = "✓" if sharpe >= RELAXED_THRESHOLDS["min_cross_ticker_sharpe"] else "✗"
                print(f"    {marker} {name:<35} sharpe={sharpe:>7.3f}")

        if error_strategies:
            print(f"\n  Module/Data Errors:")
            for r in error_strategies:
                print(f"    ⚠️  {r['strategy']}: {r.get('error', 'unknown')[:80]}")

    print(f"\n{'=' * 90}")
    print(f"STRATEGY_REGISTRY final count: {len(STRATEGY_REGISTRY)}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
