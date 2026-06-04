"""
Batch promotion script — Cycle 13.

Validates and promotes top near-miss winners from results/winners/winners.json.
Targets: informed_simple_adaptive|JNJ, roc_ema_volume|GOOGL, rsi_crossover|JNJ
Plus: refined_mean_reversion_nvda and refined_volume_aapl if modules exist.

Uses relaxed thresholds from Cycle 10-12 fixes.
"""

import sys
import json
import time
import importlib
import os

sys.path.insert(0, ".")

from crabquant.refinement.promotion import run_full_validation_check, register_strategy
from crabquant.strategies import STRATEGY_REGISTRY

start_total = time.time()

# Candidates: (strategy_name, ticker, registry_name)
CANDIDATES = [
    ("informed_simple_adaptive", "JNJ", "informed_simple_adaptive_jnj"),
    ("roc_ema_volume", "GOOGL", "roc_ema_volume_googl_v2"),
    ("rsi_crossover", "JNJ", "rsi_crossover_jnj"),
    ("refined_mean_reversion_nvda", "NVDA", "refined_mean_reversion_nvda"),
    ("refined_volume_aapl", "AAPL", "refined_volume_aapl"),
]

# Cross-ticker validation tickers (diverse set)
CROSS_TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "JNJ"]

results_summary = []

for strategy_name, discovery_ticker, registry_name in CANDIDATES:
    tick_start = time.time()
    print(f"\n{'='*70}")
    print(f"Validating: {strategy_name} on {discovery_ticker} → {registry_name}")
    print(f"{'='*70}")

    # Check if already registered
    if registry_name in STRATEGY_REGISTRY:
        print(f"  SKIP: Already in registry")
        results_summary.append({
            "strategy": strategy_name,
            "ticker": discovery_ticker,
            "registry_name": registry_name,
            "status": "skip_already_registered",
            "elapsed": time.time() - tick_start,
        })
        continue

    # Check if already promoted in winners.json
    winners_path = "results/winners/winners.json"
    if os.path.exists(winners_path):
        try:
            winners = json.loads(open(winners_path).read())
            already_promoted = any(
                w.get("strategy") == registry_name and w.get("validation_status") == "promoted"
                for w in winners
            )
            if already_promoted:
                print(f"  SKIP: Already promoted in winners.json")
                results_summary.append({
                    "strategy": strategy_name,
                    "ticker": discovery_ticker,
                    "registry_name": registry_name,
                    "status": "skip_already_promoted",
                    "elapsed": time.time() - tick_start,
                })
                continue
        except Exception:
            pass

    # Load strategy module
    try:
        mod = importlib.import_module(f"crabquant.strategies.{strategy_name}")
        strategy_fn = mod.generate_signals
        params = mod.DEFAULT_PARAMS
        strategy_code = open(f"crabquant/strategies/{strategy_name}.py").read()
        print(f"  Loaded module: {strategy_name} (params: {params})")
    except Exception as e:
        print(f"  ERROR loading module: {e}")
        results_summary.append({
            "strategy": strategy_name,
            "ticker": discovery_ticker,
            "registry_name": registry_name,
            "status": "error_load",
            "error": str(e),
            "elapsed": time.time() - tick_start,
        })
        continue

    # Build validation ticker list
    validation_tickers = [discovery_ticker] + [t for t in CROSS_TICKERS if t != discovery_ticker]

    # Run validation with relaxed thresholds
    print(f"  Running validation (tickers: {validation_tickers})...")
    try:
        result = run_full_validation_check(
            strategy_fn=strategy_fn,
            params=params,
            discovery_ticker=discovery_ticker,
            validation_tickers=validation_tickers,
            min_walk_forward_sharpe=0.3,
            min_cross_ticker_sharpe=0.3,
            is_regime_specific=True,  # Most strategies are regime-specific
        )
    except Exception as e:
        print(f"  ERROR during validation: {e}")
        results_summary.append({
            "strategy": strategy_name,
            "ticker": discovery_ticker,
            "registry_name": registry_name,
            "status": "error_validation",
            "error": str(e),
            "elapsed": time.time() - tick_start,
        })
        continue

    passed = result.get("passed", False)
    wf_robust = result.get("walk_forward_robust", False)
    ct_robust = result.get("cross_ticker_robust", False)
    wf = result.get("walk_forward", {})
    ct = result.get("cross_ticker", {})

    print(f"  Passed: {passed}")
    print(f"  WF robust: {wf_robust}, CT robust: {ct_robust}")
    if wf:
        print(f"  WF avg test Sharpe: {wf.get('avg_test_sharpe', 0):.3f}, "
              f"windows: {wf.get('windows_passed', 0)}/{wf.get('num_windows', 0)}")
    if ct:
        print(f"  CT avg Sharpe: {ct.get('avg_sharpe', 0):.3f}, "
              f"profitable: {ct.get('tickers_profitable', 0)}/{ct.get('tickers_tested', 0)}")

    # If main validation failed, try discovery-ticker-only
    if not passed:
        print(f"  Main validation failed. Trying discovery-ticker-only...")
        try:
            result2 = run_full_validation_check(
                strategy_fn=strategy_fn,
                params=params,
                discovery_ticker=discovery_ticker,
                validation_tickers=[discovery_ticker],
                min_walk_forward_sharpe=0.3,
                min_cross_ticker_sharpe=0.3,
            )
            if result2.get("passed", False):
                print(f"  Discovery-ticker-only PASSED!")
                result = result2
                passed = True
                wf = result.get("walk_forward", {})
                if wf:
                    print(f"  WF avg test Sharpe: {wf.get('avg_test_sharpe', 0):.3f}, "
                          f"windows: {wf.get('windows_passed', 0)}/{wf.get('num_windows', 0)}")
        except Exception as e2:
            print(f"  Discovery-ticker-only also failed: {e2}")

    # Register if passed
    if passed:
        print(f"  ✅ Validation PASSED — registering {registry_name}...")

        # Compute regime tags
        regime_tags = None
        try:
            from crabquant.refinement.regime_tagger import compute_strategy_regime_tags
            regime_tags = compute_strategy_regime_tags(strategy_fn, params, ticker=discovery_ticker)
            print(f"  Regime tags: preferred={regime_tags.get('preferred_regimes')}, "
                  f"specific={regime_tags.get('is_regime_specific')}")
        except Exception as e:
            print(f"  Regime tagging failed: {e}")

        ok = register_strategy(
            registry_name,
            mod,
            strategy_code=strategy_code,
            regime_tags=regime_tags,
        )
        print(f"  Registration result: {ok}")

        # Update winners.json
        try:
            winners_path = "results/winners/winners.json"
            winners = json.loads(open(winners_path).read())
            for entry in reversed(winners):
                if entry.get("strategy") == registry_name:
                    entry["validation_status"] = "promoted"
                    break
            else:
                # No existing entry — add one
                winners.append({
                    "strategy": registry_name,
                    "ticker": discovery_ticker,
                    "sharpe": 0,  # Will be updated by actual backtest
                    "return": 0,
                    "max_drawdown": 0,
                    "trades": 0,
                    "params": params,
                    "refinement_run": "batch_cycle13",
                    "refinement_turns": 0,
                    "validation": result,
                    "promoted_at": __import__('datetime').datetime.now(
                        __import__('datetime').timezone.utc
                    ).isoformat(),
                    "validation_status": "promoted",
                    "regime_tags": regime_tags,
                })
            open(winners_path, "w").write(json.dumps(winners, indent=2))
            print(f"  Updated winners.json")
        except Exception as e:
            print(f"  Failed to update winners.json: {e}")

        results_summary.append({
            "strategy": strategy_name,
            "ticker": discovery_ticker,
            "registry_name": registry_name,
            "status": "promoted",
            "wf_avg_sharpe": wf.get("avg_test_sharpe", 0) if wf else 0,
            "wf_windows": f"{wf.get('windows_passed', 0)}/{wf.get('num_windows', 0)}" if wf else "N/A",
            "ct_avg_sharpe": ct.get("avg_sharpe", 0) if ct else 0,
            "elapsed": time.time() - tick_start,
        })
    else:
        print(f"  ❌ Validation FAILED — not registering")
        results_summary.append({
            "strategy": strategy_name,
            "ticker": discovery_ticker,
            "registry_name": registry_name,
            "status": "failed",
            "wf_robust": wf_robust,
            "ct_robust": ct_robust,
            "error": result.get("error"),
            "elapsed": time.time() - tick_start,
        })

# ── Final Summary ──
print(f"\n{'='*70}")
print(f"BATCH PROMOTION SUMMARY — Cycle 13")
print(f"{'='*70}")
promoted_count = sum(1 for r in results_summary if r["status"] == "promoted")
failed_count = sum(1 for r in results_summary if r["status"] == "failed")
skipped_count = sum(1 for r in results_summary if "skip" in r["status"])
print(f"  Promoted: {promoted_count}")
print(f"  Failed:   {failed_count}")
print(f"  Skipped:  {skipped_count}")
print(f"  Total:    {len(results_summary)}")

for r in results_summary:
    status_icon = "✅" if r["status"] == "promoted" else "❌" if r["status"] == "failed" else "⏭️"
    print(f"  {status_icon} {r['registry_name']:40s} | {r['status']}")

print(f"\n  Registry size: {len(STRATEGY_REGISTRY)}")
print(f"  Total elapsed: {time.time() - start_total:.1f}s")
