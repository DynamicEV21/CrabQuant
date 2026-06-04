"""
Comprehensive strategy-ticker sweep — find all (strategy, ticker) pairs
that pass rolling walk-forward validation.

Tests all 25+ registry strategies against 10+ tickers.
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

# All tickers to test
TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JNJ", "JPM",
           "GLD", "QQQ", "CAT", "DE", "UNH"]

# All available strategy modules (exclude __init__ and _registry_compat)
strategies_dir = "crabquant/strategies"
strategy_modules = sorted([
    f.replace('.py', '') for f in os.listdir(strategies_dir)
    if f.endswith('.py') and f not in ('__init__.py', '_registry_compat.py')
    and not f.startswith('refined_')  # Skip refined_ variants (duplicates)
])

print(f"Sweeping {len(strategy_modules)} strategies × {len(TICKERS)} tickers = {len(strategy_modules)*len(TICKERS)} combinations")
print(f"Current registry size: {len(STRATEGY_REGISTRY)}")

passed_pairs = []
failed_count = 0
error_count = 0
tested = 0

for strategy_name in strategy_modules:
    # Load module once
    try:
        mod = importlib.import_module(f"crabquant.strategies.{strategy_name}")
        strategy_fn = mod.generate_signals
        params = getattr(mod, "DEFAULT_PARAMS", {})
        strategy_code = open(f"crabquant/strategies/{strategy_name}.py").read()
    except Exception as e:
        print(f"  ERROR loading {strategy_name}: {e}")
        error_count += 1
        continue

    for ticker in TICKERS:
        tested += 1
        registry_name = f"{strategy_name}_{ticker.lower()}"

        # Skip if already registered
        if registry_name in STRATEGY_REGISTRY:
            continue

        # Quick test: just rolling walk-forward on discovery ticker
        try:
            result = run_full_validation_check(
                strategy_fn=strategy_fn,
                params=params,
                discovery_ticker=ticker,
                validation_tickers=[ticker],  # Discovery-only for speed
                min_walk_forward_sharpe=0.0,  # Very relaxed for sweep
                min_cross_ticker_sharpe=0.0,
                use_rolling=True,
            )
        except Exception as e:
            error_count += 1
            continue

        wf = result.get("walk_forward", {})
        avg_test_sharpe = wf.get("avg_test_sharpe", -999) if wf else -999
        windows_passed = wf.get("windows_passed", 0) if wf else 0
        num_windows = wf.get("num_windows", 0) if wf else 0

        # Keep pairs with positive avg test Sharpe and at least half windows passing
        if avg_test_sharpe > 0.2 and windows_passed >= num_windows // 2:
            # Now run full validation with cross-ticker
            cross_tickers = [ticker] + [t for t in ["SPY", "AAPL", "MSFT", "GOOGL"] if t != ticker]
            try:
                full_result = run_full_validation_check(
                    strategy_fn=strategy_fn,
                    params=params,
                    discovery_ticker=ticker,
                    validation_tickers=cross_tickers,
                    min_walk_forward_sharpe=0.3,
                    min_cross_ticker_sharpe=0.3,
                    is_regime_specific=True,
                )
            except Exception:
                continue

            if full_result.get("passed", False):
                ct = full_result.get("cross_ticker", {})
                print(f"  ✅ {strategy_name:40s} | {ticker:6s} | "
                      f"WF Sharpe: {avg_test_sharpe:.3f} ({windows_passed}/{num_windows}) | "
                      f"CT avg: {ct.get('avg_sharpe', 0):.3f}")
                passed_pairs.append({
                    "strategy": strategy_name,
                    "ticker": ticker,
                    "registry_name": registry_name,
                    "wf_avg_test_sharpe": avg_test_sharpe,
                    "wf_windows": f"{windows_passed}/{num_windows}",
                    "ct_avg_sharpe": ct.get("avg_sharpe", 0),
                    "ct_profitable": f"{ct.get('tickers_profitable', 0)}/{ct.get('tickers_tested', 0)}",
                    "params": params,
                })
            else:
                # Near-miss: good WF but failed CT
                ct = full_result.get("cross_ticker", {})
                ct_avg = ct.get("avg_sharpe", 0) if ct else 0
                if ct_avg > 0.1:
                    print(f"  🟡 {strategy_name:40s} | {ticker:6s} | "
                          f"WF Sharpe: {avg_test_sharpe:.3f} ({windows_passed}/{num_windows}) | "
                          f"CT avg: {ct_avg:.3f} (near-miss)")

    # Progress
    if tested % 20 == 0:
        print(f"  ... tested {tested} combinations, {len(passed_pairs)} passed so far ({time.time()-start_total:.0f}s)")

# ── Summary ──
print(f"\n{'='*70}")
print(f"SWEEP RESULTS")
print(f"{'='*70}")
print(f"  Tested:   {tested}")
print(f"  Passed:   {len(passed_pairs)}")
print(f"  Errors:   {error_count}")
print(f"  Elapsed:  {time.time() - start_total:.1f}s")

if passed_pairs:
    print(f"\n  Promotable pairs:")
    for p in passed_pairs:
        print(f"    {p['registry_name']:45s} | WF: {p['wf_avg_test_sharpe']:.3f} | CT: {p['ct_avg_sharpe']:.3f}")

    # Save results for promotion
    with open("results/sweep_results_cycle13.json", "w") as f:
        json.dump(passed_pairs, f, indent=2)
    print(f"\n  Results saved to results/sweep_results_cycle13.json")
