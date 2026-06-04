"""
Focused sweep: test top strategies across multiple tickers with rolling WF validation.
Only tests strategies with Sharpe > 1.0 on SPY + those already known winners.
"""
import sys
sys.path.insert(0, ".")

import json
import time
import importlib
import os

from crabquant.refinement.promotion import run_full_validation_check, register_strategy
from crabquant.strategies import STRATEGY_REGISTRY

start = time.time()

# Top strategies from quick test + known winners
TOP_STRATEGIES = [
    "invented_volume_roc_atr_trend",  # Sharpe 2.73 on SPY
    "invented_vpt_roc_ema",           # Sharpe 1.64 on SPY
    "invented_volume_breakout_adx",   # Sharpe 1.38 on SPY
    "invented_momentum_rsi_stoch",    # Sharpe 1.38 on SPY
    "invented_volume_roc_rsi_ema",    # Sharpe 1.28 on SPY
    "invented_momentum_confluence",   # Sharpe 1.25 on SPY
    "injected_momentum_atr_volume",   # Sharpe 1.19 on SPY
    "invented_volume_adx_ema",        # Sharpe 1.10 on SPY
    "adx_pullback",                   # Sharpe 1.09 on SPY
    "ema_crossover",                  # Sharpe 1.05 on SPY, 10 trades
    "informed_simple_adaptive",       # Sharpe 1.01 on SPY
    "rsi_crossover",                  # Known winner on JNJ
    "roc_ema_volume",                 # Known winner on GOOGL
    "macd_momentum",                  # Most common winner
]

TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JNJ", "JPM",
           "GLD", "QQQ", "CAT", "DE", "UNH"]

# Cross-ticker set
CT_TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]

passed = []
near_miss = []
tested = 0

for strategy_name in TOP_STRATEGIES:
    try:
        mod = importlib.import_module(f"crabquant.strategies.{strategy_name}")
        strategy_fn = mod.generate_signals
        params = getattr(mod, "DEFAULT_PARAMS", {})
        strategy_code = open(f"crabquant/strategies/{strategy_name}.py").read()
    except Exception as e:
        print(f"  ERROR loading {strategy_name}: {e}")
        continue

    for ticker in TICKERS:
        tested += 1
        registry_name = f"{strategy_name}_{ticker.lower()}"

        # Skip already registered
        if registry_name in STRATEGY_REGISTRY:
            continue

        # Quick WF check on discovery ticker only
        try:
            result = run_full_validation_check(
                strategy_fn=strategy_fn,
                params=params,
                discovery_ticker=ticker,
                validation_tickers=[ticker],
                min_walk_forward_sharpe=0.0,
                min_cross_ticker_sharpe=0.0,
            )
        except Exception:
            continue

        wf = result.get("walk_forward", {})
        avg_sharpe = wf.get("avg_test_sharpe", -999) if wf else -999
        win_pass = wf.get("windows_passed", 0) if wf else 0
        win_total = wf.get("num_windows", 0) if wf else 0

        # Filter: need positive avg test Sharpe and at least half windows passing
        if avg_sharpe < 0.15 or win_pass < max(1, win_total // 2):
            continue

        # Run full validation with cross-ticker
        ct_list = [ticker] + [t for t in CT_TICKERS if t != ticker]
        try:
            full = run_full_validation_check(
                strategy_fn=strategy_fn,
                params=params,
                discovery_ticker=ticker,
                validation_tickers=ct_list,
                min_walk_forward_sharpe=0.3,
                min_cross_ticker_sharpe=0.3,
                is_regime_specific=True,
            )
        except Exception:
            continue

        ct = full.get("cross_ticker", {})
        ct_avg = ct.get("avg_sharpe", 0) if ct else 0
        ct_profit = ct.get("tickers_profitable", 0) if ct else 0
        ct_total = ct.get("tickers_tested", 0) if ct else 0

        if full.get("passed", False):
            print(f"  ✅ {registry_name:50s} | WF: {avg_sharpe:.3f} ({win_pass}/{win_total}) | CT: {ct_avg:.3f} ({ct_profit}/{ct_total})")
            passed.append({
                "strategy": strategy_name,
                "ticker": ticker,
                "registry_name": registry_name,
                "wf_avg_test_sharpe": avg_sharpe,
                "wf_windows": f"{win_pass}/{win_total}",
                "ct_avg_sharpe": ct_avg,
                "ct_profitable": f"{ct_profit}/{ct_total}",
                "params": params,
                "strategy_code_path": f"crabquant/strategies/{strategy_name}.py",
            })
        elif ct_avg > 0.1:
            print(f"  🟡 {registry_name:50s} | WF: {avg_sharpe:.3f} ({win_pass}/{win_total}) | CT: {ct_avg:.3f} (near-miss)")
            near_miss.append({
                "strategy": strategy_name,
                "ticker": ticker,
                "registry_name": registry_name,
                "wf_avg_test_sharpe": avg_sharpe,
                "wf_windows": f"{win_pass}/{win_total}",
                "ct_avg_sharpe": ct_avg,
            })

    if tested % 25 == 0:
        print(f"  ... {tested} tested, {len(passed)} passed ({time.time()-start:.0f}s)")

print(f"\n{'='*70}")
print(f"SWEEP RESULTS")
print(f"{'='*70}")
print(f"  Tested: {tested}")
print(f"  Passed: {len(passed)}")
print(f"  Near-misses: {len(near_miss)}")
print(f"  Elapsed: {time.time()-start:.1f}s")

if passed:
    with open("results/sweep_results_cycle13.json", "w") as f:
        json.dump({"passed": passed, "near_miss": near_miss}, f, indent=2)
    print(f"  Results saved to results/sweep_results_cycle13.json")
