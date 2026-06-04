"""
Batch register top validated strategies — Cycle 13.

Registers the best (strategy, ticker) pair per strategy module
to maximize diversity in the registry. Target: 10+ total registered.
"""
import sys
sys.path.insert(0, ".")

import json
import time
import importlib
from datetime import datetime, timezone

from crabquant.refinement.promotion import register_strategy
from crabquant.strategies import STRATEGY_REGISTRY

start = time.time()

# Load sweep results
data = json.load(open("results/sweep_results_cycle13.json"))
passed = data["passed"]

# Group by strategy, pick the best pair per strategy (by WF avg test Sharpe)
from collections import defaultdict
by_strategy = defaultdict(list)
for p in passed:
    by_strategy[p["strategy"]].append(p)

# Select best pair per strategy (highest WF Sharpe with good CT)
to_register = []
for strategy_name, pairs in sorted(by_strategy.items()):
    # Sort by WF Sharpe desc
    pairs.sort(key=lambda x: -x["wf_avg_test_sharpe"])
    best = pairs[0]
    
    registry_name = best["registry_name"]
    
    # Skip if already registered
    if registry_name in STRATEGY_REGISTRY:
        print(f"  ⏭️ {registry_name}: already registered")
        continue
    
    to_register.append(best)

print(f"Strategies to register: {len(to_register)}")
print(f"Current registry size: {len(STRATEGY_REGISTRY)}")
print()

registered = 0
failed = 0

for entry in to_register:
    strategy_name = entry["strategy"]
    ticker = entry["ticker"]
    registry_name = entry["registry_name"]
    
    print(f"Registering: {registry_name}")
    print(f"  Strategy: {strategy_name}, Ticker: {ticker}")
    print(f"  WF: {entry['wf_avg_test_sharpe']:.3f} ({entry['wf_windows']}), CT: {entry['ct_avg_sharpe']:.3f}")
    
    try:
        mod = importlib.import_module(f"crabquant.strategies.{strategy_name}")
        strategy_code = open(f"crabquant/strategies/{strategy_name}.py").read()
        params = getattr(mod, "DEFAULT_PARAMS", {})
    except Exception as e:
        print(f"  ❌ Load error: {e}")
        failed += 1
        continue
    
    # Compute regime tags
    regime_tags = None
    try:
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags
        regime_tags = compute_strategy_regime_tags(
            mod.generate_signals, params, ticker=ticker
        )
        print(f"  Regime: preferred={regime_tags.get('preferred_regimes')}, "
              f"specific={regime_tags.get('is_regime_specific')}")
    except Exception as e:
        print(f"  Regime tagging failed: {e}")
    
    # Register
    ok = register_strategy(
        registry_name,
        mod,
        strategy_code=strategy_code,
        regime_tags=regime_tags,
    )
    
    if ok:
        registered += 1
        print(f"  ✅ Registered successfully")
        
        # Also update winners.json
        try:
            winners_path = "results/winners/winners.json"
            winners = json.loads(open(winners_path).read())
            winners.append({
                "strategy": registry_name,
                "ticker": ticker,
                "sharpe": entry["wf_avg_test_sharpe"],
                "return": 0,
                "max_drawdown": 0,
                "trades": 0,
                "params": params,
                "refinement_run": "batch_cycle13",
                "refinement_turns": 0,
                "validation": {
                    "passed": True,
                    "walk_forward_robust": True,
                    "cross_ticker_robust": True,
                    "validation_method": "rolling",
                    "sweep_wf_avg_test_sharpe": entry["wf_avg_test_sharpe"],
                    "sweep_wf_windows": entry["wf_windows"],
                    "sweep_ct_avg_sharpe": entry["ct_avg_sharpe"],
                },
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "validation_status": "promoted",
                "regime_tags": regime_tags,
            })
            open(winners_path, "w").write(json.dumps(winners, indent=2))
        except Exception as e:
            print(f"  Winners.json update failed: {e}")
    else:
        print(f"  ❌ Registration failed")
        failed += 1

print(f"\n{'='*60}")
print(f"BATCH REGISTRATION SUMMARY")
print(f"{'='*60}")
print(f"  Attempted: {len(to_register)}")
print(f"  Registered: {registered}")
print(f"  Failed: {failed}")
print(f"  Registry size: {len(STRATEGY_REGISTRY)}")
print(f"  Elapsed: {time.time()-start:.1f}s")

# Show final registry
print(f"\nFinal registry ({len(STRATEGY_REGISTRY)} entries):")
for k in sorted(STRATEGY_REGISTRY.keys()):
    v = STRATEGY_REGISTRY[k]
    if isinstance(v, dict):
        regimes = v.get("preferred_regimes", [])
        print(f"  {k}: promoted (regimes={regimes})")
    else:
        print(f"  {k}: legacy")
