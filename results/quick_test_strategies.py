"""
Quick sweep: test each strategy module on SPY to see which ones run without errors.
"""
import sys
sys.path.insert(0, ".")

import os
import importlib
import time

from crabquant.data import load_data
from crabquant.engine import BacktestEngine

strategies_dir = "crabquant/strategies"
strategy_modules = sorted([
    f.replace('.py', '') for f in os.listdir(strategies_dir)
    if f.endswith('.py') and f not in ('__init__.py', '_registry_compat.py')
])

print(f"Testing {len(strategy_modules)} strategies on SPY...")

working = []
failing = []

for name in strategy_modules:
    try:
        mod = importlib.import_module(f"crabquant.strategies.{name}")
        fn = mod.generate_signals
        params = getattr(mod, "DEFAULT_PARAMS", {})
        
        df = load_data("SPY", "2y")
        if df is None or df.empty:
            failing.append((name, "no_data"))
            continue
        
        entries, exits = fn(df, params)
        
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, strategy_name=name, ticker="SPY")
        
        if result and result.sharpe is not None:
            print(f"  ✅ {name:45s} | Sharpe: {result.sharpe:.2f} | Trades: {result.num_trades}")
            working.append((name, result.sharpe, result.num_trades))
        else:
            print(f"  ⚠️ {name:45s} | No result")
            failing.append((name, "no_result"))
    except Exception as e:
        err = str(e)[:60]
        print(f"  ❌ {name:45s} | {err}")
        failing.append((name, err))

print(f"\n{'='*60}")
print(f"Working: {len(working)}")
print(f"Failing: {len(failing)}")
print(f"\nTop by Sharpe:")
for name, sharpe, trades in sorted(working, key=lambda x: -x[1])[:10]:
    print(f"  {name:45s} | Sharpe: {sharpe:.2f} | Trades: {trades}")
