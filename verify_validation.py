"""Quick verification that relaxed thresholds allow strategies to pass validation."""
import sys
sys.path.insert(0, '/home/Zev/development/CrabQuant-agents/worker-1')

from crabquant.validation import walk_forward_test, rolling_walk_forward
from crabquant.engine.backtest import BacktestEngine
import pandas as pd
import numpy as np

def simple_ema_crossover(df, params):
    fast = params.get('fast', 12)
    slow = params.get('slow', 26)
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    entries = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    exits = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    return entries, exits

ticker = 'AAPL'
params = {'fast': 12, 'slow': 26}
engine = BacktestEngine()

print(f"Testing {ticker} with EMA crossover...")
print()

# Test 1: walk_forward_test with DEFAULT (relaxed) thresholds
print("=== walk_forward_test (default thresholds) ===")
wf_result = walk_forward_test(simple_ema_crossover, ticker, params, engine=engine)
print(f"  Robust: {wf_result.robust}")
print(f"  Train Sharpe: {wf_result.train_sharpe:.3f}")
print(f"  Test Sharpe: {wf_result.test_sharpe:.3f}")
print(f"  Degradation: {wf_result.degradation:.3f}")
print(f"  Notes: {wf_result.notes}")
print()

# Test 2: rolling_walk_forward with DEFAULT (relaxed) thresholds
print("=== rolling_walk_forward (default thresholds) ===")
rwf_result = rolling_walk_forward(simple_ema_crossover, ticker, params, engine=engine)
print(f"  Robust: {rwf_result.robust}")
print(f"  Avg test Sharpe: {rwf_result.avg_test_sharpe:.3f}")
print(f"  Min test Sharpe: {rwf_result.min_test_sharpe:.3f}")
print(f"  Avg degradation: {rwf_result.avg_degradation:.3f}")
print(f"  Windows passed: {rwf_result.windows_passed}/{rwf_result.num_windows}")
print(f"  Notes: {rwf_result.notes}")
print()

# Test 3: Try a few more tickers
for t in ['MSFT', 'GOOGL', 'SPY']:
    try:
        rwf = rolling_walk_forward(simple_ema_crossover, t, params, engine=engine)
        wf = walk_forward_test(simple_ema_crossover, t, params, engine=engine)
        print(f"{t}: WF robust={wf.robust}, RWF robust={rwf.robust} (avg_test_sharpe={rwf.avg_test_sharpe:.3f}, windows={rwf.windows_passed}/{rwf.num_windows})")
    except Exception as e:
        print(f"{t}: ERROR - {e}")

print()
print("=== SUMMARY ===")
print("If any strategies show robust=True, the threshold relaxation worked!")
