#!/usr/bin/env python3
"""
Simple strategy test to check if it produces trades on target tickers.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from crabquant.strategies import STRATEGY_REGISTRY
from crabquant.data import load_data

def test_strategy_on_tickers(strategy_name: str, tickers: list, period: str = "2y"):
    """Test a strategy on multiple tickers and return trade counts."""
    
    # Import the strategy
    strategy_fn, _, _, _, _ = STRATEGY_REGISTRY[strategy_name]
    
    results = {}
    
    for ticker in tickers:
        print(f"Testing {ticker}...")
        
        try:
            # Load data
            df = load_data(ticker, period=period)
            if df.empty:
                results[ticker] = 0
                print(f"  No data for {ticker}")
                continue
            
            # Generate signals with default params
            entries, exits = strategy_fn(df, STRATEGY_REGISTRY[strategy_name][1])
            
            # Count trades
            num_entries = entries.sum()
            num_exits = exits.sum()
            
            results[ticker] = {
                'entries': int(num_entries),
                'exits': int(num_exits),
                'total_trades': int(num_entries // 2),  # Rough estimate
                'data_points': len(df)
            }
            
            print(f"  {ticker}: {num_entries} entries, {num_exits} exits")
            
        except Exception as e:
            print(f"  Error testing {ticker}: {e}")
            results[ticker] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    strategy_name = "invented_rsi_volume_atr"
    target_tickers = ["AAPL", "NVDA", "CAT", "SPY"]
    
    print(f"Testing strategy: {strategy_name}")
    print("=" * 50)
    
    results = test_strategy_on_tickers(strategy_name, target_tickers, period="2y")
    
    print("\nResults Summary:")
    print("=" * 50)
    total_trades = 0
    
    for ticker, result in results.items():
        if 'error' in result:
            print(f"{ticker}: ERROR - {result['error']}")
        else:
            trades = result['total_trades']
            total_trades += trades
            print(f"{ticker}: {trades} trades")
    
    print(f"\nTotal trades across all tickers: {total_trades}")
    
    if total_trades == 0:
        print("⚠️  No trades generated! Strategy needs fixing.")
        sys.exit(1)
    else:
        print("✅ Strategy produces trades on target tickers!")
        sys.exit(0)