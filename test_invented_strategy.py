#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crabquant.strategies.invented_volume_momentum_trend import generate_signals, DEFAULT_PARAMS
from crabquant.data import load_data
import pandas as pd

def test_strategy(tickers):
    """Test the strategy on given tickers and return results"""
    results = {}
    
    for ticker in tickers:
        try:
            print(f"Testing {ticker}...")
            
            # Get historical data using CrabQuant's data loader
            data = load_data(ticker, period="2y")
            
            if len(data) < 50:  # Not enough data
                print(f"  - {ticker}: Not enough data")
                results[ticker] = {"trades": 0, "message": "Not enough data"}
                continue
                
            # Generate signals
            entries, exits = generate_signals(data, DEFAULT_PARAMS)
            
            # Count trades
            trade_count = 0
            in_position = False
            
            for i in range(1, len(entries)):
                if entries.iloc[i] and not in_position:
                    trade_count += 1
                    in_position = True
                elif exits.iloc[i] and in_position:
                    in_position = False
            
            results[ticker] = {
                "trades": trade_count,
                "message": "OK" if trade_count > 0 else "No trades generated"
            }
            
            print(f"  - {ticker}: {trade_count} trades")
            
        except Exception as e:
            print(f"  - {ticker}: Error - {str(e)}")
            results[ticker] = {"trades": 0, "message": f"Error: {str(e)}"}
    
    return results

if __name__ == "__main__":
    test_tickers = ['AAPL', 'NVDA', 'CAT', 'SPY']
    results = test_strategy(test_tickers)
    
    print("\n=== RESULTS ===")
    for ticker, result in results.items():
        print(f"{ticker}: {result['trades']} trades - {result['message']}")
    
    # Check if strategy produces >0 trades on at least 2 tickers
    successful_tickers = [t for t, r in results.items() if r['trades'] > 0]
    print(f"\nSuccessful tickers: {len(successful_tickers)}/4")
    
    if len(successful_tickers) < 2:
        print("Strategy fails - produces trades on <2 tickers. Need to fix.")
        sys.exit(1)
    else:
        print("Strategy passes - produces trades on >=2 tickers.")
        sys.exit(0)