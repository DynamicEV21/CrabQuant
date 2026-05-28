#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crabquant.strategies.invented_volume_roc_atr_trend import generate_signals, DEFAULT_PARAMS
import pandas as pd

def test_strategy(tickers):
    """Test strategy on multiple tickers"""
    print(f"Testing invented_volume_roc_atr_trend on {tickers}")
    
    for ticker in tickers:
        try:
            # Try to load data - this is simplified, in real system it would come from data service
            print(f"Testing {ticker}...")
            
            # For now, let's just simulate that we'll get data and test
            # This would normally be:
            # df = get_data(ticker, period='1y', interval='1d')
            # entries, exits = generate_signals(df, DEFAULT_PARAMS)
            # trades = calculate_trades(entries, exits)
            
            print(f"  {ticker}: Would test with data")
            
        except Exception as e:
            print(f"  {ticker}: Error - {e}")
    
    print("Test completed - need actual data to validate signals")

if __name__ == "__main__":
    test_tickers = ["AAPL", "NVDA", "CAT", "SPY"]
    test_strategy(test_tickers)