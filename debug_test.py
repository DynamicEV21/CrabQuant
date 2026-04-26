#!/usr/bin/env python3
"""
Debug test script for invented_volume_adx_ema strategy
"""
import pandas as pd
import yfinance as yf
from crabquant.strategies.invented_volume_adx_ema import generate_signals, DEFAULT_PARAMS

def test_strategy_debug(ticker, start_date='2023-01-01', end_date='2024-12-31'):
    """Test strategy with debug output"""
    print(f"\nTesting {ticker}...")
    
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date)
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {list(data.columns)}")
        print(f"Data head:\n{data.head()}")
        
        if len(data) < 100:  # Insufficient data
            print(f"  Insufficient data for {ticker}")
            return {'trades': 0, 'message': 'Insufficient data'}
            
        # Reset index to make sure we have Date as index
        data = data.reset_index()
        print(f"After reset - Data columns: {list(data.columns)}")
        print(f"Data head:\n{data.head()}")
        
        # Generate signals
        entries, exits = generate_signals(data, DEFAULT_PARAMS)
        
        print(f"Entries shape: {entries.shape}")
        print(f"Exits shape: {exits.shape}")
        print(f"Entries sum: {entries.sum()}")
        print(f"Exits sum: {exits.sum()}")
        
        # Count trades
        trade_signals = entries.sum()
        exit_signals = exits.sum()
        
        if trade_signals > 0:
            return {'trades': trade_signals, 'message': 'Success'}
        else:
            return {'trades': 0, 'message': 'No trades generated'}
                
    except Exception as e:
        print(f"  Error testing {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return {'trades': 0, 'message': f'Error: {str(e)}'}

if __name__ == "__main__":
    ticker = 'AAPL'
    result = test_strategy_debug(ticker)
    
    print("\n" + "="*50)
    print("TEST RESULTS:")
    print("="*50)
    
    status = "✓" if result['trades'] > 0 else "✗"
    print(f"{status} {ticker}: {result['trades']} trades - {result['message']}")