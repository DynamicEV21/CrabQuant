#!/usr/bin/env python3
"""
Test script for invented_volume_adx_ema strategy
"""
import pandas as pd
import yfinance as yf
from crabquant.strategies.invented_volume_adx_ema import generate_signals, DEFAULT_PARAMS

def test_strategy(tickers, start_date='2023-01-01', end_date='2024-12-31'):
    """Test strategy on multiple tickers"""
    results = {}
    
    for ticker in tickers:
        print(f"\nTesting {ticker}...")
        
        try:
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date)
            if len(data) < 100:  # Insufficient data
                print(f"  Insufficient data for {ticker}")
                results[ticker] = {'trades': 0, 'message': 'Insufficient data'}
                continue
                
            # Generate signals
            entries, exits = generate_signals(data, DEFAULT_PARAMS)
            
            # Count trades
            trade_signals = entries.sum()
            exit_signals = exits.sum()
            
            print(f"  Entries: {trade_signals}")
            print(f"  Exits: {exit_signals}")
            
            if trade_signals > 0:
                results[ticker] = {'trades': trade_signals, 'message': 'Success'}
            else:
                results[ticker] = {'trades': 0, 'message': 'No trades generated'}
                
        except Exception as e:
            print(f"  Error testing {ticker}: {e}")
            results[ticker] = {'trades': 0, 'message': f'Error: {str(e)}'}
    
    return results

if __name__ == "__main__":
    tickers = ['AAPL', 'NVDA', 'CAT', 'SPY']
    results = test_strategy(tickers)
    
    print("\n" + "="*50)
    print("TEST RESULTS:")
    print("="*50)
    
    for ticker, result in results.items():
        status = "✓" if result['trades'] > 0 else "✗"
        print(f"{status} {ticker}: {result['trades']} trades - {result['message']}")