#!/usr/bin/env python3
"""
Test script for invented_volume_roc_rsi_ema strategy
"""

import pandas as pd
import yfinance as yf
from crabquant.strategies.invented_volume_roc_rsi_ema import generate_signals, DEFAULT_PARAMS

def test_strategy(ticker, params=None):
    """Test strategy on a single ticker"""
    if params is None:
        params = DEFAULT_PARAMS
    
    print(f"\n=== Testing {ticker} ===")
    
    # Download data
    data = yf.download(ticker, period="2y", interval="1d")
    if len(data) == 0:
        print(f"No data for {ticker}")
        return 0, 0
    
    # Generate signals
    entries, exits = generate_signals(data, params)
    
    # Count trades
    entries_count = entries.sum()
    exits_count = exits.sum()
    
    print(f"Entries: {entries_count}")
    print(f"Exits: {exits_count}")
    print(f"Net trades: {min(entries_count, exits_count)}")
    
    return min(entries_count, exits_count)

def main():
    """Test on all required tickers"""
    tickers = ["AAPL", "NVDA", "CAT", "SPY"]
    total_trades = 0
    
    for ticker in tickers:
        trades = test_strategy(ticker)
        total_trades += trades
        print(f"{ticker}: {trades} trades")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total trades across all tickers: {total_trades}")
    print(f"Average trades per ticker: {total_trades / len(tickers):.1f}")
    
    if total_trades == 0:
        print("❌ No trades generated - strategy needs improvement")
    else:
        print("✅ Strategy generates trades")

if __name__ == "__main__":
    main()