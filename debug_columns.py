#!/usr/bin/env python3
"""
Debug script to check column names in downloaded data
"""

import yfinance as yf

def check_columns(ticker):
    """Check what columns are in the downloaded data"""
    data = yf.download(ticker, period="2y", interval="1d", progress=False)
    print(f"\n=== {ticker} columns ===")
    print(data.columns.tolist())
    print(f"Shape: {data.shape}")
    print("\nFirst few rows:")
    print(data.head())

if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "CAT", "SPY"]
    for ticker in tickers:
        check_columns(ticker)