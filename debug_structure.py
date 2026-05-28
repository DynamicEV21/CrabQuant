#!/usr/bin/env python3
"""
Debug script to understand DataFrame structure better
"""

import yfinance as yf
import pandas as pd

def debug_structure(ticker):
    """Check DataFrame structure in detail"""
    data = yf.download(ticker, period="2y", interval="1d", progress=False)
    
    print(f"\n=== {ticker} DataFrame Info ===")
    print(f"Columns: {data.columns}")
    print(f"Column types:\n{data.dtypes}")
    print(f"Index: {data.index}")
    print(f"Shape: {data.shape}")
    
    # Check what happens when we try to access different columns
    print(f"\nTrying to access by position:")
    print(f"First column: {data.columns[0]}")
    print(f"First column type: {type(data.columns[0])}")
    
    # Try to rename columns
    print(f"\nTrying to rename columns:")
    try:
        renamed = data.rename(columns={data.columns[0]: 'close'})
        print(f"Renamed successfully. New columns: {renamed.columns}")
    except Exception as e:
        print(f"Rename failed: {e}")
    
    # Try dropping the multi-level index
    print(f"\nTrying to reset index:")
    try:
        reset = data.reset_index()
        print(f"Reset successful. New shape: {reset.shape}")
        print(f"New columns: {reset.columns}")
    except Exception as e:
        print(f"Reset failed: {e}")

if __name__ == "__main__":
    debug_structure("AAPL")