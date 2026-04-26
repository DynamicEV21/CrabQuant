#!/usr/bin/env python3
"""
Fix column names for yfinance data
"""
import pandas as pd
import yfinance as yf

def fix_yfinance_columns(df):
    """Fix yfinance multi-level column names"""
    if isinstance(df.columns, pd.MultiIndex):
        # Extract just the price level (first level)
        df.columns = df.columns.get_level_values(0)
    
    # Ensure we have lowercase columns
    df.columns = [col.lower() for col in df.columns]
    
    # Make sure we have the right columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df

def test_column_fix():
    ticker = 'AAPL'
    data = yf.download(ticker, start='2023-01-01', end='2024-12-31')
    print(f"Original columns: {data.columns}")
    print(f"Original columns type: {type(data.columns)}")
    
    fixed_data = fix_yfinance_columns(data.copy())
    print(f"Fixed columns: {fixed_data.columns}")
    print(f"Fixed data head:\n{fixed_data.head()}")

if __name__ == "__main__":
    test_column_fix()