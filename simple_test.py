#!/usr/bin/env python3
"""
Simple test to debug OBV calculation
"""
import pandas as pd
import yfinance as yf
import pandas_ta as ta

def test_obv():
    ticker = 'AAPL'
    data = yf.download(ticker, start='2023-01-01', end='2024-12-31')
    
    # Fix column names
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [col.lower() for col in data.columns]
    
    print("Data shape:", data.shape)
    print("Data columns:", data.columns)
    print("Data types:")
    print(data.dtypes)
    
    # Test OBV calculation step by step
    print("\nTesting OBV calculation...")
    try:
        obv = ta.obv(data['close'], data['volume'])
        print("OBV calculation successful")
        print("OBV head:")
        print(obv.head(10))
        
        # Test rolling mean
        obv_fast = obv.rolling(10).mean()
        print("\nOBV fast head:")
        print(obv_fast.head(15))
        
    except Exception as e:
        print(f"OBV error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_obv()