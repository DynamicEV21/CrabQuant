#!/usr/bin/env python3
"""
Detailed debug script to find the exact error location
"""
import pandas as pd
import yfinance as yf
import pandas_ta as ta

def debug_crossover():
    ticker = 'AAPL'
    data = yf.download(ticker, start='2023-01-01', end='2024-12-31')
    
    # Fix column names
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [col.lower() for col in data.columns]
    
    print("Step 1: Calculate OBV")
    obv = ta.obv(data['close'], data['volume'])
    print("OBV dtype:", obv.dtype)
    print("OBV has NaN:", obv.isna().any())
    
    print("\nStep 2: Calculate rolling means")
    obv_fast = obv.rolling(10).mean()
    obv_slow = obv.rolling(20).mean()
    print("OBV fast dtype:", obv_fast.dtype)
    print("OBV slow dtype:", obv_slow.dtype)
    
    print("\nStep 3: Compare OBV fast and slow")
    obv_above = obv_fast > obv_slow
    print("OBV above dtype:", obv_above.dtype)
    print("OBV above has NaN:", obv_above.isna().any())
    print("OBV above head:")
    print(obv_above.head(15))
    
    print("\nStep 4: Test shift operation")
    shifted = obv_above.shift(1)
    print("Shifted dtype:", shifted.dtype)
    print("Shifted has NaN:", shifted.isna().any())
    print("Shifted head:")
    print(shifted.head(15))
    
    print("\nStep 5: Test ~ operation (this should fail)")
    try:
        test = ~obv_above.shift(1)
        print("~ operation succeeded!")
    except Exception as e:
        print(f"~ operation failed: {e}")
        print("Let's try alternatives...")
        
        # Alternative 1: Fill NaN first
        try:
            test1 = ~(obv_above.shift(1).fillna(False))
            print("Alternative 1 succeeded: ~(shifted.fillna(False))")
        except Exception as e2:
            print(f"Alternative 1 failed: {e2}")
            
        # Alternative 2: Use .notna() and boolean operations
        try:
            test2 = obv_above.shift(1).notna()
            print("Alternative 2 succeeded: .notna()")
        except Exception as e3:
            print(f"Alternative 2 failed: {e3}")

if __name__ == "__main__":
    debug_crossover()