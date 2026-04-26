#!/usr/bin/env python3

import pandas as pd
import numpy as np
from crabquant.strategies.invented_momentum_confluence import generate_signals, DEFAULT_PARAMS

def debug_strategy():
    """Debug the strategy to understand type issues."""
    print("Debugging strategy...")
    
    # Create synthetic data
    np.random.seed(42)
    n_points = 100
    
    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.02)
    volumes = np.random.randint(1000000, 5000000, n_points)
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_points)),
        'high': prices * (1 + np.random.uniform(0, 0.01, n_points)),
        'low': prices * (1 + np.random.uniform(-0.01, 0, n_points)),
        'close': prices,
        'volume': volumes
    })
    
    print("DataFrame shape:", df.shape)
    print("DataFrame dtypes:")
    print(df.dtypes)
    
    try:
        entries, exits = generate_signals(df, DEFAULT_PARAMS)
        print("\nEntries dtype:", entries.dtype)
        print("Entries NaN count:", entries.isna().sum())
        print("Exits dtype:", exits.dtype)
        print("Exits NaN count:", exits.isna().sum())
        
        # Check first few values
        print("\nFirst 10 entries:")
        print(entries.head(10))
        print("\nFirst 10 exits:")
        print(exits.head(10))
        
    except Exception as e:
        print(f"Error in generate_signals: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_strategy()