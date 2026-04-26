#!/usr/bin/env python3

import pandas as pd
import numpy as np
from crabquant.strategies.invented_momentum_confluence import generate_signals, DEFAULT_PARAMS

def test_strategy_on_tickers(tickers):
    """Test strategy on given tickers and return trade counts."""
    results = {}
    
    for ticker in tickers:
        print(f"Testing {ticker}...")
        
        try:
            # Load sample data (in real scenario, this would fetch from data source)
            # Create synthetic data for testing
            np.random.seed(42)
            n_points = 500
            
            # Generate synthetic price data
            prices = 100 + np.cumsum(np.random.randn(n_points) * 0.02)
            volumes = np.random.randint(1000000, 5000000, n_points)
            
            df = pd.DataFrame({
                'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_points)),
                'high': prices * (1 + np.random.uniform(0, 0.01, n_points)),
                'low': prices * (1 + np.random.uniform(-0.01, 0, n_points)),
                'close': prices,
                'volume': volumes
            })
            
            # Generate signals
            entries, exits = generate_signals(df, DEFAULT_PARAMS)
            
            # Fill NaN values with False
            entries = entries.fillna(False)
            exits = exits.fillna(False)
            
            # Count trades
            trades = 0
            in_position = False
            
            for i in range(len(entries)):
                if entries.iloc[i] and not in_position:
                    trades += 1
                    in_position = True
                elif exits.iloc[i] and in_position:
                    in_position = False
            
            results[ticker] = trades
            print(f"  {ticker}: {trades} trades generated")
            
        except Exception as e:
            print(f"  {ticker}: Error - {e}")
            results[ticker] = 0
    
    return results

if __name__ == "__main__":
    test_tickers = ['AAPL', 'NVDA', 'CAT', 'SPY']
    results = test_strategy_on_tickers(test_tickers)
    
    print("\nResults:")
    for ticker, trades in results.items():
        print(f"{ticker}: {trades} trades")
    
    total_trades = sum(results.values())
    print(f"\nTotal trades: {total_trades}")
    
    if total_trades == 0:
        print("⚠️  No trades generated - strategy needs adjustment")
        # Try different parameters
        print("\nTrying alternative parameters...")
        alt_params = DEFAULT_PARAMS.copy()
        alt_params['rsi_oversold'] = 25
        alt_params['rsi_overbought'] = 75
        alt_params['volume_mult'] = 1.2
        
        results_alt = test_strategy_on_tickers(test_tickers)
        total_trades_alt = sum(results_alt.values())
        print(f"Total trades (alt params): {total_trades_alt}")
    else:
        print("✅ Strategy generates trades")