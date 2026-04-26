"""
Invented Volume Momentum Trend Strategy

Combines ROC momentum with volume confirmation and ATR-based exits.
Trend filter ensures we only trade with the broader market direction.
"""

import pandas as pd
import pandas_ta as ta

def generate_signals(df, params):
    """
    Generate entry/exit signals using volume momentum trend strategy.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        params: Dict with strategy parameters
        
    Returns:
        entries: pd.Series[bool] - entry signals
        exits: pd.Series[bool] - exit signals  
    """
    # Calculate indicators
    roc = ta.roc(df['close'], length=params['roc_len'])
    volume_sma = ta.sma(df['volume'], length=params['volume_window'])
    atr = ta.atr(df['high'], df['low'], df['close'], length=params['atr_len'])
    
    # Trend filter using EMA
    ema_fast = ta.ema(df['close'], length=params['ema_fast'])
    ema_slow = ta.ema(df['close'], length=params['ema_slow'])
    trend = ema_fast > ema_slow
    
    # Volume confirmation
    volume_ratio = df['volume'] / volume_sma
    volume_surge = volume_ratio > params['volume_threshold']
    
    # Entry conditions: ROC positive + volume surge + trend filter
    entries = (roc > params['roc_threshold']) & volume_surge & trend
    
    # Exit conditions: ROC reversal + ATR stop-loss
    roc_reversal = roc < -params['roc_exit_threshold']
    atr_stop = (df['close'] < (df['close'].rolling(params['atr_len']).max() - 
                              params['atr_mult'] * atr))
    
    exits = roc_reversal | atr_stop
    
    # Prevent consecutive entries
    entries = entries & ~entries.shift(1).fillna(False)
    
    return entries, exits

# Default parameters for the strategy
DEFAULT_PARAMS = {
    'roc_len': 12,
    'roc_threshold': 0.5,
    'roc_exit_threshold': 0.3,
    'volume_window': 20,
    'volume_threshold': 1.5,
    'atr_len': 14,
    'atr_mult': 2.0,
    'ema_fast': 10,
    'ema_slow': 30,
}

# Parameter grid for optimization
PARAM_GRID = {
    'roc_len': [8, 12, 16, 20],
    'roc_threshold': [0.3, 0.5, 0.7, 1.0],
    'roc_exit_threshold': [0.2, 0.3, 0.5],
    'volume_window': [15, 20, 25, 30],
    'volume_threshold': [1.2, 1.5, 2.0, 2.5],
    'atr_len': [10, 14, 20],
    'atr_mult': [1.5, 2.0, 2.5, 3.0],
    'ema_fast': [5, 10, 15, 20],
    'ema_slow': [20, 30, 40, 50],
}

# Strategy description
DESCRIPTION = """
Volume Momentum Trend Strategy

This strategy combines ROC momentum with volume confirmation and trend filtering to identify high-probability trading opportunities.

Key Components:
1. ROC (Rate of Change) for momentum detection
2. Volume SMA ratio for volume confirmation 
3. Dual EMA crossover for trend direction
4. ATR-based stop-loss for risk management

Best For:
- Medium-term trend following
- Volume-confirmed breakouts
- Stocks with consistent volume patterns

Parameters:
- roc_len: ROC lookback period
- roc_threshold: Minimum ROC for entry signal
- volume_window: Volume moving average length
- volume_threshold: Volume surge multiplier
- atr_mult: ATR multiplier for stop-loss
- ema_fast/slow: Trend filter EMA periods
"""

def generate_signals_matrix(df, param_grid):
    """
    Generate signals matrix for parameter optimization.
    
    Args:
        df: DataFrame with price data
        param_grid: Dict of parameter options
        
    Returns:
        entries_df: DataFrame of entry signals for each parameter set
        exits_df: DataFrame of exit signals for each parameter set  
        param_list: List of parameter dicts tested
    """
    import itertools
    
    # Get all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    entries_list = []
    exits_list = []
    param_list = []
    
    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))
        entries, exits = generate_signals(df, params)
        
        entries_list.append(entries)
        exits_list.append(exits)
        param_list.append(params)
    
    # Convert to DataFrames
    entries_df = pd.DataFrame(entries_list).T
    exits_df = pd.DataFrame(exits_list).T
    
    return entries_df, exits_df, param_list