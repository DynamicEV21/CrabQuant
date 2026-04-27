"""
Invented Strategy: Volume Breakout with ADX Trend Confirmation

This strategy identifies volume breakouts that occur during strong trends,
using ADX to confirm trend direction and ATR for dynamic exits.

Entry conditions:
- Volume spike above 20-period SMA with multiplier > 2.0
- ADX > 25 confirms strong trend
- RSI < 70 (avoid overbought conditions in uptrends, > 30 in downtrends)

Exit conditions:
- ATR-based trailing stop (2.5x ATR)
- RSI reversal signals
- Volume dry-up confirmation

Best for: AAPL, NVDA, CAT, SPY - especially volatile tech and industrial stocks
"""

import pandas as pd
import pandas_ta as ta


def generate_signals(df, params):
    """
    Generate entry and exit signals for volume breakout with ADX trend confirmation.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        params: Dictionary with strategy parameters
        
    Returns:
        entries: pd.Series[bool] - True where entry signal occurs
        exits: pd.Series[bool] - True where exit signal occurs
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Calculate indicators
    # Volume indicators
    df['vol_sma'] = ta.sma(df['volume'], length=params['vol_sma_len'])
    df['vol_spike'] = df['volume'] > (df['vol_sma'] * params['vol_mult'])
    
    # Trend indicators  
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=params['adx_len'])['ADX_14']
    df['rsi'] = ta.rsi(df['close'], length=params['rsi_len'])
    
    # Volatility indicator
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=params['atr_len'])
    
    # Moving averages for trend direction
    df['sma_fast'] = ta.sma(df['close'], length=params['sma_fast'])
    df['sma_slow'] = ta.sma(df['close'], length=params['sma_slow'])
    df['trend_up'] = df['sma_fast'] > df['sma_slow']
    df['trend_down'] = df['sma_fast'] < df['sma_slow']
    
    # Entry signals
    entries = pd.Series(False, index=df.index)
    
    # Uptrend entries
    uptrend_mask = (df['trend_up']) & (df['adx'] > params['adx_threshold']) & (df['rsi'] < 70)
    entries_uptrend = (df['vol_spike']) & (uptrend_mask)
    
    # Downtrend entries  
    downtrend_mask = (df['trend_down']) & (df['adx'] > params['adx_threshold']) & (df['rsi'] > 30)
    entries_downtrend = (df['vol_spike']) & (downtrend_mask)
    
    entries = entries_uptrend | entries_downtrend
    
    # Exit signals - ATR trailing stops
    exits = pd.Series(False, index=df.index)
    
    # Initialize trailing stop variables
    if 'long_stop' not in df.columns:
        df['long_stop'] = df['low'] - (df['atr'] * params['atr_mult'])
        df['short_stop'] = df['high'] + (df['atr'] * params['atr_mult'])
    
    # Update trailing stops
    df.loc[df['trend_up'], 'long_stop'] = df['low'] - (df['atr'] * params['atr_mult'])
    df.loc[df['trend_down'], 'short_stop'] = df['high'] + (df['atr'] * params['atr_mult'])
    
    # Exit conditions for long positions
    long_exits = entries_uptrend.shift(1) & (df['close'] < df['long_stop'])
    long_rsi_exit = entries_uptrend.shift(1) & (df['rsi'] > 80)
    long_vol_exit = entries_uptrend.shift(1) & (df['volume'] < (df['vol_sma'] * 0.5))
    
    # Exit conditions for short positions
    short_exits = entries_downtrend.shift(1) & (df['close'] > df['short_stop'])
    short_rsi_exit = entries_downtrend.shift(1) & (df['rsi'] < 20)
    short_vol_exit = entries_downtrend.shift(1) & (df['volume'] < (df['vol_sma'] * 0.5))
    
    # Combine all exit signals
    exits = long_exits | long_rsi_exit | long_vol_exit | short_exits | short_rsi_exit | short_vol_exit
    
    return entries, exits


def generate_signals_matrix(df, param_grid):
    """
    Generate signals for all parameter combinations in the grid.
    
    Args:
        df: DataFrame with price and volume data
        param_grid: Dictionary of parameter options
        
    Returns:
        entries_df: DataFrame with entry signals for each parameter set
        exits_df: DataFrame with exit signals for each parameter set  
        param_list: List of parameter dictionaries used
    """
    entries_list = []
    exits_list = []
    param_list = []
    
    # Generate all parameter combinations
    from itertools import product
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for param_values_combo in product(*param_values):
        params = dict(zip(param_names, param_values_combo))
        param_list.append(params)
        
        entries, exits = generate_signals(df, params)
        entries_list.append(entries)
        exits_list.append(exits)
    
    # Create DataFrames
    entries_df = pd.DataFrame(entries_list, index=[str(p) for p in param_list]).T
    exits_df = pd.DataFrame(exits_list, index=[str(p) for p in param_list]).T
    
    return entries_df, exits_df, param_list


DEFAULT_PARAMS = {
    'vol_sma_len': 20,      # Volume SMA period for breakout detection
    'vol_mult': 2.0,        # Volume multiplier for breakout threshold  
    'adx_len': 14,          # ADX calculation period
    'adx_threshold': 25,    # ADX threshold for trend confirmation
    'rsi_len': 14,          # RSI period for overbought/oversold detection
    'atr_len': 14,          # ATR calculation period
    'atr_mult': 2.5,        # ATR multiplier for trailing stops
    'sma_fast': 10,         # Fast moving average period for trend direction
    'sma_slow': 30,         # Slow moving average period for trend direction
}

PARAM_GRID = {
    'vol_sma_len': [10, 20, 30],
    'vol_mult': [1.5, 2.0, 2.5],
    'adx_len': [14, 21], 
    'adx_threshold': [20, 25, 30],
    'rsi_len': [14, 21],
    'atr_len': [14, 21],
    'atr_mult': [2.0, 2.5, 3.0],
    'sma_fast': [5, 10, 15],
    'sma_slow': [20, 30, 50],
}

DESCRIPTION = """
Volume Breakout with ADX Trend Confirmation Strategy

This strategy identifies volume breakouts that occur during strong confirmed trends,
using ADX to filter for only high-quality trend setups. It combines volume confirmation 
with trend filtering to improve signal quality.

Entry Rules:
- Volume spike above N-period SMA (multiplier dependent on ticker)
- ADX > threshold confirms strong trend direction  
- RSI < 70 for uptrends, RSI > 30 for downtrends (avoid extremes)
- Moving average crossover confirms primary trend direction

Exit Rules:  
- ATR-based trailing stop for dynamic risk management
- RSI reversal signals (RSI > 80 for long, RSI < 20 for short)
- Volume dry-up confirmation (volume < 50% of average)

Target Tickers: AAPL, NVDA, CAT, SPY
Best for volatile stocks with clear volume patterns and strong trends.

Edge: Filters out false breakouts by requiring trend confirmation and avoids
extreme overbought/oversold conditions. ATR trailing stops provide dynamic
risk management suitable for volatile markets.
"""