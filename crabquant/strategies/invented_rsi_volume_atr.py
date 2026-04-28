"""
Invented Strategy: RSI Volume ATR Confluence

Combines RSI signals with volume confirmation and ATR-based trailing exits for 
reliable entries with dynamic risk management.

Key features:
- RSI oversold/overbought signals
- Volume spike confirmation
- ATR trailing stops for exits
- MACD trend filtering
"""
import pandas as pd
import pandas_ta as ta

DEFAULT_PARAMS = {
    "rsi_len": 14,
    "rsi_oversold": 35,  # More lenient oversold
    "rsi_overbought": 65,  # More lenient overbought
    "volume_ma_len": 20,
    "volume_spike_mult": 1.5,  # Lower volume spike threshold
    "atr_len": 14,
    "atr_mult": 2.0,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "macd_filter": False,  # Disable MACD filter for more entries
}

PARAM_GRID = {
    "rsi_len": [10, 14, 20, 25],
    "rsi_oversold": [20, 25, 30, 35],
    "rsi_overbought": [65, 70, 75, 80],
    "volume_ma_len": [10, 20, 30],
    "volume_spike_mult": [1.5, 2.0, 2.5, 3.0],
    "atr_len": [10, 14, 20],
    "atr_mult": [1.5, 2.0, 2.5, 3.0],
    "macd_fast": [8, 12, 16],
    "macd_slow": [21, 26, 31],
    "macd_signal": [6, 9, 12],
    "macd_filter": [True, False],
}

DESCRIPTION = """
RSI Volume ATR Confluence Strategy

Entry signals:
- RSI crosses above oversold level (RSI > 35)
- Volume spike > 1.5x 20-day average volume
- Optional: MACD histogram positive for trend confirmation

Exit signals:
- RSI crosses below overbought level (RSI < 65) - take profit
- Price touches ATR trailing stop - dynamic stop loss
- ATR trailing stop = close - (ATR * 2.0)

Works well in:
- Mean-reverting markets with volume confirmation
- Stocks with clear support/resistance levels
- Medium-term swing trading (1-4 weeks holding period)

Strategy strengths:
- Volume reduces false signals
- ATR trailing stops adapt to volatility
- MACD filter improves timing in trends
"""

def generate_signals(df, params):
    """
    Generate entry and exit signals for RSI Volume ATR strategy.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        params: Dict of strategy parameters
        
    Returns:
        entries: pd.Series[bool] - entry signals
        exits: pd.Series[bool] - exit signals  
    """
    params = params or DEFAULT_PARAMS

    # Calculate indicators
    rsi = ta.rsi(df['close'], length=params['rsi_len'])
    volume_ma = ta.sma(df['volume'], length=params['volume_ma_len'])
    atr = ta.atr(df['high'], df['low'], df['close'], length=params['atr_len'])
    
    # MACD for trend filtering
    macd = ta.macd(df['close'], fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'])
    macd_hist = macd['MACDh_12_26_9']
    
    # Volume spike detection - handle NaN values
    volume_ma = volume_ma.fillna(volume_ma.mean())
    volume_ratio = df['volume'] / volume_ma
    volume_spike = volume_ratio > params['volume_spike_mult']
    
    # RSI signals - handle NaN values
    rsi = rsi.fillna(50)  # Neutral value for NaN
    rsi_cross_up = (rsi > params['rsi_oversold']) & (rsi.shift(1) <= params['rsi_oversold'])
    rsi_cross_down = (rsi < params['rsi_overbought']) & (rsi.shift(1) >= params['rsi_overbought'])
    
    # MACD trend filter - handle NaN values
    macd_bullish = macd_hist > 0 if params['macd_filter'] else pd.Series(True, index=df.index)
    macd_bullish = macd_bullish.fillna(True)
    
    # Entry signals: RSI oversold + volume spike + MACD bullish
    entries = rsi_cross_up & volume_spike & macd_bullish
    
    # Exit signals: RSI overbought or ATR trailing stop
    exits = rsi_cross_down
    
    # ATR trailing stop for additional exit - handle NaN values
    atr = atr.fillna(atr.mean())
    atr_stop = df['close'] - (atr * params['atr_mult'])
    atr_stop_trigger = df['low'] <= atr_stop
    
    # Add ATR stops to exits (logical OR)
    exits = exits | atr_stop_trigger
    
    # Ensure boolean dtype and handle any remaining NaN
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)
    
    return entries, exits

def generate_signals_matrix(df, param_grid):
    """
    Generate signals for all parameter combinations in the grid.
    
    Args:
        df: DataFrame with price data
        param_grid: Dict of parameter grids
        
    Returns:
        entries_df: DataFrame with entry signals for each param combination
        exits_df: DataFrame with exit signals for each param combination  
        param_list: List of parameter dicts used
    """
    import itertools
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    param_combinations = list(itertools.product(*param_values))
    param_list = [dict(zip(param_names, combo)) for combo in param_combinations]
    
    # Initialize DataFrames
    n_combinations = len(param_list)
    entries_df = pd.DataFrame(index=df.index, columns=range(n_combinations))
    exits_df = pd.DataFrame(index=df.index, columns=range(n_combinations))
    
    # Generate signals for each parameter combination
    for i, params in enumerate(param_list):
        entries, exits = generate_signals(df, params)
        entries_df[i] = entries
        exits_df[i] = exits
    
    return entries_df, exits_df, param_list