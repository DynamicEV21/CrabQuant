import pandas as pd
import pandas_ta as ta
import itertools

# Strategy: Volume Momentum Trend
# Combines volume breakout with trend filtering and momentum confirmation
# Works well in trending markets with volume confirmation

def generate_signals(df, params):
    """
    Generate trading signals based on volume momentum trend strategy.
    
    Entry conditions:
    1. Volume breakout above moving average
    2. ADX > trend_threshold (confirming trend)
    3. RSI confirms momentum direction
    
    Exit conditions:
    1. ATR-based trailing stop
    2. RSI reversal signal
    
    Parameters:
    - volume_sma_len: period for volume moving average
    - volume_mult: volume multiplier for breakout
    - adx_len: ADX period
    - adx_threshold: ADX value to confirm trend
    - rsi_len: RSI period
    - rsi_oversold: RSI oversold threshold
    - rsi_overbought: RSI overbought threshold
    - atr_len: ATR period for stop loss
    - atr_mult: ATR multiplier for stop loss
    """
    params = params or DEFAULT_PARAMS
    # Calculate indicators
    volume_sma = df['volume'].rolling(window=params['volume_sma_len']).mean()
    volume_breakout = df['volume'] > volume_sma * params['volume_mult']
    
    adx = ta.adx(df['high'], df['low'], df['close'], length=params['adx_len'])
    adx_trend = adx[f'ADX_{params["adx_len"]}'] > params['adx_threshold']
    
    rsi = ta.rsi(df['close'], length=params['rsi_len'])
    
    # ATR for trailing stop
    atr = ta.atr(df['high'], df['low'], df['close'], length=params['atr_len'])
    atr_stop = df['close'] - atr * params['atr_mult']
    
    # Entry signals - only when volume confirms and trend is established
    entries = pd.Series(False, index=df.index)
    
    # Bullish entries: uptrend + momentum + volume
    bullish_entries = (
        (df['close'] > df['open']) &  # green candle
        adx_trend &                   # strong trend
        (rsi > params['rsi_oversold']) &  # not oversold
        volume_breakout &            # volume breakout
        (df['close'] > df['close'].shift(params['adx_len']))  # price above X bars ago
    )
    
    # Bearish entries: downtrend + momentum + volume
    bearish_entries = (
        (df['close'] < df['open']) &  # red candle
        adx_trend &                   # strong trend
        (rsi < params['rsi_overbought']) &  # not overbought
        volume_breakout &            # volume breakout
        (df['close'] < df['close'].shift(params['adx_len']))  # price below X bars ago
    )
    
    entries[bullish_entries] = True
    entries[bearish_entries] = True
    
    # Exit signals
    exits = pd.Series(False, index=df.index)
    
    # Trailing stop exit
    long_trail = df['close'] < atr_stop
    short_trail = df['close'] > atr_stop
    
    # RSI reversal exits
    rsi_oversold = rsi < params['rsi_oversold']
    rsi_overbought = rsi > params['rsi_overbought']
    
    # Exit long positions
    exits[long_trail] = True
    exits[rsi_oversold] = True
    
    # Exit short positions  
    exits[short_trail] = True
    exits[rsi_overbought] = True
    
    return entries, exits

def generate_signals_matrix(df, param_grid):
    """
    Generate signals matrix for parameter optimization.
    Returns entries_df, exits_df, and param_list.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(*param_values))
    
    entries_dfs = []
    exits_dfs = []
    param_list = []
    
    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))
        
        entries, exits = generate_signals(df, params)
        
        entries_dfs.append(entries)
        exits_dfs.append(exits)
        param_list.append(params)
    
    # Combine all signals
    entries_df = pd.DataFrame(entries_dfs).T
    exits_df = pd.DataFrame(exits_dfs).T
    
    return entries_df, exits_df, param_list

DEFAULT_PARAMS = {
    'volume_sma_len': 20,
    'volume_mult': 1.5,
    'adx_len': 14,
    'adx_threshold': 25,
    'rsi_len': 14,
    'rsi_oversold': 35,
    'rsi_overbought': 65,
    'atr_len': 14,
    'atr_mult': 2.0
}

PARAM_GRID = {
    'volume_sma_len': [10, 20, 30],
    'volume_mult': [1.3, 1.5, 2.0],
    'adx_len': [14, 21],
    'adx_threshold': [20, 25, 30],
    'rsi_len': [14, 21],
    'rsi_oversold': [30, 35, 40],
    'rsi_overbought': [60, 65, 70],
    'atr_len': [14, 21],
    'atr_mult': [1.5, 2.0, 2.5]
}

DESCRIPTION = """
Invented Volume Momentum Trend Strategy

This strategy combines volume confirmation with trend filtering and momentum analysis:
- Uses volume breakout above moving average to confirm interest
- ADX indicator to establish trend strength and direction
- RSI to time entries and identify reversals
- ATR-based trailing stops for risk management

Ideal for trending markets where volume precedes price movements.
Works well on liquid stocks with consistent volume patterns.

Entry: Volume breakout + ADX trend + RSI momentum confirmation
Exit: ATR trailing stop or RSI reversal signal
"""