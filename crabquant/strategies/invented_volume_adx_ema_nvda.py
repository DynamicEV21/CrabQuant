import pandas as pd
import pandas_ta as ta

def _normalize_columns(df):
    """Normalize yfinance multi-level columns to lowercase single level"""
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

def generate_signals(df, params):
    """
    Volume-ADX-EMA Strategy
    
    Entry conditions:
    1. OBV momentum (bullish/bearish divergence)
    2. ADX > trend threshold (confirming trend strength)
    3. Price above/below EMA based on trend direction
    
    Exit conditions:
    1. ATR-based trailing stop
    2. RSI reversal levels
    """
    params = params or DEFAULT_PARAMS

    # Normalize column names
    df = _normalize_columns(df)
    
    # Extract parameters
    obv_fast = params.get('obv_fast', 10)
    obv_slow = params.get('obv_slow', 20)
    adx_len = params.get('adx_len', 14)
    adx_threshold = params.get('adx_threshold', 25)
    ema_len = params.get('ema_len', 50)
    rsi_len = params.get('rsi_len', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    atr_mult = params.get('atr_mult', 2.0)
    
    # Calculate indicators
    df_copy = df.copy()
    
    # On Balance Volume (OBV)
    df_copy['obv'] = ta.obv(df_copy['close'], df_copy['volume'])
    df_copy['obv_fast'] = df_copy['obv'].rolling(obv_fast).mean()
    df_copy['obv_slow'] = df_copy['obv'].rolling(obv_slow).mean()
    
    # OBV crossover detection (handle NaN values properly)
    obv_above = df_copy['obv_fast'] > df_copy['obv_slow']
    prev_obv_above = obv_above.shift(1)
    prev_obv_above = prev_obv_above.where(prev_obv_above.notna(), other=False)
    obv_crossover = obv_above & (prev_obv_above == False)
    obv_crossover = obv_crossover.where(obv_crossover.notna(), other=False)
    
    # ADX for trend strength
    adx = ta.adx(df_copy['high'], df_copy['low'], df_copy['close'], length=adx_len)
    df_copy['adx'] = adx['ADX_14']
    strong_trend = df_copy['adx'] > adx_threshold
    
    # EMA for trend direction
    df_copy['ema'] = ta.ema(df_copy['close'], length=ema_len)
    above_ema = df_copy['close'] > df_copy['ema']
    below_ema = df_copy['close'] < df_copy['ema']
    
    # RSI for exit conditions
    rsi = ta.rsi(df_copy['close'], length=rsi_len)
    df_copy['rsi'] = rsi
    
    # ATR for stops
    atr = ta.atr(df_copy['high'], df_copy['low'], df_copy['close'], length=14)
    df_copy['atr'] = atr
    df_copy['atr_stop'] = df_copy['close'] - (atr_mult * df_copy['atr'])
    
    # Generate signals
    entries = pd.Series(False, index=df_copy.index)
    exits = pd.Series(False, index=df_copy.index)
    
    # Long entries: OBV crossover + strong uptrend + price above EMA
    long_entry = obv_crossover & strong_trend & above_ema
    entries = entries | long_entry
    
    # Short entries: OBV crossover + strong downtrend + price below EMA
    short_entry = obv_crossover & strong_trend & below_ema
    # Note: For now, we'll focus on long entries only
    # entries = entries | short_entry
    
    # Exits: RSI overbought/oversold or ATR stop hit
    long_exit = (df_copy['rsi'] > rsi_overbought) | (df_copy['close'] < df_copy['atr_stop'])
    short_exit = (df_copy['rsi'] < rsi_oversold) | (df_copy['close'] > df_copy['atr_stop'])
    
    # Apply exits to long positions only for now
    exits = exits | (long_exit & long_entry.shift(1))
    # exits = exits | (short_exit & short_entry.shift(1))
    
    return entries, exits

def generate_signals_matrix(df, param_grid):
    """Generate signals for all parameter combinations"""
    from itertools import product
    
    entries_list = []
    exits_list = []
    param_list = []
    
    # Get all parameter combinations
    param_names = param_grid.keys()
    param_values = param_grid.values()
    
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        
        entries, exits = generate_signals(df, params)
        entries_list.append(entries)
        exits_list.append(exits)
        param_list.append(params)
    
    # Combine all signals
    entries_df = pd.concat(entries_list, axis=1, keys=range(len(entries_list)))
    exits_df = pd.concat(exits_list, axis=1, keys=range(len(exits_list)))
    
    return entries_df, exits_df, param_list

DEFAULT_PARAMS = {
    'obv_fast': 10,
    'obv_slow': 20,
    'adx_len': 14,
    'adx_threshold': 25,
    'ema_len': 50,
    'rsi_len': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'atr_mult': 2.0
}

PARAM_GRID = {
    'obv_fast': [8, 10, 12],
    'obv_slow': [15, 20, 25],
    'adx_len': [14, 21, 28],
    'adx_threshold': [20, 25, 30],
    'ema_len': [20, 50, 100],
    'rsi_len': [10, 14, 21],
    'rsi_oversold': [25, 30, 35],
    'rsi_overbought': [65, 70, 75],
    'atr_mult': [1.5, 2.0, 2.5]
}

DESCRIPTION = """
Volume-ADX-EMA Strategy: Combines volume momentum (OBV), trend strength (ADX), 
and directional bias (EMA) for trend-following entries. Uses ATR-based stops
and RSI for exit timing. Works best in trending markets with volume confirmation.

Entry:
- OBV crossover (fast > slow)
- ADX > threshold (trend strength > 25)
- Price above EMA (uptrend) or below EMA (downtrend)

Exit:
- RSI > 70 (overbought) or RSI < 30 (oversold)
- ATR trailing stop (price - 2*ATR)

Best for: AAPL, NVDA, CAT, SPY - trending tickers with good volume
"""