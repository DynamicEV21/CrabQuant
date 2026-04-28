import pandas as pd
import pandas_ta as ta
from itertools import product
from crabquant.indicator_cache import cached_indicator

def generate_signals(df, params):
    """
    Generate trading signals using Volume + ROC + RSI + EMA combination
    
    Strategy logic:
    - Enter: High volume spike + ROC bullish + RSI oversold reversal + price above EMA
    - Exit: Volume dries up + ROC bearish + RSI overbought + price below EMA
    
    Params:
    - volume_window: Volume SMA lookback (default: 20)
    - volume_mult: Volume multiplier for spike detection (default: 1.5)
    - roc_len: ROC lookback period (default: 21)
    - roc_threshold: ROC threshold for confirmation (default: 1.0)
    - rsi_len: RSI lookback period (default: 14)
    - rsi_oversold: RSI oversold threshold (default: 30)
    - rsi_overbought: RSI overbought threshold (default: 70)
    - ema_len: EMA lookback period (default: 20)
    """
    params = params or DEFAULT_PARAMS

    df = df.copy()
    
    # Reset index to flatten MultiIndex columns
    df = df.reset_index()
    
    # Create simple column names
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # Calculate indicators
    volume_sma = df['volume'].rolling(window=params['volume_window']).mean()
    volume_spike = df['volume'] > (volume_sma * params['volume_mult'])
    roc = ta.roc(df['close'], length=params['roc_len'])
    rsi = ta.rsi(df['close'], length=params['rsi_len'])
    ema = ta.ema(df['close'], length=params['ema_len'])
    
    # Generate signals
    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    
    # Entry conditions: volume spike + ROC bullish + RSI oversold reversal + price above EMA
    roc_bullish = roc > params['roc_threshold']
    rsi_oversold_reversal = (rsi.shift(1) <= params['rsi_oversold']) & (rsi > params['rsi_oversold'])
    price_above_ema = df['close'] > ema
    
    entries = volume_spike & roc_bullish & rsi_oversold_reversal & price_above_ema
    
    # Exit conditions: low volume + ROC bearish + RSI overbought + price below EMA
    volume_low = df['volume'] < (volume_sma * 0.5)  # Volume dried up
    roc_bearish = roc < -params['roc_threshold']
    rsi_overbought = rsi > params['rsi_overbought']
    price_below_ema = df['close'] < ema
    
    exits = volume_low & roc_bearish & rsi_overbought & price_below_ema
    
    return entries, exits

def generate_signals_matrix(df, param_grid=None):
    """
    Generate signals for ALL param combinations at once (vectorized).
    
    Returns:
        (entries_df, exits_df, param_list) where each DataFrame has one column per combo.
    """
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))
    
    df = df.copy()
    
    # Reset index to flatten MultiIndex columns
    df = df.reset_index()
    
    # Create simple column names
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    close = df['close']
    volume = df['volume']
    
    # Pre-compute all unique volume window lengths
    all_volume_windows = sorted(set(pg['volume_window']))
    volume_sma_cache = {}
    for volume_window in all_volume_windows:
        volume_sma_cache[volume_window] = volume.rolling(window=volume_window).mean()
    
    # Pre-compute all unique ROC lengths  
    all_roc_lens = sorted(set(pg['roc_len']))
    roc_cache = {}
    for roc_len in all_roc_lens:
        roc_cache[roc_len] = ta.roc(close, length=roc_len)
    
    # Pre-compute all unique RSI lengths
    all_rsi_lens = sorted(set(pg['rsi_len']))
    rsi_cache = {}
    for rsi_len in all_rsi_lens:
        rsi_cache[rsi_len] = ta.rsi(close, length=rsi_len)
    
    # Pre-compute all unique EMA lengths
    all_ema_lens = sorted(set(pg['ema_len']))
    ema_cache = {}
    for ema_len in all_ema_lens:
        ema_cache[ema_len] = ta.ema(close, length=ema_len)
    
    entries_cols = {}
    exits_cols = {}
    param_list = []
    
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        volume_window = params['volume_window']
        volume_mult = params['volume_mult']
        roc_len = params['roc_len']
        roc_threshold = params['roc_threshold']
        rsi_len = params['rsi_len']
        rsi_oversold = params['rsi_oversold']
        rsi_overbought = params['rsi_overbought']
        ema_len = params['ema_len']
        
        volume_sma = volume_sma_cache[volume_window]
        roc = roc_cache[roc_len]
        rsi = rsi_cache[rsi_len]
        ema = ema_cache[ema_len]
        
        volume_spike = volume > (volume_sma * volume_mult)
        volume_low = volume < (volume_sma * 0.5)
        
        # Entry conditions
        entries = (
            volume_spike &
            (roc > roc_threshold) &
            ((rsi.shift(1) <= rsi_oversold) & (rsi > rsi_oversold)) &
            (close > ema)
        ).fillna(False)
        
        # Exit conditions  
        exits = (
            volume_low &
            (roc < -roc_threshold) &
            (rsi > rsi_overbought) &
            (close < ema)
        ).fillna(False)
        
        entries_cols[f'c{i}'] = entries
        exits_cols[f'c{i}'] = exits
        param_list.append(params)
    
    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list

DEFAULT_PARAMS = {
    'volume_window': 20,
    'volume_mult': 1.5,
    'roc_len': 21,
    'roc_threshold': 1.0,
    'rsi_len': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'ema_len': 20
}

PARAM_GRID = {
    'volume_window': [10, 15, 20, 25, 30, 35, 40, 45, 50],
    'volume_mult': [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    'roc_len': [7, 10, 14, 21, 28, 35],
    'roc_threshold': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
    'rsi_len': [7, 10, 14, 21, 28],
    'rsi_oversold': [20, 25, 30, 35, 40],
    'rsi_overbought': [60, 65, 70, 75, 80],
    'ema_len': [10, 15, 20, 25, 30, 35, 40, 50]
}

DESCRIPTION = """Volume + ROC + RSI + EMA Confluence Strategy

Combines volume spikes, momentum, mean reversion, and trend for robust signals:
- Entries: High volume + ROC bullish + RSI oversold reversal + price above EMA
- Exits: Low volume + ROC bearish + RSI overbought + price below EMA
- Captures breakout momentum with mean reversion potential

Best for: Range-bound markets with occasional breakout opportunities

Inspired by: invented_vpt_roc_ema (76% win rate) and rsi_regime_dip (2.7% win rate) patterns"""