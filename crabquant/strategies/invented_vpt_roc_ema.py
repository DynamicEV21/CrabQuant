import pandas as pd
import pandas_ta as ta
from itertools import product
from crabquant.indicator_cache import cached_indicator

def generate_signals(df, params):
    """
    Generate trading signals using Volume Price Trend (VPT) + ROC + EMA combination
    
    Strategy logic:
    - Enter: VPT crossover above signal line + ROC bullish + price above EMA
    - Exit: VPT crossover below signal line + ROC bearish + price below EMA
    
    Params:
    - vpt_len: VPT signal line lookback period (default: 14)
    - roc_len: ROC lookback period (default: 21)  
    - ema_len: EMA lookback period (default: 20)
    - roc_threshold: ROC threshold for confirmation (default: 1.0)
    """
    df = df.copy()
    
    # Calculate indicators
    vpt = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
    vpt_signal = vpt.rolling(window=params['vpt_len']).mean()
    roc = ta.roc(df['close'], length=params['roc_len'])
    ema = ta.ema(df['close'], length=params['ema_len'])
    
    # Generate signals
    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    
    # Entry conditions: VPT > signal + ROC positive + price above EMA
    vpt_bullish = vpt > vpt_signal
    roc_bullish = roc > params['roc_threshold'] 
    price_above_ema = df['close'] > ema
    
    entries = vpt_bullish & roc_bullish & price_above_ema
    
    # Exit conditions: VPT < signal + ROC negative + price below EMA  
    vpt_bearish = vpt < vpt_signal
    roc_bearish = roc < -params['roc_threshold']
    price_below_ema = df['close'] < ema
    
    exits = vpt_bearish & roc_bearish & price_below_ema
    
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
    close = df['close']
    volume = df['volume']
    
    # Pre-compute all unique VPT lengths
    all_vpt_lens = sorted(set(pg['vpt_len']))
    vpt_cache = {}
    vpt_signal_cache = {}
    for vpt_len in all_vpt_lens:
        vpt = (volume * ((close - close.shift(1)) / close.shift(1))).cumsum()
        vpt_cache[vpt_len] = vpt
        vpt_signal_cache[vpt_len] = vpt.rolling(window=vpt_len).mean()
    
    # Pre-compute all unique ROC lengths  
    all_roc_lens = sorted(set(pg['roc_len']))
    roc_cache = {}
    for roc_len in all_roc_lens:
        roc_cache[roc_len] = ta.roc(close, length=roc_len)
    
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
        vpt_len = params['vpt_len']
        roc_len = params['roc_len']
        ema_len = params['ema_len']
        roc_threshold = params['roc_threshold']
        
        vpt = vpt_cache[vpt_len]
        vpt_signal = vpt_signal_cache[vpt_len]
        roc = roc_cache[roc_len]
        ema = ema_cache[ema_len]
        
        # Entry conditions
        entries = (
            (vpt > vpt_signal) & 
            (roc > roc_threshold) & 
            (close > ema)
        ).fillna(False)
        
        # Exit conditions  
        exits = (
            (vpt < vpt_signal) & 
            (roc < -roc_threshold) & 
            (close < ema)
        ).fillna(False)
        
        entries_cols[f'c{i}'] = entries
        exits_cols[f'c{i}'] = exits
        param_list.append(params)
    
    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list

DEFAULT_PARAMS = {
    'vpt_len': 14,
    'roc_len': 21, 
    'ema_len': 20,
    'roc_threshold': 1.0
}

PARAM_GRID = {
    'vpt_len': [10, 14, 20, 25],
    'roc_len': [14, 21, 28, 35],
    'ema_len': [15, 20, 25, 30],
    'roc_threshold': [0.5, 1.0, 1.5, 2.0]
}

DESCRIPTION = """VPT + ROC + EMA Confluence Strategy

Combines Volume Price Trend (VPT), Rate of Change (ROC), and EMA for trend-following signals:
- Entries: VPT above signal line + bullish ROC + price above EMA
- Exits: VPT below signal line + bearish ROC + price below EMA
- Optimized for capturing strong momentum with volume confirmation

Best for: trending stocks with high volume momentum

Inspired by: roc_ema_volume (127k wins) and vpt_crossover (18.9% win rate) patterns"""