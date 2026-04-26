"""
Invented Momentum Confluence Strategy

Combines multiple momentum indicators with volume confirmation and volatility filtering.
Uses ROC, RSI, and ADX to identify momentum shifts with volume confirmation.
"""

from crabquant.indicator_cache import cached_indicator
from itertools import product

import pandas as pd

def generate_signals(df, params):
    """
    Generate trading signals using momentum confluence.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        params: dict with parameters
        
    Returns:
        entries: pd.Series[bool] - entry signals
        exits: pd.Series[bool] - exit signals  
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    
    # Calculate indicators using cached_indicator
    roc = cached_indicator("roc", df["close"], length=p['roc_len'])
    rsi = cached_indicator("rsi", df["close"], length=p['rsi_len'])
    adx_df = cached_indicator("adx", df["high"], df["low"], df["close"], length=p['adx_len'])
    atr = cached_indicator("atr", df["high"], df["low"], df["close"], length=p['atr_len'])
    vol_sma = df["volume"].rolling(window=p['vol_sma_len']).mean()
    
    # Extract ADX value from the DataFrame
    adx = adx_df[f'ADX_{p["adx_len"]}']
    
    # Fill NaN values for indicators (fill with 0 for ROC, 50 for RSI/ADX, volume fill with mean)
    roc = roc.fillna(0)
    rsi = rsi.fillna(50)
    adx = adx.fillna(20)
    atr = atr.fillna(df['close'].rolling(window=20).mean())  # Fill ATR with rolling mean
    vol_sma = vol_sma.fillna(df['volume'].mean())
    
    # Calculate conditions
    momentum_up = roc > 0
    momentum_down = roc < 0
    
    rsi_oversold = rsi < p['rsi_oversold']
    rsi_overbought = rsi > p['rsi_overbought']
    
    adx_strong = adx > p['adx_threshold']
    adx_weak = adx < p['adx_weak_threshold']
    
    volume_expansion = df['volume'] > vol_sma * p['volume_mult']
    
    # Entry conditions - more permissive
    entries = pd.Series(False, index=df.index)
    
    # Simple momentum entries (easier to trigger)
    momentum_long = momentum_up & volume_expansion
    momentum_short = momentum_down & volume_expansion
    
    # RSI mean reversion entries
    rsi_long = rsi_oversold & volume_expansion
    rsi_short = rsi_overbought & volume_expansion
    
    # ADX regime-based entries
    trending_long = adx_strong & momentum_up
    trending_short = adx_strong & momentum_down
    
    # Combine all entry signals
    entries = momentum_long | momentum_short | rsi_long | rsi_short | trending_long | trending_short
    
    # Exit conditions using ATR stops
    exits = pd.Series(False, index=df.index)
    
    # Calculate dynamic exits
    atr_stop_long = df['close'] - (atr * p['atr_mult'])
    atr_stop_short = df['close'] + (atr * p['atr_mult'])
    
    # Simple exit conditions
    price_stop_long = df['close'] < atr_stop_long
    price_stop_short = df['close'] > atr_stop_short
    
    # RSI reversal exits
    rsi_exit_long = rsi_overbought & (rsi.shift(1) <= p['rsi_overbought'])
    rsi_exit_short = rsi_oversold & (rsi.shift(1) >= p['rsi_oversold'])
    
    # Combine exit conditions
    exits = (entries.shift(1) & price_stop_long) | (entries.shift(1) & price_stop_short) | \
            (entries.shift(1) & rsi_exit_long) | (entries.shift(1) & rsi_exit_short)
    
    # Ensure all outputs are boolean with no NaN
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)
    
    return entries, exits

def generate_signals_matrix(df, param_grid):
    """
    Generate signals for ALL param combinations at once (vectorized).
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        param_grid: dict of parameter options for optimization
        
    Returns:
        entries_df: DataFrame with entries for each param combination
        exits_df: DataFrame with exits for each param combination
        param_list: list of parameter dicts
    """
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))
    
    # Pre-compute indicators for all unique lengths
    all_roc_lens = sorted(set(pg['roc_len']))
    all_rsi_lens = sorted(set(pg['rsi_len']))
    all_adx_lens = sorted(set(pg['adx_len']))
    all_atr_lens = sorted(set(pg['atr_len']))
    all_vol_sma_lens = sorted(set(pg['vol_sma_len']))
    
    # Cache all indicators
    roc_cache = {l: cached_indicator("roc", df["close"], length=l) for l in all_roc_lens}
    rsi_cache = {l: cached_indicator("rsi", df["close"], length=l) for l in all_rsi_lens}
    adx_cache = {}
    for l in all_adx_lens:
        adx_df = cached_indicator("adx", df["high"], df["low"], df["close"], length=l)
        adx_cache[l] = adx_df[f'ADX_{l}']
    atr_cache = {l: cached_indicator("atr", df["high"], df["low"], df["close"], length=l) for l in all_atr_lens}
    vol_sma_cache = {l: df["volume"].rolling(window=l).mean() for l in all_vol_sma_lens}
    
    entries_cols = {}
    exits_cols = {}
    param_list = []
    
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        
        # Get cached indicators
        roc = roc_cache[params['roc_len']].fillna(0)
        rsi = rsi_cache[params['rsi_len']].fillna(50)
        adx = adx_cache[params['adx_len']].fillna(20)
        atr = atr_cache[params['atr_len']].fillna(df['close'].rolling(window=20).mean())
        vol_sma = vol_sma_cache[params['vol_sma_len']].fillna(df['volume'].mean())
        
        # Calculate conditions
        momentum_up = roc > 0
        momentum_down = roc < 0
        rsi_oversold = rsi < params['rsi_oversold']
        rsi_overbought = rsi > params['rsi_overbought']
        adx_strong = adx > params['adx_threshold']
        adx_weak = adx < params['adx_weak_threshold']
        volume_expansion = df['volume'] > vol_sma * params['volume_mult']
        
        # Entry conditions
        momentum_long = momentum_up & volume_expansion
        momentum_short = momentum_down & volume_expansion
        rsi_long = rsi_oversold & volume_expansion
        rsi_short = rsi_overbought & volume_expansion
        trending_long = adx_strong & momentum_up
        trending_short = adx_strong & momentum_down
        
        entries = momentum_long | momentum_short | rsi_long | rsi_short | trending_long | trending_short
        
        # Exit conditions
        atr_stop_long = df['close'] - (atr * params['atr_mult'])
        atr_stop_short = df['close'] + (atr * params['atr_mult'])
        price_stop_long = df['close'] < atr_stop_long
        price_stop_short = df['close'] > atr_stop_short
        rsi_exit_long = rsi_overbought & (rsi.shift(1) <= params['rsi_overbought'])
        rsi_exit_short = rsi_oversold & (rsi.shift(1) >= params['rsi_oversold'])
        
        exits = (entries.shift(1) & price_stop_long) | (entries.shift(1) & price_stop_short) | \
                (entries.shift(1) & rsi_exit_long) | (entries.shift(1) & rsi_exit_short)
        
        # Clean up NaN values
        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)
        
        entries_cols[f"c{i}"] = entries
        exits_cols[f"c{i}"] = exits
        param_list.append(params)
    
    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list

DEFAULT_PARAMS = {
    'roc_len': 10,
    'rsi_len': 14,
    'adx_len': 14,
    'atr_len': 14,
    'vol_sma_len': 20,
    'rsi_oversold': 35,
    'rsi_overbought': 65,
    'adx_threshold': 25,
    'adx_weak_threshold': 20,
    'volume_mult': 1.5,
    'atr_mult': 2.0
}

PARAM_GRID = {
    'roc_len': [7, 10, 14, 21],
    'rsi_len': [10, 14, 21],
    'adx_len': [14, 21],
    'atr_len': [14, 21],
    'vol_sma_len': [15, 20, 25],
    'rsi_oversold': [25, 30, 35],
    'rsi_overbought': [65, 70, 75],
    'adx_threshold': [20, 25, 30],
    'adx_weak_threshold': [15, 20, 25],
    'volume_mult': [1.2, 1.5, 2.0],
    'atr_mult': [1.5, 2.0, 2.5]
}

DESCRIPTION = """
Invented Momentum Confluence Strategy

Combines multiple momentum indicators with volume confirmation and volatility filtering.

- Uses ROC for momentum direction
- RSI for mean reversion levels
- ADX for trend strength
- Volume expansion for confirmation
- ATR-based dynamic exits for risk management

Entry signals (multiple conditions):
- Momentum: ROC > 0 or < 0 + volume expansion
- RSI mean reversion: RSI oversold/overbought + volume
- Trending: ADX strong + momentum direction

Exit signals:
- ATR price stops
- RSI reversals

Works well in trending markets with occasional mean-reversion phases.
"""