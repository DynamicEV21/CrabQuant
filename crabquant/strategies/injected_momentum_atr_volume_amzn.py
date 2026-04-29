"""
Injected Momentum ATR Volume Strategy

Combines momentum with volume confirmation and ATR-based exits.
Uses RSI regime filter to avoid entries in weak markets.

- ROC for momentum detection
- Volume spike confirmation
- RSI regime filter (avoid RSI < 30 in uptrend, > 70 in downtrend)
- ATR trailing stops for exits
"""

from itertools import product

import pandas as pd
import pandas_ta as ta

from crabquant.indicator_cache import cached_indicator


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Generate entry/exit signals for injected momentum ATR volume strategy.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        params: dict with strategy parameters
        
    Returns:
        entries: pd.Series[bool] - entry signals
        exits: pd.Series[bool] - exit signals
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Calculate indicators
    roc = close.pct_change(p['roc_len']) * 100
    volume_sma = volume.rolling(p['vol_sma_len']).mean()
    volume_ratio = volume / volume_sma
    rsi = cached_indicator('rsi', close, length=p['rsi_len'])
    atr = cached_indicator('atr', high, low, close, length=p['atr_len'])
    
    # RSI regime detection
    ema_short = cached_indicator('ema', close, length=p['ema_short_len'])
    ema_long = cached_indicator('ema', close, length=p['ema_long_len'])
    is_uptrend = ema_short > ema_long
    
    # Entry conditions
    long_momentum = roc > p['roc_threshold']
    volume_spike = volume_ratio > p['vol_threshold']
    rsi_healthy = ((is_uptrend & (rsi > p['rsi_min_uptrend'])) | 
                  (~is_uptrend & (rsi < p['rsi_max_downtrend'])))
    
    entries = (long_momentum & volume_spike & rsi_healthy).fillna(False)
    
    # Exit conditions - ATR trailing stop
    trailing_stop = close - (atr * p['atr_mult'])
    exits = ((close <= trailing_stop) & entries.shift(1)).fillna(False)
    
    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Generate signals for ALL param combinations at once (vectorized)."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Pre-compute ROC for all unique lengths
    all_roc_lens = sorted(set(pg['roc_len']))
    roc_cache = {l: close.pct_change(l) * 100 for l in all_roc_lens}

    # Pre-compute volume SMA for all unique lengths
    all_vol_sma_lens = sorted(set(pg['vol_sma_len']))
    vol_sma_cache = {w: volume.rolling(w).mean() for w in all_vol_sma_lens}

    # Pre-compute RSI for all unique lengths
    all_rsi_lens = sorted(set(pg['rsi_len']))
    rsi_cache = {l: cached_indicator('rsi', close, length=l) for l in all_rsi_lens}

    # Pre-compute ATR for all unique lengths
    all_atr_lens = sorted(set(pg['atr_len']))
    atr_cache = {l: cached_indicator('atr', high, low, close, length=l) for l in all_atr_lens}

    # Pre-compute EMA for all unique lengths (short + long combined)
    all_ema_lens = sorted(set(pg['ema_short_len']) | set(pg['ema_long_len']))
    ema_cache = {l: cached_indicator('ema', close, length=l) for l in all_ema_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        rl = params['roc_len']
        vsl = params['vol_sma_len']
        al = params['atr_len']
        esl = params['ema_short_len']
        ell = params['ema_long_len']

        roc = roc_cache[rl]
        volume_sma = vol_sma_cache[vsl]
        volume_ratio = volume / volume_sma
        rsi = rsi_cache[params['rsi_len']]
        atr = atr_cache[al]

        ema_short = ema_cache[esl]
        ema_long = ema_cache[ell]
        is_uptrend = ema_short > ema_long

        long_momentum = roc > params['roc_threshold']
        volume_spike = volume_ratio > params['vol_threshold']
        rsi_healthy = ((is_uptrend & (rsi > params['rsi_min_uptrend'])) | 
                      (~is_uptrend & (rsi < params['rsi_max_downtrend'])))

        e = (long_momentum & volume_spike & rsi_healthy).fillna(False)

        trailing_stop = close - (atr * params['atr_mult'])
        x = ((close <= trailing_stop) & e.shift(1)).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list


DEFAULT_PARAMS = {
    'roc_len': 5,        # Shorter ROC for more sensitivity
    'roc_threshold': 0.5, # Lower threshold for momentum
    'vol_sma_len': 10,   # Shorter volume window
    'vol_threshold': 1.2, # Lower volume threshold
    'rsi_len': 10,       # Shorter RSI
    'rsi_min_uptrend': 25,  # Lower RSI threshold for uptrend
    'rsi_max_downtrend': 75, # Higher RSI threshold for downtrend  
    'ema_short_len': 10,  # Shorter EMAs
    'ema_long_len': 20,
    'atr_len': 10,       # Shorter ATR
    'atr_mult': 1.5      # Lower ATR multiplier
}


PARAM_GRID = {
    'roc_len': [3, 5, 8],
    'roc_threshold': [0.3, 0.5, 1.0],
    'vol_sma_len': [5, 10, 15],
    'vol_threshold': [1.1, 1.2, 1.5],
    'rsi_len': [8, 10, 15],
    'rsi_min_uptrend': [20, 25, 30],
    'rsi_max_downtrend': [70, 75, 80],
    'ema_short_len': [5, 10, 15],
    'ema_long_len': [15, 20, 25],
    'atr_len': [8, 10, 15],
    'atr_mult': [1.0, 1.5, 2.0]
}


DESCRIPTION = """Injected Momentum ATR Volume Strategy

Combines momentum with volume confirmation and ATR-based exits.
Uses RSI regime filter to avoid entries in weak markets.

Strategy logic:
- ROC for momentum detection (positive price change)
- Volume spike confirmation (above moving average)
- RSI regime filter (avoid extreme values in opposite trend)
- ATR trailing stops for exits

Best for: Momentum-driven markets with volume confirmation
Timeframe: 1D+
Risk management: ATR-based trailing stops
"""