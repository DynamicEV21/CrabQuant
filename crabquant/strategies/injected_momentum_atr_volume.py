"""
Injected Momentum ATR Volume Strategy

Combines momentum with volume confirmation and ATR-based exits.
Uses RSI regime filter to avoid entries in weak markets.

- ROC for momentum detection
- Volume spike confirmation
- RSI regime filter (avoid RSI < 30 in uptrend, > 70 in downtrend)
- ATR trailing stops for exits
"""

import pandas as pd
import pandas_ta as ta


def generate_signals(df, params):
    """
    Generate entry/exit signals for injected momentum ATR volume strategy.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        params: dict with strategy parameters
        
    Returns:
        entries: pd.Series[bool] - entry signals
        exits: pd.Series[bool] - exit signals
    """
    # Calculate indicators
    roc = df['close'].pct_change(params['roc_len']) * 100
    volume_sma = df['volume'].rolling(params['vol_sma_len']).mean()
    volume_ratio = df['volume'] / volume_sma
    rsi = ta.rsi(df['close'], length=params['rsi_len'])
    atr = ta.atr(df['high'], df['low'], df['close'], length=params['atr_len'])
    
    # RSI regime detection
    ema_short = df['close'].ewm(span=params['ema_short_len']).mean()
    ema_long = df['close'].ewm(span=params['ema_long_len']).mean()
    is_uptrend = ema_short > ema_long
    
    # Entry conditions - more aggressive thresholds
    long_momentum = roc > params['roc_threshold']
    volume_spike = volume_ratio > params['vol_threshold']
    rsi_healthy = ((is_uptrend & (rsi > params['rsi_min_uptrend'])) | 
                  (~is_uptrend & (rsi < params['rsi_max_downtrend'])))
    
    entries = long_momentum & volume_spike & rsi_healthy
    
    # Exit conditions - ATR trailing stop
    trailing_stop = df['close'] - (atr * params['atr_mult'])
    exits = (df['close'] <= trailing_stop) & entries.shift(1)
    
    return entries, exits


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