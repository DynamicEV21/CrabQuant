"""
Momentum RSI Volume Strategy

Combines ROC momentum, RSI mean reversion, and volume confirmation for 
trend-following entries with mean-reverse exits.

Entry: ROC bullish momentum + RSI not overbought + volume spike
Exit: RSI oversold or ATR-based trailing stop

Best for: Durable trends with volume confirmation
"""

import pandas as pd
import pandas_ta as ta
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "roc_len": 12,
    "roc_threshold": 5,
    "rsi_len": 14,
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "volume_sma_len": 20,
    "volume_threshold": 1.5,
    "atr_len": 14,
    "atr_mult": 2.0,
    "trailing_len": 10
}

PARAM_GRID = {
    "roc_len": [8, 12, 16, 20],
    "roc_threshold": [3, 5, 8],
    "rsi_len": [10, 14, 20],
    "rsi_oversold": [25, 35, 45],
    "rsi_overbought": [55, 65, 75],
    "volume_sma_len": [10, 20, 30],
    "volume_threshold": [1.2, 1.5, 2.0],
    "atr_mult": [1.5, 2.0, 2.5]
}

DESCRIPTION = """Momentum RSI Volume Strategy

Trend-following strategy using ROC for momentum detection, RSI for 
mean-reversion timing, and volume for confirmation.

Entry Signals:
- ROC(roc_len) > roc_threshold (bullish momentum)
- RSI(rsi_len) < rsi_overbought (not overbought)
- Volume > volume_sma_len * volume_threshold (volume spike)

Exit Signals:
- RSI(rsi_len) < rsi_oversold (mean reversion)
- ATR trailing stop with multiplier (volatility-based exit)

Best for: Durable trends with strong volume confirmation across
multiple tickers and timeframes."""


def generate_signals(df, params):
    """Generate entry and exit signals using momentum, RSI, and volume."""
    
    # Calculate indicators using cached_indicator
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    roc = cached_indicator("roc", close, length=params['roc_len'])
    rsi = cached_indicator("rsi", close, length=params['rsi_len'])
    atr = cached_indicator("atr", high, low, close, length=params['atr_len'])
    
    # Calculate volume SMA with pandas rolling mean
    volume_sma = volume.rolling(window=params['volume_sma_len']).mean()
    
    # Initialize signals
    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    
    # Entry conditions: momentum + not overbought + volume spike
    momentum_bullish = (roc > params['roc_threshold']).fillna(False)
    not_overbought = (rsi < params['rsi_overbought']).fillna(False)
    volume_spike = (df['volume'] > (volume_sma * params['volume_threshold'])).fillna(False)
    
    entries = momentum_bullish & not_overbought & volume_spike
    
    # Exit conditions: oversold or ATR trailing stop
    oversold = (rsi < params['rsi_oversold']).fillna(False)
    
    # ATR trailing stop logic
    highest_close = df['close'].rolling(window=params['trailing_len']).max()
    trailing_stop = highest_close - (atr * params['atr_mult'])
    
    # Exit when price drops below trailing stop or RSI oversold
    below_trailing = (df['close'] < trailing_stop).fillna(False)
    exits = oversold | below_trailing
    
    return entries, exits


def generate_signals_matrix(df, param_grid):
    """Generate signals matrix for parameter optimization."""
    
    param_list = []
    entries_list = []
    exits_list = []
    
    # Generate parameter combinations
    param_combinations = [
        {k: v_list[i] for k, v_list in param_grid.items()}
        for i in range(len(param_grid['roc_len']))
    ]
    
    for params in param_combinations:
        entries, exits = generate_signals(df, params)
        param_list.append(params)
        entries_list.append(entries)
        exits_list.append(exits)
    
    return entries_list, exits_list, param_list