"""
Invented Volatility RSI Breakout Strategy

Combines Bollinger Bands for volatility detection with RSI for mean reversion
timing and volume confirmation. Uses ATR-based exits for risk management.

Entry: Price breaks Bollinger Band lower band AND RSI oversold AND volume spike
Exit: Price crosses back above BB middle band OR ATR-based stop
"""

import pandas as pd
import pandas_ta as ta

DEFAULT_PARAMS = {
    "bb_length": 20,
    "bb_std": 2.0,
    "rsi_length": 14,
    "rsi_oversold": 30,
    "volume_window": 20,
    "volume_multiplier": 2.0,
    "atr_length": 14,
    "atr_multiplier": 2.0,
}

PARAM_GRID = {
    "bb_length": [15, 20, 25],
    "bb_std": [1.5, 2.0, 2.5],
    "rsi_length": [10, 14, 20],
    "rsi_oversold": [25, 30, 35],
    "volume_window": [10, 20, 30],
    "volume_multiplier": [1.5, 2.0, 2.5],
    "atr_length": [10, 14, 20],
    "atr_multiplier": [1.5, 2.0, 2.5],
}

DESCRIPTION = """
Volatility RSI Breakout Strategy - detects mean reversion opportunities after 
volatility spikes. Entry when price breaks below Bollinger Band lower band with
oversold RSI and volume confirmation. Exit on mean reversion or ATR stop.
Works well in volatile, mean-reverting markets.
"""


def generate_signals(df, params):
    """
    Generate entry and exit signals for Volatility RSI Breakout strategy.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        params: Strategy parameters
        
    Returns:
        entries: pd.Series[bool] - True when entry signal occurs
        exits: pd.Series[bool] - True when exit signal occurs
    """
    # Calculate indicators
    bb = ta.bbands(df['close'], length=params['bb_length'], std=params['bb_std'])
    rsi = ta.rsi(df['close'], length=params['rsi_length'])
    atr = ta.atr(df['high'], df['low'], df['close'], length=params['atr_length'])
    
    # Volume spike detection
    volume_ma = df['volume'].rolling(params['volume_window']).mean()
    volume_spike = df['volume'] > (volume_ma * params['volume_multiplier'])
    
    # Entry conditions: BB breakdown + oversold RSI + volume spike
    bb_lower = bb[f'BBU_{params["bb_length"]}_{params["bb_std"]}_2.0']
    entries = (
        (df['close'] < bb_lower) &  # Price below BB lower
        (rsi < params['rsi_oversold']) &  # Oversold RSI
        (volume_spike) &  # Volume spike
        (rsi.notna()) &  # Valid RSI
        (bb_lower.notna())  # Valid BB
    )
    
    # Exit conditions: mean reversion or ATR stop
    bb_middle = bb[f'BBM_{params["bb_length"]}_{params["bb_std"]}_2.0']
    atr_stop = df['low'] - (atr * params['atr_multiplier'])
    
    exits = (
        (df['close'] > bb_middle) |  # Mean reversion
        (df['low'] < atr_stop)  # ATR stop loss
    )
    
    # Clean up
    entries = entries.fillna(False)
    exits = exits.fillna(False)
    
    return entries, exits


def generate_signals_matrix(df, param_grid):
    """
    Generate signals matrix for parameter optimization.
    """
    param_list = []
    entries_list = []
    exits_list = []
    
    # Generate parameter combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for param_values_tuple in product(*param_values):
        params = dict(zip(param_names, param_values_tuple))
        param_list.append(params)
        
        entries, exits = generate_signals(df, params)
        entries_list.append(entries)
        exits_list.append(exits)
    
    entries_df = pd.DataFrame(entries_list).T
    exits_df = pd.DataFrame(exits_list).T
    
    return entries_df, exits_df, param_list