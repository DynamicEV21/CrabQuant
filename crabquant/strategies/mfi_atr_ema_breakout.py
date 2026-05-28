"""
mfi_atr_ema_breakout
Volume-weighted momentum breakout using EMA price action, MFI volume confirmation, and ATR volatility validation

This strategy identifies swing trading opportunities by combining three core indicators:
1. EMA (Exponential Moving Average) - tracks trend direction and price momentum
2. MFI (Money Flow Index) - confirms high volume participation in price moves
3. ATR (Average True Range) - validates sufficient volatility for meaningful moves

Entry: Price breaks above EMA AND MFI shows strong buying pressure (above threshold) AND ATR confirms adequate volatility
Exit: Price falls below EMA OR MFI shows weakening buying pressure (below threshold)

Designed for 2-10 day swing trades with volume-weighted edge detection.
"""
from itertools import product
import pandas as pd
import pandas_ta
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_length": 20,
    "mfi_length": 14,
    "mfi_high": 70,
    "atr_length": 14,
    "atr_threshold_pct": 0.5,
}

PARAM_GRID = {
    "ema_length": [9, 12, 15, 20, 25],
    "mfi_length": [10, 14, 20],
    "mfi_high": [60, 70, 80],
    "atr_length": [14],
    "atr_threshold_pct": [0.3, 0.5, 0.7],
}

DESCRIPTION = (
    "Entry: Price closes above EMA AND MFI > mfi_high (strong volume-driven buying pressure) "
    "AND ATR > atr_threshold_pct% of price (sufficient volatility for breakout). "
    "Exit: Price closes below EMA OR MFI < 40 (buying pressure weakened). "
    "This volume-weighted breakout strategy ensures entries only when price action, "
    "volume participation, and volatility align for 2-10 day swing trades."
)

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    """Generate entry/exit signals."""
    p = {**DEFAULT_PARAMS, **(params or {})}

    # Core indicators
    ema = cached_indicator('ema', df['close'], length=p['ema_length'])
    mfi = cached_indicator('mfi', df['high'], df['low'], df['close'], df['volume'], length=p['mfi_length'])
    atr = cached_indicator('atr', df['high'], df['low'], df['close'], length=p['atr_length'])

    # Calculate ATR threshold as percentage of price
    atr_threshold = df['close'] * p['atr_threshold_pct'] / 100

    # Entry signals
    price_above_ema = df['close'] > ema
    mfi_bullish = mfi > p['mfi_high']
    volatility_sufficient = atr > atr_threshold

    entries = (price_above_ema & mfi_bullish & volatility_sufficient).fillna(False)

    # Exit signals
    price_below_ema = df['close'] < ema
    mfi_weak = mfi < 40

    exits = (price_below_ema | mfi_weak).fillna(False)

    return entries, exits

def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Generate signals for ALL param combinations."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))
    entries_cols = {}
    exits_cols = {}
    param_list = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        e, x = generate_signals(df, params)
        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)
    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
