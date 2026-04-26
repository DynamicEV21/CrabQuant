"""
Invented Momentum RSI Stochastic Strategy

Combines RSI with Stochastic oscillator for momentum confirmation.
Targets momentum stocks like CAT, JPM, SPY with volume confirmation.
Best in trending markets with pullback reversals.
"""

from itertools import product

import pandas as pd
import pandas_ta


DEFAULT_PARAMS = {
    "rsi_len": 14,
    "rsi_oversold": 35,
    "volume_window": 20,
    "volume_mult": 1.2,
}

PARAM_GRID = {
    "rsi_len": [7, 14, 21],
    "rsi_oversold": [25, 30, 35],
    "volume_window": [10, 20, 30],
    "volume_mult": [1.0, 1.2, 1.5],
}

DESCRIPTION = (
    "Simple RSI oversold strategy with volume confirmation. "
    "Enters when RSI is oversold with volume spike. "
    "Exits when RSI becomes overbought. "
    "Best for momentum stocks (CAT, JPM, SPY) with pullback reversals."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Generate entry/exit signals.

    Args:
        df: DataFrame with columns open, high, low, close, volume
        params: Strategy parameters (uses defaults if None)

    Returns:
        (entries, exits) as boolean Series
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    # Calculate indicators
    rsi = pandas_ta.rsi(close, length=p["rsi_len"])
    
    # Volume filter
    volume_avg = pandas_ta.sma(volume, length=p["volume_window"])
    volume_spike = volume > (volume_avg * p["volume_mult"])

    # Entry conditions: RSI oversold with volume spike
    entries = (
        (rsi < p["rsi_oversold"]) &
        (volume_spike)
    ).fillna(False)

    # Exit conditions: RSI overbought
    exits = (rsi > 70).fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Generate signals for ALL param combinations at once (vectorized)."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    close = df["close"]
    volume = df["volume"]

    # Deduplicate
    all_rsi_lens = sorted(set(pg["rsi_len"]))
    all_vol_windows = sorted(set(pg["volume_window"]))

    rsi_cache = {l: pandas_ta.rsi(close, length=l) for l in all_rsi_lens}
    vol_avg_cache = {w: pandas_ta.sma(volume, length=w) for w in all_vol_windows}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        rsi = rsi_cache[params["rsi_len"]]
        volume_avg = vol_avg_cache[params["volume_window"]]
        volume_spike = volume > (volume_avg * params["volume_mult"])

        e = ((rsi < params["rsi_oversold"]) & volume_spike).fillna(False)
        x = (rsi > 70).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list