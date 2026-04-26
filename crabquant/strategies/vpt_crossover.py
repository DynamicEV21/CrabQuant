"""
VPT Crossover Strategy

Volume Price Trend crossover with RSI filter and volume confirmation.
VPT = cumulative sum of (volume * ((close - prev_close) / prev_close)).
Entry when VPT crosses above its SMA signal with RSI and volume confirmation.
Exit when VPT crosses below signal or RSI overbought.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "vpt_signal_len": 20,
    "rsi_len": 14,
    "vol_sma_len": 20,
    "rsi_entry": 40,
    "rsi_exit": 80,
}

PARAM_GRID = {
    "vpt_signal_len": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 42, 44, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80],
    "rsi_len": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 42, 44, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75],
    "vol_sma_len": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 42, 44, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80, 85, 90],
    "rsi_entry": [10, 12, 14, 16, 18, 20, 22, 24, 25, 26, 28, 30, 32, 33, 34, 35, 36, 38, 40, 42, 44, 45, 46, 48, 50, 52, 54, 55, 56, 58, 60, 62, 64, 65],
    "rsi_exit": [58, 60, 62, 64, 65, 66, 68, 70, 72, 74, 75, 76, 78, 80, 82, 84, 85, 86, 88, 90, 92, 94, 95, 96, 98],
}

DESCRIPTION = (
    "Volume Price Trend crossover with RSI filter and volume confirmation. "
    "Enters when VPT crosses above its SMA signal line with RSI above entry threshold "
    "and volume above its SMA. Exits on VPT cross below signal or RSI overbought. "
    "Works well in volume-confirmed trending markets."
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

    # VPT computed manually (pandas_ta has no VPT)
    vpt = (volume * ((close - close.shift(1)) / close.shift(1))).cumsum()
    vpt_signal = vpt.rolling(window=p["vpt_signal_len"]).mean()

    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    vol_sma = volume.rolling(window=p["vol_sma_len"]).mean()

    entries = (
        (vpt.shift(1) <= vpt_signal.shift(1))
        & (vpt > vpt_signal)
        & (rsi > p["rsi_entry"])
        & (volume > vol_sma)
    ).fillna(False)

    exits = (
        ((vpt.shift(1) >= vpt_signal.shift(1)) & (vpt < vpt_signal))
        | (rsi > p["rsi_exit"])
    ).fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Generate signals for ALL param combinations at once (vectorized).

    Returns:
        (entries_df, exits_df, param_list) where each DataFrame has one column per combo.
    """
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    # Deduplicate: VPT is the same for all combos (no length param), only signal SMA varies
    close = df["close"]
    volume = df["volume"]

    vpt = (volume * ((close - close.shift(1)) / close.shift(1))).cumsum()

    # Pre-compute VPT signals for all unique signal lengths
    all_vpt_signal_lens = sorted(set(pg["vpt_signal_len"]))
    vpt_signals = {l: vpt.rolling(window=l).mean() for l in all_vpt_signal_lens}

    # Pre-compute RSI for all unique lengths
    all_rsi_lens = sorted(set(pg["rsi_len"]))
    rsi_cache = {l: cached_indicator("rsi", close, length=l) for l in all_rsi_lens}

    # Pre-compute volume SMA for all unique lengths
    all_vol_sma_lens = sorted(set(pg["vol_sma_len"]))
    vol_sma_cache = {l: volume.rolling(window=l).mean() for l in all_vol_sma_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        vsl = params["vpt_signal_len"]
        rl = params["rsi_len"]
        vsl_vol = params["vol_sma_len"]
        rsi_entry = params["rsi_entry"]
        rsi_exit = params["rsi_exit"]

        vs = vpt_signals[vsl]
        rsi = rsi_cache[rl]
        vsm = vol_sma_cache[vsl_vol]

        e = (
            (vpt.shift(1) <= vs.shift(1))
            & (vpt > vs)
            & (rsi > rsi_entry)
            & (volume > vsm)
        ).fillna(False)
        x = (
            ((vpt.shift(1) >= vs.shift(1)) & (vpt < vs))
            | (rsi > rsi_exit)
        ).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
