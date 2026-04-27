"""
ROC EMA Volume Momentum Strategy

Simple momentum strategy combining Rate of Change, EMA trend filter,
and volume confirmation with ATR trailing stop exits.

Entry: ROC positive + price above EMA + volume above average
Exit: ATR trailing stop (rolling high - ATR * multiplier)
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 14,
    "ema_len": 50,
    "volume_window": 20,
    "volume_mult": 1.0,
    "atr_len": 14,
    "atr_mult": 2.0,
}

PARAM_GRID = {
    "roc_len": [10, 14, 20],
    "ema_len": [30, 50, 75],
    "volume_window": [15, 20, 25],
    "volume_mult": [0.8, 1.0, 1.2],
    "atr_len": [10, 14, 20],
    "atr_mult": [1.5, 2.0, 2.5],
}

DESCRIPTION = (
    "ROC EMA Volume momentum strategy. "
    "Enters when ROC is positive, price above EMA, and volume above average. "
    "Exits via ATR trailing stop. "
    "Simple trend-following with volume confirmation."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ROC - momentum indicator
    roc = cached_indicator("roc", close, length=p["roc_len"])
    momentum_ok = roc > 0

    # EMA trend filter
    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    # Volume confirmation
    volume_avg = cached_indicator("sma", volume, length=p["volume_window"])
    volume_ok = volume > (volume_avg * p["volume_mult"])

    # Entry: all three conditions met
    entries = (momentum_ok & trend_ok & volume_ok).fillna(False)

    # ATR trailing stop exit
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    rolling_high = close.rolling(window=p["atr_len"], min_periods=1).max()
    trailing_stop = rolling_high - (atr * p["atr_mult"])
    exits = (close < trailing_stop).fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Generate signals for ALL param combinations at once (vectorized)."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Deduplicate indicator calculations
    all_roc_lens = sorted(set(pg["roc_len"]))
    all_ema_lens = sorted(set(pg["ema_len"]))
    all_vol_windows = sorted(set(pg["volume_window"]))
    all_atr_lens = sorted(set(pg["atr_len"]))

    roc_cache = {l: cached_indicator("roc", close, length=l) for l in all_roc_lens}
    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_ema_lens}
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_windows}
    atr_cache = {l: cached_indicator("atr", high, low, close, length=l) for l in all_atr_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))

        roc = roc_cache[params["roc_len"]]
        momentum_ok = roc > 0

        ema = ema_cache[params["ema_len"]]
        trend_ok = close > ema

        volume_avg = vol_avg_cache[params["volume_window"]]
        volume_ok = volume > (volume_avg * params["volume_mult"])

        e = (momentum_ok & trend_ok & volume_ok).fillna(False)

        atr = atr_cache[params["atr_len"]]
        rolling_high = close.rolling(window=params["atr_len"], min_periods=1).max()
        trailing_stop = rolling_high - (atr * params["atr_mult"])
        x = (close < trailing_stop).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
