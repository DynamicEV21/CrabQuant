"""
ROC + EMA + Volume Momentum Strategy

Combines momentum (ROC), trend (EMA), and volume confirmation for entries.
ATR trailing stop for dynamic exits. Designed for AAPL momentum capture.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 12,
    "ema_len": 20,
    "volume_len": 20,
    "volume_mult": 1.0,
    "atr_len": 14,
    "atr_mult": 2.0,
}

PARAM_GRID = {
    "roc_len": [8, 12, 16, 20],
    "ema_len": [15, 20, 25, 30],
    "volume_len": [15, 20, 25],
    "volume_mult": [0.9, 1.0, 1.1],
    "atr_len": [10, 14, 18],
    "atr_mult": [1.5, 2.0, 2.5],
}

DESCRIPTION = (
    "ROC momentum with EMA trend filter and volume confirmation. "
    "Enters when ROC positive, price above EMA, volume above average. "
    "Exits via ATR trailing stop. Designed for AAPL momentum capture."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Momentum: ROC positive
    roc = cached_indicator("roc", close, length=p["roc_len"])
    momentum_ok = roc > 0

    # Trend filter: price above EMA
    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    # Volume confirmation
    vol_avg = cached_indicator("sma", volume, length=p["volume_len"])
    volume_ok = volume > (vol_avg * p["volume_mult"])

    # Entry: all three conditions
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
    all_vol_lens = sorted(set(pg["volume_len"]))
    all_atr_lens = sorted(set(pg["atr_len"]))

    roc_cache = {l: cached_indicator("roc", close, length=l) for l in all_roc_lens}
    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_ema_lens}
    vol_avg_cache = {l: cached_indicator("sma", volume, length=l) for l in all_vol_lens}
    atr_cache = {l: cached_indicator("atr", high, low, close, length=l) for l in all_atr_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))

        roc = roc_cache[params["roc_len"]]
        ema = ema_cache[params["ema_len"]]
        vol_avg = vol_avg_cache[params["volume_len"]]
        atr = atr_cache[params["atr_len"]]

        momentum_ok = roc > 0
        trend_ok = close > ema
        volume_ok = volume > (vol_avg * params["volume_mult"])

        e = (momentum_ok & trend_ok & volume_ok).fillna(False)

        rolling_high = close.rolling(window=params["atr_len"], min_periods=1).max()
        trailing_stop = rolling_high - (atr * params["atr_mult"])
        x = (close < trailing_stop).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
