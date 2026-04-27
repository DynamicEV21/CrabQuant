"""
ROC + EMA + Volume Momentum Strategy for SPY

Combines momentum (ROC), trend direction (EMA), and volume confirmation.
ATR trailing stop for risk-adaptive exits.
Optimized for SPY's smooth trend characteristics.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 12,
    "ema_len": 50,
    "vol_len": 20,
    "vol_mult": 1.1,
    "atr_len": 14,
    "atr_mult": 2.5,
}

PARAM_GRID = {
    "roc_len": [8, 10, 12, 14, 16],
    "ema_len": [30, 40, 50, 60],
    "vol_len": [15, 20, 25],
    "vol_mult": [1.0, 1.1, 1.2],
    "atr_len": [10, 14, 20],
    "atr_mult": [2.0, 2.5, 3.0],
}

DESCRIPTION = (
    "ROC momentum with EMA trend filter and volume confirmation. "
    "Enters when ROC > 0 (positive momentum), price above EMA (uptrend), "
    "and volume above average (confirmation). "
    "Exits via ATR trailing stop (rolling high minus ATR * multiplier). "
    "Designed for SPY's smooth momentum characteristics."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    volume = df["volume"]

    # Momentum: Rate of Change
    roc = cached_indicator("roc", close, length=p["roc_len"])
    momentum_ok = roc > 0

    # Trend filter: Price above EMA
    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    # Volume confirmation: Above average
    vol_avg = cached_indicator("sma", volume, length=p["vol_len"])
    volume_ok = volume > (vol_avg * p["vol_mult"])

    # Entry: All three conditions
    entries = (momentum_ok & trend_ok & volume_ok).fillna(False)

    # Exit: ATR trailing stop
    atr = cached_indicator("atr", high, df["low"], close, length=p["atr_len"])
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
    volume = df["volume"]

    # Deduplicate indicator calculations
    all_roc_lens = sorted(set(pg["roc_len"]))
    all_ema_lens = sorted(set(pg["ema_len"]))
    all_vol_lens = sorted(set(pg["vol_len"]))
    all_atr_lens = sorted(set(pg["atr_len"]))

    roc_cache = {l: cached_indicator("roc", close, length=l) for l in all_roc_lens}
    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_ema_lens}
    vol_avg_cache = {l: cached_indicator("sma", volume, length=l) for l in all_vol_lens}
    atr_cache = {l: cached_indicator("atr", high, df["low"], close, length=l) for l in all_atr_lens}
    rolling_high_cache = {l: close.rolling(window=l, min_periods=1).max() for l in all_atr_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))

        roc = roc_cache[params["roc_len"]]
        momentum_ok = roc > 0

        ema = ema_cache[params["ema_len"]]
        trend_ok = close > ema

        vol_avg = vol_avg_cache[params["vol_len"]]
        volume_ok = volume > (vol_avg * params["vol_mult"])

        e = (momentum_ok & trend_ok & volume_ok).fillna(False)

        atr = atr_cache[params["atr_len"]]
        rolling_high = rolling_high_cache[params["atr_len"]]
        trailing_stop = rolling_high - (atr * params["atr_mult"])
        x = (close < trailing_stop).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
