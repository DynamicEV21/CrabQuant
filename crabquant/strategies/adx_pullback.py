"""
ADX Trend Pullback Strategy

ADX trend strength + pullback to EMA entry.
Best performer: NFLX (Sharpe 2.09, 105% return, only 14.2% drawdown).
Most efficient risk-adjusted strategy found.
"""

from itertools import product

import pandas as pd
import pandas_ta


DEFAULT_PARAMS = {
    "adx_len": 14,
    "adx_threshold": 25,
    "ema_len": 20,
    "take_atr": 3,
}

PARAM_GRID = {
    "adx_len": [12, 14, 20],
    "adx_threshold": [20, 25, 30],
    "ema_len": [15, 20, 25],
    "take_atr": [2, 3, 4],
}

DESCRIPTION = (
    "ADX trend strength with pullback to EMA entry. "
    "Enters when ADX confirms strong trend and price pulls back to EMA. "
    "Exits when price reaches take-profit (EMA + ATR multiplier). "
    "Best risk-adjusted returns — NFLX achieved Sharpe 2.09 with only 14.2% drawdown."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    adx = pandas_ta.adx(high, low, close, length=p["adx_len"])
    adx_col = [c for c in adx.columns if "ADX" in c and "DI" not in c][0]

    ema = pandas_ta.ema(close, length=p["ema_len"])
    atr = pandas_ta.atr(high, low, close, length=14)

    # Strong trend
    strong_trend = adx[adx_col] > p["adx_threshold"]
    # Pullback to EMA (price crosses below EMA)
    pullback = (close < ema) & (close.shift(1) >= ema.shift(1))

    entries = (strong_trend & pullback).fillna(False)
    exits = (close > ema + atr * p["take_atr"]).fillna(False)

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

    # Deduplicate indicators
    all_adx_lens = sorted(set(pg["adx_len"]))
    all_ema_lens = sorted(set(pg["ema_len"]))

    adx_cache = {}
    for l in all_adx_lens:
        adx = pandas_ta.adx(high, low, close, length=l)
        adx_col = [c for c in adx.columns if "ADX" in c and "DI" not in c][0]
        adx_cache[l] = adx[adx_col]

    ema_cache = {l: pandas_ta.ema(close, length=l) for l in all_ema_lens}
    atr = pandas_ta.atr(high, low, close, length=14)

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        adx_val = adx_cache[params["adx_len"]]
        ema_val = ema_cache[params["ema_len"]]

        strong_trend = adx_val > params["adx_threshold"]
        pullback = (close < ema_val) & (close.shift(1) >= ema_val.shift(1))

        e = (strong_trend & pullback).fillna(False)
        x = (close > ema_val + atr * params["take_atr"]).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
