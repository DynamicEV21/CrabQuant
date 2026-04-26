"""
Momentum RSI ATR Confluence Strategy

Combines momentum detection (ROC), RSI pullback entry in uptrends,
and ATR-based trailing stop exits.

Entry logic:
- Price above 50 EMA (trend filter)
- ROC(14) > 0 (positive momentum)
- RSI crossed above pullback threshold from below (pullback recovery)

Exit logic:
- ATR trailing stop: exit when close falls below (EMA - atr_mult * ATR)
- Or RSI exceeds overbought level (take profit on extended moves)
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator, clear_cache


DEFAULT_PARAMS = {
    "rsi_len": 14,
    "rsi_pullback": 45,
    "rsi_overbought": 78,
    "roc_len": 14,
    "roc_threshold": 0,
    "ema_len": 50,
    "atr_len": 14,
    "atr_exit_mult": 3.0,
}

PARAM_GRID = {
    "rsi_len": [10, 14, 21],
    "rsi_pullback": [40, 45, 50],
    "rsi_overbought": [75, 78, 82],
    "roc_len": [10, 14, 21],
    "roc_threshold": [-1, 0, 1],
    "ema_len": [30, 50, 70],
    "atr_len": [10, 14, 21],
    "atr_exit_mult": [2.0, 2.5, 3.0, 3.5],
}

DESCRIPTION = (
    "Momentum RSI ATR confluence strategy. "
    "Enters on RSI recovery from pullback in a confirmed uptrend "
    "(price > EMA, ROC > 0). "
    "Exits on ATR-based trailing stop or RSI overbought. "
    "Designed for trending stocks with regular pullbacks — targets "
    "momentum continuation after healthy dips."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Indicators
    rsi = pandas_ta.rsi(close, length=p["rsi_len"])
    roc = pandas_ta.roc(close, length=p["roc_len"])
    ema = pandas_ta.ema(close, length=p["ema_len"])
    atr = pandas_ta.atr(high, low, close, length=p["atr_len"])
    adx = pandas_ta.adx(high, low, close, length=14)
    adx_col = [c for c in adx.columns if "ADX" in c and "DI" not in c][0]

    # Entry conditions
    trend_ok = close > ema
    momentum_ok = roc > p["roc_threshold"]
    adx_ok = adx[adx_col] > 20

    # RSI pullback: dipped below pullback level, now recovering
    rsi_turning_up = (rsi.shift(1) < p["rsi_pullback"]) & (rsi >= p["rsi_pullback"])
    rsi_rising = rsi > rsi.shift(1)
    rsi_in_zone = rsi.between(p["rsi_pullback"], 60)
    rsi_pullback_entry = rsi_turning_up | (rsi_in_zone & rsi_rising & (rsi.shift(2) < p["rsi_pullback"]))

    entries = (trend_ok & momentum_ok & adx_ok & rsi_pullback_entry).fillna(False)

    # Exit conditions
    atr_stop = ema - (p["atr_exit_mult"] * atr)
    atr_exit = close < atr_stop
    rsi_exit = rsi > p["rsi_overbought"]
    exits = (atr_exit | rsi_exit).fillna(False)

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

    # Deduplicate indicators (cached across calls)
    all_rsi_lens = sorted(set(pg["rsi_len"]))
    all_roc_lens = sorted(set(pg["roc_len"]))
    all_ema_lens = sorted(set(pg["ema_len"]))
    all_atr_lens = sorted(set(pg["atr_len"]))

    rsi_cache = {l: cached_indicator("rsi", close, length=l) for l in all_rsi_lens}
    roc_cache = {l: cached_indicator("roc", close, length=l) for l in all_roc_lens}
    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_ema_lens}
    atr_cache = {l: cached_indicator("atr", high, low, close, length=l) for l in all_atr_lens}
    adx = cached_indicator("adx", high, low, close, length=14)
    adx_col = [c for c in adx.columns if "ADX" in c and "DI" not in c][0]
    adx_val = adx[adx_col]

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        rsi = rsi_cache[params["rsi_len"]]
        roc = roc_cache[params["roc_len"]]
        ema = ema_cache[params["ema_len"]]
        atr = atr_cache[params["atr_len"]]

        trend_ok = close > ema
        momentum_ok = roc > params["roc_threshold"]
        adx_ok = adx_val > 20

        rsi_turning_up = (rsi.shift(1) < params["rsi_pullback"]) & (rsi >= params["rsi_pullback"])
        rsi_rising = rsi > rsi.shift(1)
        rsi_in_zone = rsi.between(params["rsi_pullback"], 60)
        rsi_pullback_entry = rsi_turning_up | (rsi_in_zone & rsi_rising & (rsi.shift(2) < params["rsi_pullback"]))

        e = (trend_ok & momentum_ok & adx_ok & rsi_pullback_entry).fillna(False)

        atr_stop = ema - (params["atr_exit_mult"] * atr)
        atr_exit = close < atr_stop
        rsi_exit = rsi > params["rsi_overbought"]
        x = (atr_exit | rsi_exit).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
