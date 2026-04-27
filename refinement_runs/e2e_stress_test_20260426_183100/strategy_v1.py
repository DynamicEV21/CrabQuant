"""
ROC EMA Volume Momentum Strategy for SPY

Momentum strategy with volume confirmation and ATR trailing stops.
Entry: ROC positive + price above EMA + volume above average
Exit: ATR trailing stop (rolling high minus ATR * multiplier)

Optimized for SPY with controlled drawdown via ATR stops.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 12,
    "ema_len": 50,
    "volume_len": 20,
    "volume_mult": 1.0,
    "atr_len": 14,
    "atr_mult": 2.0,
    "roc_threshold": 0,
}

PARAM_GRID = {
    "roc_len": [8, 10, 12, 14, 16],
    "ema_len": [30, 50, 75, 100],
    "volume_len": [15, 20, 25],
    "volume_mult": [0.8, 1.0, 1.2],
    "atr_len": [10, 14, 20],
    "atr_mult": [1.5, 2.0, 2.5, 3.0],
    "roc_threshold": [-0.5, 0, 0.5],
}

DESCRIPTION = (
    "ROC EMA Volume momentum strategy for SPY. "
    "Enters when ROC is positive, price above EMA, and volume above average. "
    "Exits via ATR trailing stop (rolling high minus ATR * multiplier). "
    "Volume confirmation filters false breakouts. "
    "ATR trailing stops control drawdown."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ROC calculation
    roc = cached_indicator("roc", close, length=p["roc_len"])

    # Trend filter - EMA
    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    # Volume confirmation
    volume_avg = cached_indicator("sma", volume, length=p["volume_len"])
    volume_ok = volume > (volume_avg * p["volume_mult"])

    # Momentum condition
    momentum_ok = roc > p["roc_threshold"]

    # Entry: all conditions met
    entries = (trend_ok & volume_ok & momentum_ok).fillna(False)

    # ATR trailing stop
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    rolling_high = close.rolling(window=p["atr_len"]).max()
    trailing_stop = rolling_high - (atr * p["atr_mult"])

    # Exit when price falls below trailing stop
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
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_lens}
    atr_cache = {l: cached_indicator("atr", high, low, close, length=l) for l in all_atr_lens}
    rolling_high_cache = {l: close.rolling(window=l).max() for l in all_atr_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))

        roc = roc_cache[params["roc_len"]]
        ema = ema_cache[params["ema_len"]]
        volume_avg = vol_avg_cache[params["volume_len"]]
        atr = atr_cache[params["atr_len"]]
        rolling_high = rolling_high_cache[params["atr_len"]]

        trend_ok = close > ema
        volume_ok = volume > (volume_avg * params["volume_mult"])
        momentum_ok = roc > params["roc_threshold"]

        e = (trend_ok & volume_ok & momentum_ok).fillna(False)

        trailing_stop = rolling_high - (atr * params["atr_mult"])
        x = (close < trailing_stop).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list