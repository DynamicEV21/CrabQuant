"""
Momentum Volume Strategy for SPY

Simple momentum strategy with volume confirmation.
- Entry: ROC crosses above zero (momentum shift) + price above EMA (trend) + volume spike
- Exit: ROC crosses below zero (momentum loss)
"""

from itertools import product

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 10,
    "ema_len": 50,
    "vol_len": 20,
    "vol_mult": 0.8,
}

PARAM_GRID = {
    "roc_len": [5, 10, 15, 20],
    "ema_len": [20, 50, 100],
    "vol_len": [10, 20, 30],
    "vol_mult": [0.5, 0.8, 1.0, 1.2],
}

DESCRIPTION = (
    "Momentum strategy with volume confirmation for SPY. "
    "Enters when ROC crosses above zero (momentum shift), price is above EMA (trend), "
    "and volume is above average. Exits when ROC crosses below zero. "
    "Simple and robust for trending markets."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    # ROC for momentum detection
    roc = cached_indicator("roc", close, length=p["roc_len"])

    # EMA for trend filter
    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    # Volume confirmation
    vol_avg = cached_indicator("sma", volume, length=p["vol_len"])
    vol_ok = volume > (vol_avg * p["vol_mult"])

    # Entry: ROC crosses above 0 + trend + volume
    roc_prev = roc.shift(1)
    entries = (roc_prev <= 0) & (roc > 0) & trend_ok & vol_ok

    # Exit: ROC crosses below 0
    exits = (roc_prev >= 0) & (roc < 0)

    return entries.fillna(False), exits.fillna(False)


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Generate signals for ALL param combinations at once (vectorized)."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    close = df["close"]
    volume = df["volume"]

    # Deduplicate indicator calculations
    all_roc_lens = sorted(set(pg["roc_len"]))
    all_ema_lens = sorted(set(pg["ema_len"]))
    all_vol_lens = sorted(set(pg["vol_len"]))

    roc_cache = {l: cached_indicator("roc", close, length=l) for l in all_roc_lens}
    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_ema_lens}
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        roc = roc_cache[params["roc_len"]]
        ema = ema_cache[params["ema_len"]]
        vol_avg = vol_avg_cache[params["vol_len"]]

        trend_ok = close > ema
        vol_ok = volume > (vol_avg * params["vol_mult"])
        roc_prev = roc.shift(1)

        e = ((roc_prev <= 0) & (roc > 0) & trend_ok & vol_ok).fillna(False)
        x = ((roc_prev >= 0) & (roc < 0)).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list