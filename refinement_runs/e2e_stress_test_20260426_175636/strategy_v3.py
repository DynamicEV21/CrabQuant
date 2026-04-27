"""
ROC + EMA + Volume Momentum Strategy for SPY

Adapted from roc_ema_volume (Sharpe 2.21) with SPY-specific tuning:
- Lower volume multiplier (0.9) since SPY always has high volume
- Dual exit: ROC turns negative OR price crosses below EMA
- Moderate lookback periods suited to SPY's trend cadence
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 14,
    "ema_len": 50,
    "volume_window": 20,
    "volume_mult": 0.9,
}

PARAM_GRID = {
    "roc_len": [5, 10, 14, 21],
    "ema_len": [20, 50, 100],
    "volume_window": [10, 20, 30],
    "volume_mult": [0.8, 0.9, 1.0],
}

DESCRIPTION = (
    "ROC + EMA + Volume momentum for SPY. "
    "Enters when ROC is positive, price above EMA, volume above avg. "
    "Exits when ROC turns negative or price falls below EMA. "
    "Adapted for SPY with permissive volume filter."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    roc = cached_indicator("roc", close, length=p["roc_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    vol_avg = cached_indicator("sma", volume, length=p["volume_window"])

    roc_positive = roc > 0
    above_ema = close > ema
    vol_ok = volume > (vol_avg * p["volume_mult"])

    entries = (roc_positive & above_ema & vol_ok).fillna(False)

    roc_negative = roc < 0
    below_ema = close < ema
    exits = (roc_negative | below_ema).fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    close = df["close"]
    volume = df["volume"]

    all_roc_lens = sorted(set(pg["roc_len"]))
    all_ema_lens = sorted(set(pg["ema_len"]))
    all_vol_windows = sorted(set(pg["volume_window"]))

    roc_cache = {l: cached_indicator("roc", close, length=l) for l in all_roc_lens}
    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_ema_lens}
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_windows}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        roc = roc_cache[params["roc_len"]]
        ema = ema_cache[params["ema_len"]]
        vol_avg = vol_avg_cache[params["volume_window"]]

        roc_positive = roc > 0
        above_ema = close > ema
        vol_ok = volume > (vol_avg * params["volume_mult"])

        e = (roc_positive & above_ema & vol_ok).fillna(False)

        roc_negative = roc < 0
        below_ema = close < ema
        x = (roc_negative | below_ema).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
