"""
ATR Channel Breakout Strategy

Keltner Channel (EMA + ATR) breakout with volume and trend confirmation.
Best performer: ORCL (Sharpe 1.59, 84% return).
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "ema_len": 20,
    "atr_len": 10,
    "mult": 2.0,
    "vol_mult": 1.2,
}

PARAM_GRID = {
    "ema_len": [15, 20],
    "atr_len": [10, 14],
    "mult": [1.5, 2.0, 2.5],
    "vol_mult": [1.0, 1.2, 1.5],
}

DESCRIPTION = (
    "ATR-based Keltner Channel breakout with volume and trend confirmation. "
    "Enters when price breaks above upper channel with above-average volume, "
    "and price is above 50 EMA (trend filter). "
    "Exits when price falls back below channel midpoint (EMA). "
    "Good for catching momentum breakouts from consolidation."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema = cached_indicator("ema", close, length=p["ema_len"])
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])

    upper = ema + atr * p["mult"]

    # Breakout above upper channel (prior bar)
    breakout = close > upper.shift(1)
    # Volume confirmation
    vol_confirm = volume > volume.rolling(20).mean() * p["vol_mult"]
    # Trend filter
    trend = close > cached_indicator("ema", close, length=50)

    entries = (breakout & vol_confirm & trend).fillna(False)
    exits = (close < ema).fillna(False)

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

    # Deduplicate
    all_ema_lens = sorted(set(pg["ema_len"]))
    all_atr_lens = sorted(set(pg["atr_len"]))
    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_ema_lens}
    atr_cache = {l: cached_indicator("atr", high, low, close, length=l) for l in all_atr_lens}
    trend_ema = cached_indicator("ema", close, length=50)
    vol_avg_20 = volume.rolling(20).mean()

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        ema_val = ema_cache[params["ema_len"]]
        atr_val = atr_cache[params["atr_len"]]

        upper = ema_val + atr_val * params["mult"]
        breakout = close > upper.shift(1)
        vol_confirm = volume > vol_avg_20 * params["vol_mult"]
        trend = close > trend_ema

        e = (breakout & vol_confirm & trend).fillna(False)
        x = (close < ema_val).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
