"""
Multi-RSI Confluence Strategy

Multiple timeframe RSI confluence with volume confirmation.
Enters when all three RSIs are oversold and the fastest starts turning up.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "rsi1": 7,
    "rsi2": 14,
    "rsi3": 28,
    "thresh": 35,
    "vol_mult": 1.0,
    "exit_thresh": 65,
}

PARAM_GRID = {
    "rsi1": [5, 7, 10],
    "rsi2": [14, 21],
    "rsi3": [28, 35],
    "thresh": [30, 35, 40],
    "vol_mult": [0.8, 1.0, 1.2],
    "exit_thresh": [60, 65, 70],
}

DESCRIPTION = (
    "Multiple timeframe RSI confluence with volume confirmation. "
    "Enters when RSI-7, RSI-14, and RSI-28 are all below threshold "
    "and the fastest RSI starts turning up. "
    "Exits when fastest RSI recovers above exit threshold. "
    "Designed for catching deep pullback reversals in uptrends."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    rsi1 = cached_indicator("rsi", close, length=p["rsi1"])
    rsi2 = cached_indicator("rsi", close, length=p["rsi2"])
    rsi3 = cached_indicator("rsi", close, length=p["rsi3"])

    all_oversold = (rsi1 < p["thresh"]) & (rsi2 < p["thresh"]) & (rsi3 < p["thresh"])
    rsi_turning = rsi1 > rsi1.shift(1)
    vol_avg = volume.rolling(20).mean()
    vol_confirm = volume > vol_avg * p["vol_mult"]

    entries = (all_oversold & rsi_turning & vol_confirm).fillna(False)
    exits = (rsi1 > p["exit_thresh"]).fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Generate signals for ALL param combinations at once (vectorized)."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    close = df["close"]
    volume = df["volume"]

    # Deduplicate RSI lengths
    all_rsi_lens = sorted(set(pg["rsi1"]) | set(pg["rsi2"]) | set(pg["rsi3"]))
    rsi_cache = {l: cached_indicator("rsi", close, length=l) for l in all_rsi_lens}
    vol_avg_20 = volume.rolling(20).mean()

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        rsi1 = rsi_cache[params["rsi1"]]
        rsi2 = rsi_cache[params["rsi2"]]
        rsi3 = rsi_cache[params["rsi3"]]

        all_oversold = (rsi1 < params["thresh"]) & (rsi2 < params["thresh"]) & (rsi3 < params["thresh"])
        rsi_turning = rsi1 > rsi1.shift(1)
        vol_confirm = volume > vol_avg_20 * params["vol_mult"]

        e = (all_oversold & rsi_turning & vol_confirm).fillna(False)
        x = (rsi1 > params["exit_thresh"]).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
