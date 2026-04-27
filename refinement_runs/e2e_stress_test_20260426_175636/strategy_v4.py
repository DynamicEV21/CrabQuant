"""
RSI Regime Dip Strategy for SPY

Dual RSI timeframe approach avoiding volume filter issues:
- Long RSI (21) determines bullish regime (above threshold)
- Short RSI (7) times pullback entries (below dip level)
- Exits when short RSI recovers above exit level

Proven pattern: rsi_regime_dip (Sharpe 2.18 in QF backtests).
Adapted for SPY by removing volume dependency entirely.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "rsi_long": 21,
    "rsi_short": 7,
    "regime_threshold": 50,
    "dip_level": 40,
    "exit_level": 65,
}

PARAM_GRID = {
    "rsi_long": [14, 21, 28],
    "rsi_short": [5, 7, 10],
    "regime_threshold": [45, 50, 55],
    "dip_level": [30, 35, 40],
    "exit_level": [55, 60, 65, 70],
}

DESCRIPTION = (
    "RSI regime dip for SPY. "
    "Long RSI determines bullish regime, short RSI times dip entries. "
    "Exits when short RSI recovers above exit level. "
    "No volume filter — avoids SPY liquidity noise."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    rsi_long = cached_indicator("rsi", close, length=p["rsi_long"])
    rsi_short = cached_indicator("rsi", close, length=p["rsi_short"])

    bullish_regime = rsi_long > p["regime_threshold"]
    dip = rsi_short < p["dip_level"]

    entries = (bullish_regime & dip).fillna(False)

    recovered = rsi_short > p["exit_level"]
    exits = recovered.fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    close = df["close"]

    all_rsi_long = sorted(set(pg["rsi_long"]))
    all_rsi_short = sorted(set(pg["rsi_short"]))

    rsi_long_cache = {l: cached_indicator("rsi", close, length=l) for l in all_rsi_long}
    rsi_short_cache = {l: cached_indicator("rsi", close, length=l) for l in all_rsi_short}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        rsi_long = rsi_long_cache[params["rsi_long"]]
        rsi_short = rsi_short_cache[params["rsi_short"]]

        bullish_regime = rsi_long > params["regime_threshold"]
        dip = rsi_short < params["dip_level"]

        e = (bullish_regime & dip).fillna(False)

        recovered = rsi_short > params["exit_level"]
        x = recovered.fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
