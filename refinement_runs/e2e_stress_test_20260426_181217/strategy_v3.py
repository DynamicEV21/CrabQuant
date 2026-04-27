"""
RSI Regime Dip Strategy for SPY

Uses a long-period RSI to determine market regime (bullish when above threshold).
Uses a short-period RSI to time entries on dips within the bullish regime.
Exits when short RSI recovers above exit level.

Designed for SPY's tendency to have healthy pullbacks within uptrends.
Removes volume filter that was killing signals in previous MACD strategy.
"""

from itertools import product

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "rsi_long": 50,
    "rsi_short": 7,
    "regime_threshold": 50,
    "dip_level": 35,
    "exit_level": 65,
}

PARAM_GRID = {
    "rsi_long": [30, 40, 50, 60],
    "rsi_short": [5, 7, 10, 14],
    "regime_threshold": [45, 50, 55],
    "dip_level": [25, 30, 35, 40],
    "exit_level": [55, 60, 65, 70],
}

DESCRIPTION = (
    "RSI regime filter with dip timing for SPY. "
    "Long RSI determines bullish regime (above threshold). "
    "Short RSI times entry when it crosses up from oversold dip level. "
    "Exits when short RSI recovers above exit level. "
    "Designed to catch pullback reversals in SPY's smooth uptrends."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    # RSI calculations
    rsi_long = cached_indicator("rsi", close, length=p["rsi_long"])
    rsi_short = cached_indicator("rsi", close, length=p["rsi_short"])

    rsi_short_prev = rsi_short.shift(1)

    # Regime filter: long RSI above threshold = bullish
    regime_bullish = rsi_long > p["regime_threshold"]

    # Entry: short RSI crosses up from below dip level in bullish regime
    dip_cross_up = (rsi_short_prev <= p["dip_level"]) & (rsi_short > p["dip_level"])
    entries = (regime_bullish & dip_cross_up).fillna(False)

    # Exit: short RSI rises above exit level
    exits = (rsi_short > p["exit_level"]).fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Generate signals for ALL param combinations at once (vectorized)."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    close = df["close"]

    # Deduplicate RSI lengths
    all_rsi_lens = sorted(set(pg["rsi_long"]) | set(pg["rsi_short"]))
    rsi_cache = {l: cached_indicator("rsi", close, length=l) for l in all_rsi_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        rsi_long = rsi_cache[params["rsi_long"]]
        rsi_short = rsi_cache[params["rsi_short"]]
        rsi_short_prev = rsi_short.shift(1)

        regime_bullish = rsi_long > params["regime_threshold"]
        dip_cross_up = (rsi_short_prev <= params["dip_level"]) & (rsi_short > params["dip_level"])

        e = (regime_bullish & dip_cross_up).fillna(False)
        x = (rsi_short > params["exit_level"]).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
