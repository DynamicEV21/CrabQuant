"""
RSI Regime Dip Strategy with Volume Confirmation for SPY

Uses long RSI as regime filter (bullish when above threshold).
Enters on short RSI dip recovery in bullish regime with volume confirmation.
Exits when short RSI recovers above exit level or regime turns bearish.
Designed for SPY's pattern of trending with regular pullbacks.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "rsi_long": 50,
    "rsi_short": 7,
    "regime_threshold": 50,
    "dip_level": 30,
    "exit_level": 55,
    "volume_window": 20,
    "volume_mult": 0.9,
}

PARAM_GRID = {
    "rsi_long": [30, 40, 50, 60],
    "rsi_short": [5, 7, 10, 14],
    "regime_threshold": [45, 50, 55],
    "dip_level": [25, 30, 35],
    "exit_level": [55, 60, 65, 70],
    "volume_window": [15, 20, 30],
    "volume_mult": [0.8, 0.9, 1.0],
}

DESCRIPTION = (
    "RSI regime dip strategy with volume confirmation. "
    "Long RSI determines market regime (bullish when above threshold). "
    "Short RSI times entry on dip recovery in bullish regime. "
    "Volume above average confirms genuine buying opportunity. "
    "Exits when short RSI recovers above exit level or regime turns bearish. "
    "Designed for SPY's trending pattern with regular pullbacks."
)


def _valid_combos(param_grid: dict) -> list[dict]:
    """Filter param combos where rsi_short < rsi_long and dip_level < exit_level."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    valid = []
    for vals in product(*(pg[k] for k in keys)):
        params = dict(zip(keys, vals))
        if params["rsi_short"] < params["rsi_long"] and params["dip_level"] < params["exit_level"]:
            valid.append(params)
    return valid


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Generate entry/exit signals.

    Args:
        df: DataFrame with columns open, high, low, close, volume
        params: Strategy parameters (uses defaults if None)

    Returns:
        (entries, exits) as boolean Series
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    # Regime detection: long RSI above threshold = bullish
    rsi_long = cached_indicator("rsi", close, length=p["rsi_long"])
    regime_bullish = rsi_long > p["regime_threshold"]

    # Entry timing: short RSI recovers from dip (crosses back above dip_level)
    rsi_short = cached_indicator("rsi", close, length=p["rsi_short"])
    rsi_short_prev = rsi_short.shift(1)
    dip_recovery = (rsi_short_prev < p["dip_level"]) & (rsi_short >= p["dip_level"])

    # Volume confirmation
    volume_avg = cached_indicator("sma", volume, length=p["volume_window"])
    volume_ok = volume > (volume_avg * p["volume_mult"])

    # Entry: bullish regime + dip recovery with volume
    entries = (regime_bullish & dip_recovery & volume_ok).fillna(False)

    # Exit: RSI recovers above exit level OR regime turns bearish
    rsi_recovered = rsi_short > p["exit_level"]
    regime_bearish = rsi_long < p["regime_threshold"]
    exits = (rsi_recovered | regime_bearish).fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Generate signals for ALL param combinations at once (vectorized).
    Only includes combos where rsi_short < rsi_long and dip_level < exit_level.

    Returns:
        (entries_df, exits_df, param_list) where each DataFrame has one column per combo.
    """
    pg = param_grid or PARAM_GRID
    valid_combos = _valid_combos(pg)

    close = df["close"]
    volume = df["volume"]

    # Deduplicate indicator calculations
    all_rsi_lens = sorted(set(
        c["rsi_long"] for c in valid_combos
    ) | set(
        c["rsi_short"] for c in valid_combos
    ))
    all_vol_windows = sorted(set(c["volume_window"] for c in valid_combos))

    rsi_cache = {l: cached_indicator("rsi", close, length=l) for l in all_rsi_lens}
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_windows}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, params in enumerate(valid_combos):
        rsi_long = rsi_cache[params["rsi_long"]]
        rsi_short = rsi_cache[params["rsi_short"]]
        volume_avg = vol_avg_cache[params["volume_window"]]

        regime_bullish = rsi_long > params["regime_threshold"]
        rsi_short_prev = rsi_short.shift(1)
        dip_recovery = (rsi_short_prev < params["dip_level"]) & (rsi_short >= params["dip_level"])
        volume_ok = volume > (volume_avg * params["volume_mult"])

        e = (regime_bullish & dip_recovery & volume_ok).fillna(False)

        rsi_recovered = rsi_short > params["exit_level"]
        regime_bearish = rsi_long < params["regime_threshold"]
        x = (rsi_recovered | regime_bearish).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
