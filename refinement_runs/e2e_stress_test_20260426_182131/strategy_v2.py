"""
RSI Regime Dip + Volume Strategy for SPY

Regime-adaptive dip buying with volume confirmation:
- RSI(long) determines market regime (bullish when > threshold)
- RSI(short) times entry on dips (below dip_level in bullish regime)
- Volume above SMA confirms institutional participation
- Exits on RSI(short) recovery OR regime turning bearish

Designed to avoid regime fragility by sitting out bear markets entirely.
Target Sharpe 1.5+ for SPY over 2-year periods.
"""

from itertools import product

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "rsi_long": 28,
    "rsi_short": 7,
    "regime_threshold": 50,
    "dip_level": 30,
    "exit_level": 55,
    "volume_window": 20,
    "volume_mult": 0.8,
}

PARAM_GRID = {
    "rsi_long": [21, 28, 35],
    "rsi_short": [5, 7, 10],
    "regime_threshold": [45, 50, 55],
    "dip_level": [25, 30, 35],
    "exit_level": [50, 55, 60, 65],
    "volume_window": [15, 20, 25],
    "volume_mult": [0.0, 0.8, 1.0],
}

DESCRIPTION = (
    "RSI regime-filtered dip buying with volume confirmation. "
    "Enters when long-term RSI confirms bullish regime, short-term RSI "
    "detects a pullback, and volume confirms institutional buying. "
    "Exits on RSI recovery or regime shift to bearish. "
    "Designed to avoid regime fragility by not trading in bear markets."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    # Regime detection: long-term RSI above threshold = bullish
    rsi_long = cached_indicator("rsi", close, length=p["rsi_long"])
    bullish_regime = rsi_long > p["regime_threshold"]

    # Dip detection: short-term RSI below dip level
    rsi_short = cached_indicator("rsi", close, length=p["rsi_short"])
    dip_detected = rsi_short < p["dip_level"]

    # Volume confirmation
    volume_avg = cached_indicator("sma", volume, length=p["volume_window"])
    volume_ok = volume > (volume_avg * p["volume_mult"])

    # Entry: bullish regime + dip + volume
    entries = (bullish_regime & dip_detected & volume_ok).fillna(False)

    # Exit: RSI short recovers OR regime turns bearish
    rsi_recovered = rsi_short > p["exit_level"]
    regime_bearish = rsi_long < p["regime_threshold"]
    exits = (rsi_recovered | regime_bearish).fillna(False)

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

    # Deduplicate indicator calculations
    all_rsi_lens = sorted(set(pg["rsi_long"]) | set(pg["rsi_short"]))
    all_vol_windows = sorted(set(pg["volume_window"]))

    rsi_cache = {l: cached_indicator("rsi", close, length=l) for l in all_rsi_lens}
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_windows}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))

        rsi_long = rsi_cache[params["rsi_long"]]
        rsi_short = rsi_cache[params["rsi_short"]]
        volume_avg = vol_avg_cache[params["volume_window"]]

        bullish_regime = rsi_long > params["regime_threshold"]
        dip_detected = rsi_short < params["dip_level"]
        volume_ok = volume > (volume_avg * params["volume_mult"])

        e = (bullish_regime & dip_detected & volume_ok).fillna(False)

        rsi_recovered = rsi_short > params["exit_level"]
        regime_bearish = rsi_long < params["regime_threshold"]
        x = (rsi_recovered | regime_bearish).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
