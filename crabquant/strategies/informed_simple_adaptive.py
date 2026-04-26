"""
Informed Simple Adaptive Strategy

Basic adaptive strategy combining trend detection with mean reversion.
Uses ADX for regime detection and RSI for entries.
Targets AAPL, NVDA, CAT, SPY.
"""

from itertools import product

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "adx_len": 14,
    "adx_threshold": 25,
    "rsi_len": 14,
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "volume_window": 20,
    "volume_mult": 1.3,
}

PARAM_GRID = {
    "adx_len": [14, 21],
    "adx_threshold": [20, 25, 30],
    "rsi_len": [10, 14, 21],
    "rsi_oversold": [25, 30, 35],
    "rsi_overbought": [60, 65, 70],
    "volume_window": [15, 20, 25],
    "volume_mult": [1.2, 1.3, 1.5],
}

DESCRIPTION = (
    "Simple adaptive strategy using ADX to detect market regime. "
    "Entries on RSI extremes in mean-reverting markets, "
    "RSI pullbacks in trending markets with volume confirmation. "
    "Designed for mixed-regime stocks like AAPL, NVDA, CAT, SPY."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Generate entry/exit signals based on market regime.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    # Calculate indicators using cached_indicator
    adx_df = cached_indicator("adx", df["high"], df["low"], close, length=p["adx_len"])
    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    volume_avg = cached_indicator("sma", volume, length=p["volume_window"])
    
    # Extract ADX value from the DataFrame (ADX column)
    adx = adx_df[f'ADX_{p["adx_len"]}']
    
    # Volume indicators
    volume_spike = volume > (volume_avg * p["volume_mult"])

    # Market regime detection
    trending = adx > p["adx_threshold"]
    mean_reverting = ~trending

    # Entry conditions based on regime
    entries = (
        # Mean-reverting: RSI extremes with volume
        (mean_reverting & ((rsi < p["rsi_oversold"]) | (rsi > p["rsi_overbought"])) & volume_spike) |
        # Trending: RSI not overbought with volume (pullback opportunity)
        (trending & (rsi < p["rsi_overbought"]) & volume_spike)
    ).fillna(False)

    # Exit conditions: RSI mean reversion
    exits = (
        # Mean-reverting: RSI returning to neutral
        (mean_reverting & (rsi > 50) & (rsi.shift(1) <= 50)) |
        # Trending: RSI overbought
        (trending & (rsi > p["rsi_overbought"]))
    ).fillna(False)

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

    # Deduplicate and cache indicators
    all_adx_lens = sorted(set(pg["adx_len"]))
    all_rsi_lens = sorted(set(pg["rsi_len"]))
    all_vol_windows = sorted(set(pg["volume_window"]))

    # Cache all indicators
    adx_cache = {}
    for l in all_adx_lens:
        adx_df = cached_indicator("adx", high, low, close, length=l)
        adx_cache[l] = adx_df[f'ADX_{l}']
    
    rsi_cache = {l: cached_indicator("rsi", close, length=l) for l in all_rsi_lens}
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_windows}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        
        # Get cached indicators
        adx = adx_cache[params["adx_len"]]
        rsi = rsi_cache[params["rsi_len"]]
        volume_avg = vol_avg_cache[params["volume_window"]]
        volume_spike = volume > (volume_avg * params["volume_mult"])

        # Market regime detection
        trending = adx > params["adx_threshold"]
        mean_reverting = ~trending

        # Entry conditions
        entries = (
            (mean_reverting & ((rsi < params["rsi_oversold"]) | (rsi > params["rsi_overbought"])) & volume_spike) |
            (trending & (rsi < params["rsi_overbought"]) & volume_spike)
        ).fillna(False)

        # Exit conditions
        exits = (
            (mean_reverting & (rsi > 50) & (rsi.shift(1) <= 50)) |
            (trending & (rsi > params["rsi_overbought"]))
        ).fillna(False)

        entries_cols[f"c{i}"] = entries
        exits_cols[f"c{i}"] = exits
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list