"""
Informed Adaptive Trend Reversion Strategy

Combines trend detection with mean reversion using volatility-adjusted parameters.
Adapts to different market regimes using ADX for trend strength and ATR for volatility normalization.
Targets AAPL, NVDA, CAT, SPY with regime-specific entry/exit logic.

Best for stocks with mixed characteristics (trending vs mean-reverting).
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
    "ema_len": 50,
    "atr_len": 14,
    "atr_mult": 2.0,
    "volume_window": 20,
    "volume_mult": 1.3,
}

PARAM_GRID = {
    "adx_len": [14, 21],
    "adx_threshold": [20, 25, 30],
    "rsi_len": [10, 14, 21],
    "rsi_oversold": [25, 30, 35],
    "rsi_overbought": [60, 65, 70],
    "ema_len": [30, 50, 70],
    "atr_len": [14, 21],
    "atr_mult": [1.5, 2.0, 2.5],
    "volume_window": [15, 20, 25],
    "volume_mult": [1.2, 1.3, 1.5],
}

DESCRIPTION = (
    "Adaptive strategy that detects market regime (trending vs mean-reverting) "
    "using ADX, then applies appropriate entry logic with ATR-based stops. "
    "For trending markets: pullbacks to EMA with volume confirmation. "
    "For mean-reverting markets: RSI extremes with ATR stop. "
    "Designed for mixed-regime stocks like AAPL, NVDA, CAT, SPY."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Generate entry/exit signals based on market regime.

    Args:
        df: DataFrame with columns open, high, low, close, volume
        params: Strategy parameters (uses defaults if None)

    Returns:
        (entries, exits) as boolean Series
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Calculate indicators using cached_indicator
    adx = cached_indicator("adx", high, low, close, length=p["adx_len"])
    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    volume_avg = cached_indicator("sma", volume, length=p["volume_window"])
    
    # Volume indicators
    volume_spike = volume > (volume_avg * p["volume_mult"])

    # Market regime detection
    trending = adx > p["adx_threshold"]
    mean_reverting = ~trending  # Inverse of trending

    # Trending regime logic (for NVDA, strong trends)
    trending_entries = (
        (close < ema) &  # Pullback to EMA
        (rsi < p["rsi_overbought"]) &  # Not overbought
        volume_spike &  # Volume confirmation
        trending
    ).fillna(False)

    # Mean-reverting regime logic (for AAPL, CAT, SPY)
    mean_rev_entries = (
        ((rsi < p["rsi_oversold"]) | (rsi > p["rsi_overbought"])) &  # RSI extremes
        volume_spike &  # Volume confirmation
        mean_reverting
    ).fillna(False)

    # Combined entries
    entries = trending_entries | mean_rev_entries

    # Exits: ATR-based stop loss + take profit
    current_atr_stop = close - (atr * p["atr_mult"])
    current_atr_target = close + (atr * p["atr_mult"])

    exits = (
        (close <= current_atr_stop) |  # ATR stop loss
        (close >= current_atr_target) |  # ATR take profit
        (trending & (close > ema * 1.02))  # Trend break (for regime switch)
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
    all_ema_lens = sorted(set(pg["ema_len"]))
    all_atr_lens = sorted(set(pg["atr_len"]))
    all_vol_windows = sorted(set(pg["volume_window"]))

    # Cache all indicators
    adx_cache = {}
    for l in all_adx_lens:
        adx_cache[l] = cached_indicator("adx", high, low, close, length=l)
    
    rsi_cache = {l: cached_indicator("rsi", close, length=l) for l in all_rsi_lens}
    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_ema_lens}
    atr_cache = {l: cached_indicator("atr", high, low, close, length=l) for l in all_atr_lens}
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_windows}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        
        # Get cached indicators
        adx = adx_cache[params["adx_len"]]
        rsi = rsi_cache[params["rsi_len"]]
        ema = ema_cache[params["ema_len"]]
        atr = atr_cache[params["atr_len"]]
        volume_avg = vol_avg_cache[params["volume_window"]]
        volume_spike = volume > (volume_avg * params["volume_mult"])

        # Market regime detection
        trending = adx > params["adx_threshold"]
        mean_reverting = ~trending

        # Trending regime entries
        trending_entries = (
            (close < ema) &
            (rsi < params["rsi_overbought"]) &
            volume_spike &
            trending
        )

        # Mean-reverting regime entries  
        mean_rev_entries = (
            ((rsi < params["rsi_oversold"]) | (rsi > params["rsi_overbought"])) &
            volume_spike &
            mean_reverting
        )

        # Combined entries
        entries = trending_entries | mean_rev_entries

        # ATR-based exits
        current_atr_stop = close - (atr * params["atr_mult"])
        current_atr_target = close + (atr * params["atr_mult"])
        exits = (
            (close <= current_atr_stop) |
            (close >= current_atr_target) |
            (trending & (close > ema * 1.02))
        )

        entries_cols[f"c{i}"] = entries.fillna(False)
        exits_cols[f"c{i}"] = exits.fillna(False)
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list