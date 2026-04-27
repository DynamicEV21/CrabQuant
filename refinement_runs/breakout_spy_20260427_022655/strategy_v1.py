"""
Broad Market Breakout Strategy

Donchian Channel breakout with ATR trailing stops and volume confirmation.
Optimized for index ETFs (SPY, QQQ, IWM).
"""

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "dc_len": 20,
    "atr_len": 14,
    "vol_len": 20,
    "vol_mult": 1.3,
    "trend_ema": 50,
    "trail_mult": 2.5,
}


DESCRIPTION = (
    "Broad market breakout using Donchian Channel with volume confirmation. "
    "Enters when price breaks above N-day high with volume spike and price "
    "above 50 EMA trend filter. Exits on ATR trailing stop from rolling high. "
    "Designed for index ETFs like SPY, QQQ, IWM with smoother trends."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Donchian Channel
    dc_high = high.rolling(p["dc_len"]).max()

    # ATR for trailing stop
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])

    # Volume confirmation
    vol_avg = volume.rolling(p["vol_len"]).mean()
    vol_spike = volume > vol_avg * p["vol_mult"]

    # Trend filter
    trend_ema = cached_indicator("ema", close, length=p["trend_ema"])
    in_uptrend = close > trend_ema

    # Breakout: close above prior bar's channel high
    breakout = close > dc_high.shift(1)

    entries = (breakout & vol_spike & in_uptrend).fillna(False)

    # ATR trailing stop from rolling high
    rolling_high = close.rolling(p["dc_len"], min_periods=1).max()
    trailing_stop = rolling_high - atr * p["trail_mult"]
    exits = (close < trailing_stop).fillna(False)

    return entries, exits