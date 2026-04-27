"""
High Beta Semiconductor Breakout Strategy

ATR Keltner Channel breakout with momentum and volume confirmation.
Tight ATR trailing stops for risk management on volatile semis.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_len": 15,
    "atr_len": 10,
    "channel_mult": 1.8,
    "vol_mult": 1.3,
    "roc_len": 10,
    "trail_mult": 2.5,
    "trend_ema": 50,
}

DESCRIPTION = (
    "High beta breakout for semiconductors using ATR Keltner Channel breakout "
    "with ROC momentum and volume confirmation. Enters when price breaks above "
    "upper channel with positive ROC and volume spike. Uses tight ATR trailing "
    "stop (2.5x ATR) for risk management. Optimized for AMD's explosive moves."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema = cached_indicator("ema", close, length=p["ema_len"])
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    upper = ema + atr * p["channel_mult"]

    breakout = close > upper.shift(1)
    roc = close.pct_change(p["roc_len"])
    momentum = roc > 0
    vol_confirm = volume > volume.rolling(20).mean() * p["vol_mult"]
    trend = close > cached_indicator("ema", close, length=p["trend_ema"])

    entries = (breakout & momentum & vol_confirm & trend).fillna(False)

    rolling_high = close.rolling(10).max()
    trailing_stop = rolling_high - atr * p["trail_mult"]
    exits = (close < trailing_stop).fillna(False)

    return entries, exits