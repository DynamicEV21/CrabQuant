"""
Broad Market Breakout Strategy

Donchian Channel breakout with ATR trailing stops for index ETFs.
Optimized for SPY/QQQ/IWM smooth trend capture.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "dc_len": 15,
    "atr_len": 14,
    "atr_mult": 2.5,
    "vol_len": 20,
    "vol_mult": 1.3,
    "trend_ema": 50,
}

DESCRIPTION = (
    "Donchian Channel breakout with ATR trailing stops for index ETFs. "
    "Enters when price breaks above N-day high with volume confirmation "
    "and price above 50 EMA (trend filter). "
    "Exits via ATR trailing stop from rolling high. "
    "Designed for SPY/QQQ/IWM smooth trend capture."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    donch_high = high.rolling(p["dc_len"]).max()
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    trend_ema = cached_indicator("ema", close, length=p["trend_ema"])

    breakout = close > donch_high.shift(1)
    vol_confirm = volume > volume.rolling(p["vol_len"]).mean() * p["vol_mult"]
    trend_filter = close > trend_ema

    entries = (breakout & vol_confirm & trend_filter).fillna(False)

    rolling_high = close.rolling(p["dc_len"]).max()
    trailing_stop = rolling_high - atr * p["atr_mult"]
    exits = (close < trailing_stop).fillna(False)

    return entries, exits