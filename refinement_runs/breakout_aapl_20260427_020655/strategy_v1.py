"""
ATR Donchian Breakout with Volume Confirmation

Breakout entry on Donchian high penetration with volume spike.
ATR trailing stop for dynamic exits. Trend filter via 50 EMA.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "dc_len": 20,
    "atr_len": 14,
    "vol_len": 20,
    "vol_mult": 1.3,
    "atr_stop_mult": 2.5,
    "ema_len": 50,
}

DESCRIPTION = (
    "Donchian channel breakout with volume confirmation and ATR trailing stop. "
    "Enters when price breaks above N-day high with volume spike and above 50 EMA. "
    "Exits on ATR-based trailing stop (rolling high minus ATR * multiplier). "
    "Designed for AAPL range expansion plays with volatility-adaptive risk management."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    donch_high = high.rolling(p["dc_len"]).max()
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    vol_avg = volume.rolling(p["vol_len"]).mean()
    ema = cached_indicator("ema", close, length=p["ema_len"])

    breakout = close > donch_high.shift(1)
    vol_spike = volume > vol_avg * p["vol_mult"]
    trend_filter = close > ema

    entries = (breakout & vol_spike & trend_filter).fillna(False)

    rolling_high = close.cummax()
    trailing_stop = rolling_high - atr * p["atr_stop_mult"]
    exits = (close < trailing_stop).fillna(False)

    return entries, exits