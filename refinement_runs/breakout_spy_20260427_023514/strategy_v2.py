"""
Keltner Channel Breakout Strategy

Tighter ATR-based channel with relaxed filters for index ETFs.
Optimized for higher trade frequency on SPY/QQQ/IWM.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_len": 10,
    "atr_len": 10,
    "mult": 1.5,
    "trend_ema": 30,
}

DESCRIPTION = (
    "Keltner Channel breakout with relaxed filters for index ETFs. "
    "Uses shorter EMA and lower ATR multiplier to generate more signals than Donchian. "
    "Enters when price breaks above upper channel with price above trend EMA. "
    "Exits when price falls below channel midpoint (EMA). "
    "Designed for SPY/QQQ/IWM higher-frequency trend capture."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema = cached_indicator("ema", close, length=p["ema_len"])
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    trend_ema = cached_indicator("ema", close, length=p["trend_ema"])

    upper = ema + atr * p["mult"]

    breakout = close > upper.shift(1)
    trend_filter = close > trend_ema

    entries = (breakout & trend_filter).fillna(False)
    exits = (close < ema).fillna(False)

    return entries, exits