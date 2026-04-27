"""
Keltner Channel Breakout for Index ETFs

ATR-based channel breakout with relaxed filters for broad market ETFs.
Optimized for SPY, QQQ, IWM.
"""

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "ema_len": 15,
    "atr_len": 10,
    "mult": 1.5,
    "vol_mult": 1.0,
    "trend_ema": 30,
}


DESCRIPTION = (
    "Keltner Channel breakout for index ETFs with relaxed volume filter. "
    "Enters when price breaks above upper channel (EMA + ATR*mult) with "
    "volume above average and price above trend EMA. Shorter lookback (15) "
    "than Donchian for more signals. Exits at channel midpoint (EMA)."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema = cached_indicator("ema", close, length=p["ema_len"])
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])

    upper = ema + atr * p["mult"]

    breakout = close > upper.shift(1)
    vol_confirm = volume > volume.rolling(20).mean() * p["vol_mult"]
    trend_filter = close > cached_indicator("ema", close, length=p["trend_ema"])

    entries = (breakout & vol_confirm & trend_filter).fillna(False)
    exits = (close < ema).fillna(False)

    return entries, exits