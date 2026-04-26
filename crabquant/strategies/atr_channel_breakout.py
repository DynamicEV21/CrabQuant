"""
ATR Channel Breakout Strategy

Keltner Channel (EMA + ATR) breakout with volume and trend confirmation.
Best performer: ORCL (Sharpe 1.59, 84% return).
"""

import pandas as pd
import pandas_ta


DEFAULT_PARAMS = {
    "ema_len": 20,
    "atr_len": 10,
    "mult": 2.0,
    "vol_mult": 1.2,
}

PARAM_GRID = {
    "ema_len": [15, 20],
    "atr_len": [10, 14],
    "mult": [1.5, 2.0, 2.5],
    "vol_mult": [1.0, 1.2, 1.5],
}

DESCRIPTION = (
    "ATR-based Keltner Channel breakout with volume and trend confirmation. "
    "Enters when price breaks above upper channel with above-average volume, "
    "and price is above 50 EMA (trend filter). "
    "Exits when price falls back below channel midpoint (EMA). "
    "Good for catching momentum breakouts from consolidation."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema = pandas_ta.ema(close, length=p["ema_len"])
    atr = pandas_ta.atr(high, low, close, length=p["atr_len"])

    upper = ema + atr * p["mult"]

    # Breakout above upper channel (prior bar)
    breakout = close > upper.shift(1)
    # Volume confirmation
    vol_confirm = volume > volume.rolling(20).mean() * p["vol_mult"]
    # Trend filter
    trend = close > pandas_ta.ema(close, length=50)

    entries = (breakout & vol_confirm & trend).fillna(False)
    exits = (close < ema).fillna(False)

    return entries, exits
