"""
Keltner Channel Breakout with RSI Pullback Entry

Tight Keltner channels for frequent breakout signals.
RSI pullback entries in uptrends to boost trade count.
EMA-based exits for simplicity.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_len": 15,
    "atr_len": 10,
    "mult": 1.5,
    "vol_mult": 1.0,
    "trend_ema": 30,
    "rsi_len": 14,
    "rsi_level": 40,
}

DESCRIPTION = (
    "Keltner Channel breakout with RSI pullback supplement for higher trade frequency. "
    "Primary entry: price breaks above tight upper channel (EMA + 1.5*ATR) with minimal volume filter. "
    "Secondary entry: RSI dips below 40 then recovers when price above trend EMA. "
    "Exits on close below channel EMA midpoint. Designed for AAPL with relaxed filters to meet minimum trade count."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema = cached_indicator("ema", close, length=p["ema_len"])
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    trend_ema = cached_indicator("ema", close, length=p["trend_ema"])
    rsi = cached_indicator("rsi", close, length=p["rsi_len"])

    upper = ema + atr * p["mult"]
    vol_avg = volume.rolling(20).mean()

    breakout = close > upper.shift(1)
    vol_ok = volume > vol_avg * p["vol_mult"]
    trend_ok = close > trend_ema
    entry_breakout = breakout & vol_ok & trend_ok

    rsi_dip = rsi.shift(1) < p["rsi_level"]
    rsi_recover = rsi > p["rsi_level"]
    entry_pullback = rsi_dip & rsi_recover & trend_ok

    entries = (entry_breakout | entry_pullback).fillna(False)
    exits = (close < ema).fillna(False)

    return entries, exits