"""
MACD Momentum with RSI Regime Filter and ATR Stops

Combines MACD histogram momentum with RSI regime filtering.
Enters on MACD histogram turning positive when RSI < 70 and price above EMA.
Exits on ATR trailing stop or MACD histogram turning negative.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_len": 14,
    "rsi_max": 70,
    "ema_len": 50,
    "atr_len": 14,
    "atr_mult": 2.0,
}

DESCRIPTION = (
    "MACD momentum with RSI regime filter and ATR trailing stops. "
    "Enters when MACD histogram crosses above zero, price above EMA, "
    "and RSI below 70 (avoid overbought). Exits on ATR trailing stop "
    "or MACD histogram turning negative. Designed for momentum stocks."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # MACD
    macd = cached_indicator("macd", close, fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"])
    hist_col = f"MACDh_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"
    hist = macd[hist_col]
    hist_prev = hist.shift(1)

    # RSI regime filter
    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    rsi_ok = rsi < p["rsi_max"]

    # Trend filter
    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    # Entry: MACD histogram crosses above zero with filters
    hist_cross_up = (hist_prev <= 0) & (hist > 0)
    entries = (hist_cross_up & trend_ok & rsi_ok).fillna(False)

    # ATR trailing stop
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    rolling_high = close.rolling(20).max()
    trailing_stop = rolling_high - (atr * p["atr_mult"])
    atr_exit = (close < trailing_stop).fillna(False)

    # MACD exit: histogram turns negative
    macd_exit = (hist_prev >= 0) & (hist < 0)

    exits = (atr_exit | macd_exit).fillna(False)

    return entries, exits