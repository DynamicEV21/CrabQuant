"""
MACD RSI Momentum with ATR Stop

Combines MACD momentum direction with RSI timing and volume confirmation.
Uses ATR trailing stop for exits to limit drawdowns on AAPL/SPY/NVDA.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_len": 14,
    "rsi_entry_max": 65,
    "rsi_exit_min": 75,
    "atr_len": 14,
    "atr_mult": 2.0,
}

DESCRIPTION = (
    "MACD momentum direction with RSI timing and ATR trailing stop. "
    "Enters when MACD histogram turns positive, RSI is not overbought, "
    "and volume is above average. Exits on ATR trailing stop or RSI overbought. "
    "Designed for AAPL momentum capture with controlled drawdowns."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    macd = cached_indicator("macd", close, fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"])
    hist_col = f"MACDh_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"
    hist = macd[hist_col]
    hist_prev = hist.shift(1)

    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    rsi_ok = rsi < p["rsi_entry_max"]
    rsi_overbought = rsi > p["rsi_exit_min"]

    vol_avg = cached_indicator("sma", volume, length=20)
    vol_ok = volume > vol_avg

    hist_turn_positive = (hist_prev <= 0) & (hist > 0)
    entries = (hist_turn_positive & rsi_ok & vol_ok).fillna(False)

    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    rolling_high = close.cummax()
    trailing_stop = rolling_high - (atr * p["atr_mult"])
    atr_exit = (close < trailing_stop).fillna(False)
    exits = (atr_exit | rsi_overbought).fillna(False)

    return entries, exits