"""
EMA Trend RSI Pullback for SPY

Bullish regime via EMA crossover, RSI dip timing for entries.
Dual exit: RSI recovery and ATR trailing stop.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "fast_ema": 20,
    "slow_ema": 50,
    "rsi_len": 14,
    "rsi_dip": 42,
    "rsi_exit": 65,
    "atr_len": 14,
    "atr_mult": 2.0,
}

DESCRIPTION = (
    "EMA trend filter with RSI pullback entry for SPY. "
    "Bullish when fast EMA > slow EMA. "
    "Enters on RSI dip with upward turn in bullish regime. "
    "Exits on RSI recovery or ATR trailing stop."
)

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]

    ema_fast = cached_indicator("ema", close, length=p["fast_ema"])
    ema_slow = cached_indicator("ema", close, length=p["slow_ema"])
    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    atr = cached_indicator("atr", df["high"], df["low"], close, length=p["atr_len"])

    bullish = ema_fast > ema_slow
    rsi_turning_up = (rsi < p["rsi_dip"]) & (rsi > rsi.shift(1))

    entries = (bullish & rsi_turning_up).fillna(False)

    rsi_exit = rsi > p["rsi_exit"]
    rolling_high = high.rolling(window=p["atr_len"], min_periods=1).max()
    trailing_stop = rolling_high - (atr * p["atr_mult"])
    atr_exit = close < trailing_stop

    exits = (rsi_exit | atr_exit).fillna(False)

    return entries, exits