"""
MSFT Momentum Strategy

MACD histogram crossover with RSI momentum filter and EMA trend confirmation.
Targets sustained trend acceleration in large cap tech.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_len": 14,
    "rsi_min": 45,
    "rsi_max": 75,
    "ema_len": 50,
}

DESCRIPTION = (
    "Large cap tech momentum using MACD histogram crossover with RSI momentum "
    "filter and EMA trend confirmation. Enters when MACD histogram turns positive, "
    "RSI shows strength (between thresholds), and price is above EMA. "
    "Exits when MACD histogram turns negative. Designed for MSFT sustained trends."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    macd = cached_indicator(
        "macd", close,
        fast=p["macd_fast"],
        slow=p["macd_slow"],
        signal=p["macd_signal"]
    )
    hist_key = f"MACDh_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"
    hist = macd[hist_key]

    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])

    hist_prev = hist.shift(1)
    macd_cross_up = (hist_prev <= 0) & (hist > 0)
    rsi_ok = (rsi >= p["rsi_min"]) & (rsi <= p["rsi_max"])
    trend_ok = close > ema

    entries = (macd_cross_up & rsi_ok & trend_ok).fillna(False)

    macd_cross_down = (hist_prev > 0) & (hist <= 0)
    exits = macd_cross_down.fillna(False)

    return entries, exits