"""
MACD Momentum with EMA Trend Filter

Relaxed MACD entry (histogram positive, not just crossing) with EMA trend
filter to generate more trades while maintaining quality. RSI-based exits.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "ema_len": 50,
    "rsi_len": 14,
    "rsi_exit": 75,
    "volume_window": 20,
    "volume_mult": 0.9,
}

DESCRIPTION = (
    "MACD momentum with EMA trend filter. Enters when MACD histogram is positive, "
    "price is above EMA, and volume is near/above average. Exits on RSI overbought "
    "or MACD histogram turning negative. Relaxed entry for more trade frequency."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    macd = cached_indicator("macd", close, fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"])
    hist_col = f"MACDh_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"
    hist = macd[hist_col]
    hist_prev = hist.shift(1)

    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    vol_avg = cached_indicator("sma", volume, length=p["volume_window"])
    vol_ok = volume > (vol_avg * p["volume_mult"])

    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    rsi_overbought = rsi > p["rsi_exit"]

    hist_positive = hist > 0
    hist_turn_positive = (hist_prev <= 0) & (hist > 0)
    entries = (trend_ok & vol_ok & (hist_positive | hist_turn_positive)).fillna(False)

    hist_negative = hist < 0
    exits = (hist_negative | rsi_overbought).fillna(False)

    return entries, exits