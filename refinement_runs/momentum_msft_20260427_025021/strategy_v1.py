"""
MSFT Momentum Strategy

MACD momentum acceleration with RSI pullback timing and EMA trend filter.
Designed for large cap tech sustained trending moves.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_len": 50,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_len": 14,
    "rsi_max": 65,
    "volume_window": 20,
    "volume_mult": 1.1,
}

DESCRIPTION = (
    "MACD momentum acceleration with RSI pullback timing and EMA trend filter. "
    "Enters when MACD histogram turns positive, price above EMA, RSI below overbought, "
    "and volume above average. Exits on MACD histogram turning negative. "
    "Optimized for MSFT sustained trend moves."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    macd = cached_indicator("macd", close, fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"])
    hist_col = f"MACDh_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"
    hist = macd[hist_col]
    hist_prev = hist.shift(1)
    mom_turn = (hist_prev <= 0) & (hist > 0)

    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    rsi_ok = rsi < p["rsi_max"]

    vol_avg = cached_indicator("sma", volume, length=p["volume_window"])
    vol_ok = volume > (vol_avg * p["volume_mult"])

    entries = (trend_ok & mom_turn & rsi_ok & vol_ok).fillna(False)
    exits = ((hist_prev > 0) & (hist <= 0)).fillna(False)

    return entries, exits