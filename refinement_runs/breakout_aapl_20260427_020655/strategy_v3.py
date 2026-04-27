"""
Keltner Channel Breakout with ROC Momentum Filter

Tighter Keltner channels generate more breakout candidates.
ROC momentum filter ensures only quality entries proceed.
RSI pullback entries also require positive momentum.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_len": 15,
    "atr_len": 10,
    "mult": 1.3,
    "vol_mult": 1.0,
    "trend_ema": 30,
    "rsi_len": 14,
    "rsi_level": 40,
    "roc_len": 10,
}

DESCRIPTION = (
    "Keltner Channel breakout with ROC momentum confirmation. "
    "Tighter channel (1.3x ATR) generates more breakout candidates; "
    "ROC > 0 filters for genuine momentum. "
    "Primary: price above upper channel + volume + trend + ROC. "
    "Secondary: RSI pullback recovery + trend + ROC. "
    "Exits on close below channel EMA."
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
    roc = cached_indicator("roc", close, length=p["roc_len"])

    upper = ema + atr * p["mult"]
    vol_avg = volume.rolling(20).mean()

    breakout = close > upper.shift(1)
    vol_ok = volume > vol_avg * p["vol_mult"]
    trend_ok = close > trend_ema
    momentum_ok = roc > 0
    entry_breakout = breakout & vol_ok & trend_ok & momentum_ok

    rsi_dip = rsi.shift(1) < p["rsi_level"]
    rsi_recover = rsi > p["rsi_level"]
    entry_pullback = rsi_dip & rsi_recover & trend_ok & momentum_ok

    entries = (entry_breakout | entry_pullback).fillna(False)
    exits = (close < ema).fillna(False)

    return entries, exits