"""
ROC EMA Volume Momentum Strategy

Momentum strategy combining ROC direction, EMA trend filter,
volume confirmation, and ATR trailing stop exits.
Designed for E2E integration test on AAPL.
"""

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 10,
    "ema_len": 20,
    "volume_len": 20,
    "volume_mult": 1.0,
    "atr_len": 14,
    "atr_mult": 2.0,
}


DESCRIPTION = (
    "ROC momentum with EMA trend filter and volume confirmation. "
    "Enters when ROC is positive, price above EMA, and volume above average. "
    "Exits via ATR trailing stop (rolling high minus ATR * multiplier). "
    "Simple momentum design for E2E testing on AAPL."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    roc = cached_indicator("roc", close, length=p["roc_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    vol_avg = cached_indicator("sma", volume, length=p["volume_len"])
    atr = cached_indicator("atr", df["high"], df["low"], close, length=p["atr_len"])

    momentum_ok = roc > 0
    trend_ok = close > ema
    volume_ok = volume > (vol_avg * p["volume_mult"])

    entries = (momentum_ok & trend_ok & volume_ok).fillna(False)

    rolling_high = close.rolling(window=p["roc_len"], min_periods=1).max()
    trailing_stop = rolling_high - (atr * p["atr_mult"])
    exits = (close < trailing_stop).fillna(False)

    return entries, exits