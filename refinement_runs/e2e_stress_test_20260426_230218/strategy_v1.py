"""
ROC EMA Volume Momentum Strategy for SPY

Momentum entry with trend filter and volume confirmation.
ATR trailing stop for drawdown control.
"""

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 21,
    "ema_len": 50,
    "volume_len": 20,
    "volume_mult": 1.1,
    "atr_len": 14,
    "atr_mult": 2.5,
}


DESCRIPTION = (
    "ROC momentum with EMA trend filter and volume confirmation. "
    "Enters when ROC turns positive, price above EMA, volume above average. "
    "Exits via ATR trailing stop (high minus ATR * mult). "
    "Designed for SPY with strict drawdown control."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    volume = df["volume"]

    roc = cached_indicator("roc", close, length=p["roc_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    vol_avg = cached_indicator("sma", volume, length=p["volume_len"])
    atr = cached_indicator("atr", df["high"], df["low"], close, length=p["atr_len"])

    trend_ok = close > ema
    vol_ok = volume > (vol_avg * p["volume_mult"])
    roc_positive = roc > 0
    roc_turning = (roc.shift(1) <= 0) & (roc > 0)

    entries = (trend_ok & vol_ok & (roc_positive | roc_turning)).fillna(False)

    rolling_high = high.rolling(window=p["atr_len"], min_periods=1).max()
    trailing_stop = rolling_high - (atr * p["atr_mult"])
    exits = (close < trailing_stop).fillna(False)

    return entries, exits