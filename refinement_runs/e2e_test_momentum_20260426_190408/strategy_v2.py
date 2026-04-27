"""
ROC EMA Volume Momentum Strategy

Momentum strategy using ROC for direction, EMA for trend filter,
volume for confirmation, and ATR trailing stop for exits.
"""

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 10,
    "ema_len": 20,
    "vol_len": 20,
    "vol_mult": 1.0,
    "atr_len": 14,
    "atr_mult": 2.0,
}


DESCRIPTION = (
    "ROC + EMA + Volume momentum strategy. "
    "Enters when ROC is positive, price above EMA, and volume above average. "
    "Exits via ATR trailing stop (rolling high minus ATR * multiplier). "
    "Simple trend-following with volume confirmation."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    volume = df["volume"]

    roc = cached_indicator("roc", close, length=p["roc_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    vol_sma = cached_indicator("sma", volume, length=p["vol_len"])
    atr = cached_indicator("atr", high, df["low"], close, length=p["atr_len"])

    momentum_ok = roc > 0
    trend_ok = close > ema
    volume_ok = volume > (vol_sma * p["vol_mult"])

    entries = (momentum_ok & trend_ok & volume_ok).fillna(False)

    rolling_high = high.rolling(window=p["atr_len"], min_periods=1).max()
    trailing_stop = rolling_high - (atr * p["atr_mult"])
    exits = (close < trailing_stop).fillna(False)

    return entries, exits