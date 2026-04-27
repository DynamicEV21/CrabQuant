"""
Momentum ROC EMA Strategy with ATR Trailing Stop

ROC momentum with EMA trend filter, volume confirmation, and ATR trailing exit.
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
    "Momentum ROC with EMA trend filter, volume confirmation, and ATR trailing stop. "
    "Enters when ROC positive, price above EMA, volume above average. "
    "Exits via ATR trailing stop to let winners run."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    roc = cached_indicator("roc", close, length=p["roc_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    vol_avg = cached_indicator("sma", volume, length=p["volume_len"])
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])

    trend_ok = close > ema
    vol_ok = volume > (vol_avg * p["volume_mult"])

    entries = (roc > 0) & trend_ok & vol_ok

    # ATR trailing stop: exit when close falls below rolling high minus ATR*mult
    rolling_high = close.cummax()
    trailing_stop = rolling_high - (atr * p["atr_mult"])
    exits = close < trailing_stop

    # Only exit if we're actually in a position (entry occurred and not yet exited)
    # Track position state to avoid spurious exits before entry
    position = entries.fillna(False).cumsum() - exits.fillna(False).cumsum()
    exits = exits & (position > 0)

    return entries.fillna(False), exits.fillna(False)