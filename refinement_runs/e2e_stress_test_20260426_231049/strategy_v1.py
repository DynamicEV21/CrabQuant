"""
Momentum ROC with Volume and ATR Stops

ROC momentum + EMA trend + volume confirmation + ATR trailing stop.
Designed for SPY stress test with robust risk management.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "roc_len": 21,
    "ema_len": 50,
    "volume_len": 20,
    "volume_mult": 1.2,
    "atr_len": 14,
    "atr_mult": 2.5,
    "rsi_len": 14,
    "rsi_max": 75,
}

DESCRIPTION = (
    "Momentum ROC with EMA trend filter, volume confirmation, and ATR trailing stops. "
    "Enters when ROC is positive, price above EMA, volume above average, and RSI not overbought. "
    "Exits via ATR trailing stop or RSI overbought. "
    "Designed for SPY momentum with risk management."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    roc = cached_indicator("roc", close, length=p["roc_len"])
    momentum_ok = roc > 0

    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    vol_avg = cached_indicator("sma", volume, length=p["volume_len"])
    volume_ok = volume > (vol_avg * p["volume_mult"])

    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    rsi_ok = rsi < p["rsi_max"]

    entries = (momentum_ok & trend_ok & volume_ok & rsi_ok).fillna(False)

    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    rolling_high = close.cummax()
    trailing_stop = rolling_high - (atr * p["atr_mult"])

    exit_atr = close < trailing_stop
    rsi_overbought = rsi > p["rsi_max"]
    exits = (exit_atr | rsi_overbought).fillna(False)

    return entries, exits