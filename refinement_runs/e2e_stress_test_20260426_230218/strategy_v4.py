import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "roc_len": 10,
    "ema_len": 50,
    "volume_len": 20,
    "volume_mult": 0.8,
    "atr_len": 14,
    "atr_mult": 2.5,
}

DESCRIPTION = (
    "ROC momentum with EMA-50 trend filter, MACD histogram confirmation, "
    "mild volume filter, and wide ATR trailing stop for SPY. "
    "MACD filter removes false ROC signals in choppy markets. "
    "Wider stop lets winners run to improve win rate."
)

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    volume = df["volume"]

    roc = cached_indicator("roc", close, length=p["roc_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    atr = cached_indicator("atr", high, df["low"], close, length=p["atr_len"])
    volume_avg = cached_indicator("sma", volume, length=p["volume_len"])
    macd = cached_indicator("macd", close, fast=12, slow=26, signal=9)

    hist = macd["MACDh_12_26_9"]
    momentum_ok = roc > 0
    trend_ok = close > ema
    volume_ok = volume > (volume_avg * p["volume_mult"])
    macd_ok = hist > 0

    entries = (momentum_ok & trend_ok & volume_ok & macd_ok).fillna(False)

    rolling_high = high.rolling(window=p["atr_len"], min_periods=1).max()
    trailing_stop = rolling_high - (atr * p["atr_mult"])
    exits = (close < trailing_stop).fillna(False)

    return entries, exits