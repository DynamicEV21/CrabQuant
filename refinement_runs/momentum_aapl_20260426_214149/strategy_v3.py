import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "roc_len": 12,
    "ema_len": 50,
    "vol_window": 20,
    "vol_mult": 1.0,
    "atr_len": 14,
    "atr_mult": 2.5,
}

DESCRIPTION = "ROC momentum with EMA trend filter and volume confirmation. Enters when ROC positive, price above EMA, volume above average. Exits via ATR trailing stop."


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    roc = cached_indicator("roc", close, length=p["roc_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    vol_avg = cached_indicator("sma", volume, length=p["vol_window"])
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])

    trend_ok = close > ema
    momentum_ok = roc > 0
    vol_ok = volume > (vol_avg * p["vol_mult"])

    entries = (trend_ok & momentum_ok & vol_ok).fillna(False)

    rolling_max = close.cummax()
    trailing_stop = rolling_max - (atr * p["atr_mult"])
    exits = (close < trailing_stop).fillna(False)

    return entries, exits