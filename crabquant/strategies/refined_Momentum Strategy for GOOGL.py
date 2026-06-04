import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "ema_len": 20,
    "atr_len": 14,
    "atr_mult": 2.0,
    "trailing_len": 20,
}

DESCRIPTION = "MACD histogram positive signals momentum acceleration. EMA trend filter prevents counter-trend entries. ATR trailing stop locks in profits by exiting before major reversals."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    macd = cached_indicator("macd", close, fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"])
    hist = macd.iloc[:, 1]

    ema = cached_indicator("ema", close, length=p["ema_len"])
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])

    atr_stop = close.rolling(window=p["trailing_len"]).max() - atr * p["atr_mult"]

    entries = (
        (hist > 0)
        & (close > ema)
    ).fillna(False)

    exits = (close < atr_stop).fillna(False)

    return entries, exits
