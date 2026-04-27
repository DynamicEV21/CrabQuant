import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_len": 10,
    "atr_len": 10,
    "mult": 1.5,
    "trend_ema": 30,
    "rsi_len": 14,
    "rsi_max": 65,
    "vol_sma_len": 20,
}

DESCRIPTION = (
    "Keltner Channel breakout with volatility regime and RSI filters. "
    "Enters when price breaks above upper channel, price above trend EMA, "
    "ATR expanding above its SMA (volatility regime), and RSI below 65. "
    "Exits when price falls below channel midpoint. "
    "Reduces false breakouts in choppy markets for SPY/QQQ/IWM."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema = cached_indicator("ema", close, length=p["ema_len"])
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    trend_ema = cached_indicator("ema", close, length=p["trend_ema"])
    rsi = cached_indicator("rsi", close, length=p["rsi_len"])

    upper = ema + atr * p["mult"]
    atr_sma = atr.rolling(p["vol_sma_len"]).mean()

    breakout = close > upper.shift(1)
    trend_filter = close > trend_ema
    vol_filter = atr > atr_sma
    rsi_filter = rsi < p["rsi_max"]

    entries = (breakout & trend_filter & vol_filter & rsi_filter).fillna(False)
    exits = (close < ema).fillna(False)

    return entries, exits