import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_len": 15,
    "atr_len": 10,
    "mult": 1.5,
    "vol_mult": 1.2,
    "fast_ema": 20,
    "slow_ema": 50,
    "roc_len": 10,
    "exit_buffer": 1.0,
}

DESCRIPTION = (
    "Keltner Channel breakout with dual EMA trend confirmation and buffered exit. "
    "Requires EMA20 > EMA50 for strong trend alignment, ROC > 0 for momentum, "
    "volume above 1.2x average, and price above upper channel (1.5x ATR). "
    "Exits on close below EMA minus ATR buffer, giving winners more room to run."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema = cached_indicator("ema", close, length=p["ema_len"])
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    fast_ema = cached_indicator("ema", close, length=p["fast_ema"])
    slow_ema = cached_indicator("ema", close, length=p["slow_ema"])
    roc = cached_indicator("roc", close, length=p["roc_len"])

    upper = ema + atr * p["mult"]
    vol_avg = volume.rolling(20).mean()

    breakout = close > upper.shift(1)
    vol_ok = volume > vol_avg * p["vol_mult"]
    trend_ok = fast_ema > slow_ema
    momentum_ok = roc > 0

    entries = (breakout & vol_ok & trend_ok & momentum_ok).fillna(False)

    exit_level = ema - atr * p["exit_buffer"]
    exits = (close < exit_level).fillna(False)

    return entries, exits