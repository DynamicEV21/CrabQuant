"""
ROC Momentum with ADX Regime Filter and Volume Confirmation

Uses ADX to filter for trending regimes, ROC for momentum direction,
EMA for trend bias, and volume for signal confirmation. Exits on
ATR trailing stop. Designed to be robust across market regimes.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "roc_len": 21,
    "ema_len": 50,
    "adx_len": 14,
    "adx_min": 20,
    "vol_len": 20,
    "vol_mult": 1.0,
    "atr_len": 14,
    "atr_mult": 2.0,
}

DESCRIPTION = (
    "ROC momentum with ADX regime filter and volume confirmation. "
    "Enters when ADX confirms trending market, ROC positive, "
    "price above EMA, and volume above average. Exits on ATR trailing stop. "
    "Designed to avoid choppy markets and be robust across regimes."
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

    adx = cached_indicator("adx", high, low, close, length=p["adx_len"])
    adx_col = f"ADX_{p['adx_len']}"
    regime_ok = adx[adx_col] > p["adx_min"]

    vol_sma = cached_indicator("sma", volume, length=p["vol_len"])
    vol_ok = volume > (vol_sma * p["vol_mult"])

    entries = (momentum_ok & trend_ok & regime_ok & vol_ok).fillna(False)

    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])
    rolling_high = close.rolling(20).max()
    trailing_stop = rolling_high - (atr * p["atr_mult"])
    exits = (close < trailing_stop).fillna(False)

    return entries, exits