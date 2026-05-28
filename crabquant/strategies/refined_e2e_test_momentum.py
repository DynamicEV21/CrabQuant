"""
ROC Momentum with EMA Trend Filter

Simple momentum strategy using Rate of Change, EMA trend filter,
and volume confirmation. Designed for AAPL E2E integration test.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "roc_len": 14,
    "ema_len": 21,
    "vol_len": 20,
    "vol_mult": 1.1,
    "exit_roc": -1.0,
}

DESCRIPTION = (
    "ROC momentum with EMA trend filter and volume confirmation. "
    "Enters when ROC is positive, price above EMA, and volume above average. "
    "Exits when ROC turns negative or price falls below EMA. "
    "Simple momentum strategy optimized for trending stocks like AAPL."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    roc = cached_indicator("roc", close, length=p["roc_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    vol_sma = cached_indicator("sma", volume, length=p["vol_len"])

    trend_ok = close > ema
    momentum_ok = roc > 0
    vol_ok = volume > (vol_sma * p["vol_mult"])

    entries = (trend_ok & momentum_ok & vol_ok).fillna(False)

    roc_negative = roc < p["exit_roc"]
    below_ema = close < ema
    exits = (roc_negative | below_ema).fillna(False)

    return entries, exits