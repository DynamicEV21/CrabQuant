"""
Momentum ROC EMA Strategy for AAPL E2E Test

Simple momentum strategy using ROC, EMA trend filter, and volume confirmation.
Designed for reliable signal generation with minimal complexity.
"""

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 10,
    "ema_len": 20,
    "volume_len": 20,
    "volume_mult": 1.0,
    "exit_roc": 0,
}


DESCRIPTION = (
    "Simple momentum strategy for AAPL E2E test. "
    "Enters when ROC is positive, price above EMA, and volume above average. "
    "Exits when ROC turns negative. "
    "Designed for reliable signal generation with minimal complexity."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    roc = cached_indicator("roc", close, length=p["roc_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    vol_avg = cached_indicator("sma", volume, length=p["volume_len"])

    trend_ok = close > ema
    vol_ok = volume > (vol_avg * p["volume_mult"])

    entries = (roc > 0) & trend_ok & vol_ok
    exits = (roc < p["exit_roc"]) & (roc.shift(1) >= p["exit_roc"])

    return entries.fillna(False), exits.fillna(False)