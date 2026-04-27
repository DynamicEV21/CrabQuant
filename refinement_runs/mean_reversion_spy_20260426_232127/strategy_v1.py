"""
Mean Reversion Strategy for SPY/QQQ/IWM

Bollinger Band lower penetration with RSI oversold confirmation.
Exits on reversion to middle band or RSI recovery.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "bb_len": 20,
    "bb_std": 2.0,
    "rsi_len": 14,
    "rsi_oversold": 30,
    "rsi_exit": 55,
}

DESCRIPTION = (
    "Mean reversion using Bollinger Bands and RSI. "
    "Enters when price closes below lower BB with RSI oversold. "
    "Exits when price reverts to middle band or RSI recovers above exit level. "
    "Designed for index ETFs (SPY, QQQ, IWM) which mean-revert short-term."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    bb = cached_indicator("bbands", close, length=p["bb_len"], std=p["bb_std"])
    bbl_col = [c for c in bb.columns if c.startswith("BBL")][0]
    bbm_col = [c for c in bb.columns if c.startswith("BBM")][0]
    bbl = bb[bbl_col]
    bbm = bb[bbm_col]

    rsi = cached_indicator("rsi", close, length=p["rsi_len"])

    entries = (close < bbl) & (rsi < p["rsi_oversold"])
    exits = (close > bbm) | (rsi > p["rsi_exit"])

    return entries.fillna(False), exits.fillna(False)