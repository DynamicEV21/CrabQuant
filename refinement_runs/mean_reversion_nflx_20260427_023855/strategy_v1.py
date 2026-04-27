"""
Mean Reversion Strategy for NFLX

Uses Bollinger Band lower band touches with RSI and Stochastic
confirmation for extreme deviation entries. Exits on mean reversion.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "bb_len": 20,
    "bb_std": 2.0,
    "rsi_len": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "stoch_len": 14,
    "stoch_oversold": 20,
}

DESCRIPTION = (
    "Mean reversion using Bollinger Bands, RSI, and Stochastic. "
    "Enters when price touches lower BB with RSI oversold and Stoch K below threshold. "
    "Exits when price reverts to BB middle or RSI overbought. "
    "Designed for NFLX volatile swings."
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
    stoch = cached_indicator("stoch", df["high"], df["low"], close, k=p["stoch_len"], d=3, smooth_k=3)
    stoch_k_col = [c for c in stoch.columns if c.startswith("STOCHk")][0]
    stoch_k = stoch[stoch_k_col]

    bb_touch = close <= bbl
    rsi_oversold = rsi < p["rsi_oversold"]
    stoch_oversold = stoch_k < p["stoch_oversold"]

    entries = (bb_touch & rsi_oversold & stoch_oversold).fillna(False)

    mean_revert = close >= bbm
    rsi_overbought = rsi > p["rsi_overbought"]
    exits = (mean_revert | rsi_overbought).fillna(False)

    return entries, exits