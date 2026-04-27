"""
Mean Reversion NFLX Strategy

Bollinger Band + RSI + Stochastic mean reversion.
Enters on extreme oversold conditions, exits on mean reversion.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "bb_len": 20,
    "bb_std": 2.0,
    "rsi_len": 14,
    "rsi_oversold": 30,
    "stoch_len": 14,
    "stoch_oversold": 20,
    "rsi_exit": 55,
    "bb_exit_mid": True,
}

DESCRIPTION = (
    "Mean reversion using Bollinger Bands, RSI, and Stochastic. "
    "Enters when price below lower BB, RSI oversold, and Stoch K < threshold. "
    "Exits when price reverts to BB middle or RSI recovers. "
    "Designed for volatile mean-reverting stocks like NFLX."
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

    stoch = cached_indicator("stoch", df["high"], df["low"], close, k=p["stoch_len"], d=3)
    stoch_k_col = [c for c in stoch.columns if c.startswith("STOCHk")][0]
    stoch_k = stoch[stoch_k_col]

    entries = (
        (close < bbl)
        & (rsi < p["rsi_oversold"])
        & (stoch_k < p["stoch_oversold"])
    ).fillna(False)

    exits = ((close > bbm) | (rsi > p["rsi_exit"])).fillna(False)

    return entries, exits