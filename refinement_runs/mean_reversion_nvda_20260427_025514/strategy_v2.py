"""
Mean Reversion BB RSI Stoch Strategy

Relaxed confluence: 2 of 3 conditions trigger entry.
Designed for volatile stocks (NVDA, AMD, SMCI).
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "bb_len": 20,
    "bb_std": 2.0,
    "rsi_len": 14,
    "rsi_oversold": 38,
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_oversold": 28,
    "exit_rsi": 62,
}

DESCRIPTION = (
    "Mean reversion using BB lower band, RSI oversold, and stochastic oversold. "
    "Requires 2 of 3 conditions to trigger entry for reliable signal generation. "
    "Exits on RSI recovery or price returning to BB middle. "
    "Designed for volatile mean-reverting stocks like NVDA."
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

    stoch = cached_indicator("stoch", df["high"], df["low"], close, k=p["stoch_k"], d=p["stoch_d"])
    stoch_k_col = [c for c in stoch.columns if c.startswith("STOCHk")][0]
    stoch_k = stoch[stoch_k_col]

    below_bb = (close <= bbl).fillna(False)
    rsi_low = (rsi < p["rsi_oversold"]).fillna(False)
    stoch_low = (stoch_k < p["stoch_oversold"]).fillna(False)

    score = below_bb.astype(int) + rsi_low.astype(int) + stoch_low.astype(int)
    entries = (score >= 2).fillna(False)

    exits = ((rsi > p["exit_rsi"]) | (close > bbm)).fillna(False)

    return entries, exits