"""
Relaxed Mean Reversion Strategy for SPY/QQQ/IWM

Multiple entry pathways to increase trade frequency
while maintaining mean reversion edge.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "bb_len": 20,
    "bb_std_narrow": 1.5,
    "bb_std_wide": 2.0,
    "rsi_len": 14,
    "rsi_moderate": 35,
    "rsi_deep": 25,
    "rsi_exit": 55,
}

DESCRIPTION = (
    "Relaxed mean reversion with multiple entry pathways. "
    "Enters on: (1) price below 1.5std BB with RSI<35, "
    "(2) price below 2std BB with RSI<40, or (3) RSI<25 deep oversold. "
    "Exits on reversion to middle band or RSI>55."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    bb_narrow = cached_indicator("bbands", close, length=p["bb_len"], std=p["bb_std_narrow"])
    bb_wide = cached_indicator("bbands", close, length=p["bb_len"], std=p["bb_std_wide"])

    bbl_narrow = bb_narrow[[c for c in bb_narrow.columns if c.startswith("BBL")][0]]
    bbl_wide = bb_wide[[c for c in bb_wide.columns if c.startswith("BBL")][0]]
    bbm = bb_wide[[c for c in bb_wide.columns if c.startswith("BBM")][0]]

    rsi = cached_indicator("rsi", close, length=p["rsi_len"])

    entry1 = (close < bbl_narrow) & (rsi < p["rsi_moderate"])
    entry2 = (close < bbl_wide) & (rsi < 40)
    entry3 = rsi < p["rsi_deep"]

    entries = (entry1 | entry2 | entry3).fillna(False)
    exits = ((close > bbm) | (rsi > p["rsi_exit"])).fillna(False)

    return entries, exits