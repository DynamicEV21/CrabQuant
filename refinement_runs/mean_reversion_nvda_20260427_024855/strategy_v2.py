import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "bb_len": 20,
    "bb_std": 2.0,
    "rsi_len": 14,
    "rsi_entry": 40,
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_entry": 30,
    "exit_rsi": 60,
}

DESCRIPTION = (
    "Mean reversion using Bollinger Bands with RSI and Stochastic filter on volatile semis. "
    "Enters when price below lower BB with RSI below entry threshold or Stochastic oversold. "
    "Exits when RSI recovers above exit threshold. "
    "Designed for NVDA, AMD, SMCI high-volatility mean reversion."
)

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    bb = cached_indicator("bbands", close, length=p["bb_len"], std=p["bb_std"])
    bbl_col = [c for c in bb.columns if c.startswith("BBL")][0]
    bbl = bb[bbl_col]

    rsi = cached_indicator("rsi", close, length=p["rsi_len"])

    stoch = cached_indicator("stoch", df["high"], df["low"], close, k=p["stoch_k"], d=p["stoch_d"])
    stoch_k_col = [c for c in stoch.columns if c.startswith("STOCHk")][0]
    stoch_k = stoch[stoch_k_col]

    bb_oversold = close < bbl
    rsi_oversold = rsi < p["rsi_entry"]
    stoch_oversold = stoch_k < p["stoch_entry"]

    entries = (bb_oversold & (rsi_oversold | stoch_oversold)).fillna(False)
    exits = (rsi > p["exit_rsi"]).fillna(False)

    return entries, exits