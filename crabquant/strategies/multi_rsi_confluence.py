"""
Multi-RSI Confluence Strategy

Multiple timeframe RSI confluence with volume confirmation.
Enters when all three RSIs are oversold and the fastest starts turning up.
"""

import pandas as pd
import pandas_ta


DEFAULT_PARAMS = {
    "rsi1": 7,
    "rsi2": 14,
    "rsi3": 28,
    "thresh": 35,
    "vol_mult": 1.0,
    "exit_thresh": 65,
}

PARAM_GRID = {
    "rsi1": [5, 7, 10],
    "rsi2": [14, 21],
    "rsi3": [28, 35],
    "thresh": [30, 35, 40],
    "vol_mult": [0.8, 1.0, 1.2],
    "exit_thresh": [60, 65, 70],
}

DESCRIPTION = (
    "Multiple timeframe RSI confluence with volume confirmation. "
    "Enters when RSI-7, RSI-14, and RSI-28 are all below threshold "
    "and the fastest RSI starts turning up. "
    "Exits when fastest RSI recovers above exit threshold. "
    "Designed for catching deep pullback reversals in uptrends."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    rsi1 = pandas_ta.rsi(close, length=p["rsi1"])
    rsi2 = pandas_ta.rsi(close, length=p["rsi2"])
    rsi3 = pandas_ta.rsi(close, length=p["rsi3"])

    all_oversold = (rsi1 < p["thresh"]) & (rsi2 < p["thresh"]) & (rsi3 < p["thresh"])
    rsi_turning = rsi1 > rsi1.shift(1)
    vol_avg = volume.rolling(20).mean()
    vol_confirm = volume > vol_avg * p["vol_mult"]

    entries = (all_oversold & rsi_turning & vol_confirm).fillna(False)
    exits = (rsi1 > p["exit_thresh"]).fillna(False)

    return entries, exits
