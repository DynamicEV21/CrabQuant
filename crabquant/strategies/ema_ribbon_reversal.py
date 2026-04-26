"""
EMA Ribbon Reversal Strategy

Multiple EMA alignment (ribbon) with RSI pullback entry.
Enters when EMAs are in perfect bullish order and RSI dips.
"""

import pandas as pd
import pandas_ta


DEFAULT_PARAMS = {
    "dip_level": 40,
}

PARAM_GRID = {
    "dip_level": [35, 40, 45],
}

DESCRIPTION = (
    "EMA ribbon alignment with RSI pullback entry. "
    "Enters when 10/20/30/50 EMAs are perfectly aligned (bullish order) "
    "and RSI dips below dip_level. "
    "Exits on RSI recovery above 60 or alignment break. "
    "Designed for strong trends with regular healthy pullbacks."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    ema10 = pandas_ta.ema(close, length=10)
    ema20 = pandas_ta.ema(close, length=20)
    ema30 = pandas_ta.ema(close, length=30)
    ema50 = pandas_ta.ema(close, length=50)

    # Perfect bullish alignment
    aligned = (ema10 > ema20) & (ema20 > ema30) & (ema30 > ema50)

    # RSI dip
    rsi = pandas_ta.rsi(close, length=14)
    dip = rsi < p["dip_level"]

    entries = (aligned & dip).fillna(False)
    exits = ((rsi > 60) | ~(ema10 > ema20)).fillna(False)

    return entries, exits
