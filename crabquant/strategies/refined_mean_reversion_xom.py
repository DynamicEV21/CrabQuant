import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "rsi_period": 10,
    "rsi_entry": 55,
    "rsi_exit": 60,
}

DESCRIPTION = "Fast RSI mean reversion for energy stocks. Enters when RSI(10) turns up from below 55, exits at RSI above 60. Wider entry threshold captures more mean-reversion oscillations in the 42-55 RSI band that the previous version missed."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    rsi = cached_indicator("rsi", close, length=p["rsi_period"])

    rsi_oversold = rsi < p["rsi_entry"]
    rsi_turning_up = rsi > rsi.shift(1)

    entries = (rsi_oversold & rsi_turning_up).fillna(False)
    exits = (rsi > p["rsi_exit"]).fillna(False)

    return entries, exits
