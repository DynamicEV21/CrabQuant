import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "rsi_period": 14,
    "rsi_entry": 60,
    "rsi_exit": 68,
    "bb_period": 18,
    "bb_std": 1.8,
}

DESCRIPTION = "Mean reversion entering when RSI turns up from below 60 or price falls below lower Bollinger Band (1.8 std). Exits at BB midline or RSI above 68. Uses proven RSI+BBands indicator family for XOM instead of negatively-correlated EMA/SMA. Wider RSI threshold and tighter bands boost trade frequency to 30-40 while preserving mean-reversion edge."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    rsi = cached_indicator("rsi", close, length=p["rsi_period"])
    bb = cached_indicator("bbands", close, length=p["bb_period"], std=p["bb_std"])
    bb_lower = bb.iloc[:, 0]
    bb_mid = bb.iloc[:, 1]

    rsi_pullback = (rsi < p["rsi_entry"]) & (rsi > rsi.shift(1))
    below_band = close < bb_lower

    entries = (rsi_pullback | below_band).fillna(False)
    exits = ((close > bb_mid) | (rsi > p["rsi_exit"])).fillna(False)

    return entries, exits
