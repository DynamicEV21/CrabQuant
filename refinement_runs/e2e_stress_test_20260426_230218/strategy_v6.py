import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "fast_len": 12,
    "slow_len": 26,
}

DESCRIPTION = (
    "Pure EMA fast/slow crossover for SPY. Enters on golden cross "
    "(fast EMA crosses above slow EMA). Exits on death cross "
    "(fast EMA crosses below slow EMA). No volume filter or trailing "
    "stop to maximize trade count and avoid over-filtering."
)

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    ema_fast = cached_indicator("ema", close, length=p["fast_len"])
    ema_slow = cached_indicator("ema", close, length=p["slow_len"])

    entries = (
        (ema_fast.shift(1) <= ema_slow.shift(1))
        & (ema_fast > ema_slow)
    ).fillna(False)

    exits = (
        (ema_fast.shift(1) >= ema_slow.shift(1))
        & (ema_fast < ema_slow)
    ).fillna(False)

    return entries, exits