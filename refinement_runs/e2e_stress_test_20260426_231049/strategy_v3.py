"""
EMA Crossover with Volume Trend Confirmation for SPY

Proven EMA 9/21 crossover base (Sharpe 2.10 in QF backtests).
Volume trend filter confirms building interest without killing signals.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "fast_len": 9,
    "slow_len": 21,
    "volume_window": 20,
    "volume_trend_len": 5,
}

DESCRIPTION = (
    "EMA 9/21 crossover with rising volume trend confirmation for SPY. "
    "Enters on golden cross when 20-day volume SMA is rising over 5 days. "
    "Exits on death cross. Volume trend filter confirms building momentum "
    "without depending on single-day volume spikes that miss valid crossovers."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    ema_fast = cached_indicator("ema", close, length=p["fast_len"])
    ema_slow = cached_indicator("ema", close, length=p["slow_len"])
    volume_sma = cached_indicator("sma", volume, length=p["volume_window"])

    golden_cross = (
        (ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)
    )
    death_cross = (
        (ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)
    )

    volume_rising = volume_sma > volume_sma.shift(p["volume_trend_len"])

    entries = (golden_cross & volume_rising).fillna(False)
    exits = death_cross.fillna(False)

    return entries, exits