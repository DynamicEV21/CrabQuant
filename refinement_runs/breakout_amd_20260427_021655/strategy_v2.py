"""
Donchian Volume Breakout with Trend Filter

Donchian Channel breakout with volume confirmation and EMA trend filter.
Exits at channel midpoint for tight risk control on volatile semis.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "dc_len": 15,
    "vol_len": 20,
    "vol_mult": 1.2,
    "trend_ema": 30,
}

DESCRIPTION = (
    "Donchian Channel breakout with volume confirmation and EMA trend filter. "
    "Enters when price breaks above N-day high with volume spike and price above "
    "trend EMA. Exits at channel midpoint for fast risk control on failed breakouts. "
    "Shorter channel and relaxed filters increase trade count for AMD volatility."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    donch_high = high.rolling(p["dc_len"]).max()
    donch_low = low.rolling(p["dc_len"]).min()
    dc_mid = (donch_high + donch_low) / 2

    breakout = close > donch_high.shift(1)
    vol_spike = volume > volume.rolling(p["vol_len"]).mean() * p["vol_mult"]
    trend = close > cached_indicator("ema", close, length=p["trend_ema"])

    entries = (breakout & vol_spike & trend).fillna(False)
    exits = (close < dc_mid).fillna(False)

    return entries, exits