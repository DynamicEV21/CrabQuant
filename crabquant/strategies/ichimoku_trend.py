"""
Ichimoku Trend Strategy

Simplified Ichimoku Cloud trend following.
Enters on Tenkan/Kijun cross above the cloud.
"""

import pandas as pd


DEFAULT_PARAMS = {}

PARAM_GRID = {}

DESCRIPTION = (
    "Simplified Ichimoku Cloud trend following. "
    "Enters when Tenkan-sen crosses above Kijun-sen and price is above the cloud. "
    "Exits when price falls below Senkou Span A (cloud top). "
    "Parameter-free — pure price action. Works best on trending instruments."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    _ = params  # No parameters
    close = df["close"]
    high = df["high"]
    low = df["low"]

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    above_cloud = close > span_a
    tk_cross = (tenkan.shift(1) < kijun.shift(1)) & (tenkan > kijun)

    entries = (above_cloud & tk_cross).fillna(False)
    exits = (close < span_a).fillna(False)

    return entries, exits
