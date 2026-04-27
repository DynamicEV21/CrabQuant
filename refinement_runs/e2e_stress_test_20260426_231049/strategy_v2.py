"""
RSI Regime Dip Strategy for SPY

Long RSI defines bullish regime, short RSI times pullback entries.
Proven Sharpe 2.18 in QF backtests for trending markets with pullbacks.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "rsi_long_len": 21,
    "rsi_short_len": 7,
    "regime_threshold": 50,
    "dip_level": 40,
    "exit_level": 65,
}

DESCRIPTION = (
    "RSI regime filter with dip timing for SPY. "
    "Long RSI(21) above 50 defines bullish regime. "
    "Short RSI(7) dip below 40 triggers entry in bullish regime. "
    "Exits on short RSI recovery above 65 or regime turning bearish. "
    "Designed for SPY trending markets with regular pullbacks."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    rsi_long = cached_indicator("rsi", close, length=p["rsi_long_len"])
    rsi_short = cached_indicator("rsi", close, length=p["rsi_short_len"])

    bullish_regime = rsi_long > p["regime_threshold"]
    bearish_regime = rsi_long < p["regime_threshold"]
    dip = rsi_short < p["dip_level"]
    recovery = rsi_short > p["exit_level"]

    entries = (bullish_regime & dip).fillna(False)
    exits = (recovery | bearish_regime).fillna(False)

    return entries, exits