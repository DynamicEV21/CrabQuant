import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "rsi_long_len": 50,
    "rsi_short_len": 7,
    "regime_threshold": 50,
    "dip_level": 35,
    "exit_level": 55,
    "ema_len": 50,
}

DESCRIPTION = "RSI regime filter with dip timing. Long RSI determines bullish regime. Short RSI times entry on dips below dip_level in bullish regime. Exits when short RSI recovers above exit_level or regime turns bearish. Mean-reversion within trends."


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    rsi_long = cached_indicator("rsi", close, length=p["rsi_long_len"])
    rsi_short = cached_indicator("rsi", close, length=p["rsi_short_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])

    bullish_regime = (rsi_long > p["regime_threshold"]) & (close > ema)
    dip_signal = rsi_short < p["dip_level"]

    entries = (bullish_regime & dip_signal).fillna(False)

    recovery = rsi_short > p["exit_level"]
    regime_break = (rsi_long < p["regime_threshold"]) | (close < ema)
    exits = (recovery | regime_break).fillna(False)

    return entries, exits