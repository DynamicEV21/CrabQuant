import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "rsi_long": 28,
    "rsi_short": 7,
    "regime_threshold": 55,
    "dip_level": 30,
    "ema_len": 50,
    "rsi_exit": 72,
}

DESCRIPTION = (
    "RSI regime dip with EMA trend filter. Long RSI confirms bullish regime, "
    "short RSI times pullback recovery entries. EMA ensures directional alignment. "
    "Exits on RSI overbought or EMA trend break."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    rsi_long = cached_indicator("rsi", close, length=p["rsi_long"])
    rsi_short = cached_indicator("rsi", close, length=p["rsi_short"])
    ema = cached_indicator("ema", close, length=p["ema_len"])

    bullish_regime = rsi_long > p["regime_threshold"]
    trend_ok = close > ema
    dip_recovery = (rsi_short.shift(1) < p["dip_level"]) & (rsi_short >= p["dip_level"])

    entries = (bullish_regime & trend_ok & dip_recovery).fillna(False)
    exits = ((rsi_short > p["rsi_exit"]) | (close < ema)).fillna(False)

    return entries, exits