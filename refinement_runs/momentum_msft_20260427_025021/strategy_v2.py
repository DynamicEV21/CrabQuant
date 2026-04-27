"""
MSFT MACD Trend Strategy

MACD histogram momentum with EMA trend filter. Relaxed entry conditions
to generate sufficient trades while maintaining trend alignment.
"""

import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_len": 50,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "exit_hist": -0.3,
}

DESCRIPTION = (
    "MACD histogram momentum with EMA trend filter. "
    "Enters on histogram crossing zero or strengthening while positive, "
    "when price is above EMA. No volume or RSI filters to maximize "
    "trade frequency. Exits when histogram drops below threshold."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    macd = cached_indicator("macd", close, fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"])
    hist_col = f"MACDh_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"
    hist = macd[hist_col]
    hist_prev = hist.shift(1)
    hist_prev2 = hist.shift(2)

    mom_cross = (hist_prev <= 0) & (hist > 0)
    mom_strong = (hist_prev < hist_prev2) & (hist > hist_prev) & (hist > 0)

    entries = (trend_ok & (mom_cross | mom_strong)).fillna(False)
    exits = (hist < p["exit_hist"]).fillna(False)

    return entries, exits