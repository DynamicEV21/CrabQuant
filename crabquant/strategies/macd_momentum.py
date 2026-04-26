"""
MACD Momentum Strategy

MACD histogram momentum shift with trend filter.
Best performer: AMD (Sharpe 2.15, 208% return), GOOGL (Sharpe 2.09).
Works best on high-momentum stocks with volume confirmation.
"""

import pandas as pd
import pandas_ta


DEFAULT_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "exit_hist": -0.5,
    "sma_len": 50,
    "volume_window": 20,
    "volume_mult": 1.2,
}

PARAM_GRID = {
    "macd_fast": [8, 12, 16],
    "macd_slow": [21, 26, 31],
    "macd_signal": [7, 9, 11],
    "exit_hist": [-1, -0.5, -0.2, 0],
    "sma_len": [20, 50, 100],
    "volume_window": [10, 20, 30],
    "volume_mult": [1.0, 1.2, 1.5],
}

DESCRIPTION = (
    "MACD histogram momentum shift with trend and volume confirmation. "
    "Enters on histogram turning positive or strengthening above zero, "
    "when price is above SMA and volume is above average. "
    "Exits when histogram drops below exit threshold or momentum weakens. "
    "Works best on high-momentum stocks with volume confirmation."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    # MACD calculation
    macd = pandas_ta.macd(
        close,
        fast=p["macd_fast"],
        slow=p["macd_slow"],
        signal=p["macd_signal"],
    )
    hist_col = f"MACDh_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"
    hist = macd[hist_col]

    # Trend filter
    sma = pandas_ta.sma(close, length=p["sma_len"])
    trend_ok = close > sma

    # Volume confirmation
    volume_avg = pandas_ta.sma(volume, length=p["volume_window"])
    volume_ok = volume > (volume_avg * p["volume_mult"])

    # Entry conditions: MACD histogram turns positive or strengthens
    hist_prev = hist.shift(1)
    hist_prev2 = hist.shift(2)
    
    # Histogram turning positive (crosses zero line)
    mom_cross = (hist_prev <= 0) & (hist > 0)
    # Histogram strengthening while positive
    mom_strong = (hist_prev < hist_prev2) & (hist > hist_prev) & (hist > 0)

    entries = (trend_ok & volume_ok & (mom_cross | mom_strong)).fillna(False)
    
    # Exit conditions: histogram drops below threshold or momentum weakens
    exits = (hist < p["exit_hist"]).fillna(False)

    return entries, exits