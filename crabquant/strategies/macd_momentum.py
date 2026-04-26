"""
MACD Momentum Strategy

MACD histogram momentum shift with 200 SMA trend filter.
Best performer: AMD (Sharpe 2.15, 208% return), GOOGL (Sharpe 2.09).
The strongest strategy found in initial testing.
"""

import pandas as pd
import pandas_ta


DEFAULT_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "exit_hist": -1,
}

PARAM_GRID = {
    "macd_fast": [8, 12],
    "macd_slow": [21, 26],
    "macd_signal": [7, 9],
    "exit_hist": [-2, -1, -0.5],
}

DESCRIPTION = (
    "MACD histogram momentum shift with 200 SMA trend filter. "
    "Enters on histogram turning positive or strengthening above zero, "
    "only when price is above the 200 SMA (long-term uptrend). "
    "Exits when histogram drops below exit threshold. "
    "Strongest strategy in initial testing — works best on high-momentum stocks."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    macd = pandas_ta.macd(
        close,
        fast=p["macd_fast"],
        slow=p["macd_slow"],
        signal=p["macd_signal"],
    )
    hist_col = f"MACDh_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"
    hist = macd[hist_col]

    hist_prev = hist.shift(1)
    hist_prev2 = hist.shift(2)

    # Histogram turning positive
    mom_shift = (hist_prev < 0) & (hist > 0)
    # Or histogram strengthening while positive
    mom_strong = (hist_prev < hist_prev2) & (hist > hist_prev) & (hist > 0)

    # Trend filter
    sma200 = pandas_ta.sma(close, length=200)
    trend_ok = close > sma200

    entries = (trend_ok & (mom_shift | mom_strong)).fillna(False)
    exits = (hist < p["exit_hist"]).fillna(False)

    return entries, exits
