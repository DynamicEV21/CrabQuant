import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "regime_rsi_len": 50,
    "regime_rsi_thresh": 48,
    "dip_rsi_len": 14,
    "dip_rsi_level": 45,
    "exit_rsi_level": 65,
    "atr_len": 14,
    "atr_mult": 2.0,
}

DESCRIPTION = (
    "RSI regime dip strategy for SPY. 50-period RSI defines bullish regime, "
    "14-period RSI times entry on dips below 45. Exits on RSI recovery above 65, "
    "regime reversal, or ATR trailing stop. Avoids bear markets via regime filter."
)

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]

    regime_rsi = cached_indicator("rsi", close, length=p["regime_rsi_len"])
    dip_rsi = cached_indicator("rsi", close, length=p["dip_rsi_len"])
    atr = cached_indicator("atr", high, df["low"], close, length=p["atr_len"])

    bullish = regime_rsi > p["regime_rsi_thresh"]
    dip = dip_rsi < p["dip_rsi_level"]
    entries = (bullish & dip).fillna(False)

    rsi_exit = dip_rsi > p["exit_rsi_level"]
    regime_exit = regime_rsi < (p["regime_rsi_thresh"] - 8)
    rolling_high = high.rolling(window=p["atr_len"], min_periods=1).max()
    trailing_stop = rolling_high - (atr * p["atr_mult"])
    atr_exit = close < trailing_stop

    exits = (rsi_exit | regime_exit | atr_exit).fillna(False)
    return entries, exits