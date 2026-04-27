import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "bb_len": 20,
    "bb_std": 1.5,
    "rsi_len": 14,
    "rsi_oversold": 40,
    "stoch_len": 14,
    "stoch_oversold": 30,
    "near_band_pct": 0.02,
}

DESCRIPTION = (
    "Relaxed mean reversion using 2-of-3 confluence: BB near lower band, "
    "RSI oversold, Stochastic oversold. Wider BB (1.5 std) and looser "
    "thresholds increase trade frequency. Exits on BB middle reversion."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    bb = cached_indicator("bbands", close, length=p["bb_len"], std=p["bb_std"])
    bbl = bb[[c for c in bb.columns if c.startswith("BBL")][0]]
    bbm = bb[[c for c in bb.columns if c.startswith("BBM")][0]]

    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    stoch = cached_indicator("stoch", df["high"], df["low"], close, k=p["stoch_len"], d=3, smooth_k=3)
    stoch_k = stoch[[c for c in stoch.columns if c.startswith("STOCHk")][0]]

    band_range = bbm - bbl
    near_lower = close <= (bbl + band_range * p["near_band_pct"])
    rsi_oversold = rsi < p["rsi_oversold"]
    stoch_oversold = stoch_k < p["stoch_oversold"]

    confluence = rsi_oversold.astype(int) + stoch_oversold.astype(int) + near_lower.astype(int)
    entries = (near_lower & (confluence >= 2)).fillna(False)

    exits = (close >= bbm).fillna(False)

    return entries, exits