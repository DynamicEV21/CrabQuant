"""
Bollinger Squeeze Strategy

Bollinger Band squeeze (low volatility) followed by breakout with volume.
Enters when bands narrow then price breaks above upper band.
"""

import pandas as pd
import pandas_ta


DEFAULT_PARAMS = {
    "bb_len": 20,
    "bb_std": 2.0,
    "squeeze_len": 50,
    "squeeze_mult": 0.8,
    "vol_mult": 1.2,
}

PARAM_GRID = {
    "bb_len": [15, 20, 25],
    "bb_std": [1.5, 2.0, 2.5],
    "squeeze_len": [30, 50],
    "squeeze_mult": [0.7, 0.8, 0.9],
    "vol_mult": [1.0, 1.2, 1.5],
}

DESCRIPTION = (
    "Bollinger Band squeeze breakout with volume confirmation. "
    "Enters when BB width narrows below average (squeeze) then price "
    "breaks above upper band with above-average volume. "
    "Exits when price falls back to middle band. "
    "Classic volatility expansion play — works well before earnings or major moves."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    bb = pandas_ta.bbands(close, length=p["bb_len"], std=p["bb_std"])
    # pandas_ta column format: BBU_<len>_<std>_<std> (std duplicated)
    bbu_col = [c for c in bb.columns if c.startswith("BBU")][0]
    bbl_col = [c for c in bb.columns if c.startswith("BBL")][0]
    bbm_col = [c for c in bb.columns if c.startswith("BBM")][0]
    bbu = bb[bbu_col]
    bbl = bb[bbl_col]
    bbm = bb[bbm_col]

    bb_width = (bbu - bbl) / bbm
    bb_width_avg = bb_width.rolling(p["squeeze_len"]).mean()

    squeeze = bb_width < bb_width_avg * p["squeeze_mult"]
    breakout_up = close > bbu
    vol_confirm = volume > volume.rolling(20).mean() * p["vol_mult"]

    entries = (squeeze.shift(1) & breakout_up & vol_confirm).fillna(False)
    exits = (close < bbm).fillna(False)

    return entries, exits
