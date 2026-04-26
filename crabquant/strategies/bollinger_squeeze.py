"""
Bollinger Squeeze Strategy

Bollinger Band squeeze (low volatility) followed by breakout with volume.
Enters when bands narrow then price breaks above upper band.
"""

from itertools import product

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


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Generate signals for ALL param combinations at once (vectorized)."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    close = df["close"]
    volume = df["volume"]

    # Deduplicate BB combos
    all_bb_combos = set()
    for vals in combos:
        params = dict(zip(keys, vals))
        all_bb_combos.add((params["bb_len"], params["bb_std"]))

    bb_cache = {}
    for bl, bs in all_bb_combos:
        bb = pandas_ta.bbands(close, length=bl, std=bs)
        bbu_col = [c for c in bb.columns if c.startswith("BBU")][0]
        bbl_col = [c for c in bb.columns if c.startswith("BBL")][0]
        bbm_col = [c for c in bb.columns if c.startswith("BBM")][0]
        bb_cache[(bl, bs)] = (bb[bbu_col], bb[bbl_col], bb[bbm_col])

    # Deduplicate squeeze lengths
    all_squeeze_lens = sorted(set(pg["squeeze_len"]))
    vol_avg_20 = volume.rolling(20).mean()

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        bbu, bbl, bbm = bb_cache[(params["bb_len"], params["bb_std"])]

        bb_width = (bbu - bbl) / bbm
        bb_width_avg = bb_width.rolling(params["squeeze_len"]).mean()
        squeeze = bb_width < bb_width_avg * params["squeeze_mult"]
        breakout_up = close > bbu
        vol_confirm = volume > vol_avg_20 * params["vol_mult"]

        e = (squeeze.shift(1) & breakout_up & vol_confirm).fillna(False)
        x = (close < bbm).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
