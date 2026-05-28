"""
BB + Stochastic + MACD Triple Confluence Strategy

Mean-reversion entry requiring confluence from Bollinger Bands, Stochastic,
and MACD histogram. Sharpe 2.02 in QuantFactory backtests. Strong
multi-indicator confirmation reduces false entries.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "bb_len": 20,
    "bb_std": 2.0,
    "stoch_k": 14,
    "stoch_d": 3,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
}

PARAM_GRID = {
    "bb_len": [15, 20, 25],
    "bb_std": [1.5, 2.0, 2.5],
    "stoch_k": [14, 21],
    "stoch_d": [3, 5],
    "macd_fast": [12],
    "macd_slow": [26],
    "macd_signal": [9],
}

DESCRIPTION = (
    "BB + Stochastic + MACD triple confluence mean reversion. "
    "Enters when price below BB mid, Stoch K < 20 with K > D, and MACD histogram rising. "
    "Exits on Stoch K > 80 with K < D, or price above BB upper. "
    "Sharpe 2.02 in QF backtests. Best for range-bound or mean-reverting markets."
)


# pandas_ta column indices (positional, immune to naming quirks):
# bbands: 0=BBL, 1=BBM, 2=BBU, 3=BBB, 4=BBP
# stoch:  0=STOCHk, 1=STOCHd, 2=STOCHh
# macd:   0=MACD, 1=MACDh, 2=MACDs
_BB_LOWER, _BB_MID, _BB_UPPER = 0, 1, 2
_STOCH_K, _STOCH_D = 0, 1
_MACD_HIST = 1


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Generate entry/exit signals.

    Args:
        df: DataFrame with columns open, high, low, close, volume
        params: Strategy parameters (uses defaults if None)

    Returns:
        (entries, exits) as boolean Series
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    bb = cached_indicator("bbands", close, length=p["bb_len"], std=p["bb_std"])
    stoch = cached_indicator("stoch", high, low, close, k=p["stoch_k"], d=p["stoch_d"])
    macd = cached_indicator("macd", close, fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"])

    bb_mid = bb.iloc[:, _BB_MID]
    bb_upper = bb.iloc[:, _BB_UPPER]
    stoch_k = stoch.iloc[:, _STOCH_K]
    stoch_d = stoch.iloc[:, _STOCH_D]
    macd_h = macd.iloc[:, _MACD_HIST]

    entries = (
        (close < bb_mid)
        & (stoch_k < 20)
        & (stoch_k > stoch_d)
        & (macd_h > macd_h.shift(1))
    ).fillna(False)

    exits = (
        ((stoch_k > 80) & (stoch_k < stoch_d))
        | (close > bb_upper)
    ).fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Generate signals for ALL param combinations at once (vectorized).

    Returns:
        (entries_df, exits_df, param_list) where each DataFrame has one column per combo.
    """
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Deduplicate indicator computations
    all_bb_configs = sorted(set((bl, bs) for bl in pg["bb_len"] for bs in pg["bb_std"]))
    bb_cache = {}
    for bl, bs in all_bb_configs:
        bb = cached_indicator("bbands", close, length=bl, std=bs)
        bb_cache[(bl, bs)] = {
            "mid": bb.iloc[:, _BB_MID],
            "lower": bb.iloc[:, _BB_LOWER],
            "upper": bb.iloc[:, _BB_UPPER],
        }

    all_stoch_configs = sorted(set((sk, sd) for sk in pg["stoch_k"] for sd in pg["stoch_d"]))
    stoch_cache = {}
    for sk, sd in all_stoch_configs:
        stoch = cached_indicator("stoch", high, low, close, k=sk, d=sd)
        stoch_cache[(sk, sd)] = {
            "k": stoch.iloc[:, _STOCH_K],
            "d": stoch.iloc[:, _STOCH_D],
        }

    # MACD params are fixed in the grid but compute once anyway
    all_macd_configs = set(
        (mf, ms, msi)
        for mf in pg["macd_fast"]
        for ms in pg["macd_slow"]
        for msi in pg["macd_signal"]
    )
    macd_cache = {}
    for mf, ms, msi in all_macd_configs:
        macd = cached_indicator("macd", close, fast=mf, slow=ms, signal=msi)
        macd_cache[(mf, ms, msi)] = macd.iloc[:, _MACD_HIST]

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        bl = params["bb_len"]
        bs = params["bb_std"]
        sk = params["stoch_k"]
        sd = params["stoch_d"]
        mf = params["macd_fast"]
        ms = params["macd_slow"]
        msi = params["macd_signal"]

        bb_mid = bb_cache[(bl, bs)]["mid"]
        bb_upper = bb_cache[(bl, bs)]["upper"]
        stk = stoch_cache[(sk, sd)]["k"]
        std_ = stoch_cache[(sk, sd)]["d"]
        macd_h = macd_cache[(mf, ms, msi)]

        e = (
            (close < bb_mid)
            & (stk < 20)
            & (stk > std_)
            & (macd_h > macd_h.shift(1))
        ).fillna(False)
        x = (
            ((stk > 80) & (stk < std_))
            | (close > bb_upper)
        ).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
