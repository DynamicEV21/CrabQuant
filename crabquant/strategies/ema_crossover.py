"""
EMA Crossover Strategy

Pure EMA fast/slow crossover — the simplest winning strategy. Sharpe 2.10 in
QuantFactory backtests. Entry on golden cross (fast crosses above slow),
exit on death cross (fast crosses below slow).
"""

from itertools import product

import pandas as pd
import pandas_ta


DEFAULT_PARAMS = {
    "fast_len": 9,
    "slow_len": 21,
}

PARAM_GRID = {
    "fast_len": [5, 7, 9, 12],
    "slow_len": [15, 21, 26, 50],
}

DESCRIPTION = (
    "Pure EMA fast/slow crossover. "
    "Enters when fast EMA crosses above slow EMA (golden cross). "
    "Exits when fast EMA crosses below slow EMA (death cross). "
    "Sharpe 2.10 in QF backtests. Simplest winning strategy — "
    "works best in cleanly trending markets."
)


def _valid_combos(param_grid: dict) -> list[dict]:
    """Filter param combos where fast_len < slow_len."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    valid = []
    for vals in product(*(pg[k] for k in keys)):
        params = dict(zip(keys, vals))
        if params["fast_len"] < params["slow_len"]:
            valid.append(params)
    return valid


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

    ema_fast = pandas_ta.ema(close, length=p["fast_len"])
    ema_slow = pandas_ta.ema(close, length=p["slow_len"])

    entries = (
        (ema_fast.shift(1) < ema_slow.shift(1))
        & (ema_fast > ema_slow)
    ).fillna(False)

    exits = (
        (ema_fast.shift(1) > ema_slow.shift(1))
        & (ema_fast < ema_slow)
    ).fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Generate signals for ALL param combinations at once (vectorized).
    Only includes combos where fast_len < slow_len.

    Returns:
        (entries_df, exits_df, param_list) where each DataFrame has one column per combo.
    """
    pg = param_grid or PARAM_GRID
    valid_combos = _valid_combos(pg)

    close = df["close"]

    # Deduplicate: compute EMA once per unique length
    all_fast_lens = sorted(set(c["fast_len"] for c in valid_combos))
    all_slow_lens = sorted(set(c["slow_len"] for c in valid_combos))
    all_lens = sorted(set(all_fast_lens) | set(all_slow_lens))

    ema_cache = {l: pandas_ta.ema(close, length=l) for l in all_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, params in enumerate(valid_combos):
        fl = params["fast_len"]
        sl = params["slow_len"]
        ef = ema_cache[fl]
        es = ema_cache[sl]

        e = (
            (ef.shift(1) < es.shift(1))
            & (ef > es)
        ).fillna(False)
        x = (
            (ef.shift(1) > es.shift(1))
            & (ef < es)
        ).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
