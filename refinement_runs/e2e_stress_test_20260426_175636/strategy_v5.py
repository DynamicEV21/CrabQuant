"""
EMA Crossover Strategy for SPY

Pure EMA fast/slow crossover optimized for SPY.
- Shorter EMAs than default to generate sufficient trade count in 2y window
- Golden cross entry, death cross exit
- No volume filter — avoids SPY liquidity noise that killed previous attempts

Proven pattern: ema_crossover (Sharpe 2.10 in QF backtests).
SPY-specific tuning: shorter fast EMA (5-12) for more crossover signals.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "fast_len": 8,
    "slow_len": 21,
}

PARAM_GRID = {
    "fast_len": [3, 5, 7, 8, 9, 10, 12],
    "slow_len": [10, 15, 18, 21, 25, 30, 40, 50],
}

DESCRIPTION = (
    "EMA crossover for SPY. "
    "Enters on golden cross (fast EMA crosses above slow EMA). "
    "Exits on death cross (fast EMA crosses below slow EMA). "
    "Shorter EMAs optimized for SPY's 2-year window to maximize trade count. "
    "No volume filter — avoids SPY liquidity noise."
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
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    ema_fast = cached_indicator("ema", close, length=p["fast_len"])
    ema_slow = cached_indicator("ema", close, length=p["slow_len"])

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
    pg = param_grid or PARAM_GRID
    valid_combos = _valid_combos(pg)

    close = df["close"]

    all_fast_lens = sorted(set(c["fast_len"] for c in valid_combos))
    all_slow_lens = sorted(set(c["slow_len"] for c in valid_combos))
    all_lens = sorted(set(all_fast_lens) | set(all_slow_lens))

    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_lens}

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
