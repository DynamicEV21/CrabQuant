"""
EMA Crossover with Volume Confirmation for SPY

Fast/slow EMA crossover for momentum direction.
Volume above moving average for confirmation.
Simple and robust - adapts to all regimes by following trend.
"""

from itertools import product

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "fast_len": 9,
    "slow_len": 21,
    "volume_len": 20,
    "volume_mult": 0.8,
}

PARAM_GRID = {
    "fast_len": [5, 8, 9, 12, 15],
    "slow_len": [15, 20, 21, 30, 50],
    "volume_len": [10, 20, 30],
    "volume_mult": [0.6, 0.8, 1.0, 1.2],
}

DESCRIPTION = (
    "EMA crossover with volume confirmation. "
    "Enters on golden cross (fast EMA crosses above slow EMA) with above-average volume. "
    "Exits on death cross (fast EMA crosses below slow EMA). "
    "Simple trend-following momentum with volume filter to reduce false signals."
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
    volume = df["volume"]

    ema_fast = cached_indicator("ema", close, length=p["fast_len"])
    ema_slow = cached_indicator("ema", close, length=p["slow_len"])
    vol_sma = cached_indicator("sma", volume, length=p["volume_len"])

    golden_cross = (
        (ema_fast.shift(1) < ema_slow.shift(1))
        & (ema_fast > ema_slow)
    )
    volume_ok = volume > (vol_sma * p["volume_mult"])

    entries = (golden_cross & volume_ok).fillna(False)

    exits = (
        (ema_fast.shift(1) > ema_slow.shift(1))
        & (ema_fast < ema_slow)
    ).fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Generate signals for ALL valid param combinations at once (vectorized)."""
    pg = param_grid or PARAM_GRID
    valid_combos = _valid_combos(pg)

    close = df["close"]
    volume = df["volume"]

    all_lens = sorted(set(
        c["fast_len"] for c in valid_combos
    ) | set(
        c["slow_len"] for c in valid_combos
    ))
    all_vol_lens = sorted(set(c["volume_len"] for c in valid_combos))

    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_lens}
    vol_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, params in enumerate(valid_combos):
        ef = ema_cache[params["fast_len"]]
        es = ema_cache[params["slow_len"]]
        vs = vol_cache[params["volume_len"]]

        golden = (ef.shift(1) < es.shift(1)) & (ef > es)
        vol_ok = volume > (vs * params["volume_mult"])
        e = (golden & vol_ok).fillna(False)

        x = (
            (ef.shift(1) > es.shift(1))
            & (ef < es)
        ).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
