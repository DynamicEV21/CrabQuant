"""
RSI Crossover Strategy

Fast/slow RSI crossover with regime filter.
Best performers: TXN (Sharpe 1.56), GOOGL (Sharpe 1.68).
Works well in trending markets with mean-reverting pullbacks.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator, clear_cache


DEFAULT_PARAMS = {
    "fast_len": 7,
    "slow_len": 21,
    "regime_len": 50,
    "regime_bull": 55,
    "exit_level": 40,
}

PARAM_GRID = {
    "fast_len": [3, 5, 7, 10, 14, 21],
    "slow_len": [14, 21, 28, 35, 42],
    "regime_len": [30, 50, 70, 100],
    "regime_bull": [45, 50, 55, 60, 65],
    "exit_level": [30, 35, 40, 45, 50],
}

DESCRIPTION = (
    "Fast/slow RSI crossover with regime filter. "
    "Enters when fast RSI crosses above slow RSI in a bullish regime. "
    "Exits when fast RSI drops below exit_level. "
    "Best in trending markets with regular pullbacks."
)


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

    rsi_fast = pandas_ta.rsi(close, length=p["fast_len"])
    rsi_slow = pandas_ta.rsi(close, length=p["slow_len"])
    regime = pandas_ta.rsi(close, length=p["regime_len"])

    entries = (
        (rsi_fast.shift(1) < rsi_slow.shift(1))
        & (rsi_fast > rsi_slow)
        & (regime > p["regime_bull"])
    ).fillna(False)

    exits = (rsi_fast < p["exit_level"]).fillna(False)

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
    keys = list(pg.keys())  # fast_len, slow_len, regime_len, regime_bull, exit_level
    combos = list(product(*(pg[k] for k in keys)))

    # Deduplicate indicator lengths
    all_fast_lens = sorted(set(pg["fast_len"]))
    all_slow_lens = sorted(set(pg["slow_len"]))
    all_regime_lens = sorted(set(pg["regime_len"]))

    close = df["close"]
    rsi_fast = {l: cached_indicator("rsi", close, length=l) for l in all_fast_lens}
    rsi_slow = {l: cached_indicator("rsi", close, length=l) for l in all_slow_lens}
    regime = {l: cached_indicator("rsi", close, length=l) for l in all_regime_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        fl, sl, rl = params["fast_len"], params["slow_len"], params["regime_len"]
        rb, el = params["regime_bull"], params["exit_level"]

        e = (
            (rsi_fast[fl].shift(1) < rsi_slow[sl].shift(1))
            & (rsi_fast[fl] > rsi_slow[sl])
            & (regime[rl] > rb)
        ).fillna(False)
        x = (rsi_fast[fl] < el).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list