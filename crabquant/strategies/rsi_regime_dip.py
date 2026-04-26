"""
RSI Regime Filter + Dip Timing Strategy

Uses a long-period RSI as regime filter (bullish/bearish) and a short-period
RSI for timing entries on dips within bullish regimes. Sharpe 2.18 in
QuantFactory backtests. Simple but effective trend-following with pullback entry.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "regime_len": 50,
    "timing_len": 14,
    "dip_level": 40,
    "recovery_level": 60,
    "regime_bull": 50,
}

PARAM_GRID = {
    "regime_len": [30, 40, 50, 60],
    "timing_len": [10, 14, 21],
    "dip_level": [30, 35, 40, 45],
    "recovery_level": [55, 60, 65],
    "regime_bull": [45, 50, 55],
}

DESCRIPTION = (
    "RSI regime filter with dip timing. "
    "Long RSI determines market regime (bullish when above threshold). "
    "Short RSI times entry on dips (below dip_level in bullish regime). "
    "Exits when short RSI recovers or regime turns bearish. "
    "Sharpe 2.18 in QF backtests. Works well in trending markets with pullbacks."
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

    regime_rsi = cached_indicator("rsi", close, length=p["regime_len"])
    timing_rsi = cached_indicator("rsi", close, length=p["timing_len"])

    bullish = regime_rsi > p["regime_bull"]

    entries = (
        bullish
        & (timing_rsi < p["dip_level"])
    ).fillna(False)

    exits = (
        (timing_rsi > p["recovery_level"])
        | (~bullish)
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

    # Pre-compute RSI for all unique regime lengths
    all_regime_lens = sorted(set(pg["regime_len"]))
    regime_rsi_cache = {l: cached_indicator("rsi", close, length=l) for l in all_regime_lens}

    # Pre-compute RSI for all unique timing lengths
    all_timing_lens = sorted(set(pg["timing_len"]))
    timing_rsi_cache = {l: cached_indicator("rsi", close, length=l) for l in all_timing_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        rl = params["regime_len"]
        tl = params["timing_len"]
        dip = params["dip_level"]
        recovery = params["recovery_level"]
        bull = params["regime_bull"]

        regime = regime_rsi_cache[rl]
        timing = timing_rsi_cache[tl]

        bullish = regime > bull

        e = (bullish & (timing < dip)).fillna(False)
        x = ((timing > recovery) | (~bullish)).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
