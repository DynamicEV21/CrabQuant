"""
ROC Momentum + EMA Trend + Volume Strategy for SPY

Momentum-driven strategy with trend filter and volume confirmation:
- ROC crossing above zero signals momentum shift
- Price above EMA confirms established uptrend
- Volume above SMA confirms institutional participation
- Exits cleanly on momentum reversal (ROC drops below threshold) or trend break (price < EMA)

Clean stateless logic - no trailing stop complexity.
Designed to avoid regime fragility by exiting when momentum or trend breaks.
"""

from itertools import product

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 12,
    "ema_len": 50,
    "volume_window": 20,
    "volume_mult": 1.0,
    "exit_roc": 0.0,
}

PARAM_GRID = {
    "roc_len": [5, 8, 10, 12, 15, 20],
    "ema_len": [20, 30, 50],
    "volume_window": [10, 20, 30],
    "volume_mult": [0.0, 0.8, 1.0, 1.2],
    "exit_roc": [-2.0, -1.0, -0.5, 0.0],
}

DESCRIPTION = (
    "ROC momentum crossover with EMA trend filter and volume confirmation. "
    "Enters when ROC crosses above zero, price is above EMA, and volume is above average. "
    "Exits when ROC drops below exit threshold or price falls below EMA. "
    "Clean stateless logic avoids trailing stop bugs and naturally adapts to regime changes."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    # Momentum: ROC crossing above zero
    roc = cached_indicator("roc", close, length=p["roc_len"])
    roc_cross_up = (roc.shift(1) <= 0) & (roc > 0)

    # Trend filter: price above EMA
    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    # Volume confirmation
    volume_avg = cached_indicator("sma", volume, length=p["volume_window"])
    volume_ok = volume > (volume_avg * p["volume_mult"])

    # Entry: momentum crossover + trend + volume
    entries = (roc_cross_up & trend_ok & volume_ok).fillna(False)

    # Exit: momentum reversal OR trend break
    momentum_dead = roc < p["exit_roc"]
    trend_broken = close < ema
    exits = (momentum_dead | trend_broken).fillna(False)

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

    # Deduplicate indicator calculations
    all_roc_lens = sorted(set(pg["roc_len"]))
    all_ema_lens = sorted(set(pg["ema_len"]))
    all_vol_windows = sorted(set(pg["volume_window"]))

    roc_cache = {l: cached_indicator("roc", close, length=l) for l in all_roc_lens}
    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_ema_lens}
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_windows}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))

        roc = roc_cache[params["roc_len"]]
        ema = ema_cache[params["ema_len"]]
        volume_avg = vol_avg_cache[params["volume_window"]]

        roc_cross_up = (roc.shift(1) <= 0) & (roc > 0)
        trend_ok = close > ema
        volume_ok = volume > (volume_avg * params["volume_mult"])

        e = (roc_cross_up & trend_ok & volume_ok).fillna(False)

        momentum_dead = roc < params["exit_roc"]
        trend_broken = close < ema
        x = (momentum_dead | trend_broken).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
