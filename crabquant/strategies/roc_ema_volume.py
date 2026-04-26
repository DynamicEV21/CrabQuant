"""
ROC + EMA + Volume + ATR Trailing Stop Strategy

Momentum entry via ROC, EMA trend filter, volume confirmation, with ATR trailing stop exit.
Sharpe 2.21 in QuantFactory backtests. Combines trend-following entry with
protective exit that locks in profits.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "roc_len": 10,
    "ema_len": 20,
    "vol_sma_len": 20,
    "atr_len": 14,
    "atr_mult": 2.0,
    "trailing_len": 20,
}

PARAM_GRID = {
    "roc_len": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 25],
    "ema_len": [8, 10, 12, 14, 15, 16, 18, 20, 22, 25, 28, 30, 35, 40, 45],
    "vol_sma_len": [8, 10, 12, 14, 15, 16, 18, 20, 22, 25, 28, 30, 35, 40, 45],
    "atr_len": [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 23, 25],
    "atr_mult": [1.0, 1.25, 1.4, 1.5, 1.6, 1.75, 1.9, 2.0, 2.1, 2.25, 2.4, 2.5, 2.6, 2.75, 2.9, 3.0, 3.2],
    "trailing_len": [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 28],
}

DESCRIPTION = (
    "ROC + EMA + Volume entry with ATR trailing stop exit. "
    "Enters when ROC is positive, price above EMA, and volume above its SMA. "
    "Exits via ATR trailing stop (rolling high minus ATR * multiplier). "
    "Sharpe 2.21 in QF backtests. Best for trending stocks with momentum."
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
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    roc = cached_indicator("roc", close, length=p["roc_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    vol_sma = volume.rolling(window=p["vol_sma_len"]).mean()
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])

    # ATR trailing stop: rolling max of close minus ATR * multiplier
    atr_stop = close.rolling(window=p["trailing_len"]).max() - atr * p["atr_mult"]

    entries = (
        (roc > 0)
        & (close > ema)
        & (volume > vol_sma)
    ).fillna(False)

    exits = (close < atr_stop).fillna(False)

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
    volume = df["volume"]

    # Pre-compute ROC for all unique lengths
    all_roc_lens = sorted(set(pg["roc_len"]))
    roc_cache = {l: cached_indicator("roc", close, length=l) for l in all_roc_lens}

    # Pre-compute EMA for all unique lengths
    all_ema_lens = sorted(set(pg["ema_len"]))
    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_ema_lens}

    # Pre-compute volume SMA for all unique lengths
    all_vol_sma_lens = sorted(set(pg["vol_sma_len"]))
    vol_sma_cache = {l: volume.rolling(window=l).mean() for l in all_vol_sma_lens}

    # Pre-compute ATR for all unique lengths
    all_atr_lens = sorted(set(pg["atr_len"]))
    atr_cache = {l: cached_indicator("atr", high, low, close, length=l) for l in all_atr_lens}

    # Pre-compute rolling max for all unique trailing lengths
    all_trailing_lens = sorted(set(pg["trailing_len"]))
    rolling_max_cache = {l: close.rolling(window=l).max() for l in all_trailing_lens}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        rl = params["roc_len"]
        el = params["ema_len"]
        vsl = params["vol_sma_len"]
        al = params["atr_len"]
        am = params["atr_mult"]
        tl = params["trailing_len"]

        roc = roc_cache[rl]
        ema = ema_cache[el]
        vsm = vol_sma_cache[vsl]
        atr = atr_cache[al]
        rmax = rolling_max_cache[tl]

        atr_stop = rmax - atr * am

        e = (
            (roc > 0)
            & (close > ema)
            & (volume > vsm)
        ).fillna(False)
        x = (close < atr_stop).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
