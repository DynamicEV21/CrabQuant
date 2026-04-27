"""
MACD Momentum + Trend + Volume Strategy for SPY

Dual-entry MACD momentum strategy optimized for SPY's moderate volatility:
- Histogram crossing above zero OR strengthening while positive = entry trigger
- Shorter MACD params (10/21/7) for more responsive signals than standard
- EMA trend filter ensures we only trade in established uptrends
- Relaxed volume filter (0.0-1.0x) — optimizer can disable if volume hurts on liquid SPY
- Exit on histogram dropping below threshold for responsive risk management

Key improvement over ROC approach: MACD histogram has TWO entry paths
(cross zero + strengthen while positive), significantly increasing signal
frequency from ~9 trades to potentially 20-40 while maintaining quality.
"""

from itertools import product

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "macd_fast": 10,
    "macd_slow": 21,
    "macd_signal": 7,
    "sma_len": 30,
    "volume_window": 20,
    "volume_mult": 0.8,
    "exit_hist": -0.3,
}

PARAM_GRID = {
    "macd_fast": [6, 8, 10, 12],
    "macd_slow": [17, 21, 26],
    "macd_signal": [5, 7, 9],
    "sma_len": [20, 30, 50],
    "volume_window": [10, 20],
    "volume_mult": [0.0, 0.6, 0.8, 1.0],
    "exit_hist": [-0.5, -0.3, -0.1, 0.0],
}

DESCRIPTION = (
    "MACD histogram dual-entry momentum with EMA trend filter and volume confirmation. "
    "Enters on histogram crossing above zero OR strengthening while positive, "
    "when price is above SMA and optionally volume is above average. "
    "Exits when histogram drops below exit threshold. "
    "Shorter MACD params tuned for SPY's moderate volatility profile."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    # MACD calculation
    macd = cached_indicator(
        "macd",
        close,
        fast=p["macd_fast"],
        slow=p["macd_slow"],
        signal=p["macd_signal"],
    )
    hist_col = f"MACDh_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"
    hist = macd[hist_col]

    # Trend filter: price above SMA
    sma = cached_indicator("sma", close, length=p["sma_len"])
    trend_ok = close > sma

    # Volume confirmation (can be disabled with volume_mult=0.0)
    volume_avg = cached_indicator("sma", volume, length=p["volume_window"])
    if p["volume_mult"] > 0:
        volume_ok = volume > (volume_avg * p["volume_mult"])
    else:
        volume_ok = pd.Series(True, index=close.index)

    # Entry: dual-path momentum detection
    hist_prev = hist.shift(1)
    hist_prev2 = hist.shift(2)

    # Path 1: Histogram crossing above zero (momentum shift)
    mom_cross = (hist_prev <= 0) & (hist > 0)

    # Path 2: Histogram strengthening while already positive (momentum acceleration)
    mom_strong = (hist_prev < hist_prev2) & (hist > hist_prev) & (hist > 0)

    entries = (trend_ok & volume_ok & (mom_cross | mom_strong)).fillna(False)

    # Exit: histogram drops below threshold
    exits = (hist < p["exit_hist"]).fillna(False)

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

    # Deduplicate MACD combos
    all_macd_combos = set()
    for vals in combos:
        params = dict(zip(keys, vals))
        all_macd_combos.add((params["macd_fast"], params["macd_slow"], params["macd_signal"]))

    macd_cache = {}
    for mf, ms, msig in all_macd_combos:
        macd = cached_indicator("macd", close, fast=mf, slow=ms, signal=msig)
        hist_col = f"MACDh_{mf}_{ms}_{msig}"
        macd_cache[(mf, ms, msig)] = macd[hist_col]

    # Deduplicate SMA and volume windows
    all_sma_lens = sorted(set(pg["sma_len"]))
    all_vol_windows = sorted(set(pg["volume_window"]))
    sma_cache = {l: cached_indicator("sma", close, length=l) for l in all_sma_lens}
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_windows}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        mf, ms, msig = params["macd_fast"], params["macd_slow"], params["macd_signal"]
        hist = macd_cache[(mf, ms, msig)]

        sma = sma_cache[params["sma_len"]]
        trend_ok = close > sma

        volume_avg = vol_avg_cache[params["volume_window"]]
        if params["volume_mult"] > 0:
            volume_ok = volume > (volume_avg * params["volume_mult"])
        else:
            volume_ok = pd.Series(True, index=close.index)

        hist_prev = hist.shift(1)
        hist_prev2 = hist.shift(2)
        mom_cross = (hist_prev <= 0) & (hist > 0)
        mom_strong = (hist_prev < hist_prev2) & (hist > hist_prev) & (hist > 0)

        e = (trend_ok & volume_ok & (mom_cross | mom_strong)).fillna(False)
        x = (hist < params["exit_hist"]).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
