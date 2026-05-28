"""
MACD Momentum Strategy

MACD histogram momentum shift with trend filter.
Best performer: AMD (Sharpe 2.15, 208% return), GOOGL (Sharpe 2.09).
Works best on high-momentum stocks with volume confirmation.
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator, clear_cache


DEFAULT_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "exit_hist": -0.5,
    "sma_len": 50,
    "volume_window": 20,
    "volume_mult": 1.2,
}

PARAM_GRID = {
    "macd_fast": [8, 12, 16],
    "macd_slow": [21, 26, 31],
    "macd_signal": [7, 9, 11],
    "exit_hist": [-1, -0.5, -0.2, 0],
    "sma_len": [20, 50, 100],
    "volume_window": [10, 20, 30],
    "volume_mult": [1.0, 1.2, 1.5],
}

DESCRIPTION = (
    "MACD histogram momentum shift with trend and volume confirmation. "
    "Enters on histogram turning positive or strengthening above zero, "
    "when price is above SMA and volume is above average. "
    "Exits when histogram drops below exit threshold or momentum weakens. "
    "Works best on high-momentum stocks with volume confirmation."
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

    # Trend filter
    sma = cached_indicator("sma", close, length=p["sma_len"])
    trend_ok = close > sma

    # Volume confirmation
    volume_avg = cached_indicator("sma", volume, length=p["volume_window"])
    volume_ok = volume > (volume_avg * p["volume_mult"])

    # Entry conditions: MACD histogram turns positive or strengthens
    hist_prev = hist.shift(1)
    hist_prev2 = hist.shift(2)
    
    # Histogram turning positive (crosses zero line)
    mom_cross = (hist_prev <= 0) & (hist > 0)
    # Histogram strengthening while positive
    mom_strong = (hist_prev < hist_prev2) & (hist > hist_prev) & (hist > 0)

    entries = (trend_ok & volume_ok & (mom_cross | mom_strong)).fillna(False)
    
    # Exit conditions: histogram drops below threshold or momentum weakens
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
        volume_ok = volume > (volume_avg * params["volume_mult"])

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