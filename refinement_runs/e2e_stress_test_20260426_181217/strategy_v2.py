"""
MACD Momentum Strategy for SPY

MACD histogram momentum shift with EMA trend filter and volume confirmation.
Replaces ROC state-based entries with MACD event-based triggers.
Exits on histogram weakening instead of whipsaw-prone ATR trailing stop.
Optimized for SPY's smooth trend characteristics.
"""

from itertools import product

import pandas as pd

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "ema_len": 50,
    "volume_window": 20,
    "volume_mult": 1.0,
    "exit_hist": -0.3,
}

PARAM_GRID = {
    "macd_fast": [8, 10, 12, 15],
    "macd_slow": [21, 26, 32],
    "macd_signal": [7, 9, 11],
    "ema_len": [30, 50, 100],
    "volume_window": [15, 20, 30],
    "volume_mult": [0.9, 1.0, 1.1],
    "exit_hist": [-0.5, -0.3, -0.1, 0],
}

DESCRIPTION = (
    "MACD histogram momentum shift with EMA trend filter and volume confirmation. "
    "Enters on histogram crossing zero (event trigger) or strengthening while positive, "
    "when price is above EMA and volume is above average. "
    "Exits when histogram drops below exit threshold (momentum exhaustion). "
    "Designed for SPY's smooth momentum characteristics."
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
    hist_prev = hist.shift(1)
    hist_prev2 = hist.shift(2)

    # Trend filter: Price above EMA
    ema = cached_indicator("ema", close, length=p["ema_len"])
    trend_ok = close > ema

    # Volume confirmation
    volume_avg = cached_indicator("sma", volume, length=p["volume_window"])
    volume_ok = volume > (volume_avg * p["volume_mult"])

    # Entry event 1: Histogram crosses above zero
    mom_cross = (hist_prev <= 0) & (hist > 0)
    # Entry event 2: Histogram strengthening while already positive
    mom_strong = (hist_prev < hist_prev2) & (hist > hist_prev) & (hist > 0)

    entries = (trend_ok & volume_ok & (mom_cross | mom_strong)).fillna(False)

    # Exit: Histogram drops below threshold (momentum exhaustion)
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

    all_ema_lens = sorted(set(pg["ema_len"]))
    all_vol_windows = sorted(set(pg["volume_window"]))
    ema_cache = {l: cached_indicator("ema", close, length=l) for l in all_ema_lens}
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_windows}

    entries_cols = {}
    exits_cols = {}
    param_list = []

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        mf, ms, msig = params["macd_fast"], params["macd_slow"], params["macd_signal"]
        hist = macd_cache[(mf, ms, msig)]
        hist_prev = hist.shift(1)
        hist_prev2 = hist.shift(2)

        ema = ema_cache[params["ema_len"]]
        trend_ok = close > ema
        volume_avg = vol_avg_cache[params["volume_window"]]
        volume_ok = volume > (volume_avg * params["volume_mult"])

        mom_cross = (hist_prev <= 0) & (hist > 0)
        mom_strong = (hist_prev < hist_prev2) & (hist > hist_prev) & (hist > 0)

        e = (trend_ok & volume_ok & (mom_cross | mom_strong)).fillna(False)
        x = (hist < params["exit_hist"]).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
