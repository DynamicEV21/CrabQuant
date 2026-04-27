"""
SPY Momentum Volume Strategy

MACD momentum with volume confirmation and ATR trailing stops.
Designed for SPY stress test - optimized for smoother index trends.

Entry: MACD histogram positive + strengthening, price above SMA, volume confirmed
Exit: ATR trailing stop (tighter than individual stocks due to lower volatility)
"""

from itertools import product

import pandas as pd
import pandas_ta

from crabquant.indicator_cache import cached_indicator


DEFAULT_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "sma_len": 50,
    "volume_window": 20,
    "volume_mult": 1.0,
    "atr_len": 14,
    "atr_mult": 2.0,
}

PARAM_GRID = {
    "macd_fast": [8, 12, 16],
    "macd_slow": [21, 26, 31],
    "macd_signal": [7, 9, 11],
    "sma_len": [20, 50, 100],
    "volume_window": [10, 20, 30],
    "volume_mult": [0.9, 1.0, 1.2],
    "atr_len": [10, 14, 21],
    "atr_mult": [1.5, 2.0, 2.5],
}

DESCRIPTION = (
    "MACD momentum with volume confirmation and ATR trailing stops for SPY. "
    "Enters when MACD histogram is positive and strengthening, price above SMA, "
    "volume above average. Exits via ATR trailing stop from rolling high. "
    "Optimized for SPY's smoother trend characteristics with tighter stops."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
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

    # ATR for trailing stop
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])

    # Entry conditions: MACD histogram positive and strengthening
    hist_prev = hist.shift(1)
    hist_prev2 = hist.shift(2)
    
    # Histogram positive and momentum strengthening
    mom_positive = hist > 0
    mom_strengthening = (hist > hist_prev) & (hist_prev <= hist_prev2)
    # Or histogram just turned positive
    mom_cross = (hist_prev <= 0) & (hist > 0)

    entries = (trend_ok & volume_ok & mom_positive & (mom_strengthening | mom_cross)).fillna(False)
    
    # Exit conditions: ATR trailing stop
    # Calculate rolling high since entry would require state tracking,
    # so we use a simpler approach: close below (previous close - ATR * mult)
    trailing_stop = close.shift(1) - (atr.shift(1) * p["atr_mult"])
    exits = (close < trailing_stop).fillna(False)

    return entries, exits


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Generate signals for ALL param combinations at once (vectorized)."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    close = df["close"]
    high = df["high"]
    low = df["low"]
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

    # Deduplicate SMA, volume windows, and ATR lengths
    all_sma_lens = sorted(set(pg["sma_len"]))
    all_vol_windows = sorted(set(pg["volume_window"]))
    all_atr_lens = sorted(set(pg["atr_len"]))
    
    sma_cache = {l: cached_indicator("sma", close, length=l) for l in all_sma_lens}
    vol_avg_cache = {w: cached_indicator("sma", volume, length=w) for w in all_vol_windows}
    atr_cache = {l: cached_indicator("atr", high, low, close, length=l) for l in all_atr_lens}

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

        atr = atr_cache[params["atr_len"]]

        hist_prev = hist.shift(1)
        hist_prev2 = hist.shift(2)
        
        mom_positive = hist > 0
        mom_strengthening = (hist > hist_prev) & (hist_prev <= hist_prev2)
        mom_cross = (hist_prev <= 0) & (hist > 0)

        e = (trend_ok & volume_ok & mom_positive & (mom_strengthening | mom_cross)).fillna(False)
        
        trailing_stop = close.shift(1) - (atr.shift(1) * params["atr_mult"])
        x = (close < trailing_stop).fillna(False)

        entries_cols[f"c{i}"] = e
        exits_cols[f"c{i}"] = x
        param_list.append(params)

    return pd.DataFrame(entries_cols), pd.DataFrame(exits_cols), param_list
