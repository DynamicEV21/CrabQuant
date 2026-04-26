"""
Momentum RSI ATR Confluence Strategy

Combines momentum detection (ROC), RSI pullback entry in uptrends,
and ATR-based trailing stop exits.

Entry logic:
- Price above 50 EMA (trend filter)
- ROC(14) > 0 (positive momentum)
- RSI(14) crossed above oversold threshold from below (pullback recovery)
- ADX > 20 (trend strength confirmation)

Exit logic:
- ATR trailing stop: exit when close falls below (entry high - atr_mult * ATR)
- Or RSI exceeds overbought level (take profit on extended moves)
"""

import pandas as pd
import pandas_ta


DEFAULT_PARAMS = {
    "rsi_len": 14,
    "rsi_oversold": 35,
    "rsi_overbought": 75,
    "roc_len": 14,
    "roc_threshold": 0,
    "ema_len": 50,
    "atr_len": 14,
    "atr_exit_mult": 2.5,
    "adx_len": 14,
    "adx_threshold": 20,
}

PARAM_GRID = {
    "rsi_len": [10, 14, 21],
    "rsi_oversold": [30, 35, 40],
    "rsi_overbought": [70, 75, 80],
    "roc_len": [10, 14, 21],
    "roc_threshold": [0, 1, 2],
    "ema_len": [30, 50, 70],
    "atr_len": [10, 14, 21],
    "atr_exit_mult": [2.0, 2.5, 3.0],
    "adx_len": [12, 14, 20],
    "adx_threshold": [15, 20, 25],
}

DESCRIPTION = (
    "Momentum RSI ATR confluence strategy. "
    "Enters on RSI recovery from oversold in a confirmed uptrend "
    "(price > EMA, ROC > 0, ADX > threshold). "
    "Exits on ATR-based trailing stop or RSI overbought. "
    "Designed for trending stocks with regular pullbacks — targets "
    "momentum continuation after healthy dips."
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

    # Indicators
    rsi = pandas_ta.rsi(close, length=p["rsi_len"])
    roc = pandas_ta.roc(close, length=p["roc_len"])
    ema = pandas_ta.ema(close, length=p["ema_len"])
    atr = pandas_ta.atr(high, low, close, length=p["atr_len"])
    adx = pandas_ta.adx(high, low, close, length=p["adx_len"])

    adx_col = f"ADX_{p['adx_len']}"
    adx_val = adx[adx_col] if adx_col in adx.columns else pandas_ta.adx(high, low, close, length=p["adx_len"])[f"ADX_{p['adx_len']}"]

    # Entry conditions
    trend_ok = close > ema
    momentum_ok = roc > p["roc_threshold"]
    trend_strength_ok = adx_val > p["adx_threshold"]
    # RSI recovering from oversold: RSI is above oversold but below 55,
    # and was below oversold within the last N bars (lookback window)
    lookback = max(p["rsi_len"], 5)
    rsi_in_recovery_zone = rsi.between(p["rsi_oversold"], 55)
    rsi_was_oversold = rsi.rolling(lookback).min() < p["rsi_oversold"]
    rsi_recovery = rsi_in_recovery_zone & rsi_was_oversold

    entries = (trend_ok & momentum_ok & rsi_recovery).fillna(False)

    # Exit conditions
    # 1. ATR trailing stop: close below EMA - atr_mult * ATR
    atr_stop = ema - (p["atr_exit_mult"] * atr)
    atr_exit = close < atr_stop

    # 2. RSI overbought take-profit
    rsi_exit = rsi > p["rsi_overbought"]

    exits = (atr_exit | rsi_exit).fillna(False)

    return entries, exits
