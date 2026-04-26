"""
Invented Momentum RSI Stochastic Strategy

Combines RSI with Stochastic oscillator for momentum confirmation.
Targets momentum stocks like CAT, JPM, SPY with volume confirmation.
Best in trending markets with pullback reversals.
"""

import pandas as pd
import pandas_ta


DEFAULT_PARAMS = {
    "rsi_len": 14,
    "rsi_oversold": 35,
    "volume_window": 20,
    "volume_mult": 1.2,
}

PARAM_GRID = {
    "rsi_len": [7, 14, 21],
    "rsi_oversold": [25, 30, 35],
    "volume_window": [10, 20, 30],
    "volume_mult": [1.0, 1.2, 1.5],
}

DESCRIPTION = (
    "Simple RSI oversold strategy with volume confirmation. "
    "Enters when RSI is oversold with volume spike. "
    "Exits when RSI becomes overbought. "
    "Best for momentum stocks (CAT, JPM, SPY) with pullback reversals."
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
    volume = df["volume"]

    # Calculate indicators
    rsi = pandas_ta.rsi(close, length=p["rsi_len"])
    
    # Volume filter
    volume_avg = pandas_ta.sma(volume, length=p["volume_window"])
    volume_spike = volume > (volume_avg * p["volume_mult"])

    # Entry conditions: RSI oversold with volume spike
    entries = (
        (rsi < p["rsi_oversold"]) &
        (volume_spike)
    ).fillna(False)

    # Exit conditions: RSI overbought
    exits = (rsi > 70).fillna(False)

    return entries, exits