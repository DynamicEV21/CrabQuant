"""
Strategy Helpers — Pure pandas/numpy technical indicators.

These functions provide common crypto indicators that strategies can use
alongside pandas_ta. All functions accept pandas Series and return pandas
Series. They use only pandas and numpy — no extra dependencies.

Enhancement 13: Missing Crypto Indicators (from VISION.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range.

    Measures market volatility as the moving average of the True Range
    over *period* bars.  First ``period - 1`` values are NaN.

    Parameters
    ----------
    high, low, close : pd.Series
        Bar data (must be same length).
    period : int
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        ATR values with NaN for the first ``period - 1`` entries.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index.

    Measures trend strength on a 0–100 scale.  Values above 25 typically
    indicate a strong trend.  First ``2 * period - 1`` values are NaN
    (due to the smoothed Wilder moving average).

    Parameters
    ----------
    high, low, close : pd.Series
        Bar data (must be same length).
    period : int
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        ADX values (0–100) with NaN for the initial warm-up period.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = high.diff()
    minus_dm = -low.diff()

    # +DM and -DM: only record when directional move is positive and dominant
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_high).abs()
    tr3 = (low - prev_low).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing (equivalent to EMA with alpha = 1/period)
    alpha = 1.0 / period
    atr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # +DI and -DI
    plus_di = 100.0 * (plus_smooth / atr_smooth)
    minus_di = 100.0 * (minus_smooth / atr_smooth)

    # DX
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * ((plus_di - minus_di).abs() / di_sum)

    # ADX = smoothed DX
    result = dx.ewm(alpha=alpha, adjust=False).mean()
    # Set the initial warm-up period to NaN (2 * period - 1 bars)
    warmup = 2 * period - 1
    result.iloc[:warmup] = np.nan
    return result


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume.

    Cumulative volume indicator that adds volume on up-days and subtracts
    on down-days.  Starts at 0 (no NaN values).

    Parameters
    ----------
    close : pd.Series
        Closing prices.
    volume : pd.Series
        Trading volume.

    Returns
    -------
    pd.Series
        OBV line starting from 0.
    """
    direction = close.diff().fillna(0)
    signed_vol = volume * np.sign(direction)
    return signed_vol.cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price (cumulative).

    Running VWAP calculated from the beginning of the series.
    Commonly used in intraday trading; here provided as a simple
    cumulative variant.

    Parameters
    ----------
    high, low, close : pd.Series
        Price data.
    volume : pd.Series
        Trading volume.

    Returns
    -------
    pd.Series
        Cumulative VWAP.  First row is NaN (no prior data to average).
    """
    typical_price = (high + low + close) / 3.0
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    result = cum_tp_vol / cum_vol
    # First value is just the first bar — mark it NaN for consistency
    result.iloc[0] = np.nan
    return result


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Supertrend indicator.

    Returns ``(supertrend_line, direction)`` where *direction* is 1 for
    bullish (price above the line) and -1 for bearish.

    Parameters
    ----------
    high, low, close : pd.Series
        Bar data (must be same length).
    period : int
        ATR look-back (default 10).
    multiplier : float
        ATR multiplier for the bands (default 3.0).

    Returns
    -------
    supertrend_line : pd.Series
        The supertrend value.  NaN for the first ``period`` bars.
    direction : pd.Series
        +1 (bullish) or -1 (bearish).  NaN for the first ``period`` bars.
    """
    atr_val = atr(high, low, close, period=period)

    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    st_line = pd.Series(np.nan, index=close.index, dtype=float)
    direction = pd.Series(np.nan, index=close.index, dtype=float)

    # Initialise on the first non-NaN bar
    start = period  # 0-indexed; period-th bar is the first valid ATR
    if start >= len(close):
        return st_line, direction

    # Determine initial direction from price position relative to bands
    if close.iloc[start] > upper_band.iloc[start]:
        direction.iloc[start] = 1.0
        st_line.iloc[start] = lower_band.iloc[start]
    elif close.iloc[start] < lower_band.iloc[start]:
        direction.iloc[start] = -1.0
        st_line.iloc[start] = upper_band.iloc[start]
    else:
        # Close is between bands — default bullish
        direction.iloc[start] = 1.0
        st_line.iloc[start] = lower_band.iloc[start]

    prev_st = st_line.iloc[start]
    prev_dir = direction.iloc[start]

    # Separate trackers for ratcheting: upper only goes down, lower only goes up
    ratchet_upper = upper_band.iloc[start]
    ratchet_lower = lower_band.iloc[start]

    for i in range(start + 1, len(close)):
        # Ratchet: upper band can only decrease, lower band can only increase
        ratchet_upper = min(upper_band.iloc[i], ratchet_upper)
        ratchet_lower = max(lower_band.iloc[i], ratchet_lower)

        # When bands cross, reset the ratchets to raw values
        if ratchet_lower > ratchet_upper:
            ratchet_lower = lower_band.iloc[i]
            ratchet_upper = upper_band.iloc[i]

        # Determine direction using the ratcheted bands
        if prev_dir == -1.0 and close.iloc[i] > ratchet_upper:
            # Close crossed above ratcheted upper band → flip bullish
            curr_dir = 1.0
            curr_st = ratchet_lower
        elif prev_dir == 1.0 and close.iloc[i] < ratchet_lower:
            # Close crossed below ratcheted lower band → flip bearish
            curr_dir = -1.0
            curr_st = ratchet_upper
        elif prev_dir == 1.0:
            # Still bullish: support line ratchets up
            curr_dir = 1.0
            curr_st = max(ratchet_lower, prev_st)
        else:
            # Still bearish: resistance line ratchets down
            curr_dir = -1.0
            curr_st = min(ratchet_upper, prev_st)

        st_line.iloc[i] = curr_st
        direction.iloc[i] = curr_dir

        prev_st = curr_st
        prev_dir = curr_dir

    return st_line, direction
