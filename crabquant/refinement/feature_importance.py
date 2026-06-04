"""
Feature Importance Analysis for CrabQuant refinement pipeline.

After a backtest, analyzes which indicators in the strategy code actually
contributed to (or hurt) the Sharpe ratio. Uses correlation-based analysis:
for each indicator, computes its rolling correlation with forward returns
to determine predictive power.

The LLM receives this feedback so it can:
- Double down on indicators that are actually driving returns
- Remove or replace indicators that are noise or harmful
- Understand WHY a strategy is performing well or poorly

Phase 5.6 — Continuous Improvement item 5.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from crabquant.data import load_data

logger = logging.getLogger(__name__)

# Forward return window for correlation analysis (5 trading days ≈ 1 week)
_FORWARD_WINDOW = 5
# Minimum data points required for meaningful correlation
_MIN_OBSERVATIONS = 60
# Correlation threshold for "contributing" classification
_CONTRIBUTING_THRESHOLD = 0.05
# Correlation threshold for "harmful" classification
_HARMFUL_THRESHOLD = -0.05


# ── Indicator computation helpers ──────────────────────────────────────────

# Map indicator names to their computation functions.
# These are lightweight wrappers around pandas_ta that produce a single Series
# from OHLCV data, suitable for correlation analysis.
_INDICATOR_COMPUTE = {}


def _register_indicator(name: str):
    """Decorator to register an indicator computation function."""
    def decorator(fn):
        _INDICATOR_COMPUTE[name.lower()] = fn
        return fn
    return decorator


@_register_indicator("ema")
def _compute_ema(df: pd.DataFrame, **kwargs) -> pd.Series:
    period = kwargs.get("length", kwargs.get("span", 20))
    return df["close"].ewm(span=int(period)).mean()


@_register_indicator("sma")
def _compute_sma(df: pd.DataFrame, **kwargs) -> pd.Series:
    period = kwargs.get("length", 20)
    return df["close"].rolling(window=int(period)).mean()


@_register_indicator("rsi")
def _compute_rsi(df: pd.DataFrame, **kwargs) -> pd.Series:
    period = kwargs.get("length", 14)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=int(period)).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=int(period)).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


@_register_indicator("macd")
def _compute_macd(df: pd.DataFrame, **kwargs) -> pd.Series:
    fast = kwargs.get("fast", 12)
    slow = kwargs.get("slow", 26)
    signal = kwargs.get("signal", 9)
    ema_fast = df["close"].ewm(span=int(fast)).mean()
    ema_slow = df["close"].ewm(span=int(slow)).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=int(signal)).mean()
    return macd_line - signal_line  # MACD histogram


@_register_indicator("atr")
def _compute_atr(df: pd.DataFrame, **kwargs) -> pd.Series:
    period = kwargs.get("length", 14)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=int(period)).mean()


@_register_indicator("bbands")
def _compute_bbands(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Returns %B (price position within Bollinger Bands)."""
    period = kwargs.get("length", 20)
    std_dev = kwargs.get("std", 2)
    sma = df["close"].rolling(window=int(period)).mean()
    std = df["close"].rolling(window=int(period)).std()
    upper = sma + float(std_dev) * std
    lower = sma - float(std_dev) * std
    # %B: where price is within the bands (0 = lower, 1 = upper)
    bandwidth = upper - lower
    return (df["close"] - lower) / bandwidth.replace(0, np.nan)


@_register_indicator("roc")
def _compute_roc(df: pd.DataFrame, **kwargs) -> pd.Series:
    period = kwargs.get("length", 12)
    return df["close"].pct_change(periods=int(period))


@_register_indicator("stoch")
def _compute_stoch(df: pd.DataFrame, **kwargs) -> pd.Series:
    period = kwargs.get("length", kwargs.get("k", 14))
    high = df["high"].rolling(window=int(period)).max()
    low = df["low"].rolling(window=int(period)).min()
    return 100 * (df["close"] - low) / (high - low).replace(0, np.nan)


@_register_indicator("adx")
def _compute_adx(df: pd.DataFrame, **kwargs) -> pd.Series:
    period = kwargs.get("length", 14)
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = _compute_atr(df, length=int(period)) * (
        df["close"].rolling(window=int(period)).mean().notna()
    )
    atr = tr.rolling(window=int(period)).mean()
    plus_di = 100 * (plus_dm.rolling(window=int(period)).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=int(period)).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=int(period)).mean()


@_register_indicator("obv")
def _compute_obv(df: pd.DataFrame, **kwargs) -> pd.Series:
    """On-Balance Volume — returns OBV as a Series."""
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


@_register_indicator("vwap")
def _compute_vwap(df: pd.DataFrame, **kwargs) -> pd.Series:
    """VWAP — returns VWAP as a Series (approximate for daily data)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
    return (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)


@_register_indicator("cci")
def _compute_cci(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Commodity Channel Index."""
    period = kwargs.get("length", 20)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical.rolling(window=int(period)).mean()
    mad = typical.rolling(window=int(period)).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return (typical - sma) / (0.015 * mad.replace(0, np.nan))


@_register_indicator("wma")
def _compute_wma(df: pd.DataFrame, **kwargs) -> pd.Series:
    period = kwargs.get("length", 20)
    weights = np.arange(1, int(period) + 1)
    def weighted_mean(x):
        if len(x) < len(weights):
            return np.nan
        return np.dot(x[-len(weights):], weights) / weights.sum()
    return df["close"].rolling(window=int(period)).apply(weighted_mean, raw=True)


@_register_indicator("supertrend")
def _compute_supertrend(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Supertrend — returns direction: +1 for uptrend, -1 for downtrend."""
    period = kwargs.get("length", kwargs.get("atr_period", 10))
    multiplier = kwargs.get("multiplier", 3.0)
    atr = _compute_atr(df, length=int(period))
    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + float(multiplier) * atr
    lower_band = hl2 - float(multiplier) * atr
    
    direction = pd.Series(1.0, index=df.index)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i]:
            direction.iloc[i] = 1.0
        elif df["close"].iloc[i] < lower_band.iloc[i]:
            direction.iloc[i] = -1.0
        else:
            direction.iloc[i] = direction.iloc[i - 1]
    return direction


@_register_indicator("williams")
def _compute_williams(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Williams %R."""
    period = kwargs.get("length", 14)
    high = df["high"].rolling(window=int(period)).max()
    low = df["low"].rolling(window=int(period)).min()
    return -100 * (high - df["close"]) / (high - low).replace(0, np.nan)


@_register_indicator("mfi")
def _compute_mfi(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Money Flow Index."""
    period = kwargs.get("length", 14)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
    raw_mf = typical * volume
    mf_change = typical.diff()
    pos_mf = raw_mf.where(mf_change > 0, 0.0).rolling(window=int(period)).sum()
    neg_mf = raw_mf.where(mf_change < 0, 0.0).rolling(window=int(period)).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


@_register_indicator("dmi")
def _compute_dmi(df: pd.DataFrame, **kwargs) -> pd.Series:
    """DMI+ - DMI- difference (directional movement)."""
    return _compute_adx(df, **kwargs)  # Simplified: use ADX as proxy


# ── Core analysis functions ────────────────────────────────────────────────


def extract_indicators_from_code(code: str) -> list[str]:
    """Extract indicator names from strategy source code.

    Looks for patterns like:
    - cached_indicator("ema", ...)
    - cached_indicator('rsi', ...)
    - ta.rsi(...)
    - pandas_ta.macd(...)

    Args:
        code: Strategy source code string.

    Returns:
        List of unique indicator function names found.
    """
    cached = re.findall(r'cached_indicator\s*\(\s*[\'"](\w+)', code)
    direct = re.findall(r'(?:ta|pandas_ta)\.(\w+)\s*\(', code)
    return list(dict.fromkeys(cached + direct))  # dedup preserving order


def compute_indicator_series(
    indicator_name: str,
    df: pd.DataFrame,
    **kwargs,
) -> Optional[pd.Series]:
    """Compute an indicator series from OHLCV data.

    Args:
        indicator_name: Name of the indicator (e.g., 'ema', 'rsi').
        df: OHLCV DataFrame with columns: open, high, low, close, (volume).
        **kwargs: Additional parameters for the indicator.

    Returns:
        pd.Series of indicator values, or None if computation fails.
    """
    compute_fn = _INDICATOR_COMPUTE.get(indicator_name.lower())
    if compute_fn is None:
        logger.debug("No compute function for indicator: %s", indicator_name)
        return None
    try:
        result = compute_fn(df, **kwargs)
        if isinstance(result, pd.DataFrame):
            # Some indicators return DataFrames — take the first column
            result = result.iloc[:, 0]
        return result
    except Exception as e:
        logger.warning("Failed to compute indicator %s: %s", indicator_name, e)
        return None


def compute_feature_importance(
    strategy_code: str,
    ticker: str,
    period: str = "2y",
    forward_window: int = _FORWARD_WINDOW,
) -> dict:
    """Analyze which indicators contribute most to the strategy's predictive power.

    For each indicator found in the strategy code:
    1. Compute the indicator values on the price data
    2. Compute forward returns over the specified window
    3. Calculate the rolling correlation between indicator and forward returns
    4. Rank by average correlation strength

    This is NOT a perfect ablation analysis — it measures how well each indicator
    predicts future returns independently, not how it interacts with other
    indicators in the strategy. But it's fast (no backtest re-runs) and gives
    the LLM actionable feedback about which indicators are actually useful.

    Args:
        strategy_code: The strategy's Python source code.
        ticker: Ticker symbol to analyze on.
        period: Data period string (e.g., '2y').
        forward_window: Days ahead to compute forward returns.

    Returns:
        Dict with keys:
            indicators: list of dicts with:
                name, correlation, classification, abs_correlation
            summary: str (human-readable summary for LLM)
            dominant_indicator: str or None
            weakest_indicator: str or None
    """
    # 1. Load data
    df = load_data(ticker, period)
    if df is None or df.empty or len(df) < _MIN_OBSERVATIONS:
        return _empty_result("insufficient_data")

    # 2. Extract indicators from code
    indicator_names = extract_indicators_from_code(strategy_code)
    if not indicator_names:
        return _empty_result("no_indicators_found")

    # 3. Compute forward returns
    forward_returns = df["close"].pct_change(forward_window).shift(-forward_window)

    # 4. Analyze each indicator
    results = []
    for name in indicator_names:
        series = compute_indicator_series(name, df)
        if series is None:
            continue

        # Align series with forward_returns
        aligned = pd.DataFrame({"indicator": series, "fwd_ret": forward_returns})
        aligned = aligned.dropna()

        if len(aligned) < _MIN_OBSERVATIONS:
            continue

        # Compute rolling correlation (21-day window ≈ 1 month)
        rolling_corr = (
            aligned["indicator"]
            .rolling(window=21, min_periods=10)
            .corr(aligned["fwd_ret"])
        )

        # Use median of rolling correlation (robust to outliers)
        median_corr = rolling_corr.median()
        if np.isnan(median_corr):
            # Fallback to simple correlation
            median_corr = aligned["indicator"].corr(aligned["fwd_ret"])
            if np.isnan(median_corr):
                median_corr = 0.0

        # Classify
        if median_corr >= _CONTRIBUTING_THRESHOLD:
            classification = "contributing"
        elif median_corr <= _HARMFUL_THRESHOLD:
            classification = "harmful"
        else:
            classification = "neutral"

        results.append({
            "name": name,
            "correlation": round(float(median_corr), 3),
            "abs_correlation": round(abs(float(median_corr)), 3),
            "classification": classification,
        })

    if not results:
        return _empty_result("all_indicators_failed")

    # Sort by absolute correlation (most impactful first)
    results.sort(key=lambda x: x["abs_correlation"], reverse=True)

    # Identify dominant and weakest indicators
    dominant = None
    for r in results:  # sorted by abs_correlation desc
        if r["classification"] == "contributing":
            dominant = r["name"]
            break
    weakest = None
    for r in reversed(results):
        if r["classification"] == "harmful":
            weakest = r["name"]
            break

    # Build summary
    summary = _format_importance_summary(results, dominant, weakest)

    return {
        "indicators": results,
        "summary": summary,
        "dominant_indicator": dominant,
        "weakest_indicator": weakest,
    }


def _empty_result(reason: str) -> dict:
    """Return an empty feature importance result."""
    return {
        "indicators": [],
        "summary": f"(Feature importance unavailable: {reason})",
        "dominant_indicator": None,
        "weakest_indicator": None,
    }


def _format_importance_summary(
    results: list[dict],
    dominant: Optional[str],
    weakest: Optional[str],
) -> str:
    """Format feature importance results as a human-readable summary for the LLM.

    Args:
        results: List of indicator analysis results.
        dominant: Name of the most contributing indicator (or None).
        weakest: Name of the most harmful indicator (or None).

    Returns:
        Formatted string for prompt injection.
    """
    lines = ["### Indicator Predictive Power Analysis"]
    lines.append(
        "Correlation of each indicator with forward returns "
        f"(median over {_FORWARD_WINDOW}-day forward window):"
    )
    lines.append("")

    for r in results:
        name = r["name"].upper()
        corr = r["correlation"]
        cls = r["classification"]

        if cls == "contributing":
            icon = "✅"
            note = "positively predicts returns"
        elif cls == "harmful":
            icon = "⚠️"
            note = "negatively correlated with returns — consider removing"
        else:
            icon = "➖"
            note = "no predictive power — noise"

        lines.append(f"  {icon} {name}: corr={corr:+.3f} ({note})")

    # Add actionable advice
    lines.append("")
    if dominant:
        lines.append(
            f"**Key driver**: {dominant.upper()} is your strongest signal — "
            "consider making it the core of your entry logic."
        )
    if weakest:
        lines.append(
            f"**Warning**: {weakest.upper()} is hurting performance — "
            "consider removing it or inverting its logic."
        )

    # Check for indicator overload
    contributing = [r for r in results if r["classification"] == "contributing"]
    harmful = [r for r in results if r["classification"] == "harmful"]
    neutral = [r for r in results if r["classification"] == "neutral"]

    if len(results) > 4:
        lines.append(
            f"**Indicator overload**: You're using {len(results)} indicators but "
            f"only {len(contributing)} are helping. Consider simplifying."
        )
    elif len(harmful) > len(contributing):
        lines.append(
            "**More harm than good**: More indicators are hurting than helping. "
            "Try a simpler approach with just the top 1-2 contributing indicators."
        )

    if neutral and not contributing and not harmful:
        lines.append(
            "**No edge detected**: None of your indicators show predictive power. "
            "This strategy may be curve-fit to historical noise. Try different "
            "indicators or a completely different approach."
        )

    return "\n".join(lines)


def format_feature_importance_for_prompt(importance: dict) -> str:
    """Format feature importance result for injection into the LLM prompt.

    Args:
        importance: Dict from compute_feature_importance().

    Returns:
        Formatted string ready for prompt injection.
    """
    if not importance.get("indicators"):
        return ""
    return importance.get("summary", "")
