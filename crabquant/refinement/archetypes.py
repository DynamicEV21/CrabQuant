"""
Strategy archetype templates for LLM-guided strategy invention.

Provides proven starting-point skeletons for common strategy families.
The LLM receives these as templates to customize rather than inventing
from scratch, which dramatically improves first-turn quality.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class Archetype(TypedDict):
    """A strategy archetype template."""
    name: str
    description: str
    skeleton_code: str
    default_params: dict[str, float | int | str]
    typical_indicators: list[str]
    trade_frequency_expectation: str
    regime_affinity: str  # "trending", "ranging", "volatile", "any"
    anti_patterns: list[str]  # common mistakes specific to this archetype


# ── Archetype Definitions ────────────────────────────────────────────────

ARCHETYPE_REGISTRY: dict[str, Archetype] = {
    "mean_reversion": {
        "name": "Mean Reversion",
        "description": (
            "Buys when price deviates below its mean (oversold) and sells "
            "when it reverts above (overbought). Works best in ranging markets."
        ),
        "skeleton_code": '''"""
Mean reversion strategy: buys oversold, sells overbought.
"""
import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "rsi_period": 14,
    "rsi_entry": 30,
    "rsi_exit": 70,
    "bb_period": 20,
    "bb_std": 2.0,
}

DESCRIPTION = "Mean reversion using RSI and Bollinger Bands."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    rsi = cached_indicator("rsi", close, length=p["rsi_period"])
    bb = cached_indicator("bbands", close, length=p["bb_period"], std=p["bb_std"])
    bb_lower = bb.iloc[:, 0]
    bb_upper = bb.iloc[:, 2]

    # Entry: RSI oversold OR price below lower band
    entries = (
        ((rsi < p["rsi_entry"]) | (close < bb_lower))
    ).fillna(False)

    # Exit: RSI overbought OR price above upper band
    exits = (
        ((rsi > p["rsi_exit"]) | (close > bb_upper))
    ).fillna(False)

    return entries, exits
''',
        "default_params": {
            "rsi_period": 14,
            "rsi_entry": 30,
            "rsi_exit": 70,
            "bb_period": 20,
            "bb_std": 2.0,
        },
        "typical_indicators": ["rsi", "bbands", "cci", "stoch", "willr"],
        "trade_frequency_expectation": "Moderate (30-60 trades/year) — signals fire when price deviates from mean",
        "regime_affinity": "ranging",
        "anti_patterns": [
            "Using RSI entry < 20 with BB exit > 80 — too restrictive, almost never triggers",
            "Adding volume filter to mean reversion — low volume often coincides with oversold (false filter)",
            "Using very long RSI periods (>30) — misses the reversion signal",
        ],
    },

    "momentum": {
        "name": "Momentum / Trend Following",
        "description": (
            "Buys on trend acceleration and sells on deceleration. "
            "Rides existing trends rather than predicting reversals."
        ),
        "skeleton_code": '''"""
Momentum strategy: buys on trend acceleration, sells on deceleration.
"""
import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_fast": 12,
    "ema_slow": 26,
    "roc_period": 10,
    "roc_threshold": 0.0,
}

DESCRIPTION = "Momentum using EMA crossover with ROC confirmation."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema_fast = cached_indicator("ema", close, length=p["ema_fast"])
    ema_slow = cached_indicator("ema", close, length=p["ema_slow"])
    roc = cached_indicator("roc", close, length=p["roc_period"])

    # Entry: EMA crossover + positive momentum
    entries = (
        (ema_fast.shift(1) < ema_slow.shift(1))
        & (ema_fast > ema_slow)
        & (roc > p["roc_threshold"])
    ).fillna(False)

    # Exit: EMA cross under OR momentum turns negative
    exits = (
        (ema_fast < ema_slow)
        & (roc < 0)
    ).fillna(False)

    return entries, exits
''',
        "default_params": {
            "ema_fast": 12,
            "ema_slow": 26,
            "roc_period": 10,
            "roc_threshold": 0.0,
        },
        "typical_indicators": ["ema", "sma", "roc", "macd", "adx", "supertrend"],
        "trade_frequency_expectation": "Low-moderate (15-40 trades/year) — trends persist, fewer signals",
        "regime_affinity": "trending",
        "anti_patterns": [
            "Using too many trend filters together (EMA + MACD + ADX + Supertrend) — over-constrained, rare entries",
            "Very fast EMA crossovers (3/5) — mostly noise, high churn",
            "Ignoring exit conditions — momentum strategies need defined exits or they give back all gains",
        ],
    },

    "breakout": {
        "name": "Breakout / Range Expansion",
        "description": (
            "Buys when price breaks out of a defined range (channel, bands, "
            "high/low). Captures the start of new trends after consolidation."
        ),
        "skeleton_code": '''"""
Breakout strategy: buys on range expansion, sells on contraction.
"""
import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "atr_period": 14,
    "atr_multiplier": 2.0,
    "lookback": 20,
}

DESCRIPTION = "Breakout using ATR-based range expansion."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    atr = cached_indicator("atr", high, low, close, length=p["atr_period"])
    highest = close.rolling(p["lookback"]).max()
    lowest = close.rolling(p["lookback"]).min()

    # Entry: price breaks above recent range
    entries = (
        (close > highest.shift(1))
        & (atr > atr.rolling(20).mean())  # vol expansion confirmation
    ).fillna(False)

    # Exit: price breaks below recent range OR ATR contracts
    exits = (
        (close < lowest.shift(1))
    ).fillna(False)

    return entries, exits
''',
        "default_params": {
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "lookback": 20,
        },
        "typical_indicators": ["atr", "bbands", "donchian", "keltner", "adx"],
        "trade_frequency_expectation": "Low (10-25 trades/year) — genuine breakouts are infrequent",
        "regime_affinity": "volatile",
        "anti_patterns": [
            "Very short lookback (< 10 bars) — captures noise, not real breakouts",
            "No volatility confirmation — most 'breakouts' in low-vol environments are false",
            "Wide ATR multiplier (>3.0) with short lookback — contradictory (wants big moves in short windows)",
        ],
    },

    "volatility": {
        "name": "Volatility Regime",
        "description": (
            "Trades based on volatility expansion/contraction cycles. "
            "Buys on low-vol breakout, sells on high-vol mean reversion."
        ),
        "skeleton_code": '''"""
Volatility regime strategy: buys on low-vol expansion, sells on high-vol contraction.
"""
import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "atr_short": 10,
    "atr_long": 50,
    "atr_ratio_entry": 0.8,
    "atr_ratio_exit": 1.5,
    "bb_period": 20,
    "bb_std": 1.5,
}

DESCRIPTION = "Volatility regime using ATR ratio and Bollinger bandwidth."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    atr_short = cached_indicator("atr", high, low, close, length=p["atr_short"])
    atr_long = cached_indicator("atr", high, low, close, length=p["atr_long"])
    atr_ratio = atr_short / atr_long

    bb = cached_indicator("bbands", close, length=p["bb_period"], std=p["bb_std"])
    bb_width = bb.iloc[:, 3]  # bandwidth

    # Entry: volatility contracting (low ATR ratio) then starting to expand
    entries = (
        (atr_ratio.shift(1) < p["atr_ratio_entry"])
        & (atr_ratio > p["atr_ratio_entry"])
    ).fillna(False)

    # Exit: volatility spikes above threshold
    exits = (
        (atr_ratio > p["atr_ratio_exit"])
    ).fillna(False)

    return entries, exits
''',
        "default_params": {
            "atr_short": 10,
            "atr_long": 50,
            "atr_ratio_entry": 0.8,
            "atr_ratio_exit": 1.5,
            "bb_period": 20,
            "bb_std": 1.5,
        },
        "typical_indicators": ["atr", "bbands", "vix", "cci", "keltner"],
        "trade_frequency_expectation": "Low (10-20 trades/year) — vol cycles are slow",
        "regime_affinity": "volatile",
        "anti_patterns": [
            "Using very short ATR periods (< 5) — too noisy for vol regime detection",
            "Setting entry ratio too low (< 0.5) — almost never triggers",
            "Combining with RSI — RSI and vol are correlated, adds little information",
        ],
    },

    "statistical_arb": {
        "name": "Statistical Arbitrage (Z-Score)",
        "description": (
            "Uses z-score normalization of price relative to its own moving statistics "
            "to identify statistically abnormal deviations. Buys when price is "
            "significantly below its rolling mean (negative z-score), sells when "
            "it reverts. More sophisticated than simple mean reversion — uses "
            "statistical thresholds instead of fixed indicator levels."
        ),
        "skeleton_code": '''""""
Statistical arbitrage strategy: buys when z-score is extreme, sells on reversion.
""""
import pandas as pd
import numpy as np
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "lookback": 20,
    "entry_z": -1.5,
    "exit_z": 0.0,
    "volume_confirm": True,
    "vol_period": 20,
}

DESCRIPTION = "Statistical arbitrage using z-score of price deviation from rolling mean."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Compute z-score of price relative to rolling window
    rolling_mean = close.rolling(p["lookback"]).mean()
    rolling_std = close.rolling(p["lookback"]).std()
    z_score = ((close - rolling_mean) / rolling_std).fillna(0)

    # Optional volume confirmation: require above-average volume
    vol_avg = cached_indicator("sma", volume, length=p["vol_period"])
    vol_above_avg = volume > vol_avg

    # Entry: z-score drops below threshold (price statistically cheap)
    if p["volume_confirm"]:
        entries = ((z_score < p["entry_z"]) & vol_above_avg).fillna(False)
    else:
        entries = (z_score < p["entry_z"]).fillna(False)

    # Exit: z-score reverts to mean or above
    exits = (z_score > p["exit_z"]).fillna(False)

    return entries, exits
''',
        "default_params": {
            "lookback": 20,
            "entry_z": -1.5,
            "exit_z": 0.0,
            "volume_confirm": True,
            "vol_period": 20,
        },
        "typical_indicators": ["sma", "rsi", "bbands", "cci", "obv"],
        "trade_frequency_expectation": "Moderate (20-50 trades/year) — statistical deviations are fairly common",
        "regime_affinity": "ranging",
        "anti_patterns": [
            "Using very short lookback (< 10) — rolling stats are unstable, z-scores meaningless",
            "Setting entry_z < -3.0 — extremely restrictive, almost never triggers",
            "Not using fillna(0) on z-score — NaN propagation causes silent failures",
            "Using z-score with trending markets — mean-reversion assumptions break in trends",
        ],
    },

    "multi_signal_ensemble": {
        "name": "Multi-Signal Ensemble",
        "description": (
            "Combines multiple independent signals (trend, momentum, volatility) "
            "using a voting/consensus mechanism. A trade fires only when N of M "
            "signals agree. Reduces false positives by requiring confirmation "
            "across different indicator families."
        ),
        "skeleton_code": '''""""
Multi-signal ensemble strategy: requires multiple indicators to agree before trading.
""""
import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_period": 14,
    "rsi_threshold": 50,
    "min_signals": 2,  # at least 2 of 3 signals must agree
    "atr_period": 14,
}

DESCRIPTION = "Multi-signal ensemble requiring consensus across EMA trend, RSI momentum, and volume."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Signal 1: Trend (EMA crossover)
    ema_fast = cached_indicator("ema", close, length=p["ema_fast"])
    ema_slow = cached_indicator("ema", close, length=p["ema_slow"])
    trend_bullish = (ema_fast > ema_slow).fillna(False)

    # Signal 2: Momentum (RSI direction)
    rsi = cached_indicator("rsi", close, length=p["rsi_period"])
    momentum_bullish = (rsi > p["rsi_threshold"]).fillna(False)

    # Signal 3: Volume confirmation
    vol_avg = cached_indicator("sma", volume, length=20)
    vol_confirm = (volume > vol_avg).fillna(False)

    # Count bullish signals
    bullish_count = (
        trend_bullish.astype(int)
        + momentum_bullish.astype(int)
        + vol_confirm.astype(int)
    )

    # Entry: enough signals agree
    entries = (bullish_count >= p["min_signals"]).fillna(False).astype(bool)

    # Exit: trend reverses AND momentum weakens
    exits = (
        (~trend_bullish) & (rsi < 40)
    ).fillna(False)

    return entries, exits
''',
        "default_params": {
            "ema_fast": 12,
            "ema_slow": 26,
            "rsi_period": 14,
            "rsi_threshold": 50,
            "min_signals": 2,
            "atr_period": 14,
        },
        "typical_indicators": ["ema", "sma", "rsi", "macd", "adx", "obv"],
        "trade_frequency_expectation": "Moderate (20-40 trades/year) — consensus reduces frequency but improves quality",
        "regime_affinity": "any",
        "anti_patterns": [
            "Requiring all signals to agree (min_signals = total) — almost never triggers",
            "Using correlated signals (EMA + MACD) — they agree by default, no diversification benefit",
            "Adding too many signals (>5) — diminishing returns, harder to debug, overfitting risk",
            "Not weighting signals — a strong trend signal should count more than a marginal volume signal",
        ],
    },

    "volatility_breakout": {
        "name": "Volatility Breakout",
        "description": (
            "Identifies periods of low volatility (squeeze) and enters when "
            "volatility expands, signaling a potential large move. Uses "
            "Bollinger Band width contraction followed by expansion, confirmed "
            "by ATR. Different from plain breakout — specifically targets the "
            "squeeze-to-expansion transition."
        ),
        "skeleton_code": '''""""
Volatility breakout strategy: buys on squeeze-to-expansion transition.
""""
import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "bb_period": 20,
    "bb_std": 2.0,
    "bb_width_percentile": 20,  # BB width below this percentile = squeeze
    "atr_period": 14,
    "atr_expansion": 1.2,  # ATR must expand by this ratio
    "atr_ma_period": 20,
}

DESCRIPTION = "Volatility breakout targeting squeeze-to-expansion transitions."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Bollinger Band width (normalized squeeze detector)
    bb = cached_indicator("bbands", close, length=p["bb_period"], std=p["bb_std"])
    bb_width = bb.iloc[:, 3]  # bandwidth

    # Squeeze: BB width in bottom percentile of its own history
    bb_width_pct = bb_width.rolling(100).rank(pct=True)
    in_squeeze = (bb_width_pct < p["bb_width_percentile"] / 100).fillna(False)

    # ATR expansion: current ATR > X times its moving average
    atr = cached_indicator("atr", high, low, close, length=p["atr_period"])
    atr_ma = atr.rolling(p["atr_ma_period"]).mean()
    atr_expanding = (atr > atr_ma * p["atr_expansion"]).fillna(False)

    # Entry: was in squeeze, now breaking out with volume expansion
    entries = (
        in_squeeze.shift(1)
        & ~in_squeeze
        & atr_expanding
        & (close > close.rolling(10).mean())  # directional bias: up
    ).fillna(False)

    # Exit: ATR contracts back or price drops below short MA
    exits = (
        (~atr_expanding)
        | (close < close.rolling(10).mean())
    ).fillna(False)

    return entries, exits
''',
        "default_params": {
            "bb_period": 20,
            "bb_std": 2.0,
            "bb_width_percentile": 20,
            "atr_period": 14,
            "atr_expansion": 1.2,
            "atr_ma_period": 20,
        },
        "typical_indicators": ["bbands", "atr", "keltner", "adx", "cci"],
        "trade_frequency_expectation": "Low (8-20 trades/year) — genuine squeezes are infrequent",
        "regime_affinity": "volatile",
        "anti_patterns": [
            "Very short percentile lookback (< 50) — squeeze detection is unstable",
            "Setting bb_width_percentile too high (> 40) — not selective enough, trades noise",
            "No directional bias — breakout direction is unknown, need trend filter",
            "Using rank() without enough lookback — percentile is meaningless with < 50 bars",
        ],
    },
}


def get_archetype(name: str) -> Archetype | None:
    """Look up an archetype by name. Case-insensitive."""
    return ARCHETYPE_REGISTRY.get(name.lower().strip())


def list_archetypes() -> list[str]:
    """Return all available archetype names."""
    return list(ARCHETYPE_REGISTRY.keys())


def format_archetype_for_prompt(archetype: Archetype) -> str:
    """Format an archetype for injection into the LLM prompt."""
    lines = [
        f"## Archetype: {archetype['name']}",
        f"",
        f"{archetype['description']}",
        f"",
        f"**Regime affinity:** {archetype['regime_affinity']}",
        f"**Expected trade frequency:** {archetype['trade_frequency_expectation']}",
        f"**Typical indicators:** {', '.join(archetype['typical_indicators'])}",
        f"",
        f"**Starting template (customize the parameters and conditions):**",
        f"```python",
        archetype["skeleton_code"],
        f"```",
        f"",
        f"**Common mistakes to AVOID with this archetype:**",
    ]
    for ap in archetype["anti_patterns"]:
        lines.append(f"- {ap}")

    return "\n".join(lines)
