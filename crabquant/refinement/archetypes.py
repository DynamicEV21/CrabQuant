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
