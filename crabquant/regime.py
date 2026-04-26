"""
CrabQuant Market Regime Detector

Classifies current market regime using SPY price action + VIX.
Used to weight strategy selection and adapt backtest thresholds.
"""

import logging
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERSION = "mean_reversion"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


def detect_regime(
    spy_data: pd.DataFrame,
    vix_data: pd.DataFrame | None = None,
    lookback: int = 50,
) -> tuple[MarketRegime, dict]:
    """
    Classify the current market regime from SPY price data.

    Uses a score-based approach combining:
    - SMA slope (20 and 50 day)
    - Price vs SMA position
    - Rate of change (20 day)
    - Bollinger Band width (volatility proxy)
    - VIX level if available
    - Realized volatility (20 day rolling std of returns)

    Args:
        spy_data: DataFrame with 'close' column and DatetimeIndex
        vix_data: Optional DataFrame with 'close' column for VIX
        lookback: Number of recent bars to analyze

    Returns:
        (MarketRegime, metadata) where metadata contains scores and confidence
    """
    close = spy_data["close"].iloc[-lookback:]
    if len(close) < 20:
        logger.warning(f"Only {len(close)} bars available, need at least 20")
        return MarketRegime.LOW_VOLATILITY, {"confidence": 0.0, "scores": {}, "reason": "insufficient_data"}

    # ── Indicators ──
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean().dropna() if len(close) >= 50 else close.rolling(len(close)).mean()

    current_price = close.iloc[-1]
    current_sma20 = sma_20.iloc[-1]
    current_sma50 = sma_50.iloc[-1]

    # SMA slopes (normalized as % per bar)
    sma20_slope = (sma_20.iloc[-1] - sma_20.iloc[-6]) / sma_20.iloc[-6] if len(sma_20) >= 6 else 0.0
    sma50_slope = (sma_50.iloc[-1] - sma_50.iloc[-6]) / sma_50.iloc[-6] if len(sma_50) >= 6 else 0.0

    # Price position relative to SMAs
    price_vs_sma20 = (current_price - current_sma20) / current_sma20
    price_vs_sma50 = (current_price - current_sma50) / current_sma50

    # Rate of change (20 day)
    roc_20 = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] if len(close) >= 21 else 0.0

    # Bollinger Band width (volatility proxy)
    std_20 = close.rolling(20).std().iloc[-1]
    bb_width = (2 * std_20) / current_sma20  # normalized

    # Realized volatility (20 day rolling std of returns)
    returns = close.pct_change().dropna()
    realized_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252) if len(returns) >= 20 else 0.0

    # VIX level
    vix_score = 0.0
    vix_value = None
    if vix_data is not None and len(vix_data) > 0:
        vix_close = vix_data["close"]
        # Align to most recent date available
        vix_value = float(vix_close.iloc[-1])
        if vix_value > 30:
            vix_score = 1.0
        elif vix_value > 20:
            vix_score = 0.5
        elif vix_value < 15:
            vix_score = -0.5
        else:
            vix_score = 0.0

    # ── Score each regime ──

    # Trending up: positive slopes, price above SMAs, positive ROC
    trend_up_score = 0.0
    trend_up_score += min(max(sma20_slope * 100, -1), 1) * 0.25
    trend_up_score += min(max(sma50_slope * 100, -1), 1) * 0.25
    trend_up_score += min(max(price_vs_sma20 * 50, -1), 1) * 0.25
    trend_up_score += min(max(roc_20 * 10, -1), 1) * 0.25

    # Trending down: negative slopes, price below SMAs, negative ROC
    trend_down_score = 0.0
    trend_down_score += min(max(-sma20_slope * 100, -1), 1) * 0.25
    trend_down_score += min(max(-sma50_slope * 100, -1), 1) * 0.25
    trend_down_score += min(max(-price_vs_sma20 * 50, -1), 1) * 0.25
    trend_down_score += min(max(-roc_20 * 10, -1), 1) * 0.25

    # Mean reversion: low trend scores, price near SMA, moderate volatility
    mean_rev_score = 0.0
    mean_rev_score += max(0, 1 - abs(trend_up_score)) * 0.3
    mean_rev_score += max(0, 1 - abs(trend_down_score)) * 0.3
    mean_rev_score += max(0, 1 - abs(price_vs_sma20) * 50) * 0.2
    # Sideways = low ROC
    mean_rev_score += max(0, 1 - abs(roc_20) * 10) * 0.2

    # High volatility: wide BB, high realized vol, high VIX
    high_vol_score = 0.0
    high_vol_score += min(max(bb_width * 20 - 0.5, 0), 1) * 0.3
    high_vol_score += min(max(realized_vol / 0.30 - 0.3, 0), 1) * 0.3
    high_vol_score += max(vix_score, 0) * 0.4

    # Low volatility: narrow BB, low realized vol, low VIX
    low_vol_score = 0.0
    low_vol_score += max(0, 1 - bb_width * 20) * 0.3
    low_vol_score += max(0, 1 - realized_vol / 0.20) * 0.3
    low_vol_score += max(-vix_score, 0) * 0.4

    # ── Pick highest-scoring regime ──
    scores = {
        MarketRegime.TRENDING_UP: trend_up_score,
        MarketRegime.TRENDING_DOWN: trend_down_score,
        MarketRegime.MEAN_REVERSION: mean_rev_score,
        MarketRegime.HIGH_VOLATILITY: high_vol_score,
        MarketRegime.LOW_VOLATILITY: low_vol_score,
    }

    best_regime = max(scores, key=scores.get)
    best_score = scores[best_regime]

    # Confidence: how much does the best score stand out?
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] > 0:
        confidence = (sorted_scores[0] - sorted_scores[1]) / (sorted_scores[0] + 0.001)
        confidence = min(max(confidence, 0.0), 1.0)
    else:
        confidence = 0.0

    metadata = {
        "confidence": round(confidence, 3),
        "scores": {k.value: round(v, 3) for k, v in scores.items()},
        "vix_value": vix_value,
        "realized_vol": round(realized_vol, 4),
        "bb_width": round(bb_width, 4),
        "sma20_slope": round(sma20_slope, 6),
        "roc_20": round(roc_20, 4),
    }

    return best_regime, metadata


# ── Strategy affinity scores per regime ──

REGIME_STRATEGY_AFFINITY = {
    MarketRegime.TRENDING_UP: {
        "ema_ribbon_reversal": 0.95,
        "macd_momentum": 0.90,
        "adx_pullback": 0.85,
        "ichimoku_trend": 0.85,
        "rsi_crossover": 0.80,
        "atr_channel_breakout": 0.80,
        "volume_breakout": 0.75,
        "invented_momentum_rsi_atr": 0.75,
        "invented_momentum_rsi_stoch": 0.70,
        "multi_rsi_confluence": 0.65,
        "bollinger_squeeze": 0.40,
    },
    MarketRegime.TRENDING_DOWN: {
        # Short-biased and conservative strategies
        "adx_pullback": 0.80,
        "ichimoku_trend": 0.75,
        "atr_channel_breakout": 0.70,
        "macd_momentum": 0.65,
        "rsi_crossover": 0.60,
        "multi_rsi_confluence": 0.55,
        "bollinger_squeeze": 0.50,
        "ema_ribbon_reversal": 0.45,
        "volume_breakout": 0.40,
        "invented_momentum_rsi_atr": 0.40,
        "invented_momentum_rsi_stoch": 0.35,
    },
    MarketRegime.MEAN_REVERSION: {
        "bollinger_squeeze": 0.90,
        "multi_rsi_confluence": 0.80,
        "rsi_crossover": 0.75,
        "invented_momentum_rsi_stoch": 0.70,
        "volume_breakout": 0.65,
        "adx_pullback": 0.60,
        "macd_momentum": 0.55,
        "atr_channel_breakout": 0.50,
        "ema_ribbon_reversal": 0.45,
        "ichimoku_trend": 0.40,
        "invented_momentum_rsi_atr": 0.35,
    },
    MarketRegime.HIGH_VOLATILITY: {
        # Conservative strategies, avoid complex ones
        "adx_pullback": 0.70,
        "atr_channel_breakout": 0.65,
        "bollinger_squeeze": 0.60,
        "ichimoku_trend": 0.55,
        "multi_rsi_confluence": 0.50,
        "rsi_crossover": 0.45,
        "volume_breakout": 0.40,
        "macd_momentum": 0.35,
        "ema_ribbon_reversal": 0.30,
        "invented_momentum_rsi_atr": 0.25,
        "invented_momentum_rsi_stoch": 0.20,
    },
    MarketRegime.LOW_VOLATILITY: {
        # Momentum strategies thrive in low vol
        "ema_ribbon_reversal": 0.90,
        "macd_momentum": 0.85,
        "ichimoku_trend": 0.85,
        "rsi_crossover": 0.80,
        "adx_pullback": 0.75,
        "atr_channel_breakout": 0.70,
        "volume_breakout": 0.65,
        "invented_momentum_rsi_atr": 0.65,
        "invented_momentum_rsi_stoch": 0.60,
        "multi_rsi_confluence": 0.55,
        "bollinger_squeeze": 0.40,
    },
}


def get_strategy_ranking(
    regime: MarketRegime,
    available_strategies: list[str] | None = None,
) -> list[tuple[str, float]]:
    """
    Get strategies ranked by affinity score for a given regime.

    Args:
        regime: Current market regime
        available_strategies: Optional list to filter to. If None, returns all.

    Returns:
        List of (strategy_name, affinity_score) sorted by score descending.
    """
    affinity = REGIME_STRATEGY_AFFINITY.get(regime, {})

    if available_strategies is not None:
        affinity = {k: v for k, v in affinity.items() if k in available_strategies}

    return sorted(affinity.items(), key=lambda x: x[1], reverse=True)
