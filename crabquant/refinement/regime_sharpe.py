"""
CrabQuant Refinement — Regime Sharpe Analysis (Phase 3).

Segments an equity curve by market regime and computes Sharpe ratio
per regime. Identifies if a strategy is regime-dependent.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeSharpeReport:
    """Report of Sharpe ratio broken down by market regime."""

    sharpe_by_regime: dict[str, float] = field(default_factory=dict)
    regime_segments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sharpe_by_regime": self.sharpe_by_regime,
            "regime_segments": self.regime_segments,
        }


def compute_regime_sharpe(
    portfolio,
    regime_labels: pd.Series,
) -> RegimeSharpeReport:
    """Compute Sharpe ratio per regime segment from portfolio returns.

    Args:
        portfolio: vbt.Portfolio object (or mock with .returns() method).
        regime_labels: pd.Series of regime labels (str) indexed by date,
                       same index as portfolio returns. Each value is a
                       regime name like 'bull', 'bear', 'sideways'.

    Returns:
        RegimeSharpeReport with per-regime Sharpe ratios and segment details.
    """
    report = RegimeSharpeReport()

    if portfolio is None or regime_labels is None or len(regime_labels) == 0:
        return report

    try:
        returns = portfolio.returns()
    except Exception as e:
        logger.warning("regime_sharpe: portfolio.returns() failed: %s", e)
        return report

    if returns is None or len(returns) == 0:
        return report

    # Align returns and regime labels
    common_idx = returns.index.intersection(regime_labels.index)
    if len(common_idx) == 0:
        return report

    aligned_returns = returns.loc[common_idx]
    aligned_labels = regime_labels.loc[common_idx]

    # Group by contiguous regime segments
    segments = _extract_contiguous_segments(aligned_labels)

    # Compute Sharpe per regime
    sharpe_by_regime: dict[str, list[float]] = {}
    detailed_segments: list[dict] = []

    for seg in segments:
        regime = seg["regime"]
        seg_returns = aligned_returns.iloc[seg["start"]:seg["end"]]
        n_bars = len(seg_returns)

        if n_bars < 10:
            continue

        std = seg_returns.std()
        if std > 1e-10:
            sharpe = round(seg_returns.mean() / std * (252 ** 0.5), 4)
        else:
            sharpe = 0.0

        sharpe_by_regime.setdefault(regime, []).append(sharpe)

        detailed_segments.append({
            "regime": regime,
            "start": str(aligned_returns.index[seg["start"]].date()),
            "end": str(aligned_returns.index[seg["end"] - 1].date()),
            "sharpe": sharpe,
            "n_bars": n_bars,
        })

    # Average Sharpe across segments of the same regime
    report.sharpe_by_regime = {
        regime: round(float(np.mean(sharpes)), 4)
        for regime, sharpes in sharpe_by_regime.items()
    }
    report.regime_segments = detailed_segments

    return report


def is_regime_dependent(
    report: RegimeSharpeReport,
    threshold: float = 2.0,
) -> bool:
    """Determine if a strategy is regime-dependent.

    A strategy is regime-dependent if:
    1. Any regime has a negative Sharpe ratio, OR
    2. The range between best and worst regime Sharpe exceeds the threshold.

    Args:
        report: RegimeSharpeReport from compute_regime_sharpe.
        threshold: Sharpe range threshold (default 2.0).

    Returns:
        True if the strategy is regime-dependent.
    """
    sharpes = list(report.sharpe_by_regime.values())

    if len(sharpes) < 2:
        return False

    # Any negative Sharpe in any regime
    if any(s < 0 for s in sharpes):
        return True

    # Wide range across regimes
    sharpe_range = max(sharpes) - min(sharpes)
    if sharpe_range > threshold:
        return True

    return False


def _extract_contiguous_segments(labels: pd.Series) -> list[dict]:
    """Extract contiguous segments of identical regime labels.

    Returns list of dicts with 'regime', 'start' (inclusive), 'end' (exclusive).
    """
    if len(labels) == 0:
        return []

    segments = []
    current_regime = labels.iloc[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels.iloc[i] != current_regime:
            segments.append({
                "regime": current_regime,
                "start": start_idx,
                "end": i,
            })
            current_regime = labels.iloc[i]
            start_idx = i

    # Final segment
    segments.append({
        "regime": current_regime,
        "start": start_idx,
        "end": len(labels),
    })

    return segments
