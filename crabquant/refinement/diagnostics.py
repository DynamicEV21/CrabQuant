"""CrabQuant Refinement — Diagnostics Helpers."""

import hashlib
import logging
from pathlib import Path
from types import ModuleType

import pandas as pd

from crabquant.data import load_data
from crabquant.engine.backtest import BacktestEngine, BacktestResult
from crabquant.regime import detect_regime

logger = logging.getLogger(__name__)


def run_backtest_safely(
    strategy_module: ModuleType,
    ticker: str,
    period: str = "2y",
    return_portfolio: bool = False,
) -> tuple:
    """Load data, generate signals, and run a backtest.

    Returns (BacktestResult, df, portfolio) on success, or (None, None, None) on any
    error.  When return_portfolio=True the engine is called with that kwarg and
    the vbt.Portfolio object is returned as the third element.
    """
    try:
        df = load_data(ticker, period)
        if df is None or df.empty:
            logger.warning("No data for %s/%s", ticker, period)
            return None, None, None

        params = strategy_module.DEFAULT_PARAMS
        entries, exits = strategy_module.generate_signals(df, params)

        strategy_name = getattr(strategy_module, "DESCRIPTION", "unknown")[:30]
        engine = BacktestEngine()

        portfolio = None
        if return_portfolio:
            try:
                result, portfolio = engine.run(
                    df, entries, exits,
                    strategy_name=strategy_name,
                    ticker=ticker,
                    return_portfolio=True,
                )
            except TypeError:
                result = engine.run(
                    df, entries, exits,
                    strategy_name=strategy_name,
                    ticker=ticker,
                )
        else:
            result = engine.run(
                df, entries, exits,
                strategy_name=strategy_name,
                ticker=ticker,
            )

        return result, df, portfolio

    except ValueError as e:
        logger.warning("Data error for %s: %s", ticker, e)
        return None, None, None
    except Exception as e:
        logger.warning("Backtest error for %s: %s", ticker, e)
        return None, None, None


def compute_sharpe_by_year(portfolio) -> dict:
    """Compute annualised Sharpe ratio per calendar year from a vbt.Portfolio.

    Returns {"2022": 1.23, "2023": 0.87, ...} or {} on any error.
    Years with fewer than 10 bars are skipped.
    """
    try:
        returns = portfolio.returns()
        idx = returns.index
        result = {}
        for year in sorted(set(idx.year)):
            mask = idx.year == year
            yr = returns[mask]
            if len(yr) < 10:
                continue
            std = yr.std()
            if std > 1e-10:
                result[str(year)] = round(yr.mean() / std * (252 ** 0.5), 4)
            else:
                result[str(year)] = 0.0
        return result
    except Exception as e:
        logger.warning("Sharpe-by-year error: %s", e)
        return {}


def compute_strategy_hash(code: str) -> str:
    """Deterministic 12-char SHA-256 hash of strategy source code.

    Normalises by stripping blank lines and per-line leading/trailing whitespace
    so trivial formatting differences don't produce different hashes.
    """
    normalized = "\n".join(line.strip() for line in code.split("\n") if line.strip())
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


# ── Tier 2 Diagnostics (Phase 3) ────────────────────────────────────────────

def compute_tier2_diagnostics(
    portfolio,
    ticker: str,
    period: str = "2y",
    top_n: int = 5,
    winners_file: str | None = None,
) -> dict:
    """Compute Tier 2 diagnostics: regime decomposition, top drawdowns,
    portfolio correlation, and benchmark comparison.

    Args:
        portfolio: vbt.Portfolio object (or None).
        ticker: Primary ticker symbol.
        period: Data period string (e.g. '2y').
        top_n: Maximum number of top drawdowns to return.
        winners_file: Path to winners.json for portfolio correlation.

    Returns:
        Dict with keys: regime_segments, top_drawdowns,
        portfolio_correlation, benchmark_return_pct.
    """
    result: dict = {
        "regime_segments": [],
        "top_drawdowns": [],
        "portfolio_correlation": None,
        "benchmark_return_pct": None,
    }

    if portfolio is None:
        return result

    try:
        returns = portfolio.returns()
    except Exception:
        return result

    if returns is None or len(returns) == 0:
        return result

    # ── Regime decomposition ──────────────────────────────────────────────
    result["regime_segments"] = _compute_regime_segments(returns, ticker, period)

    # ── Top drawdowns ─────────────────────────────────────────────────────
    result["top_drawdowns"] = _compute_top_drawdowns(returns, top_n)

    # ── Benchmark comparison (buy-and-hold) ───────────────────────────────
    result["benchmark_return_pct"] = _compute_benchmark_return(ticker, period)

    # ── Portfolio correlation ─────────────────────────────────────────────
    if winners_file is None:
        from crabquant.refinement.config import RefinementConfig
        cfg = RefinementConfig()
        winners_file = cfg.winners_file

    result["portfolio_correlation"] = _compute_portfolio_correlation(
        returns, winners_file
    )

    return result


def _compute_regime_segments(
    returns: pd.Series,
    ticker: str,
    period: str,
    window: int = 63,
) -> list[dict]:
    """Segment the equity curve by market regime using rolling windows.

    Uses detect_regime on price data to label each window, then computes
    Sharpe ratio per contiguous regime segment.
    """
    try:
        price_df = load_data(ticker, period)
        if price_df is None or price_df.empty:
            return []

        # Align returns with price data
        common_idx = returns.index.intersection(price_df.index)
        if len(common_idx) < window:
            return []

        aligned_returns = returns.loc[common_idx]
        aligned_price = price_df.loc[common_idx]

        # Label each bar by regime using a rolling window approach
        regime_labels = []
        step = max(window // 2, 1)
        for i in range(0, len(aligned_price) - window + 1, step):
            window_df = aligned_price.iloc[i:i + window]
            try:
                regime, _ = detect_regime(window_df)
                regime_labels.append((i, i + window, regime.value))
            except Exception:
                regime_labels.append((i, i + window, "unknown"))

        if not regime_labels:
            return []

        # Merge contiguous segments with the same regime
        segments = []
        current_start = regime_labels[0][0]
        current_end = regime_labels[0][1]
        current_regime = regime_labels[0][2]

        for start, end, regime in regime_labels[1:]:
            if regime == current_regime:
                current_end = end
            else:
                # Finalize current segment
                seg_returns = aligned_returns.iloc[current_start:min(current_end, len(aligned_returns))]
                if len(seg_returns) >= 10:
                    std = seg_returns.std()
                    sharpe = round(seg_returns.mean() / std * (252 ** 0.5), 4) if std > 1e-10 else 0.0
                    segments.append({
                        "regime": current_regime,
                        "start": str(aligned_returns.index[current_start].date()),
                        "end": str(aligned_returns.index[min(current_end, len(aligned_returns)) - 1].date()),
                        "sharpe": sharpe,
                    })
                current_start = start
                current_end = end
                current_regime = regime

        # Don't forget the last segment
        seg_returns = aligned_returns.iloc[current_start:min(current_end, len(aligned_returns))]
        if len(seg_returns) >= 10:
            std = seg_returns.std()
            sharpe = round(seg_returns.mean() / std * (252 ** 0.5), 4) if std > 1e-10 else 0.0
            segments.append({
                "regime": current_regime,
                "start": str(aligned_returns.index[current_start].date()),
                "end": str(aligned_returns.index[min(current_end, len(aligned_returns)) - 1].date()),
                "sharpe": sharpe,
            })

        return segments

    except Exception as e:
        logger.warning("Regime decomposition error: %s", e)
        return []


def _compute_top_drawdowns(
    returns: pd.Series,
    top_n: int = 5,
) -> list[dict]:
    """Find the top N drawdowns with dates and durations."""
    try:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        # Find drawdown periods
        in_drawdown = False
        drawdowns = []
        dd_start = None
        dd_min_idx = None
        dd_min_val = 0.0

        for i in range(len(drawdown)):
            if drawdown.iloc[i] < -0.001:  # at least 0.1% drawdown
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = i
                    dd_min_val = drawdown.iloc[i]
                    dd_min_idx = i
                elif drawdown.iloc[i] < dd_min_val:
                    dd_min_val = drawdown.iloc[i]
                    dd_min_idx = i
            else:
                if in_drawdown:
                    in_drawdown = False
                    drawdowns.append({
                        "start": str(drawdown.index[dd_start].date()),
                        "end": str(drawdown.index[dd_min_idx].date()),
                        "depth_pct": round(dd_min_val, 4),
                        "duration_bars": dd_min_idx - dd_start + 1,
                    })

        # Handle drawdown that extends to end of data
        if in_drawdown:
            drawdowns.append({
                "start": str(drawdown.index[dd_start].date()),
                "end": str(drawdown.index[dd_min_idx].date()),
                "depth_pct": round(dd_min_val, 4),
                "duration_bars": dd_min_idx - dd_start + 1,
            })

        # Sort by depth and take top N
        drawdowns.sort(key=lambda x: x["depth_pct"])
        return drawdowns[:top_n]

    except Exception as e:
        logger.warning("Top drawdowns error: %s", e)
        return []


def _compute_benchmark_return(ticker: str, period: str) -> float | None:
    """Compute buy-and-hold return for the benchmark (ticker) over the period."""
    try:
        from crabquant.data import load_data
        df = load_data(ticker, period)
        if df is None or df.empty or "close" not in df.columns:
            return None
        first_close = df["close"].iloc[0]
        last_close = df["close"].iloc[-1]
        if first_close <= 0:
            return None
        return round((last_close - first_close) / first_close, 4)
    except Exception as e:
        logger.warning("Benchmark return error: %s", e)
        return None


def _compute_portfolio_correlation(
    strategy_returns: pd.Series,
    winners_file: str,
) -> float | None:
    """Compute correlation of strategy returns to existing winners.

    Returns the average correlation across all winners, or None if no winners.
    """
    try:
        winners_path = Path(winners_file)
        if not winners_path.exists():
            return None

        import json
        winners = json.loads(winners_path.read_text())
        if not winners:
            return None

        # We can't recompute equity curves from winners.json alone
        # (that requires full backtest). Return None as a placeholder.
        # In production, this would be backed by stored equity curves.
        logger.info("Portfolio correlation: %d winners found, but equity curves not available from winners.json", len(winners))
        return None

    except Exception as e:
        logger.warning("Portfolio correlation error: %s", e)
        return None
