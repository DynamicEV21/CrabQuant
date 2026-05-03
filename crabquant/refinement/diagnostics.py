"""CrabQuant Refinement — Diagnostics Helpers."""

import hashlib
import logging
import traceback
from pathlib import Path
from types import ModuleType

import pandas as pd

from crabquant.data import load_data
from crabquant.engine.backtest import BacktestEngine, BacktestResult
from crabquant.refinement.signal_analysis import (
    analyze_signal_density,
    check_signal_density_early_exit,
)
from crabquant.regime import detect_regime

logger = logging.getLogger(__name__)


def _build_error_info(exc: Exception) -> dict:
    """Build a structured error info dict from an exception.

    Args:
        exc: The caught exception.

    Returns:
        Dict with error_type, error_message, and error_traceback (last 10 lines).
    """
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    # Take last 10 lines of traceback (the most relevant part)
    last_lines = tb_lines[-10:] if len(tb_lines) > 10 else tb_lines
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_traceback": "".join(last_lines).strip(),
    }


def run_backtest_safely(
    strategy_module: ModuleType,
    ticker: str,
    period: str = "2y",
    return_portfolio: bool = False,
    override_params: dict | None = None,
) -> tuple:
    """Load data, generate signals, and run a backtest.

    Returns (BacktestResult, df, portfolio, None) on success, or
    (None, None, None, error_info_dict) on any error.  When return_portfolio=True
    the engine is called with that kwarg and the vbt.Portfolio object is returned
    as the third element.

    The 4th element is always None on success and an error_info dict on failure,
    providing backward compatibility for callers that only unpack 3 elements.
    error_info_dict contains: error_type, error_message, error_traceback.
    """
    try:
        df = load_data(ticker, period)
        if df is None or df.empty:
            logger.warning("No data for %s/%s", ticker, period)
            error_info = {
                "error_type": "ValueError",
                "error_message": f"No data available for {ticker}/{period}",
                "error_traceback": "",
            }
            return None, None, None, error_info

        params = override_params if override_params is not None else strategy_module.DEFAULT_PARAMS
        entries, exits = strategy_module.generate_signals(df, params)

        # ── Signal density pre-check (Phase 6) ───────────────────────────
        # Skip the expensive backtest engine if signals are clearly broken.
        should_skip, skip_reason = check_signal_density_early_exit(
            entries, exits, len(df)
        )
        if should_skip:
            logger.info("Signal density pre-check: %s", skip_reason)
            # Return a synthetic result with 0 trades so the classifier
            # picks up the right failure mode.
            strategy_name = getattr(strategy_module, "DESCRIPTION", "unknown")[:30]
            result = BacktestResult(
                ticker=ticker,
                strategy_name=strategy_name,
                iteration=0,
                sharpe=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                num_trades=0,
                win_rate=0.0,
                avg_trade_return=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                profit_factor=0.0,
                avg_holding_bars=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                passed=False,
                score=0.0,
                notes=skip_reason,
            )
            error_info = {
                "error_type": "SignalDensityError",
                "error_message": skip_reason,
                "error_traceback": "",
                "signal_analysis": analyze_signal_density(
                    entries, exits, len(df)
                ),
            }
            return result, df, None, error_info

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

        return result, df, portfolio, None

    except ValueError as e:
        logger.warning("Data error for %s: %s", ticker, e)
        return None, None, None, _build_error_info(e)
    except Exception as e:
        logger.warning("Backtest error for %s: %s", ticker, e)
        return None, None, None, _build_error_info(e)


def run_multi_ticker_backtest(
    strategy_module,
    tickers: list[str],
    period: str = "2y",
    sharpe_target: float = 0.0,
) -> dict:
    """Run backtest on multiple tickers and return per-ticker results.

    This enables the refinement loop to detect strategies that are overfit to
    a single ticker.  The LLM receives per-ticker feedback so it can adjust
    strategies that work on one ticker but fail on others.

    Args:
        strategy_module: Loaded strategy module with generate_signals.
        tickers: List of ticker symbols to test on.
        period: Data period string (e.g. '2y').
        sharpe_target: Sharpe threshold for "pass" classification.

    Returns:
        Dict with keys:
            per_ticker: list of dicts with ticker, sharpe, trades, max_drawdown,
                        win_rate, passed (bool), error (str|None).
            tickers_tested: int.
            tickers_passed: int.
            avg_sharpe: float (across all tested tickers, -999 for failed).
            min_sharpe: float (worst performing ticker, -999 for failed).
            pass_rate: float (tickers_passed / tickers_tested).
            summary: str (human-readable summary for LLM context).
    """
    per_ticker = []
    sharpes = []

    for ticker in tickers:
        entry = {
            "ticker": ticker,
            "sharpe": -999.0,
            "trades": 0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "passed": False,
            "error": None,
        }

        try:
            result, df, portfolio, _ = run_backtest_safely(
                strategy_module, ticker, period, return_portfolio=True
            )
            if result is not None:
                entry["sharpe"] = result.sharpe
                entry["trades"] = result.num_trades
                entry["max_drawdown"] = result.max_drawdown
                entry["win_rate"] = result.win_rate
                entry["total_return"] = result.total_return
                entry["passed"] = result.sharpe >= sharpe_target
                sharpes.append(result.sharpe)
            else:
                entry["error"] = "backtest_failed"
        except Exception as e:
            entry["error"] = str(e)[:100]

        per_ticker.append(entry)

    tickers_tested = len(per_ticker)
    tickers_passed = sum(1 for t in per_ticker if t["passed"])
    valid_sharpes = [t["sharpe"] for t in per_ticker if t["sharpe"] > -900]
    avg_sharpe = sum(valid_sharpes) / len(valid_sharpes) if valid_sharpes else -999.0
    min_sharpe = min(valid_sharpes) if valid_sharpes else -999.0
    pass_rate = tickers_passed / tickers_tested if tickers_tested > 0 else 0.0

    # Build human-readable summary for LLM
    lines = [f"Multi-ticker backtest ({tickers_passed}/{tickers_tested} passed):"]
    for t in per_ticker:
        status = "✅" if t["passed"] else "❌"
        err = f" ({t['error']})" if t["error"] else ""
        lines.append(
            f"  {status} {t['ticker']}: Sharpe={t['sharpe']:.2f}, "
            f"Trades={t['trades']}, DD={t['max_drawdown']:.1%}{err}"
        )

    summary = "\n".join(lines)

    return {
        "per_ticker": per_ticker,
        "tickers_tested": tickers_tested,
        "tickers_passed": tickers_passed,
        "avg_sharpe": avg_sharpe,
        "min_sharpe": min_sharpe,
        "pass_rate": pass_rate,
        "summary": summary,
    }


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
