"""
CrabQuant Strategy Inventor

An LLM-powered strategy generator that:
1. Analyzes market data to identify patterns
2. Reads existing strategies to understand what works
3. Writes new strategy code from scratch
4. Tests immediately — if viable, passes to optimization pipeline

This is the Phase 1 of the CrabQuant pipeline:
  INVENT → OPTIMIZE → VALIDATE → EVOLVE
"""

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from crabquant.data import load_data
from crabquant.engine import BacktestEngine
from crabquant.strategies import STRATEGY_REGISTRY

RESULTS_DIR = Path(Path(__file__).parent.parent, "results")
INVENTED_DIR = RESULTS_DIR / "invented"
INVENTED_DIR.mkdir(parents=True, exist_ok=True)


# ─── Market Data Analysis ─────────────────────────────────────────────────────

def analyze_market_data(ticker: str) -> dict:
    """
    Analyze a ticker's market data to identify patterns for strategy ideation.
    Returns structured observations about price behavior.
    """
    try:
        df = load_data(ticker, period="2y")
    except Exception:
        return {"error": f"Cannot load {ticker}"}

    close = df["close"]
    volume = df["volume"]

    # Basic stats
    total_return = (close.iloc[-1] / close.iloc[0]) - 1
    volatility = close.pct_change().std() * (252 ** 0.5)

    # Trend analysis
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    trend_20 = (close - sma20) / sma20
    trend_50 = (close - sma50) / sma50

    # Mean reversion tendency
    mean_reversion_score = 0
    for window in [10, 20, 30]:
        rolling_mean = close.rolling(window).mean()
        deviations = close - rolling_mean
        # Count how often price reverts to mean within N bars
        reverts = 0
        for i in range(window, len(close) - window):
            if abs(deviations.iloc[i]) > deviations.std() * 0.5:
                future_mean = close.iloc[i:i+window].mean()
                if abs(future_mean - rolling_mean.iloc[i]) < abs(deviations.iloc[i]) * 0.5:
                    reverts += 1
        total_extremes = sum(1 for d in deviations if abs(d) > deviations.std() * 0.5)
        if total_extremes > 0:
            mean_reversion_score += reverts / total_extremes
    mean_reversion_score /= 3

    # Momentum tendency
    returns = close.pct_change()
    momentum_score = 0
    for lag in [5, 10, 20]:
        autocorr = returns.autocorr(lag=lag)
        if autocorr and not (autocorr != autocorr):  # not NaN
            momentum_score += autocorr
    momentum_score /= 3

    # Volatility clustering
    vol_changes = volatility * (252 ** 0.5)
    vol_regime = "high" if volatility > 0.35 else ("low" if volatility < 0.15 else "medium")

    # Volume patterns
    avg_vol = volume.mean()
    vol_spike_freq = (volume > avg_vol * 1.5).sum() / len(volume)

    # Support/resistance zones
    recent_high = close.tail(60).max()
    recent_low = close.tail(60).min()
    current_vs_range = (close.iloc[-1] - recent_low) / (recent_high - recent_low)

    # Drawdown frequency
    cummax = close.cummax()
    drawdowns = (close - cummax) / cummax
    avg_drawdown = drawdowns.min()
    drawdown_frequency = (drawdowns < -0.05).sum() / len(drawdowns)

    return {
        "ticker": ticker,
        "period_days": len(df),
        "total_return": round(total_return, 4),
        "annualized_volatility": round(volatility, 4),
        "vol_regime": vol_regime,
        "trend_20": round(trend_20.iloc[-1], 4),
        "trend_50": round(trend_50.iloc[-1], 4),
        "above_sma200": bool(close.iloc[-1] > sma200.iloc[-1]),
        "mean_reversion_tendency": round(mean_reversion_score, 3),
        "momentum_tendency": round(momentum_score, 3),
        "volume_spike_frequency": round(vol_spike_freq, 4),
        "current_vs_range": round(current_vs_range, 4),
        "avg_max_drawdown": round(avg_drawdown, 4),
        "drawdown_frequency": round(drawdown_frequency, 4),
        "recent_30d_return": round((close.iloc[-1] / close.iloc[-30]) - 1, 4),
    }


def get_strategy_catalog() -> str:
    """Get descriptions of all existing strategies for the inventor to learn from."""
    catalog = []
    for name, (_, _, _, desc) in STRATEGY_REGISTRY.items():
        catalog.append(f"  - {name}: {desc}")
    return "\n".join(catalog)


def get_top_winners_summary() -> str:
    """Get summary of top winning strategies for pattern analysis."""
    winners_file = RESULTS_DIR / "winners" / "winners.json"
    if not winners_file.exists():
        return "  No winners yet."

    with open(winners_file) as f:
        winners = json.load(f)

    lines = []
    for w in winners[:10]:
        lines.append(
            f"  - {w['ticker']}/{w['strategy']}: Sharpe={w['sharpe']:.2f}, "
            f"Return={w['return']:.1%}, Trades={w['trades']}, "
            f"Params={json.dumps(w['params'])}"
        )
    return "\n".join(lines)


# ─── Strategy Testing ─────────────────────────────────────────────────────────

def test_strategy_code(code: str, ticker: str = "AAPL") -> dict:
    """
    Test a strategy by executing its code and running a backtest.
    Returns test results.
    """
    # Create a namespace for execution
    namespace = {
        "pd": __import__("pandas"),
        "np": __import__("numpy"),
        "pandas_ta": __import__("pandas_ta"),
    }

    try:
        exec(code, namespace)
    except Exception as e:
        return {"success": False, "error": f"Code execution error: {e}"}

    # Check for generate_signals function
    if "generate_signals" not in namespace:
        return {"success": False, "error": "No generate_signals(df, params) function found"}

    fn = namespace["generate_signals"]
    if not callable(fn):
        return {"success": False, "error": "generate_signals is not callable"}

    # Get defaults
    defaults = namespace.get("DEFAULT_PARAMS", {})
    param_grid = namespace.get("PARAM_GRID", {})

    # Test on a few tickers
    test_tickers = [ticker]
    results = []

    for t in test_tickers:
        try:
            df = load_data(t, period="2y")
        except Exception:
            continue

        try:
            entries, exits = fn(df, defaults)
            engine = BacktestEngine()
            result = engine.run(df, entries, exits, "invented", t, 0, defaults)
            results.append({
                "ticker": t,
                "sharpe": result.sharpe,
                "return": result.total_return,
                "max_dd": result.max_drawdown,
                "trades": result.num_trades,
                "passed": result.passed,
                "score": result.score,
            })
        except Exception as e:
            results.append({"ticker": t, "error": str(e)})

    valid_results = [r for r in results if "error" not in r]
    has_signals = any(r["trades"] > 0 for r in valid_results)

    return {
        "success": has_signals,
        "has_signals": has_signals,
        "results": results,
        "defaults": defaults,
        "param_grid": param_grid,
        "num_tested": len(test_tickers),
        "num_profitable": sum(1 for r in valid_results if r.get("return", 0) > 0),
    }


def save_invented_strategy(name: str, code: str, test_results: dict):
    """Save an invented strategy that passes initial testing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{name}.py"
    filepath = INVENTED_DIR / filename

    with open(filepath, "w") as f:
        f.write(code)

    # Also save metadata
    meta = {
        "name": name,
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
        "test_results": test_results,
        "status": "invented",  # invented → optimizing → validated → promoted
    }

    meta_path = INVENTED_DIR / f"{timestamp}_{name}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return filepath


# ─── Main Analysis Runner ────────────────────────────────────────────────────

def run_invention_analysis(tickers: list[str] | None = None) -> dict:
    """
    Analyze market data and prepare invention context.
    This is what the improvement agent reads to invent new strategies.
    """
    if tickers is None:
        tickers = ["AAPL", "NVDA", "CAT", "XOM", "GLD", "JPM", "SPY"]

    analyses = {}
    for ticker in tickers:
        analysis = analyze_market_data(ticker)
        if "error" not in analysis:
            analyses[ticker] = analysis

    # Find patterns across tickers
    high_momentum = [t for t, a in analyses.items() if a["momentum_tendency"] > 0.02]
    high_meanrev = [t for t, a in analyses.items() if a["mean_reversion_tendency"] > 0.5]
    high_vol = [t for t, a in analyses.items() if a["vol_regime"] == "high"]
    trending = [t for t, a in analyses.items() if a["above_sma200"] and a["trend_50"] > 0]

    invention_context = {
        "timestamp": datetime.now().isoformat(),
        "market_analyses": analyses,
        "patterns": {
            "momentum_tickers": high_momentum,
            "mean_reversion_tickers": high_meanrev,
            "high_volatility_tickers": high_vol,
            "trending_tickers": trending,
        },
        "existing_strategies": get_strategy_catalog(),
        "top_winners": get_top_winners_summary(),
        "suggestion": (
            f"Based on analysis:\n"
            f"- Momentum plays: {', '.join(high_momentum) if high_momentum else 'none detected'}\n"
            f"- Mean reversion: {', '.join(high_meanrev) if high_meanrev else 'none detected'}\n"
            f"- High vol: {', '.join(high_vol) if high_vol else 'none detected'}\n"
            f"- Strong trends: {', '.join(trending) if trending else 'none detected'}\n\n"
            f"Invent strategies that exploit these specific patterns. "
            f"Look at what existing strategies work and create variations "
            f"that target the identified regime characteristics."
        ),
    }

    # Save for agent to read
    context_path = RESULTS_DIR / "invention_context.json"
    with open(context_path, "w") as f:
        json.dump(invention_context, f, indent=2, default=str)

    return invention_context


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, help="Analyze specific ticker")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers")
    parser.add_argument("--analyze", action="store_true", help="Run market analysis for invention")
    parser.add_argument("--catalog", action="store_true", help="Print strategy catalog")
    args = parser.parse_args()

    if args.catalog:
        print(get_strategy_catalog())
    elif args.analyze:
        tickers = args.tickers.split(",") if args.tickers else None
        if args.ticker:
            tickers = [args.ticker]
        ctx = run_invention_analysis(tickers)
        print(json.dumps(ctx["suggestion"], indent=2))
    else:
        parser.print_help()
