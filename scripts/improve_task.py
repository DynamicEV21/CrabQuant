#!/usr/bin/env python3
"""
CrabQuant Improvement Task

Analyzes results, invents new strategies, and improves existing ones.
Designed to be run by the improvement cron agent.

Usage:
    python scripts/improve_task.py [--invent] [--analyze] [--validate-winners] [--expand-grids]
"""

import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from crabquant.invention import (
    analyze_market_data,
    get_strategy_catalog,
    get_top_winners_summary,
    test_strategy_code,
    save_invented_strategy,
)
from crabquant.strategies import STRATEGY_REGISTRY
from crabquant.data import load_data
from crabquant.engine import BacktestEngine

RESULTS_DIR = Path(__file__).parent.parent / "results"
WINNERS_FILE = RESULTS_DIR / "winners" / "winners.json"
INSIGHTS_FILE = RESULTS_DIR / "insights.json"
LOGS_FILE = RESULTS_DIR / "logs" / "improve_results.jsonl"


def log_result(entry: dict):
    LOGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_analysis():
    """Analyze all results and produce insights."""
    # Load cron results
    cron_results = []
    logs_file = RESULTS_DIR / "logs" / "cron_results.jsonl"
    if logs_file.exists():
        with open(logs_file) as f:
            for line in f:
                cron_results.append(json.loads(line.strip()))

    if not cron_results:
        return {"error": "No cron results yet. Run wave cron first."}

    # Strategy-level stats
    strategy_stats = {}
    ticker_stats = {}
    for r in cron_results:
        sname = r["strategy_name"]
        ticker = r["ticker"]
        passed = r.get("passed", False)

        if sname not in strategy_stats:
            strategy_stats[sname] = {"tested": 0, "won": 0, "total_sharpe": 0, "total_return": 0}
        strategy_stats[sname]["tested"] += 1
        if passed:
            strategy_stats[sname]["won"] += 1
        strategy_stats[sname]["total_sharpe"] += r.get("sharpe", 0)
        strategy_stats[sname]["total_return"] += r.get("total_return", 0)

        if ticker not in ticker_stats:
            ticker_stats[ticker] = {"tested": 0, "won": 0}
        ticker_stats[ticker]["tested"] += 1
        if passed:
            ticker_stats[ticker]["won"] += 1

    # Compute rates
    for sname, stats in strategy_stats.items():
        stats["win_rate"] = stats["won"] / stats["tested"] if stats["tested"] > 0 else 0
        stats["avg_sharpe"] = stats["total_sharpe"] / stats["tested"] if stats["tested"] > 0 else 0
    for ticker, stats in ticker_stats.items():
        stats["win_rate"] = stats["won"] / stats["tested"] if stats["tested"] > 0 else 0

    insights = {
        "timestamp": datetime.now().isoformat(),
        "total_results": len(cron_results),
        "total_winners": sum(1 for r in cron_results if r.get("passed", False)),
        "strategy_stats": strategy_stats,
        "ticker_stats": ticker_stats,
        "strategies_by_win_rate": sorted(
            strategy_stats.items(), key=lambda x: x[1]["win_rate"], reverse=True
        ),
        "tickers_by_win_rate": sorted(
            ticker_stats.items(), key=lambda x: x[1]["win_rate"], reverse=True
        ),
        "top_scoring": sorted(
            cron_results, key=lambda x: x.get("score", 0), reverse=True
        )[:10],
    }

    # Save
    with open(INSIGHTS_FILE, "w") as f:
        json.dump(insights, f, indent=2)

    return insights


def generate_invention_prompt(insights: dict, market_analysis: dict) -> str:
    """Generate a prompt for strategy invention based on data."""
    top_strategies = insights.get("strategies_by_win_rate", [])[:3]
    top_tickers = insights.get("tickers_by_win_rate", [])[:5]
    winners = insights.get("top_scoring", [])[:5]

    prompt = f"""You are a quantitative strategy developer. Based on the following data, write a COMPLETE Python strategy file.

## Market Analysis
{json.dumps(market_analysis, indent=2)}

## Top Winning Strategies (by win rate)
{json.dumps(top_strategies, indent=2)}

## Top Tickers (by win rate)  
{json.dumps(top_tickers, indent=2)}

## Top Scoring Results
{json.dumps(winners, indent=2)}

## Existing Strategies
{get_strategy_catalog()}

## Requirements
Write a NEW strategy that is DIFFERENT from existing ones. Consider:
- Combine signals from 2+ indicators (confluence)
- Exploit patterns in the market analysis (momentum, mean reversion, volatility)
- Use ATR-based dynamic exits
- Volume-weighted entries
- Multi-indicator confirmation

The file must have EXACTLY this interface:
```python
import pandas as pd
import pandas_ta

DEFAULT_PARAMS = {{...}}
PARAM_GRID = {{...}}  # at least 3 values per param
DESCRIPTION = "..."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    ...
```

IMPORTANT RULES:
- df has columns: open, high, low, close, volume (all lowercase)
- entries and exits are pd.Series[bool] indexed same as df
- Use pandas_ta for ALL indicators (rsi, macd, ema, atr, adx, bbands, stoch, etc.)
- pandas_ta column naming: ADX is 'ADX_14', Bollinger Bands are 'BBU_20_2.0_2.0' (std repeated), Stochastic is 'STOCHk_14_3_3'/'STOCHd_14_3_3'
- The strategy MUST produce >0 trades on at least 2 tickers (AAPL, NVDA, CAT, SPY, GOOGL, JPM, XOM, GLD)
- Do NOT use params that aren't in DEFAULT_PARAMS
- Do NOT hardcode column names — detect them dynamically when needed

Write ONLY the Python code. No explanations."""

    return prompt


def test_on_multiple_tickers(strategy_code: str, tickers: list[str] = None) -> dict:
    """Test a strategy on multiple tickers."""
    if tickers is None:
        tickers = ["AAPL", "NVDA", "CAT", "SPY", "GOOGL", "JPM", "XOM", "GLD"]

    result = test_strategy_code(strategy_code, tickers[0])
    if not result["success"]:
        return result

    # Already tested first ticker, test the rest
    for t in tickers[1:]:
        try:
            df = load_data(t, period="2y")
        except Exception:
            continue

        # Re-execute to get the function
        ns = {"pd": __import__("pandas"), "np": __import__("numpy"), "pandas_ta": __import__("pandas_ta")}
        exec(strategy_code, ns)
        fn = ns["generate_signals"]
        defaults = ns.get("DEFAULT_PARAMS", {})

        try:
            entries, exits = fn(df, defaults)
            engine = BacktestEngine()
            r = engine.run(df, entries, exits, "test", t, 0, defaults)
            result["results"].append({
                "ticker": t,
                "sharpe": r.sharpe,
                "return": r.total_return,
                "max_dd": r.max_drawdown,
                "trades": r.num_trades,
                "passed": r.passed,
                "score": r.score,
            })
            if r.num_trades > 0:
                result["has_signals"] = True
        except Exception as e:
            result["results"].append({"ticker": t, "error": str(e)})

    result["num_tested"] = len(result["results"])
    result["num_profitable"] = sum(
        1 for r in result["results"] if "error" not in r and r.get("return", 0) > 0
    )
    return result


def main():
    args = sys.argv[1:]
    do_invent = "--invent" in args
    do_analyze = "--analyze" in args
    do_validate = "--validate-winners" in args
    do_expand = "--expand-grids" in args

    if not any([do_invent, do_analyze, do_validate, do_expand]):
        do_analyze = True
        do_invent = True

    print(f"🦀 CrabQuant Improvement Task — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Always analyze first
    if do_analyze:
        print("\n📊 Analyzing results...")
        insights = run_analysis()
        if "error" in insights:
            print(f"   {insights['error']}")
        else:
            print(f"   Total results: {insights['total_results']}")
            print(f"   Winners: {insights['total_winners']}")
            print(f"   Win rate: {insights['total_winners']/max(insights['total_results'],1)*100:.1f}%")
            print("\n   Strategy win rates:")
            for sname, stats in insights["strategies_by_win_rate"]:
                print(f"     {sname}: {stats['win_rate']:.0%} ({stats['won']}/{stats['tested']})")
            print("\n   Top tickers:")
            for ticker, stats in insights["tickers_by_win_rate"][:5]:
                print(f"     {ticker}: {stats['win_rate']:.0%} ({stats['won']}/{stats['tested']})")

    if do_invent:
        print("\n🔬 Running market analysis for invention...")
        analysis_tickers = ["AAPL", "NVDA", "CAT", "SPY", "JPM", "XOM", "GLD"]
        market_data = {}
        for t in analysis_tickers:
            a = analyze_market_data(t)
            if "error" not in a:
                market_data[t] = a

        # Find patterns
        momentum = [t for t, a in market_data.items() if a.get("momentum_tendency", 0) > 0.01]
        meanrev = [t for t, a in market_data.items() if a.get("mean_reversion_tendency", 0) > 0.45]
        highvol = [t for t, a in market_data.items() if a.get("vol_regime") == "high"]
        trending = [t for t, a in market_data.items() if a.get("above_sma200", False)]

        print(f"   Momentum: {', '.join(momentum) or 'none'}")
        print(f"   Mean-rev: {', '.join(meanrev) or 'none'}")
        print(f"   High-vol: {', '.join(highvol) or 'none'}")
        print(f"   Trending: {', '.join(trending) or 'none'}")

        # Load insights
        insights = {}
        if INSIGHTS_FILE.exists():
            with open(INSIGHTS_FILE) as f:
                insights = json.load(f)

        # Generate invention prompt
        prompt = generate_invention_prompt(insights, market_data)
        print("\n💡 Strategy invention prompt generated.")
        print(f"   Prompt length: {len(prompt)} chars")
        print("   NOTE: This prompt should be fed to an LLM to generate strategy code.")
        print("   The cron agent will use this prompt to invent strategies.")

        # Save prompt for the cron agent
        prompt_file = RESULTS_DIR / "invention_prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)
        print(f"   Saved to: {prompt_file}")

        # Show strategy catalog
        print(f"\n📚 Current strategies ({len(STRATEGY_REGISTRY)}):")
        for name in STRATEGY_REGISTRY:
            invented = " [INVENTED]" if name.startswith("invented_") else ""
            print(f"   - {name}{invented}")

    if do_validate:
        print("\n✅ Validating winners...")
        if not WINNERS_FILE.exists():
            print("   No winners to validate.")
        else:
            with open(WINNERS_FILE) as f:
                winners = json.load(f)
            print(f"   {len(winners)} winners to validate")
            for w in winners[:3]:
                print(f"   - {w['ticker']}/{w['strategy']}: Sharpe={w['sharpe']:.2f}")

    if do_expand:
        print("\n🔧 Checking param grids for expansion...")
        if INSIGHTS_FILE.exists():
            with open(INSIGHTS_FILE) as f:
                insights = json.load(f)
            for sname, stats in insights.get("strategies_by_win_rate", []):
                if stats["win_rate"] > 0.1 and sname in STRATEGY_REGISTRY:
                    _, _, grid, _ = STRATEGY_REGISTRY[sname]
                    total_combos = 1
                    for vals in grid.values():
                        total_combos *= len(vals)
                    print(f"   {sname}: win_rate={stats['win_rate']:.0%}, grid_size={total_combos}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
