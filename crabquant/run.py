"""
CrabQuant — Main Runner

Usage:
    python -m crabquant.run                    # Full discovery sweep
    python -m crabquant.run --validate          # Validate existing winners
    python -m crabquant.run --strategy macd_momentum  # Single strategy deep dive
    python -m crabquant.run --ticker AAPL       # Single ticker all strategies
"""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crabquant.data import load_data
from crabquant.engine import BacktestEngine
from crabquant.strategies import STRATEGY_REGISTRY
from crabquant.validation import walk_forward_test, cross_ticker_validation, full_validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("crabquant")

RESULTS_DIR = Path(__file__).parent.parent / "results"
LOGS_DIR = RESULTS_DIR / "logs"
WINNERS_DIR = RESULTS_DIR / "winners"
VALIDATION_DIR = RESULTS_DIR / "validation"

# Default tickers for discovery
DEFAULT_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "NFLX", "CRM", "ORCL", "ADBE", "AVGO", "TXN", "QCOM", "INTC",
    # ETFs
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    # Finance
    "JPM", "V", "MA", "GS",
    # Energy
    "XOM",
    # Industrials
    "CAT", "GE",
    # Telecom
    "T", "DIS",
]


def sample_params(param_grid: dict) -> dict:
    """Get default params from a grid."""
    return {k: v[0] for k, v in param_grid.items()}


def mutate_params(params: dict, param_grid: dict, iteration: int) -> dict:
    """Mutate parameters to explore the search space."""
    import random
    new_params = {}
    for key, values in param_grid.items():
        current = params.get(key, values[0])
        try:
            idx = values.index(current)
        except ValueError:
            idx = 0
        # Shift by 1-2 positions, direction varies
        shift = 1 if iteration % 2 == 0 else -1
        shift *= (1 + iteration % 2)
        new_idx = max(0, min(len(values) - 1, idx + shift))
        new_params[key] = values[new_idx]
    return new_params


def save_result(result, results_dir: Path):
    """Append result to JSONL log."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "backtest_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            **{k: v for k, v in result.__dict__.items() if k != "timestamp"},
            "timestamp": result.timestamp,
        }, default=str) + "\n")


def print_result(r):
    """Pretty print a result."""
    status = "🏆 PASS" if r.passed else "❌ MISS"
    print(f"\n{'='*60}")
    print(f"{status} | {r.ticker} | {r.strategy_name} (iter {r.iteration})")
    print(f"  Sharpe: {r.sharpe:.2f} | Return: {r.total_return:.1%} | MaxDD: {r.max_drawdown:.1%}")
    print(f"  WinRate: {r.win_rate:.1%} | Trades: {r.num_trades} | Calmar: {r.calmar_ratio:.2f}")
    print(f"  Sortino: {r.sortino_ratio:.2f} | PF: {r.profit_factor:.2f} | Score: {r.score:.2f}")
    print(f"  {r.notes}")
    print(f"{'='*60}")


def run_discovery(
    strategies: list[str] | None = None,
    tickers: list[str] | None = None,
    max_iterations: int = 5,
):
    """Run strategy discovery sweep."""
    engine = BacktestEngine()
    strat_names = strategies or list(STRATEGY_REGISTRY.keys())
    ticker_list = tickers or DEFAULT_TICKERS

    logger.info(f"🦀 CrabQuant Discovery Sweep")
    logger.info(f"Strategies: {strat_names}")
    logger.info(f"Tickers: {ticker_list[:10]}... ({len(ticker_list)} total)")
    logger.info(f"Max iterations per combo: {max_iterations}")

    all_results = []

    for strat_name in strat_names:
        if strat_name not in STRATEGY_REGISTRY:
            logger.warning(f"Unknown strategy: {strat_name}")
            continue

        strategy_fn, defaults, param_grid, desc = STRATEGY_REGISTRY[strat_name]

        print(f"\n{'#'*60}")
        print(f"# {strat_name}: {desc[:80]}...")
        print(f"{'#'*60}")

        for ticker in ticker_list:
            try:
                df = load_data(ticker)
            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {e}")
                continue

            params = sample_params(param_grid)
            best = None

            for iteration in range(max_iterations):
                try:
                    entries, exits = strategy_fn(df, params)
                    result = engine.run(df, entries, exits, strat_name, ticker,
                                        iteration, params)
                    save_result(result, LOGS_DIR)
                    print_result(result)

                    if result.passed:
                        best = result
                        # Save winner
                        if iteration < max_iterations - 1:
                            params = mutate_params(params, param_grid, iteration)
                    elif result.sharpe > 0 and (best is None or result.sharpe > best.sharpe):
                        best = result

                    params = mutate_params(params, param_grid, iteration)

                except Exception as e:
                    logger.error(f"Error {ticker}/{strat_name} iter {iteration}: {e}")
                    traceback.print_exc()

            if best:
                all_results.append(best)

    # Summary
    print_summary(all_results)
    return all_results


def run_validation(winning_results: list | None = None):
    """Run walk-forward and cross-ticker validation on winners."""
    engine = BacktestEngine()

    # Load from logs if no results provided
    if winning_results is None:
        winning_results = load_winners_from_log()

    if not winning_results:
        logger.warning("No winners to validate. Run discovery first.")
        return []

    logger.info(f"🔍 Validating {len(winning_results)} winning strategies...")

    validation_results = []

    for result in winning_results:
        strat_name = result.strategy_name
        if strat_name not in STRATEGY_REGISTRY:
            continue

        strategy_fn, _, _, _ = STRATEGY_REGISTRY[strat_name]

        print(f"\n{'='*60}")
        print(f"Validating: {result.ticker}/{strat_name} (Sharpe {result.sharpe:.2f})")
        print(f"{'='*60}")

        # Walk-forward test
        wf = walk_forward_test(strategy_fn, result.ticker, result.params, engine=engine)
        print(f"  Walk-Forward:")
        print(f"    Train: Sharpe {wf.train_sharpe:.2f}, Return {wf.train_return:.1%}")
        print(f"    Test:  Sharpe {wf.test_sharpe:.2f}, Return {wf.test_return:.1%}")
        print(f"    Degradation: {wf.degradation:.1%} {'✅' if wf.robust else '❌'}")

        # Cross-ticker validation
        oos_tickers = [t for t in DEFAULT_TICKERS if t != result.ticker][:15]
        ct = cross_ticker_validation(strategy_fn, result.params, oos_tickers, engine=engine)
        print(f"  Cross-Ticker ({ct.tickers_tested} tickers):")
        print(f"    Profitable: {ct.tickers_profitable}/{ct.tickers_tested} ({ct.win_rate_across_tickers:.0%})")
        print(f"    Avg Sharpe: {ct.avg_sharpe:.2f} (σ={ct.sharpe_std:.2f})")
        print(f"    Median Sharpe: {ct.median_sharpe:.2f}")
        print(f"    Robust: {'✅' if ct.robust else '❌'}")

        validation_results.append({
            "ticker": result.ticker,
            "strategy": strat_name,
            "discovery_sharpe": result.sharpe,
            "walk_forward": {
                "test_sharpe": wf.test_sharpe,
                "test_return": wf.test_return,
                "degradation": wf.degradation,
                "robust": wf.robust,
            },
            "cross_ticker": {
                "tickers_profitable": ct.tickers_profitable,
                "tickers_tested": ct.tickers_tested,
                "avg_sharpe": ct.avg_sharpe,
                "median_sharpe": ct.median_sharpe,
                "robust": ct.robust,
            },
            "overall_robust": wf.robust and ct.robust,
        })

    # Save validation results
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_DIR / "validation_results.json", "w") as f:
        json.dump(validation_results, f, indent=2)

    # Summary
    robust = [v for v in validation_results if v["overall_robust"]]
    print(f"\n{'='*60}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total validated: {len(validation_results)}")
    print(f"Overall robust: {len(robust)}")
    if robust:
        print(f"\n🛡️ Robust Strategies (passed both walk-forward AND cross-ticker):")
        for v in robust:
            print(f"  {v['ticker']:6s} | {v['strategy']:25s} | "
                  f"Discovery Sharpe {v['discovery_sharpe']:.2f} | "
                  f"OOS Sharpe {v['walk_forward']['test_sharpe']:.2f} | "
                  f"Cross-Ticker Avg {v['cross_ticker']['avg_sharpe']:.2f}")

    return validation_results


def load_winners_from_log() -> list:
    """Load winning results from backtest log."""
    log_path = LOGS_DIR / "backtest_log.jsonl"
    if not log_path.exists():
        return []

    winners = []
    with open(log_path) as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get("passed"):
                winners.append(type('Result', (), data)())

    return winners


def print_summary(results):
    """Print discovery summary."""
    if not results:
        print("\nNo results to summarize.")
        return

    passed = [r for r in results if r.passed]
    all_valid = [r for r in results if r.sharpe > 0 and r.num_trades > 0]

    print(f"\n{'='*70}")
    print("📊 CRABQUANT DISCOVERY SUMMARY")
    print(f"{'='*70}")
    print(f"Total combos: {len(results)}")
    print(f"Valid results (trades > 0): {len(all_valid)}")
    print(f"🏆 Passed target: {len(passed)}")

    if passed:
        passed.sort(key=lambda r: r.score, reverse=True)
        print(f"\n🏆 Top Winners (by composite score):")
        for r in passed[:15]:
            print(f"  {r.ticker:6s} | {r.strategy_name:25s} | "
                  f"Sharpe {r.sharpe:5.2f} | Return {r.total_return:7.1%} | "
                  f"MaxDD {r.max_drawdown:6.1%} | Trades {r.num_trades:3d} | "
                  f"Score {r.score:5.2f}")

    if all_valid:
        all_valid.sort(key=lambda r: r.sharpe, reverse=True)
        print(f"\n📈 Top 10 by Sharpe:")
        for r in all_valid[:10]:
            tag = "🏆" if r.passed else "  "
            print(f"  {tag} {r.ticker:6s} | {r.strategy_name:25s} | "
                  f"Sharpe {r.sharpe:5.2f} | Return {r.total_return:7.1%}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_combos": len(results),
        "passed": len(passed),
        "top_winners": [
            {k: v for k, v in r.__dict__.items()}
            for r in (sorted(passed, key=lambda r: r.score, reverse=True)[:15]) if r.passed
        ] if passed else [],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="CrabQuant — Autonomous Strategy Engine")
    parser.add_argument("--validate", action="store_true", help="Validate existing winners")
    parser.add_argument("--strategy", type=str, help="Run specific strategy")
    parser.add_argument("--ticker", type=str, help="Run specific ticker")
    parser.add_argument("--iterations", type=int, default=5, help="Max iterations per combo")
    parser.add_argument("--tickers", type=str, help="Comma-separated ticker list")
    args = parser.parse_args()

    if args.validate:
        run_validation()
    elif args.strategy or args.ticker:
        strategies = [args.strategy] if args.strategy else None
        tickers = args.ticker.split(",") if args.ticker else None
        run_discovery(strategies=strategies, tickers=tickers, max_iterations=args.iterations)
    else:
        run_discovery(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
