#!/usr/bin/env python3
"""
CrabQuant Sweep Task

Sweep one strategy (or all strategies) across all tickers in parallel.
Outputs results to results/sweep_results.jsonl and updates results/cron_state.json.

Usage:
    python scripts/sweep_task.py                          # Sweep ALL strategies
    python scripts/sweep_task.py rsi_crossover            # Sweep one strategy
    python scripts/sweep_task.py --strategy rsi_crossover  # Same as above
    python scripts/sweep_task.py --workers 6               # Limit parallelism
    python scripts/sweep_task.py --tickers AAPL MSFT GOOGL # Sweep specific tickers only
"""

import argparse
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from crabquant.strategies import STRATEGY_REGISTRY
from crabquant.engine.parallel import parallel_backtest

RESULTS_DIR = Path(__file__).parent.parent / "results"
STATE_FILE = RESULTS_DIR / "cron_state.json"
SWEEP_LOG = RESULTS_DIR / "sweep_results.jsonl"

ALL_TICKERS = [
    # Large-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "NFLX", "CRM", "ORCL", "ADBE", "AVGO", "TXN", "QCOM", "INTC",
    # Mid-cap / high-growth
    "PLTR", "SHOP", "SQ", "SNOW", "RBLX", "DDOG", "NET", "MDB",
    # ETFs
    "SPY", "QQQ", "IWM", "GLD", "TLT", "XLK", "XLF", "XLE",
    # Finance
    "JPM", "V", "MA", "GS", "BLK", "SCHW",
    # Energy
    "XOM", "CVX", "COP",
    # Industrials
    "CAT", "GE", "DE", "HON",
    # Consumer / Telecom
    "T", "DIS", "NKE", "BA",
    # Healthcare
    "UNH", "JNJ", "PFE", "MRK",
    # Auto
    "GM", "F",
]


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "completed_combos": [],
        "validated_winners": [],
        "total_runs": 0,
        "total_winners": 0,
        "best_score": 0,
        "last_run": None,
        "dead_combos": [],
        "round": 0,
    }


def save_state(state: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def save_sweep_results(results: list):
    """Append results to sweep_results.jsonl."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_LOG, "a") as f:
        for r in results:
            record = {
                k: v for k, v in r.__dict__.items() if not callable(v)
            }
            record["timestamp"] = datetime.now().isoformat()
            f.write(json.dumps(record, default=str) + "\n")


def update_winners(results: list, state: dict):
    """Update winners.json with any passed results."""
    from scripts.cron_task import save_winner

    passed = [r for r in results if r.passed]
    if passed:
        for p in passed:
            state["total_winners"] += 1
            save_winner(p)
            if p.score > state["best_score"]:
                state["best_score"] = p.score


def mark_dead_combos(results: list, state: dict, strategy_name: str, tickers: list[str]):
    """Mark combos with zero trades as dead."""
    ticker_has_trades = set()
    for r in results:
        if r.num_trades > 0:
            ticker_has_trades.add(r.ticker)

    dead = set(state.get("dead_combos", []))
    for t in tickers:
        if t not in ticker_has_trades:
            dead.add(f"{strategy_name}|{t}")

    if dead - set(state.get("dead_combos", [])):
        state["dead_combos"] = sorted(dead)


def sweep_strategy(strategy_name: str, tickers: list[str], max_workers: int) -> dict:
    """Sweep one strategy across all tickers in parallel."""
    if strategy_name not in STRATEGY_REGISTRY:
        print(f"❌ Unknown strategy: {strategy_name}")
        print(f"   Available: {', '.join(sorted(STRATEGY_REGISTRY.keys()))}")
        return {"strategy": strategy_name, "results": [], "errors": []}

    _, defaults, param_grid, desc, matrix_fn = STRATEGY_REGISTRY[strategy_name]

    if not param_grid:
        print(f"⏭️ {strategy_name} has no param grid — skipping")
        return {"strategy": strategy_name, "results": [], "errors": []}

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    print(f"\n🦀 Sweeping {strategy_name} across {len(tickers)} tickers ({total_combos} param combos each)")
    print(f"   Desc: {desc[:80]}...")
    print(f"   Workers: {max_workers}")

    t0 = time.time()
    results = parallel_backtest(strategy_name, tickers, param_grid, max_workers=max_workers)
    elapsed = time.time() - t0

    traded = [r for r in results if r.num_trades > 0]
    passed = [r for r in results if r.passed]
    best = max(results, key=lambda r: r.score) if results else None

    print(f"\n   ✅ Done in {elapsed:.1f}s")
    print(f"   📊 {len(results)} total results | {len(traded)} with trades | {len(passed)} passed")

    if best:
        print(f"   🏆 Best: {best.ticker}/{best.strategy_name} "
              f"Sharpe={best.sharpe:.2f} Return={best.total_return:.1%} "
              f"Score={best.score:.2f}")

    if passed:
        print(f"   🎉 {len(passed)} WINNER(S):")
        for p in sorted(passed, key=lambda r: r.score, reverse=True)[:5]:
            print(f"      {p.ticker:6s} | Sharpe {p.sharpe:.2f} | Return {p.total_return:.1%} | "
                  f"Trades {p.num_trades} | Score {p.score:.2f}")
            print(f"         Params: {p.params}")

    return {"strategy": strategy_name, "results": results, "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(description="CrabQuant Parallel Sweep")
    parser.add_argument("strategy", nargs="?", help="Strategy name (default: all)")
    parser.add_argument("--strategy", dest="strategy_flag", help="Strategy name (alt)")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to sweep")
    parser.add_argument("--period", default="2y", help="Data period (default: 2y)")
    args = parser.parse_args()

    strategy_name = args.strategy or args.strategy_flag
    tickers = args.tickers if args.tickers else ALL_TICKERS

    if args.workers:
        max_workers = args.workers
    else:
        import os
        max_workers = min(os.cpu_count() or 4, len(tickers))

    state = load_state()

    if strategy_name:
        # Sweep one strategy
        sweep_result = sweep_strategy(strategy_name, tickers, max_workers)
        results = sweep_result["results"]

        if results:
            save_sweep_results(results)
            update_winners(results, state)
            mark_dead_combos(results, state, strategy_name, tickers)

            # Mark combos as completed in state
            completed_set = set(state["completed_combos"])
            for t in tickers:
                completed_set.add(f"{strategy_name}|{t}")
            state["completed_combos"] = sorted(completed_set)
            state["total_runs"] += 1
            state["last_run"] = datetime.now().isoformat()
            save_state(state)
    else:
        # Sweep all strategies
        strategies = sorted(STRATEGY_REGISTRY.keys())
        print(f"🚀 Full sweep: {len(strategies)} strategies × {len(tickers)} tickers = "
              f"{len(strategies) * len(tickers)} combos")

        grand_total_results = 0
        grand_total_passed = 0
        grand_start = time.time()

        for i, strat in enumerate(strategies):
            print(f"\n{'='*60}")
            print(f"Strategy {i+1}/{len(strategies)}: {strat}")
            print(f"{'='*60}")

            sweep_result = sweep_strategy(strat, tickers, max_workers)
            results = sweep_result["results"]

            if results:
                save_sweep_results(results)
                update_winners(results, state)
                mark_dead_combos(results, state, strat, tickers)

                completed_set = set(state["completed_combos"])
                for t in tickers:
                    completed_set.add(f"{strat}|{t}")
                state["completed_combos"] = sorted(completed_set)
                state["total_runs"] += 1
                state["last_run"] = datetime.now().isoformat()
                save_state(state)

            grand_total_results += len(results)
            grand_total_passed += len([r for r in results if r.passed])

        grand_elapsed = time.time() - grand_start
        print(f"\n{'='*60}")
        print(f"🏁 FULL SWEEP COMPLETE in {grand_elapsed:.1f}s")
        print(f"   {grand_total_results} total results | {grand_total_passed} winners")
        print(f"   State saved to {STATE_FILE}")
        print(f"   Results logged to {SWEEP_LOG}")

    print(f"\n✅ State saved to {STATE_FILE}")


if __name__ == "__main__":
    main()
