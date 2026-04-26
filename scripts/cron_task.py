"""
CrabQuant Autonomous Cron Task

Runs one discovery unit per invocation. Designed for OpenClaw cron (every 10 min).
Picks next untested strategy/ticker combo, runs optimization, validates winners.

Usage:
    python scripts/cron_task.py                # One combo, auto-pick
    python scripts/cron_task.py --strategy macd_momentum  # Specific strategy
    python scripts/cron_task.py --ticker CAT   # Specific ticker
    python scripts/cron_task.py --validate     # Validate unvalidated winners
    python scripts/cron_task.py --status       # Print current progress
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from crabquant.data import load_data
from crabquant.engine import BacktestEngine
from crabquant.strategies import STRATEGY_REGISTRY

RESULTS_DIR = Path(__file__).parent.parent / "results"
STATE_FILE = RESULTS_DIR / "cron_state.json"
LOG_FILE = RESULTS_DIR / "logs" / "cron_results.jsonl"
WINNERS_FILE = RESULTS_DIR / "winners" / "winners.json"

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
    "TSLA", "GM", "F",
]


def load_state() -> dict:
    """Load cron state from disk."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "completed_combos": [],  # ["strategy|ticker", ...]
        "validated_winners": [],  # ["strategy|ticker|params_hash", ...]
        "total_runs": 0,
        "total_winners": 0,
        "best_score": 0,
        "last_run": None,
    }


def save_state(state: dict):
    """Save cron state to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_next_combo(state: dict, strategy_filter: str = None, ticker_filter: str = None) -> tuple[str, str] | None:
    """Get the next untested strategy/ticker combo."""
    completed = set(state["completed_combos"])

    strategies = list(STRATEGY_REGISTRY.keys())
    if strategy_filter:
        strategies = [s for s in strategies if strategy_filter in s]

    tickers = ALL_TICKERS
    if ticker_filter:
        tickers = [t for t in tickers if ticker_filter.upper() in t.upper()]

    for strat, ticker in product(strategies, tickers):
        combo_key = f"{strat}|{ticker}"
        if combo_key not in completed:
            return strat, ticker

    # All combos done — reset for a new round with different params
    print("🔄 All combos completed! Resetting for new round.")
    state["completed_combos"] = []
    save_state(state)
    return None


def optimize_combo(strategy_name: str, ticker: str, max_iters: int = 8) -> dict | None:
    """
    Run parameter optimization for a single strategy/ticker combo.
    Returns best result or None if no improvement found.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        print(f"❌ Unknown strategy: {strategy_name}")
        return None

    strategy_fn, defaults, param_grid, desc = STRATEGY_REGISTRY[strategy_name]
    engine = BacktestEngine()

    print(f"\n🔍 {ticker}/{strategy_name}: {desc[:60]}...")
    print(f"   Param grid: {list(param_grid.keys())}")

    try:
        df = load_data(ticker)
    except Exception as e:
        print(f"   ❌ Data load failed: {e}")
        return None

    params = dict(defaults)
    best_result = None
    best_score = -999

    for i in range(max_iters):
        try:
            entries, exits = strategy_fn(df, params)
            result = engine.run(df, entries, exits, strategy_name, ticker, i, params)

            if result.num_trades == 0 and i == 0:
                print(f"   ⚡ Iter {i}: Zero trades — skipping combo")
                return None

            tag = "🏆" if result.passed else "  "
            print(f"   {tag} Iter {i}: Sharpe {result.sharpe:.2f} | "
                  f"Return {result.total_return:.1%} | "
                  f"Trades {result.num_trades} | "
                  f"Score {result.score:.2f}")

            if result.score > best_score and result.num_trades > 0:
                best_score = result.score
                best_result = result

            # Early stop: if we found a winner, try 2 more iters to optimize
            if result.passed and i >= 3:
                break

            # Mutate params for next iteration
            params = mutate_params(params, param_grid, i)

        except Exception as e:
            print(f"   ❌ Iter {i} error: {e}")
            continue

    return best_result


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
        # Shift 1-2 positions, alternating direction
        shift = 1 if iteration % 2 == 0 else -1
        shift *= (1 + iteration % 2)
        new_idx = max(0, min(len(values) - 1, idx + shift))
        new_params[key] = values[new_idx]
    return new_params


def save_result(result):
    """Append result to cron results log."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            **{k: v for k, v in result.__dict__.items() if not callable(v)},
            "timestamp": datetime.now().isoformat(),
        }, default=str) + "\n")


def save_winner(result):
    """Append winner to winners file."""
    WINNERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    winners = []
    if WINNERS_FILE.exists():
        with open(WINNERS_FILE) as f:
            winners = json.load(f)

    # Check for duplicate (same strategy + ticker + params)
    key = f"{result.strategy_name}|{result.ticker}|{json.dumps(result.params, sort_keys=True)}"
    if not any(w.get("key") == key for w in winners):
        winners.append({
            "key": key,
            "ticker": result.ticker,
            "strategy": result.strategy_name,
            "sharpe": result.sharpe,
            "return": result.total_return,
            "max_dd": result.max_drawdown,
            "trades": result.num_trades,
            "score": result.score,
            "win_rate": result.win_rate,
            "calmar": result.calmar_ratio,
            "sortino": result.sortino_ratio,
            "profit_factor": result.profit_factor,
            "params": result.params,
            "discovered": datetime.now().isoformat(),
        })

    # Keep top 50 by score
    winners.sort(key=lambda w: w.get("score", 0), reverse=True)
    winners = winners[:50]

    with open(WINNERS_FILE, "w") as f:
        json.dump(winners, f, indent=2)


def print_status():
    """Print current cron progress."""
    state = load_state()
    total_possible = len(STRATEGY_REGISTRY) * len(ALL_TICKERS)
    completed = len(state["completed_combos"])
    pct = (completed / total_possible * 100) if total_possible > 0 else 0

    print(f"🦀 CrabQuant Cron Status")
    print(f"{'='*50}")
    print(f"Combos tested: {completed}/{total_possible} ({pct:.1f}%)")
    print(f"Total runs: {state['total_runs']}")
    print(f"Winners found: {state['total_winners']}")
    print(f"Best score: {state['best_score']:.2f}")
    print(f"Last run: {state.get('last_run', 'Never')}")

    if WINNERS_FILE.exists():
        with open(WINNERS_FILE) as f:
            winners = json.load(f)
        print(f"\n🏆 Top 10 Winners:")
        for w in winners[:10]:
            print(f"  {w['ticker']:6s} | {w['strategy']:25s} | "
                  f"Sharpe {w['sharpe']:5.2f} | Return {w['return']:7.1%} | "
                  f"Trades {w['trades']:3d} | Score {w['score']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="CrabQuant Cron Task")
    parser.add_argument("--strategy", type=str, help="Specific strategy")
    parser.add_argument("--ticker", type=str, help="Specific ticker")
    parser.add_argument("--validate", action="store_true", help="Validate recent winners")
    parser.add_argument("--status", action="store_true", help="Print status")
    parser.add_argument("--max-iters", type=int, default=8, help="Max optimization iterations")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    state = load_state()

    if args.validate:
        # Run walk-forward validation on recent winners
        from crabquant.validation import walk_forward_test

        if not WINNERS_FILE.exists():
            print("No winners to validate.")
            return

        with open(WINNERS_FILE) as f:
            winners = json.load(f)

        # Pick unvalidated winners
        unvalidated = [w for w in winners if w["key"] not in state["validated_winners"]]

        if not unvalidated:
            print("All winners already validated.")
            return

        w = unvalidated[0]
        strat_name = w["strategy"]
        if strat_name in STRATEGY_REGISTRY:
            strategy_fn = STRATEGY_REGISTRY[strat_name][0]
            print(f"\n🛡️ Validating {w['ticker']}/{strat_name}...")
            wf = walk_forward_test(strategy_fn, w["ticker"], w["params"])
            print(f"  Train: Sharpe {wf.train_sharpe:.2f}, Return {wf.train_return:.1%}")
            print(f"  Test:  Sharpe {wf.test_sharpe:.2f}, Return {wf.test_return:.1%}")
            print(f"  Degradation: {wf.degradation:.1%} {'✅' if wf.robust else '❌'}")

            state["validated_winners"].append(w["key"])
            save_state(state)

        return

    # Normal discovery mode — pick next combo
    combo = get_next_combo(state, args.strategy, args.ticker)
    if combo is None:
        print("No combo to run.")
        return

    strategy_name, ticker = combo
    print(f"\n🦀 CrabQuant Cron — {datetime.now().strftime('%H:%M:%S')}")
    print(f"Running {ticker}/{strategy_name} (up to {args.max_iters} iterations)...")

    start = time.time()
    result = optimize_combo(strategy_name, ticker, max_iters=args.max_iters)
    elapsed = time.time() - start

    # Update state
    state["completed_combos"].append(f"{strategy_name}|{ticker}")
    state["total_runs"] += 1
    state["last_run"] = datetime.now().isoformat()

    if result and result.passed:
        state["total_winners"] += 1
        if result.score > state["best_score"]:
            state["best_score"] = result.score
        save_result(result)
        save_winner(result)
        print(f"\n🏆 WINNER! {ticker}/{strategy_name} | "
              f"Sharpe {result.sharpe:.2f} | Return {result.total_return:.1%} | "
              f"Score {result.score:.2f} | ({elapsed:.0f}s)")
    elif result:
        save_result(result)
        print(f"\n❌ No pass — best: Sharpe {result.sharpe:.2f} | ({elapsed:.0f}s)")
    else:
        print(f"\n⏭️ Skipped — zero trades or error | ({elapsed:.0f}s)")

    save_state(state)
    print_status()


if __name__ == "__main__":
    main()
