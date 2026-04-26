"""
CrabQuant Autonomous Cron Task v3

Improvements over v2:
1. **Vectorized backtesting** — ALL param combos tested in one vbt.Portfolio.from_signals() call
2. **50-100x speedup** — replaces sequential per-combo iterations
3. **Smart combo picking** — prioritizes combos likely to succeed based on historical patterns
4. **Self-improvement analysis** — detects patterns in winners for the agent to act on
5. **Zero-trade early exit** — skips combo after zero trades across all combos

Usage:
    python scripts/cron_task.py                # One combo, auto-pick (smart ordering)
    python scripts/cron_task.py --strategy X    # Specific strategy
    python scripts/cron_task.py --ticker Y      # Specific ticker
    python scripts/cron_task.py --validate      # Validate unvalidated winners
    python scripts/cron_task.py --analyze       # Run self-improvement analysis
    python scripts/cron_task.py --status        # Print current progress
    python scripts/cron_task.py --benchmark     # Benchmark vectorized vs sequential
"""

import argparse
import json
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path

MAX_GRID_SIZE = 5000  # Sample down to this if grid is larger

sys.path.insert(0, str(Path(__file__).parent.parent))

from crabquant.data import load_data
from crabquant.engine import BacktestEngine
from crabquant.strategies import STRATEGY_REGISTRY

RESULTS_DIR = Path(__file__).parent.parent / "results"
STATE_FILE = RESULTS_DIR / "cron_state.json"
LOG_FILE = RESULTS_DIR / "logs" / "cron_results.jsonl"
WINNERS_FILE = RESULTS_DIR / "winners" / "winners.json"
INSIGHTS_FILE = RESULTS_DIR / "insights.json"
DEAD_COMBOS_FILE = RESULTS_DIR / "dead_combos.json"

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


# ─── State Management ─────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "completed_combos": [],
        "validated_winners": [],
        "total_runs": 0,
        "total_passing_iterations": 0,
        "best_score": 0,
        "last_run": None,
        "dead_combos": [],
        "round": 0,
    }


def save_state(state: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_dead_combos() -> set:
    if DEAD_COMBOS_FILE.exists():
        with open(DEAD_COMBOS_FILE) as f:
            return set(json.load(f))
    return set()


def save_dead_combos(dead: set):
    DEAD_COMBOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DEAD_COMBOS_FILE, "w") as f:
        json.dump(sorted(dead), f)


def load_winners() -> list:
    if WINNERS_FILE.exists():
        with open(WINNERS_FILE) as f:
            return json.load(f)
    return []


def load_all_results() -> list:
    results = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return results


# ─── Smart Combo Picking ─────────────────────────────────────────────────────

def get_combo_priority(strategy: str, ticker: str, state: dict, winners: list) -> float:
    """
    Score how promising a combo is. Higher = test first.
    """
    dead = set(state.get("dead_combos", []))
    key = f"{strategy}|{ticker}"
    if key in dead:
        return -999

    strat_wins = Counter(w["strategy"] for w in winners)
    ticker_wins = Counter(w["ticker"] for w in winners)

    ticker_sector_wins = 0
    for w in winners:
        if w["strategy"] == strategy and w["ticker"] == ticker:
            ticker_sector_wins += 3

    score = 0.0
    score += strat_wins.get(strategy, 0) * 1.0
    score += ticker_wins.get(ticker, 0) * 1.0
    score += ticker_sector_wins

    return score


def get_next_combo(state: dict, strategy_filter=None, ticker_filter=None) -> tuple[str, str] | None:
    """Get next untested combo, prioritized by likelihood of success."""
    completed = set(state["completed_combos"])
    dead = set(state.get("dead_combos", []))
    winners = load_winners()

    strategies = list(STRATEGY_REGISTRY.keys())
    if strategy_filter:
        strategies = [s for s in strategies if strategy_filter in s]

    tickers = ALL_TICKERS
    if ticker_filter:
        tickers = [t for t in tickers if ticker_filter.upper() in t.upper()]

    candidates = []
    for strat, ticker in product(strategies, tickers):
        key = f"{strat}|{ticker}"
        if key not in completed and key not in dead:
            priority = get_combo_priority(strat, ticker, state, winners)
            candidates.append((priority, strat, ticker))

    if not candidates:
        print(f"🔄 Round {state.get('round', 0) + 1} complete! Starting new round.")
        state["completed_combos"] = []
        state["round"] = state.get("round", 0) + 1
        save_state(state)
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)

    import random
    top = candidates[:max(10, len(candidates) // 5)]
    chosen = random.choice(top) if len(top) > 1 else top[0]

    return chosen[1], chosen[2]


# ─── Vectorized Parameter Sweep ───────────────────────────────────────────────

def sample_param_grid(param_grid: dict, max_size: int = MAX_GRID_SIZE) -> tuple[dict, int]:
    """
    If param_grid has more than max_size combos, randomly sample max_size combos.
    Returns (possibly_sampled_grid, original_total_combos).
    """
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    if total_combos <= max_size:
        return param_grid, total_combos

    # Generate all combos, then randomly sample
    keys = list(param_grid.keys())
    all_combos = list(product(*(param_grid[k] for k in keys)))
    sampled = random.sample(all_combos, max_size)

    # Rebuild a grid from sampled combos — one value per param
    sampled_grid = defaultdict(list)
    for combo in sampled:
        for k, v in zip(keys, combo):
            if v not in sampled_grid[k]:
                sampled_grid[k].append(v)
    # Sort each param list to keep deterministic ordering within the sampled subset
    sampled_grid = {k: sorted(vs) for k, vs in sampled_grid.items()}

    # Recalculate actual combos in the sampled grid
    sampled_total = 1
    for v in sampled_grid.values():
        sampled_total *= len(v)

    return dict(sampled_grid), total_combos


def vectorized_sweep(matrix_fn, param_grid, df, engine,
                     strategy_name, ticker) -> list:
    """
    Run param combos in one vectorized vbt.Portfolio.from_signals() call.
    If the grid is too large (> MAX_GRID_SIZE), randomly samples combos first.
    Returns list of BacktestResult for all combos tested.
    """
    effective_grid, original_total = sample_param_grid(param_grid)
    if original_total > MAX_GRID_SIZE:
        sampled_total = 1
        for v in effective_grid.values():
            sampled_total *= len(v)
        print(f"   🔀 Grid has {original_total:,} combos, sampling ~{sampled_total:,}")

    entries_df, exits_df, param_list = matrix_fn(df, effective_grid)

    if entries_df.empty or len(param_list) == 0:
        print(f"   ⚡ No param combos generated")
        return []

    print(f"   📊 Testing {len(param_list)} param combos in one vectorized call...")

    results = engine.run_vectorized(df, entries_df, exits_df, param_list, strategy_name, ticker)

    # Summary stats
    traded = [r for r in results if r.num_trades > 0]
    passed = [r for r in results if r.passed]
    best = max(results, key=lambda r: r.score) if results else None

    print(f"   📈 {len(traded)} combos with trades | {len(passed)} passed | "
          f"Best score: {best.score:.2f} (Sharpe {best.sharpe:.2f})" if best else
          f"   📈 {len(traded)} combos with trades | {len(passed)} passed")

    return results


def directed_search(strategy_fn, defaults, param_grid, df, engine,
                    strategy_name, ticker, max_iters=10) -> object | None:
    """
    Directed parameter search (kept for backward compatibility / single-strategy mode).
    1. Start with defaults
    2. If a promising result found (Sharpe > 0.5), narrow search around those params
    3. If a winner found, do fine-tuning around winner params
    """
    params = dict(defaults)
    best_result = None
    best_score = -999
    best_params = None

    phase = "exploration"
    narrow_params = None

    for i in range(max_iters):
        try:
            if phase == "narrowing" and narrow_params:
                params = perturb_params(narrow_params, param_grid, radius=1)
            elif phase == "fine_tuning" and best_params:
                params = perturb_params(best_params, param_grid, radius=1)

            entries, exits = strategy_fn(df, params)
            result = engine.run(df, entries, exits, strategy_name, ticker, i, params)

            if result.num_trades == 0 and i == 0:
                print(f"   ⚡ Iter {i}: Zero trades — skipping combo")
                return None

            tag = "🏆" if result.passed else ("📈" if result.sharpe > 0.5 else "  ")
            phase_tag = f"[{phase}]"
            print(f"   {tag} {phase_tag} Iter {i}: Sharpe {result.sharpe:.2f} | "
                  f"Return {result.total_return:.1%} | "
                  f"Trades {result.num_trades} | "
                  f"Score {result.score:.2f}")

            if result.score > best_score and result.num_trades > 0:
                best_score = result.score
                best_result = result
                best_params = dict(params)

            if phase == "exploration" and result.sharpe > 0.5 and narrow_params is None:
                phase = "narrowing"
                narrow_params = dict(params)
                print(f"   🔍 Promising — narrowing search around these params")

            if phase == "narrowing" and result.passed:
                phase = "fine_tuning"
                print(f"   🎯 Winner found — fine-tuning")

            if result.passed and i >= 5 and phase == "fine_tuning":
                break
            if i >= 3 and best_result and best_result.sharpe < 0.3:
                print(f"   🛑 Best Sharpe {best_result.sharpe:.2f} — unlikely to improve")
                break

        except Exception as e:
            print(f"   ❌ Iter {i} error: {e}")
            continue

    return best_result


def perturb_params(params: dict, param_grid: dict, radius: int = 1) -> dict:
    """Perturb params by ±radius positions in the grid."""
    import random
    new_params = dict(params)
    keys_to_perturb = random.sample(list(param_grid.keys()), min(2, len(param_grid)))

    for key in keys_to_perturb:
        values = param_grid[key]
        current = params.get(key, values[0])
        try:
            idx = values.index(current)
        except ValueError:
            idx = 0
        shift = random.randint(-radius, radius)
        new_idx = max(0, min(len(values) - 1, idx + shift))
        new_params[key] = values[new_idx]

    return new_params


# ─── Benchmark ────────────────────────────────────────────────────────────────

def run_benchmark():
    """Compare vectorized vs sequential approach on one strategy+ticker."""
    import numpy as np

    strategy_name = "rsi_crossover"
    ticker = "AAPL"
    print(f"🔄 Benchmark: {ticker}/{strategy_name}")

    strategy_fn, defaults, param_grid, desc, matrix_fn = STRATEGY_REGISTRY[strategy_name]

    # Use a smaller grid for benchmark
    small_grid = {k: v[:3] for k, v in param_grid.items()}
    total_combos = 1
    for v in small_grid.values():
        total_combos *= len(v)
    print(f"   Grid: {total_combos} combos (first 3 values per param)")

    try:
        df = load_data(ticker, period="2y")
    except Exception as e:
        print(f"❌ Data load failed: {e}")
        return

    engine = BacktestEngine()

    # Sequential benchmark
    from itertools import product as iproduct
    keys = list(small_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in iproduct(*(small_grid[k] for k in keys))]

    print(f"\n   🐢 Sequential ({total_combos} calls to Portfolio.from_signals)...")
    t0 = time.time()
    seq_results = []
    for i, params in enumerate(combos):
        entries, exits = strategy_fn(df, params)
        result = engine.run(df, entries, exits, strategy_name, ticker, i, params)
        seq_results.append(result)
    seq_time = time.time() - t0
    seq_best = max(seq_results, key=lambda r: r.score)

    # Vectorized benchmark
    print(f"   🚀 Vectorized (1 call to Portfolio.from_signals)...")
    t0 = time.time()
    entries_df, exits_df, param_list = matrix_fn(df, small_grid)
    vec_results = engine.run_vectorized(df, entries_df, exits_df, param_list, strategy_name, ticker)
    vec_time = time.time() - t0
    vec_best = max(vec_results, key=lambda r: r.score)

    print(f"\n{'='*60}")
    print(f"{'Metric':<25} {'Sequential':>12} {'Vectorized':>12} {'Speedup':>10}")
    print(f"{'-'*60}")
    print(f"{'Time (s)':<25} {seq_time:>12.2f} {vec_time:>12.2f} {seq_time/vec_time:>10.1f}x")
    print(f"{'Combos tested':<25} {len(seq_results):>12} {len(vec_results):>12}")
    print(f"{'Best Sharpe':<25} {seq_best.sharpe:>12.2f} {vec_best.sharpe:>12.2f}")
    print(f"{'Best Score':<25} {seq_best.score:>12.2f} {vec_best.score:>12.2f}")
    print(f"{'Passed':<25} {sum(1 for r in seq_results if r.passed):>12} {sum(1 for r in vec_results if r.passed):>12}")

    # Verify results match
    sharpe_diffs = []
    for sr, vr in zip(seq_results, vec_results):
        if sr.num_trades > 0 and vr.num_trades > 0:
            sharpe_diffs.append(abs(sr.sharpe - vr.sharpe))

    if sharpe_diffs:
        print(f"\n   ✅ Sharpe comparison (traded combos): max diff = {max(sharpe_diffs):.6f}")
    else:
        print(f"\n   ⚠️  No traded combos to compare")


# ─── Self-Improvement Analysis ────────────────────────────────────────────────

def run_analysis() -> dict:
    """Analyze all results to find patterns and generate actionable insights."""
    results = load_all_results()
    winners = load_winners()
    state = load_state()
    dead = set(state.get("dead_combos", []))

    if not results:
        return {"insights": ["No results yet — cron needs to run more cycles."]}

    insights = []
    actions = []

    strat_results = defaultdict(list)
    for r in results:
        strat_results[r["strategy_name"]].append(r)

    print(f"\n📊 Self-Improvement Analysis")
    print(f"{'='*60}")
    print(f"Total results: {len(results)}")
    print(f"Total winners: {len(winners)}")
    print(f"Dead combos: {len(dead)}")

    strat_win_rates = {}
    for strat, rs in strat_results.items():
        wins = sum(1 for r in rs if r.get("passed"))
        total = len(rs)
        rate = wins / total if total > 0 else 0
        avg_sharpe = sum(r["sharpe"] for r in rs if r["sharpe"] > 0) / max(1, sum(1 for r in rs if r["sharpe"] > 0))
        strat_win_rates[strat] = {"rate": rate, "wins": wins, "total": total, "avg_sharpe": avg_sharpe}

    print(f"\n📈 Strategy Performance:")
    for strat, stats in sorted(strat_win_rates.items(), key=lambda x: x[1]["rate"], reverse=True):
        bar = "█" * int(stats["rate"] * 20)
        print(f"  {strat:25s} {stats['rate']:5.1%} ({stats['wins']}/{stats['total']}) avg Sharpe {stats['avg_sharpe']:.2f} {bar}")

    ticker_wins = Counter(w["ticker"] for w in winners)
    print(f"\n🔥 Ticker Win Counts:")
    for ticker, count in ticker_wins.most_common(10):
        print(f"  {ticker:6s}: {count} wins")

    param_patterns = defaultdict(lambda: defaultdict(int))
    for w in winners:
        for k, v in w.get("params", {}).items():
            param_patterns[k][str(v)] += 1

    print(f"\n🎯 Most Common Winning Params:")
    for param, values in sorted(param_patterns.items()):
        top = sorted(values.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  {param:20s}: {', '.join(f'{v} ({c}x)' for v, c in top)}")

    combo_wins = Counter(f"{w['strategy']}|{w['ticker']}" for w in winners)
    if combo_wins:
        print(f"\n🔗 Strategy-Ticker Affinities:")
        for combo, count in combo_wins.most_common(10):
            print(f"  {combo}: {count} wins")

    dead_by_strat = Counter(c.split("|")[0] for c in dead)
    if dead:
        print(f"\n💀 Dead Combos (0 trades) by Strategy:")
        for strat, count in dead_by_strat.most_common():
            print(f"  {strat}: {count} tickers")

    if strat_win_rates:
        best_strat = max(strat_win_rates, key=lambda x: strat_win_rates[x]["rate"])
        worst_strat = min(strat_win_rates, key=lambda x: strat_win_rates[x]["rate"] if strat_win_rates[x]["total"] >= 5 else 1.0)
        insights.append(f"Best strategy: {best_strat} ({strat_win_rates[best_strat]['rate']:.0%} win rate)")
        if strat_win_rates[worst_strat]["total"] >= 5:
            insights.append(f"Weakest strategy: {worst_strat} ({strat_win_rates[worst_strat]['rate']:.0%} win rate) — consider removing or rewriting")

    if ticker_wins:
        top_ticker = ticker_wins.most_common(1)[0]
        insights.append(f"Most winning ticker: {top_ticker[0]} ({top_ticker[1]} wins)")

    if dead:
        insights.append(f"{len(dead)} dead combos — strategy/ticker pairs that produce 0 trades")

    if strat_win_rates:
        for strat, stats in strat_win_rates.items():
            if stats["rate"] > 0.3 and stats["total"] >= 5:
                actions.append(f"Expand {strat} param grid — high win rate suggests more winning params exist")
                break
        for w in winners:
            actions.append(f"Consider building sector-specific variant of {w['strategy']} (works well on {w['ticker']})")
            break

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_results": len(results),
        "total_winners": len(winners),
        "dead_combos": len(dead),
        "strategy_rates": strat_win_rates,
        "ticker_wins": dict(ticker_wins.most_common(10)),
        "param_patterns": {k: dict(v) for k, v in param_patterns.items()},
        "insights": insights,
        "actions": actions,
        "round": state.get("round", 0),
        "combos_completed": len(state["completed_combos"]),
        "combos_remaining": len(STRATEGY_REGISTRY) * len(ALL_TICKERS) - len(state["completed_combos"]),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(INSIGHTS_FILE, "w") as f:
        json.dump(analysis, f, indent=2)

    if actions:
        print(f"\n💡 Suggested Actions:")
        for a in actions[:5]:
            print(f"  → {a}")

    return analysis


# ─── Result Persistence ───────────────────────────────────────────────────────

def save_result(result):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            **{k: v for k, v in result.__dict__.items() if not callable(v)},
            "timestamp": datetime.now().isoformat(),
        }, default=str) + "\n")


def _detect_current_regime() -> str:
    """Detect current market regime. Returns regime value string or 'unknown'."""
    try:
        from crabquant.regime import detect_regime
        from crabquant.data import load_data
        spy_data = load_data("SPY", period="2mo")
        regime, _ = detect_regime(spy_data)
        return regime.value
    except Exception:
        return "unknown"


def save_winner(result):
    WINNERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    winners = load_winners()

    key = f"{result.strategy_name}|{result.ticker}|{json.dumps(result.params, sort_keys=True)}"
    if not any(w.get("key") == key for w in winners):
        regime = _detect_current_regime()
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
            "regime": regime,
        })

    winners.sort(key=lambda w: w.get("score", 0), reverse=True)
    winners = winners[:50]

    with open(WINNERS_FILE, "w") as f:
        json.dump(winners, f, indent=2)


# ─── Status Display ───────────────────────────────────────────────────────────

def print_status():
    state = load_state()
    total_possible = len(STRATEGY_REGISTRY) * len(ALL_TICKERS)
    completed = len(state["completed_combos"])
    dead = len(state.get("dead_combos", []))
    pct = (completed / total_possible * 100) if total_possible > 0 else 0

    # Migrate legacy key if present
    if "total_winners" in state and "total_passing_iterations" not in state:
        state["total_passing_iterations"] = state["total_winners"]
        save_state(state)

    print(f"🦀 CrabQuant Cron Status (v3 — Vectorized)")
    print(f"{'='*50}")
    print(f"Round: {state.get('round', 0)}")
    print(f"Combos tested: {completed}/{total_possible} ({pct:.1f}%)")
    print(f"Dead combos: {dead}")
    print(f"Total runs: {state['total_runs']}")
    print(f"Passing iterations: {state.get('total_passing_iterations', state.get('total_winners', 0))}")
    print(f"Best score: {state['best_score']:.2f}")
    print(f"Last run: {state.get('last_run', 'Never')}")

    winners = load_winners()
    if winners:
        print(f"\n🏆 Top 10 Winners:")
        for w in winners[:10]:
            print(f"  {w['ticker']:6s} | {w['strategy']:25s} | "
                  f"Sharpe {w['sharpe']:5.2f} | Return {w['return']:7.1%} | "
                  f"Trades {w['trades']:3d} | Score {w['score']:.2f}")

    if INSIGHTS_FILE.exists():
        with open(INSIGHTS_FILE) as f:
            insights = json.load(f)
        if insights.get("insights"):
            print(f"\n📊 Latest Insights:")
            for i in insights["insights"][:5]:
                print(f"  • {i}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CrabQuant Cron Task v3 (Vectorized)")
    parser.add_argument("--strategy", type=str, help="Specific strategy")
    parser.add_argument("--ticker", type=str, help="Specific ticker")
    parser.add_argument("--validate", action="store_true", help="Validate unvalidated winners")
    parser.add_argument("--analyze", action="store_true", help="Run self-improvement analysis")
    parser.add_argument("--status", action="store_true", help="Print current progress")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark vectorized vs sequential")
    parser.add_argument("--combos", type=int, default=1, help="Number of combos to run this cycle")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.analyze:
        run_analysis()
        return

    if args.benchmark:
        run_benchmark()
        return

    state = load_state()

    if args.validate:
        from crabquant.validation import walk_forward_test

        winners = load_winners()
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

    # Discovery mode — vectorized sweep
    for cycle in range(args.combos):
        combo = get_next_combo(state, args.strategy, args.ticker)
        if combo is None:
            print("No combo to run.")
            break

        strategy_name, ticker = combo
        strategy_fn, defaults, param_grid, desc, matrix_fn = STRATEGY_REGISTRY[strategy_name]

        # Skip strategies with empty param grids (use directed search instead)
        if not param_grid:
            print(f"\n🦀 {ticker}/{strategy_name} — no param grid, using directed search")
            try:
                df = load_data(ticker)
            except Exception as e:
                print(f"❌ Data load failed: {e}")
                state["completed_combos"].append(f"{strategy_name}|{ticker}")
                save_state(state)
                continue

            engine = BacktestEngine()
            result = directed_search(strategy_fn, defaults, param_grid, df, engine, strategy_name, ticker)
            # Handle result same as below...
            state["completed_combos"].append(f"{strategy_name}|{ticker}")
            save_state(state)
            continue

        print(f"\n🦀 CrabQuant Cron v3 (Vectorized) — {datetime.now().strftime('%H:%M:%S')}")
        print(f"Round {state.get('round', 0)} | {ticker}/{strategy_name} | Cycle {cycle+1}/{args.combos}")

        # Count total combos
        total_combos = 1
        for v in param_grid.values():
            total_combos *= len(v)
        print(f"Param grid: {total_combos} total combinations")
        print(f"Desc: {desc[:70]}...")

        start = time.time()

        try:
            df = load_data(ticker)
        except Exception as e:
            print(f"❌ Data load failed: {e}")
            state["completed_combos"].append(f"{strategy_name}|{ticker}")
            save_state(state)
            continue

        engine = BacktestEngine()

        try:
            results = vectorized_sweep(matrix_fn, param_grid, df, engine, strategy_name, ticker)
        except Exception as e:
            print(f"❌ Vectorized sweep failed: {e}")
            traceback.print_exc()
            results = []

        elapsed = time.time() - start

        # Update state
        state["completed_combos"].append(f"{strategy_name}|{ticker}")
        state["total_runs"] += 1
        state["last_run"] = datetime.now().isoformat()

        # Check for any trades across all combos
        any_trades = any(r.num_trades > 0 for r in results) if results else False

        if not any_trades:
            dead = set(state.get("dead_combos", []))
            dead.add(f"{strategy_name}|{ticker}")
            state["dead_combos"] = sorted(dead)
            print(f"\n💀 Zero trades across all {len(results)} combos — marked as dead combo")
            # Don't log zero-trade ghost entries
        elif results:
            # Only log results that have trades AND are interesting (passed or Sharpe > 0.5)
            logged = 0
            for r in results:
                if r.num_trades >= 1 and (r.passed or r.sharpe > 0.5):
                    save_result(r)
                    logged += 1

            passed = [r for r in results if r.passed]
            best = max(results, key=lambda r: r.score)

            if passed:
                for p in passed:
                    state["total_passing_iterations"] = state.get("total_passing_iterations", 0) + 1
                    save_winner(p)
                    if p.score > state["best_score"]:
                        state["best_score"] = p.score

                print(f"\n🏆 {len(passed)} WINNER(S)! {ticker}/{strategy_name}")
                for p in sorted(passed, key=lambda r: r.score, reverse=True)[:3]:
                    print(f"   Sharpe {p.sharpe:.2f} | Return {p.total_return:.1%} | "
                          f"Trades {p.num_trades} | Score {p.score:.2f} | Params {p.params}")
                print(f"   ({elapsed:.1f}s for {len(results)} combos)")
            else:
                print(f"\n❌ No pass — best: Sharpe {best.sharpe:.2f}, "
                      f"Score {best.score:.2f} | ({elapsed:.1f}s for {len(results)} combos)")
        else:
            dead = set(state.get("dead_combos", []))
            dead.add(f"{strategy_name}|{ticker}")
            state["dead_combos"] = sorted(dead)
            print(f"\n⏭️ Skipped — dead combo | ({elapsed:.1f}s)")

        save_state(state)

        # Run analysis every 10 combos
        if state["total_runs"] % 10 == 0:
            print(f"\n📊 Running analysis (every 10 runs)...")
            run_analysis()


if __name__ == "__main__":
    main()
