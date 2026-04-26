#!/usr/bin/env python3
"""
CrabQuant Meta-Learner Task

Analyzes all system data and makes optimization decisions:
- Expands param grids of winning strategies
- Retires consistently losing strategies
- Prioritizes productive tickers
- Generates meta_report.json with actionable insights

Usage:
    python scripts/meta_task.py              # Run full meta-analysis + implement changes
    python scripts/meta_task.py --report     # Only generate report, don't change files
    python scripts/meta_task.py --status     # Show current system overview
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
STRATEGIES_DIR = RESULTS_DIR.parent / "crabquant" / "strategies"
STATE_FILE = RESULTS_DIR / "cron_state.json"
WINNERS_FILE = RESULTS_DIR / "winners" / "winners.json"
VALIDATED_FILE = RESULTS_DIR / "winners" / "validated_winners.json"
INSIGHTS_FILE = RESULTS_DIR / "insights.json"
META_REPORT_FILE = RESULTS_DIR / "meta_report.json"


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_report():
    """Generate a comprehensive meta-analysis report."""
    state = load_json(STATE_FILE)
    winners = load_json(WINNERS_FILE)
    validated = load_json(VALIDATED_FILE)
    insights = load_json(INSIGHTS_FILE)

    # System overview
    total_combos = len(state.get("completed_combos", []))
    dead_combos = len(state.get("dead_combos", []))
    total_winners = state.get("total_winners", 0)

    # Strategy analysis
    strategy_results = {}
    for w in winners:
        s = w.get("strategy", "unknown")
        if s not in strategy_results:
            strategy_results[s] = {"wins": 0, "best_sharpe": 0, "best_score": 0, "tickers": set()}
        strategy_results[s]["wins"] += 1
        strategy_results[s]["best_sharpe"] = max(strategy_results[s]["best_sharpe"], w.get("sharpe", 0))
        strategy_results[s]["best_score"] = max(strategy_results[s]["best_score"], w.get("score", 0))
        strategy_results[s]["tickers"].add(w.get("ticker", ""))

    # Convert sets to lists for JSON
    for s in strategy_results.values():
        s["tickers"] = sorted(s["tickers"])

    # Ticker analysis
    ticker_results = {}
    for w in winners:
        t = w.get("ticker", "unknown")
        if t not in ticker_results:
            ticker_results[t] = {"wins": 0, "strategies": set()}
        ticker_results[t]["wins"] += 1
        ticker_results[t]["strategies"].add(w.get("strategy", ""))

    for t in ticker_results.values():
        t["strategies"] = sorted(t["strategies"])

    # Insights-based analysis
    strategy_stats = insights.get("strategy_stats", {})
    ticker_stats = insights.get("ticker_stats", {})

    # Compute win rates
    for sname, stats in strategy_stats.items():
        stats["win_rate"] = stats.get("won", 0) / max(stats.get("tested", 1), 1)

    # Recommendations
    recommendations = []

    # 1. Strategies with high win rate → expand grid
    for sname, stats in sorted(strategy_stats.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True):
        if stats.get("win_rate", 0) > 0.15 and stats.get("tested", 0) >= 3:
            recommendations.append({
                "action": "expand_grid",
                "strategy": sname,
                "reason": f"Win rate {stats['win_rate']:.0%} ({stats.get('won', 0)}/{stats.get('tested', 0)})",
                "priority": "high" if stats.get("win_rate", 0) > 0.3 else "medium",
            })

    # 2. Strategies with 0% win rate after many tests → consider retiring
    for sname, stats in strategy_stats.items():
        if stats.get("tested", 0) >= 5 and stats.get("win_rate", 0) == 0:
            recommendations.append({
                "action": "retire",
                "strategy": sname,
                "reason": f"0% win rate after {stats['tested']} tests",
                "priority": "low",
            })

    # 3. Tickers that never win → deprioritize
    for tname, stats in ticker_stats.items():
        if stats.get("tested", 0) >= 3 and stats.get("win_rate", 0) == 0:
            recommendations.append({
                "action": "deprioritize_ticker",
                "ticker": tname,
                "reason": f"0% win rate after {stats['tested']} tests",
                "priority": "low",
            })

    # 4. Hot tickers → prioritize
    for tname, stats in sorted(ticker_stats.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True):
        if stats.get("win_rate", 0) > 0.2 and stats.get("tested", 0) >= 2:
            recommendations.append({
                "action": "prioritize_ticker",
                "ticker": tname,
                "reason": f"Win rate {stats['win_rate']:.0%} ({stats.get('won', 0)}/{stats.get('tested', 0)})",
                "priority": "medium",
            })

    # 5. Invented strategies performance
    invented = [s for s in strategy_results if s.startswith("invented_")]
    original = [s for s in strategy_results if not s.startswith("invented_")]
    invented_wins = sum(strategy_results[s]["wins"] for s in invented)
    original_wins = sum(strategy_results[s]["wins"] for s in original)

    if invented:
        recommendations.append({
            "action": "note",
            "topic": "invented_vs_original",
            "reason": f"Invented strategies: {invented_wins} wins vs Original: {original_wins} wins",
            "priority": "info",
        })

    report = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "total_combos_tested": total_combos,
            "dead_combos": dead_combos,
            "total_winners": total_winners,
            "validated_robust": len(validated),
            "coverage_pct": round(total_combos / max(total_combos + dead_combos, 1) * 100, 1),
        },
        "strategy_analysis": strategy_results,
        "ticker_analysis": ticker_results,
        "strategy_stats": strategy_stats,
        "ticker_stats": ticker_stats,
        "recommendations": recommendations,
        "invented_performance": {
            "invented_strategies": len(invented),
            "invented_wins": invented_wins,
            "original_wins": original_wins,
        },
    }

    return report


def print_status():
    report = generate_report()
    sys = report["system"]

    print(f"\n🦀 CrabQuant System Overview")
    print("=" * 50)
    print(f"Combos tested:    {sys['total_combos_tested']}")
    print(f"Dead combos:      {sys['dead_combos']}")
    print(f"Total winners:    {sys['total_winners']}")
    print(f"Validated (robust): {sys['validated_robust']}")
    print(f"Coverage:         {sys['coverage_pct']:.1f}%")
    print()

    if report["strategy_analysis"]:
        print("Strategy wins:")
        for s, data in sorted(report["strategy_analysis"].items(), key=lambda x: x[1]["wins"], reverse=True):
            print(f"  {s}: {data['wins']} wins, best Sharpe {data['best_sharpe']:.2f}")

    if report["recommendations"]:
        print("\nRecommendations:")
        for r in report["recommendations"]:
            icon = {"high": "🔴", "medium": "🟡", "low": "⚪", "info": "ℹ️"}.get(r["priority"], "⚪")
            print(f"  {icon} [{r['action']}] {r.get('strategy', r.get('ticker', r.get('topic', '')))}: {r['reason']}")


def main():
    args = sys.argv[1:]
    do_report = "--report" in args
    do_status = "--status" in args

    if do_status:
        print_status()
        return

    print(f"🦀 CrabQuant Meta-Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    report = generate_report()

    if not do_report:
        print("Report generated. Recommendations:")
        for r in report["recommendations"]:
            icon = {"high": "🔴", "medium": "🟡", "low": "⚪", "info": "ℹ️"}.get(r["priority"], "⚪")
            print(f"  {icon} [{r['action']}] {r.get('strategy', r.get('ticker', r.get('topic', '')))}: {r['reason']}")

    save_json(META_REPORT_FILE, report)
    print(f"\n📄 Report saved to {META_REPORT_FILE}")

    if do_report:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
