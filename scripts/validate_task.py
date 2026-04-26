#!/usr/bin/env python3
"""
CrabQuant Validation Task

Validates winning strategies using walk-forward and cross-ticker tests.
Designed to be run by the crabquant-validate agent.

Usage:
    python scripts/validate_task.py                  # Validate up to 5 pending winners
    python scripts/validate_task.py --all            # Validate all pending
    python scripts/validate_task.py --status         # Show validation status
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib

from crabquant.data import load_data
from crabquant.engine import BacktestEngine
from crabquant.strategies import STRATEGY_REGISTRY, DEFAULT_TICKERS
from crabquant.validation import walk_forward_test, cross_ticker_validation as cross_ticker_test

RESULTS_DIR = Path(__file__).parent.parent / "results"
STATE_FILE = RESULTS_DIR / "cron_state.json"
WINNERS_FILE = RESULTS_DIR / "winners" / "winners.json"
VALIDATED_FILE = RESULTS_DIR / "winners" / "validated_winners.json"
CURVEFIT_FILE = RESULTS_DIR / "winners" / "curvefit_winners.json"
LOGS_FILE = RESULTS_DIR / "logs" / "validation_results.jsonl"


def load_state():
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
    }


def load_winners():
    if WINNERS_FILE.exists():
        with open(WINNERS_FILE) as f:
            return json.load(f)
    return []


def load_validated():
    if VALIDATED_FILE.exists():
        with open(VALIDATED_FILE) as f:
            return json.load(f)
    return []


def load_curvefit():
    if CURVEFIT_FILE.exists():
        with open(CURVEFIT_FILE) as f:
            return json.load(f)
    return []


def save_validated(winners):
    VALIDATED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATED_FILE, "w") as f:
        json.dump(winners, f, indent=2)


def save_curvefit(winners):
    CURVEFIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CURVEFIT_FILE, "w") as f:
        json.dump(winners, f, indent=2)


def log_result(entry: dict):
    LOGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


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


def validate_winner(winner: dict) -> dict:
    """Run full validation on a winner."""
    strategy_name = winner["strategy"]
    ticker = winner["ticker"]
    params = winner["params"]

    if strategy_name not in STRATEGY_REGISTRY:
        return {"status": "error", "reason": f"Strategy {strategy_name} not in registry"}

    strategy_fn = STRATEGY_REGISTRY[strategy_name][0]

    # Detect current regime at validation time
    current_regime = _detect_current_regime()

    result = {
        "ticker": ticker,
        "strategy": strategy_name,
        "params": params,
        "original_sharpe": winner.get("sharpe", 0),
        "original_return": winner.get("return", 0),
        "original_score": winner.get("score", 0),
        "discovery_regime": winner.get("regime", "unknown"),
        "validation_regime": current_regime,
        "timestamp": datetime.now().isoformat(),
    }

    # Walk-forward test
    print(f"  📊 Walk-forward test...")
    try:
        wf = walk_forward_test(strategy_fn, ticker, params)
        result["wf_train_sharpe"] = wf.train_sharpe
        result["wf_train_return"] = wf.train_return
        result["wf_test_sharpe"] = wf.test_sharpe
        result["wf_test_return"] = wf.test_return
        result["wf_degradation"] = wf.degradation
        result["wf_robust"] = wf.robust
        result["wf_train_regime"] = wf.train_regime
        result["wf_test_regime"] = wf.test_regime
        result["wf_regime_shift"] = wf.regime_shift
        print(f"    Train: Sharpe {wf.train_sharpe:.2f}, Return {wf.train_return:.1%}")
        print(f"    Test:  Sharpe {wf.test_sharpe:.2f}, Return {wf.test_return:.1%}")
        print(f"    Degradation: {wf.degradation:.1%} {'✅' if wf.robust else '❌'}")
        if wf.regime_shift:
            print(f"    ⚠️ Regime shift: {wf.train_regime} → {wf.test_regime}")
    except Exception as e:
        result["wf_error"] = str(e)
        result["wf_robust"] = False
        print(f"    ❌ Walk-forward failed: {e}")

    # Cross-ticker test (exclude discovery ticker from validation set)
    print(f"  📊 Cross-ticker test...")
    try:
        validation_tickers = [t for t in DEFAULT_TICKERS if t != ticker]
        engine = BacktestEngine()
        ct = cross_ticker_test(strategy_fn, params, validation_tickers, engine=engine)
        result["ct_tickers_tested"] = ct.tickers_tested
        result["ct_tickers_profitable"] = ct.tickers_profitable
        result["ct_avg_sharpe"] = ct.avg_sharpe
        result["ct_generalizes"] = ct.robust
        print(f"    Tested: {ct.tickers_tested}, Profitable: {ct.tickers_profitable}")
        print(f"    Avg Sharpe: {ct.avg_sharpe:.2f} {'✅' if ct.robust else '❌'}")
    except Exception as e:
        result["ct_error"] = str(e)
        result["ct_generalizes"] = False
        print(f"    ❌ Cross-ticker failed: {e}")

    # Verdict — regime-aware logic
    wf_pass = result.get("wf_robust", False)
    ct_pass = result.get("ct_generalizes", False)
    regime_shift = result.get("wf_regime_shift", False)

    if wf_pass and ct_pass:
        result["verdict"] = "ROBUST"
        result["status"] = "passed"
    elif wf_pass or ct_pass:
        result["verdict"] = "MIXED"
        result["status"] = "partial"
    elif regime_shift:
        # Walk-forward failed but regime changed — mark as regime shift, not curve-fit
        result["verdict"] = "REGIME_SHIFT"
        result["status"] = "partial"
    else:
        result["verdict"] = "CURVE_FIT"
        result["status"] = "failed"

    print(f"  🏷️  Verdict: {result['verdict']}")

    # Log
    log_result(result)

    return result


def print_status():
    winners = load_winners()
    validated = load_validated()
    curvefit = load_curvefit()
    state = load_state()

    print(f"\n🦀 CrabQuant Validation Status")
    print("=" * 50)
    print(f"Total winners:     {len(winners)}")
    print(f"Validated (robust): {len(validated)}")
    print(f"Curve-fit:         {len(curvefit)}")
    print(f"Pending:           {len(winners) - len(validated) - len(curvefit)}")
    print()

    if validated:
        print("✅ Robust strategies:")
        for v in validated:
            print(f"   {v['ticker']}/{v['strategy']}: "
                  f"WF={v.get('wf_test_sharpe', 0):.2f}, "
                  f"CT={v.get('ct_tickers_profitable', 0)}/{v.get('ct_tickers_tested', 0)}")

    if curvefit:
        print("\n❌ Curve-fit strategies:")
        for c in curvefit:
            print(f"   {c['ticker']}/{c['strategy']}: "
                  f"Degradation={c.get('wf_degradation', 0):.1%}")


def main():
    args = sys.argv[1:]
    do_all = "--all" in args
    do_status = "--status" in args

    if do_status:
        print_status()
        return

    winners = load_winners()
    validated = load_validated()
    curvefit = load_curvefit()

    # Get validated keys (include params hash for proper dedup)
    def _dedup_key(w: dict) -> str:
        params_hash = hashlib.sha256(
            json.dumps(w.get('params', {}), sort_keys=True).encode()
        ).hexdigest()[:12]
        return f"{w['ticker']}|{w['strategy']}|{params_hash}"

    validated_keys = {_dedup_key(w) for w in validated}
    curvefit_keys = {_dedup_key(w) for w in curvefit}

    # Find unvalidated
    pending = [
        w for w in winners
        if _dedup_key(w) not in validated_keys
        and _dedup_key(w) not in curvefit_keys
    ]

    if not pending:
        print("No winners to validate.")
        print_status()
        return

    print(f"🦀 CrabQuant Validation — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Pending: {len(pending)} winners to validate")
    print()

    BATCH_SIZE = 5
    to_validate = pending if do_all else pending[:BATCH_SIZE]

    for w in to_validate:
        print(f"\n🔍 Validating {w['ticker']}/{w['strategy']}...")
        result = validate_winner(w)

        if result["status"] == "passed":
            validated.append(result)
            save_validated(validated)
            print(f"\n✅ PROMOTED — {w['ticker']}/{w['strategy']} is robust")
        elif result["status"] == "failed":
            curvefit.append(result)
            save_curvefit(curvefit)
            print(f"\n❌ REJECTED — {w['ticker']}/{w['strategy']} is curve-fit")
        else:
            print(f"\n⚠️  MIXED — {w['ticker']}/{w['strategy']} needs more data")

    # Update cron state
    state = load_state()
    for w in to_validate:
        key = _dedup_key(w)
        if key not in state["validated_winners"]:
            state["validated_winners"].append(key)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    print_status()


if __name__ == "__main__":
    main()
