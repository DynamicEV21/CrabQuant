#!/usr/bin/env python3
"""
Confirmation Task Script for Cron

Reads winners from results/winners/winners.json, runs batch confirmation
on the top unconfirmed winner, and files the result.

Usage:
    python scripts/confirm_task.py
"""

import json
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crabquant.confirm import ConfirmationResult
from crabquant.confirm.batch import batch_confirm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
WINNERS_FILE = BASE_DIR / "results" / "winners" / "winners.json"
CONFIRMED_FILE = BASE_DIR / "results" / "confirmed" / "confirmed.json"
CRON_STATE_FILE = BASE_DIR / "results" / "cron_state.json"


def load_json(path: Path, default=None):
    """Load JSON file, return default if missing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default or []


def save_json(path: Path, data):
    """Save JSON to file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_cron_state() -> dict:
    """Load cron state."""
    if CRON_STATE_FILE.exists():
        with open(CRON_STATE_FILE) as f:
            return json.load(f)
    return {}


def save_cron_state(state: dict):
    """Save cron state."""
    save_json(CRON_STATE_FILE, state)


def get_confirmed_keys() -> set:
    """Get set of already-confirmed winner keys."""
    confirmed = load_json(CONFIRMED_FILE, [])
    return {w.get("key", "") for w in confirmed if w.get("key")}


def get_failed_keys() -> set:
    """Get set of winners that already failed confirmation."""
    failed_file = BASE_DIR / "results" / "confirmed" / "failed.json"
    failed = load_json(failed_file, [])
    return {w.get("key", "") for w in failed if w.get("key")}


def main():
    logger.info("=== Confirmation Task Start ===")

    # Load winners
    if not WINNERS_FILE.exists():
        logger.info("No winners file found. Nothing to confirm.")
        return

    winners = load_json(WINNERS_FILE, [])
    if not winners:
        logger.info("Winners file is empty. Nothing to confirm.")
        return

    # Get already-processed keys
    confirmed_keys = get_confirmed_keys()
    failed_keys = get_failed_keys()
    processed = confirmed_keys | failed_keys

    # Find top unconfirmed winner (sorted by score descending)
    unconfirmed = [w for w in winners if w.get("key", "") not in processed]
    unconfirmed.sort(key=lambda w: w.get("score", 0), reverse=True)

    if not unconfirmed:
        logger.info("All winners have been confirmed or failed. Nothing to do.")
        return

    # Pick the top one
    winner = unconfirmed[0]
    key = winner["key"]
    strategy = winner["strategy"]
    ticker = winner["ticker"]
    score = winner.get("score", 0)

    logger.info(f"Confirming: {strategy}/{ticker} (score={score:.4f})")
    logger.info(f"  Params: {json.dumps(winner['params'])}")

    # Run batch confirmation
    try:
        result = batch_confirm(winner, n_periods=3)
    except Exception as e:
        logger.error(f"Confirmation failed with error: {e}")
        save_cron_state({
            **load_cron_state(),
            "last_confirm_error": str(e),
            "last_confirm_attempt": datetime.now().isoformat(),
        })
        return

    logger.info(f"  Verdict: {result.verdict}")
    logger.info(f"  Sharpe: {result.sharpe:.2f}, Return: {result.total_return:.2%}, "
                f"MaxDD: {result.max_dd:.2%}, Trades: {result.trades}")
    for note in result.notes:
        logger.info(f"    {note}")

    # Build result record
    record = {
        "key": key,
        "strategy": strategy,
        "ticker": ticker,
        "vbt_score": score,
        "vbt_sharpe": winner.get("sharpe", 0),
        "vbt_return": winner.get("return", 0),
        "confirm_sharpe": result.sharpe,
        "confirm_return": result.total_return,
        "confirm_max_dd": result.max_dd,
        "confirm_trades": result.trades,
        "confirm_win_rate": result.win_rate,
        "confirm_profit_factor": result.profit_factor,
        "confirm_expectancy": result.expectancy,
        "verdict": result.verdict,
        "confirmed_at": datetime.now().isoformat(),
        "params": winner["params"],
    }

    # File result
    if result.verdict == "ROBUST":
        confirmed = load_json(CONFIRMED_FILE, [])
        confirmed.append(record)
        save_json(CONFIRMED_FILE, confirmed)
        logger.info(f"✓ ROBUST — saved to confirmed.json")
    elif result.verdict == "FRAGILE":
        # Save to separate file for review
        fragile_file = BASE_DIR / "results" / "confirmed" / "fragile.json"
        fragile = load_json(fragile_file, [])
        fragile.append(record)
        save_json(fragile_file, fragile)
        logger.info(f"⚠ FRAGILE — saved to fragile.json for review")
    else:
        failed_file = BASE_DIR / "results" / "confirmed" / "failed.json"
        failed = load_json(failed_file, [])
        failed.append(record)
        save_json(failed_file, failed)
        logger.info(f"✗ FAILED — saved to failed.json")

    # Update cron state
    state = load_cron_state()
    state["last_confirm_key"] = key
    state["last_confirm_verdict"] = result.verdict
    state["last_confirm_time"] = datetime.now().isoformat()
    state["confirmed_count"] = len(get_confirmed_keys())
    save_cron_state(state)

    logger.info("=== Confirmation Task Complete ===")


if __name__ == "__main__":
    main()
