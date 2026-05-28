"""
Production Scanner

Reads winners and confirmed results, promotes ROBUST strategies to production.
Designed to be called by cron on a regular basis.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PRODUCTION_DIR = Path(__file__).resolve().parent.parent.parent / "strategies" / "production"
REGISTRY_FILE = PRODUCTION_DIR / "registry.json"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
WINNERS_FILE = BASE_DIR / "results" / "winners" / "winners.json"
CONFIRMED_FILE = BASE_DIR / "results" / "confirmed" / "confirmed.json"


def _load_json(path: Path, default=None):
    """Load JSON file, return default if missing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default or []


def _get_promoted_keys() -> set[str]:
    """Get set of keys already in production registry."""
    try:
        if REGISTRY_FILE.exists():
            with open(REGISTRY_FILE) as f:
                registry = json.load(f)
            return {entry["key"] for entry in registry}
    except Exception:
        pass
    return set()


def _make_winner_key(winner: dict) -> str:
    """
    Create the key format that matches what confirmed.json uses.

    confirmed.json uses: "strategy|ticker|{json_params}"
    We reconstruct this from the winner dict.
    """
    import hashlib

    params = winner.get("params", {})
    serialized = json.dumps(params, sort_keys=True, separators=(",", ":"))
    params_hash = hashlib.sha256(serialized.encode()).hexdigest()[:12]
    return f"{winner['strategy']}|{winner['ticker']}|{params_hash}"


def scan_and_promote() -> list[dict]:
    """
    Scan winners and confirmed results, promote ROBUST strategies to production.

    For each ROBUST confirmed strategy not yet in the production registry,
    promote it to the production folder with a markdown report.

    Returns:
        List of newly promoted strategy info dicts with keys:
        strategy_name, ticker, key, promoted_at.
    """
    from crabquant.production.promoter import promote_strategy, _params_hash, _make_key as _promoter_make_key

    # Load confirmed ROBUST strategies
    confirmed = _load_json(CONFIRMED_FILE, [])
    robust = [c for c in confirmed if c.get("verdict") == "ROBUST"]

    if not robust:
        logger.info("No ROBUST confirmed strategies to promote.")
        return []

    # Load winners for VBT metrics
    winners = _load_json(WINNERS_FILE, [])
    winners_by_key = {}
    for w in winners:
        wkey = w.get("key", "")
        if wkey:
            winners_by_key[wkey] = w

    # Also build a lookup by (strategy, ticker, params_hash)
    winners_by_params = {}
    for w in winners:
        strategy = w.get("strategy", "")
        ticker = w.get("ticker", "")
        params = w.get("params", {})
        phash = _params_hash(params)
        winners_by_params[(strategy, ticker, phash)] = w

    # Get already-promoted keys
    promoted_keys = _get_promoted_keys()

    newly_promoted = []

    for confirmed_entry in robust:
        strategy_name = confirmed_entry.get("strategy", "")
        ticker = confirmed_entry.get("ticker", "")
        params = confirmed_entry.get("params", {})
        verdict = confirmed_entry.get("verdict", "")

        # Check if already promoted by looking up both key formats
        confirm_key = confirmed_entry.get("key", "")
        promoter_key = _promoter_make_key(strategy_name, ticker, params)

        if promoter_key in promoted_keys:
            logger.debug(f"Skipping {strategy_name}/{ticker} — already in production")
            continue

        # Find the VBT winner for this confirmed entry
        # Try the confirmed key first, then params lookup
        vbt_winner = winners_by_key.get(confirm_key)
        if not vbt_winner:
            phash = _params_hash(params)
            vbt_winner = winners_by_params.get((strategy_name, ticker, phash))

        if not vbt_winner:
            logger.warning(
                f"Skipping {strategy_name}/{ticker} — no matching VBT winner found"
            )
            continue

        # Promote
        try:
            report = promote_strategy(
                strategy_name=strategy_name,
                ticker=ticker,
                params=params,
                vbt_result=vbt_winner,
                confirm_result=confirmed_entry,
            )
            newly_promoted.append({
                "strategy_name": strategy_name,
                "ticker": ticker,
                "key": promoter_key,
                "promoted_at": report.date_promoted,
            })
            logger.info(f"Promoted {strategy_name}/{ticker} to production")
        except ValueError as e:
            logger.warning(f"Cannot promote {strategy_name}/{ticker}: {e}")
        except Exception as e:
            logger.error(f"Error promoting {strategy_name}/{ticker}: {e}")

    return newly_promoted
