#!/usr/bin/env python3
"""
Promote Task Script for Cron

Scans confirmed ROBUST strategies and promotes them to the production registry.
Minimal output for cron agent consumption.

Usage:
    python scripts/promote_task.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from crabquant.production.scanner import scan_and_promote

    newly_promoted = scan_and_promote()

    if not newly_promoted:
        # Minimal output — saves tokens when nothing happened
        print("NO_REPLY")
        return

    # Compact summary
    names = [f"{s['ticker']}/{s['strategy_name']}" for s in newly_promoted]
    print(f"Promoted {len(newly_promoted)} strategies: {', '.join(names)}")


if __name__ == "__main__":
    main()
