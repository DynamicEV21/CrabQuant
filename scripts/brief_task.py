#!/usr/bin/env python3
"""
CrabQuant Daily Brief — cron entry point.

Generates the daily market brief and prints to stdout.
The cron agent delivers the output to Telegram.

Usage:
    python scripts/brief_task.py

Exit codes:
    0 — success, brief printed (or NO_REPLY)
    1 — unrecoverable error
"""

import logging
import sys

# Ensure project root is on path
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

try:
    from crabquant.brief import generate_brief

    brief = generate_brief()
    print(brief)
except Exception as e:
    # Fallback: report error instead of crashing silently
    print(f"⚠️ Brief generation failed: {e}")
    sys.exit(1)
