"""Pytest configuration — add scripts/ to sys.path so crabquant_cron is importable."""

import sys
from pathlib import Path

# Ensure scripts/ is importable for cron_integration tests
scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
