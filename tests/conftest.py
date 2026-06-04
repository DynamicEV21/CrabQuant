"""Pytest configuration — add scripts/ to sys.path so crabquant_cron is importable."""

import sys
import warnings
from pathlib import Path

# Suppress known non-actionable warnings
warnings.filterwarnings("ignore", message="This process.*is multi-threaded", category=DeprecationWarning)

# Ensure scripts/ is importable for cron_integration tests
scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
