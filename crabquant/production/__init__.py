"""
CrabQuant Production Strategy Registry

When a strategy passes BOTH VectorBT discovery AND backtesting.py confirmation,
it can be promoted to the production registry. This provides one place to look
for "strategies we trust" vs the research folder of everything we've tested.
"""

from pathlib import Path

from crabquant.production.promoter import promote_strategy
from crabquant.production.report import StrategyReport

PRODUCTION_DIR = Path(__file__).resolve().parent.parent.parent / "strategies" / "production"
REGISTRY_FILE = PRODUCTION_DIR / "registry.json"


def get_production_strategies() -> list[dict]:
    """
    List all promoted production strategies.

    Returns:
        List of dicts with keys: key, strategy_name, ticker, params_hash,
        promoted_at, report (StrategyReport instance).
    """
    if not REGISTRY_FILE.exists():
        return []

    import json

    with open(REGISTRY_FILE) as f:
        registry = json.load(f)

    result = []
    for entry in registry:
        report_path = PRODUCTION_DIR / entry.get("report_file", "")
        report = _load_report(report_path) if report_path.exists() else None
        result.append({
            "key": entry["key"],
            "strategy_name": entry["strategy_name"],
            "ticker": entry["ticker"],
            "params_hash": entry.get("params_hash", ""),
            "promoted_at": entry.get("promoted_at", ""),
            "verdict": entry.get("verdict", ""),
            "report": report,
        })

    return result


def get_production_report(strategy_key: str) -> StrategyReport | None:
    """
    Get a production strategy's full report by key.

    Args:
        strategy_key: The strategy key (e.g. "roc_ema_volume|CAT|<params_hash>")

    Returns:
        StrategyReport instance or None if not found.
    """
    if not REGISTRY_FILE.exists():
        return None

    import json

    with open(REGISTRY_FILE) as f:
        registry = json.load(f)

    for entry in registry:
        if entry["key"] == strategy_key:
            report_path = PRODUCTION_DIR / entry.get("report_file", "")
            if report_path.exists():
                return _load_report(report_path)
            return None

    return None


def _load_report(path: Path) -> StrategyReport | None:
    """Load a StrategyReport from a markdown file's embedded metadata."""
    import json
    import re

    text = path.read_text()

    # Look for JSON metadata block at the end of the file
    match = re.search(r"<!-- METADATA\n(.+?)\n-->", text, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(1))
        return StrategyReport.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return None


__all__ = [
    "promote_strategy",
    "get_production_strategies",
    "get_production_report",
    "PRODUCTION_DIR",
    "REGISTRY_FILE",
    "StrategyReport",
]
