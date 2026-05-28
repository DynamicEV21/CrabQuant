"""Per-wave metrics tracking for refinement convergence analysis.

Tracks convergence rates, Sharpe ratios, and archetype-level performance
across optimization waves to identify the most promising strategy patterns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def compute_convergence_rate(successful: int, total: int) -> float:
    """Compute convergence rate as successful/total, returning 0.0 for zero denominator."""
    if total == 0:
        return 0.0
    return successful / total


@dataclass
class WaveMetrics:
    """Metrics for a single optimization wave."""

    wave_number: int
    total_mandates: int
    successful: int = 0
    convergence_rate: float = field(init=False)
    avg_sharpe: float = field(init=False)
    sharpe_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.convergence_rate = compute_convergence_rate(self.successful, self.total_mandates)
        if self.sharpe_values:
            self.avg_sharpe = sum(self.sharpe_values) / len(self.sharpe_values)
        else:
            self.avg_sharpe = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_number": self.wave_number,
            "total_mandates": self.total_mandates,
            "successful": self.successful,
            "convergence_rate": self.convergence_rate,
            "avg_sharpe": self.avg_sharpe,
            "sharpe_values": self.sharpe_values,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaveMetrics:
        m = cls(
            wave_number=data["wave_number"],
            total_mandates=data["total_mandates"],
            successful=data.get("successful", 0),
            sharpe_values=data.get("sharpe_values", []),
        )
        return m


@dataclass
class MandateArchetypeStats:
    """Aggregated statistics for a single mandate archetype."""

    archetype: str
    total: int = 0
    successful: int = 0
    _sharpe_values: list[float] = field(default_factory=list, repr=False)
    avg_sharpe: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._recalc()

    def _recalc(self) -> None:
        if self._sharpe_values:
            self.avg_sharpe = sum(self._sharpe_values) / len(self._sharpe_values)
        else:
            self.avg_sharpe = 0.0

    def record(self, sharpe: float, success: bool) -> None:
        """Record a single mandate result for this archetype."""
        self.total += 1
        if success:
            self.successful += 1
            self._sharpe_values.append(sharpe)
        self._recalc()

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype": self.archetype,
            "total": self.total,
            "successful": self.successful,
            "convergence_rate": compute_convergence_rate(self.successful, self.total),
            "avg_sharpe": self.avg_sharpe,
        }


def identify_best_archetypes(
    stats_dict: dict[str, MandateArchetypeStats],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return the top N archetypes sorted by convergence rate (desc), then avg_sharpe (desc)."""
    if not stats_dict:
        return []
    entries = [s.to_dict() for s in stats_dict.values()]
    entries.sort(key=lambda d: (d["convergence_rate"], d["avg_sharpe"]), reverse=True)
    return entries[:top_n]


class PerWaveMetricsTracker:
    """Tracks metrics across multiple optimization waves."""

    def __init__(self) -> None:
        self.wave_metrics: dict[int, WaveMetrics] = {}
        self.archetype_stats: dict[str, MandateArchetypeStats] = {}

    def record_wave(self, report: dict[str, Any]) -> None:
        """Record metrics from a single wave report."""
        wave_num = report["wave_number"]
        total = report["total_mandates"]
        successful = report["successful"]

        sharpe_values: list[float] = []
        for result in report.get("results", []):
            if result.get("status") == "success":
                sharpe_values.append(result.get("sharpe", 0.0))

            # Track archetype stats
            archetype = result.get("archetype")
            if archetype:
                if archetype not in self.archetype_stats:
                    self.archetype_stats[archetype] = MandateArchetypeStats(archetype=archetype)
                self.archetype_stats[archetype].record(
                    sharpe=result.get("sharpe", 0.0),
                    success=result.get("status") == "success",
                )

        self.wave_metrics[wave_num] = WaveMetrics(
            wave_number=wave_num,
            total_mandates=total,
            successful=successful,
            sharpe_values=sharpe_values,
        )

    def get_summary(self) -> dict[str, Any]:
        """Return a summary dict of all tracked metrics."""
        return {
            "total_waves": len(self.wave_metrics),
            "wave_metrics": {
                k: v.to_dict() for k, v in self.wave_metrics.items()
            },
            "archetype_stats": {
                k: v.to_dict() for k, v in self.archetype_stats.items()
            },
        }

    def save(self, path: str) -> None:
        """Save metrics to a JSON file, creating directories as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.get_summary(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> PerWaveMetricsTracker:
        """Load metrics from a JSON file. Returns empty tracker if file doesn't exist."""
        p = Path(path)
        if not p.exists():
            return cls()

        data = json.loads(p.read_text(encoding="utf-8"))
        tracker = cls()

        for wave_num_str, wave_data in data.get("wave_metrics", {}).items():
            tracker.wave_metrics[int(wave_num_str)] = WaveMetrics.from_dict(wave_data)

        for arch_name, arch_data in data.get("archetype_stats", {}).items():
            stats = MandateArchetypeStats(archetype=arch_data["archetype"])
            # Replay records to reconstruct internal state
            for _ in range(arch_data.get("total", 0)):
                stats.record(0.0, success=False)
            # Correct: overwrite the totals directly since we can't replay exact records
            stats.total = arch_data.get("total", 0)
            stats.successful = arch_data.get("successful", 0)
            stats.avg_sharpe = arch_data.get("avg_sharpe", 0.0)
            tracker.archetype_stats[arch_name] = stats

        return tracker
