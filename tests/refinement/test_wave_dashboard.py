"""Tests for crabquant.refinement.wave_dashboard — Phase 3."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from crabquant.refinement.wave_dashboard import (
    DashboardSnapshot,
    collect_running_mandates,
    compute_convergence_rate,
    compute_wave_progress,
    find_best_strategies,
    generate_dashboard,
    snapshot_to_json,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_run_state(
    name: str = "test_mandate",
    status: str = "running",
    best_sharpe: float = 1.5,
    current_turn: int = 3,
    max_turns: int = 7,
    created_at: str | None = None,
) -> dict:
    return {
        "run_id": f"run_{name}",
        "mandate_name": name,
        "status": status,
        "best_sharpe": best_sharpe,
        "current_turn": current_turn,
        "max_turns": max_turns,
        "sharpe_target": 1.5,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "tickers": ["SPY"],
        "period": "2y",
        "history": [],
    }


def make_runs_dir(states: list[dict]) -> str:
    """Create a temp directory with run state.json files."""
    tmpdir = tempfile.mkdtemp()
    for state in states:
        name = state["mandate_name"]
        run_dir = Path(tmpdir) / f"run_{name}"
        run_dir.mkdir()
        run_dir.joinpath("state.json").write_text(json.dumps(state))
    return tmpdir


# ── collect_running_mandates ──────────────────────────────────────────────────

class TestCollectRunningMandates:

    def test_finds_running_states(self):
        states = [
            make_run_state("m1", status="running"),
            make_run_state("m2", status="success"),
            make_run_state("m3", status="running"),
        ]
        tmpdir = make_runs_dir(states)
        running = collect_running_mandates(tmpdir)
        assert len(running) == 2
        names = {r["mandate_name"] for r in running}
        assert "m1" in names
        assert "m3" in names

    def test_returns_empty_for_no_running(self):
        states = [
            make_run_state("m1", status="success"),
            make_run_state("m2", status="max_turns_exhausted"),
        ]
        tmpdir = make_runs_dir(states)
        running = collect_running_mandates(tmpdir)
        assert running == []

    def test_returns_empty_for_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            running = collect_running_mandates(tmpdir)
            assert running == []

    def test_returns_empty_for_nonexistent_dir(self):
        running = collect_running_mandates("/nonexistent/path")
        assert running == []


# ── compute_convergence_rate ──────────────────────────────────────────────────

class TestComputeConvergenceRate:

    def test_all_success(self):
        states = [
            make_run_state("m1", status="success"),
            make_run_state("m2", status="success"),
            make_run_state("m3", status="success"),
        ]
        rate = compute_convergence_rate(states)
        assert rate == 1.0

    def test_none_success(self):
        states = [
            make_run_state("m1", status="failed"),
            make_run_state("m2", status="max_turns_exhausted"),
        ]
        rate = compute_convergence_rate(states)
        assert rate == 0.0

    def test_partial_success(self):
        states = [
            make_run_state("m1", status="success"),
            make_run_state("m2", status="failed"),
            make_run_state("m3", status="success"),
            make_run_state("m4", status="running"),
        ]
        rate = compute_convergence_rate(states)
        # 2 success out of 3 completed (running excluded)
        assert 0.5 <= rate <= 0.7

    def test_empty_states(self):
        rate = compute_convergence_rate([])
        assert rate == 0.0


# ── compute_wave_progress ─────────────────────────────────────────────────────

class TestComputeWaveProgress:

    def test_progress_fraction(self):
        states = [
            make_run_state("m1", status="success", current_turn=5, max_turns=7),
            make_run_state("m2", status="running", current_turn=3, max_turns=7),
        ]
        progress = compute_wave_progress(states)
        # m1 is done (100%), m2 is 3/7 (~43%), avg = ~71%
        assert 0.6 <= progress <= 0.8

    def test_all_complete(self):
        states = [
            make_run_state("m1", status="success"),
            make_run_state("m2", status="failed"),
        ]
        progress = compute_wave_progress(states)
        assert progress == 1.0

    def test_empty_states(self):
        progress = compute_wave_progress([])
        assert progress == 0.0


# ── find_best_strategies ──────────────────────────────────────────────────────

class TestFindBestStrategies:

    def test_sorts_by_sharpe(self):
        states = [
            make_run_state("m1", best_sharpe=1.2),
            make_run_state("m2", best_sharpe=2.5),
            make_run_state("m3", best_sharpe=0.8),
        ]
        best = find_best_strategies(states, top_n=2)
        assert len(best) == 2
        assert best[0]["mandate_name"] == "m2"
        assert best[1]["mandate_name"] == "m1"

    def test_respects_top_n(self):
        states = [make_run_state(f"m{i}", best_sharpe=float(i)) for i in range(5)]
        best = find_best_strategies(states, top_n=3)
        assert len(best) == 3

    def test_empty_states(self):
        best = find_best_strategies([])
        assert best == []

    def test_default_top_n(self):
        states = [make_run_state(f"m{i}", best_sharpe=float(i)) for i in range(10)]
        best = find_best_strategies(states)
        assert len(best) == 5  # default


# ── generate_dashboard ────────────────────────────────────────────────────────

class TestGenerateDashboard:

    def test_returns_snapshot(self):
        states = [
            make_run_state("m1", status="running", best_sharpe=1.2, current_turn=3, max_turns=7),
            make_run_state("m2", status="success", best_sharpe=2.1),
        ]
        tmpdir = make_runs_dir(states)
        snapshot = generate_dashboard(tmpdir)
        assert isinstance(snapshot, DashboardSnapshot)
        assert snapshot.running_count == 1
        assert snapshot.total_mandates == 2

    def test_empty_runs_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = generate_dashboard(tmpdir)
            assert snapshot.running_count == 0
            assert snapshot.total_mandates == 0

    def test_includes_best_strategies(self):
        states = [
            make_run_state("m1", best_sharpe=2.5, status="success"),
            make_run_state("m2", best_sharpe=1.0, status="running"),
        ]
        tmpdir = make_runs_dir(states)
        snapshot = generate_dashboard(tmpdir)
        assert len(snapshot.best_strategies) > 0
        assert snapshot.best_strategies[0]["mandate_name"] == "m1"


# ── snapshot_to_json ──────────────────────────────────────────────────────────

class TestSnapshotToJson:

    def test_returns_valid_json(self):
        snapshot = DashboardSnapshot(
            running_count=3,
            total_mandates=10,
            convergence_rate=0.6,
            wave_progress=0.75,
            best_strategies=[{"mandate_name": "m1", "best_sharpe": 2.5}],
            running_mandates=[{"mandate_name": "m2"}],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        json_str = snapshot_to_json(snapshot)
        parsed = json.loads(json_str)
        assert parsed["running_count"] == 3
        assert parsed["total_mandates"] == 10
        assert parsed["convergence_rate"] == 0.6
        assert "best_strategies" in parsed
        assert "running_mandates" in parsed
        assert "timestamp" in parsed

    def test_empty_snapshot(self):
        snapshot = DashboardSnapshot()
        json_str = snapshot_to_json(snapshot)
        parsed = json.loads(json_str)
        assert parsed["running_count"] == 0
        assert parsed["total_mandates"] == 0
