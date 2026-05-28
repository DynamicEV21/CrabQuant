"""Tests for crabquant.refinement.state — DaemonState persistence."""

import json
import os
import time

import pytest

from crabquant.refinement.state import DaemonState


@pytest.fixture
def state_path(tmp_path):
    return str(tmp_path / "daemon_state.json")


# ── create ─────────────────────────────────────────────────────────────────


class TestCreate:
    def test_create_fresh_state(self):
        s = DaemonState.create()
        assert s.daemon_id  # non-empty UUID
        assert s.started_at  # non-empty ISO string
        assert s.last_heartbeat
        assert s.last_wave_completed == ""
        assert s.current_wave == 0
        assert s.total_mandates_run == 0
        assert s.total_strategies_promoted == 0
        assert s.total_api_calls == 0
        assert s.pending_mandates == []
        assert s.completed_mandates == []
        assert s.failed_mandates == []
        assert s.last_error is None
        assert s.shutdown_requested is False


# ── save / load ────────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, state_path):
        s = DaemonState.create()
        s.current_wave = 3
        s.total_mandates_run = 7
        s.total_strategies_promoted = 2
        s.pending_mandates = ["a.json", "b.json"]
        s.completed_mandates = ["c.json"]
        s.failed_mandates = ["d.json"]
        s.last_error = "oops"
        s.shutdown_requested = True

        s.save(state_path)
        loaded = DaemonState.load(state_path)

        assert loaded is not None
        assert loaded.daemon_id == s.daemon_id
        assert loaded.started_at == s.started_at
        assert loaded.last_heartbeat == s.last_heartbeat
        assert loaded.last_wave_completed == s.last_wave_completed
        assert loaded.current_wave == 3
        assert loaded.total_mandates_run == 7
        assert loaded.total_strategies_promoted == 2
        assert loaded.total_api_calls == 0
        assert loaded.pending_mandates == ["a.json", "b.json"]
        assert loaded.completed_mandates == ["c.json"]
        assert loaded.failed_mandates == ["d.json"]
        assert loaded.last_error == "oops"
        assert loaded.shutdown_requested is True

    def test_atomic_write(self, state_path):
        s = DaemonState.create()
        s.save(state_path)

        # File should exist at final path
        assert os.path.exists(state_path)
        # No .tmp leftover
        assert not os.path.exists(state_path + ".tmp")

    def test_load_missing_file(self, state_path):
        assert DaemonState.load(state_path) is None

    def test_load_corrupted_json(self, state_path):
        with open(state_path, "w") as f:
            f.write("{{not valid json")
        assert DaemonState.load(state_path) is None


# ── mutations ──────────────────────────────────────────────────────────────


class TestMutations:
    def test_record_wave_start(self, state_path):
        s = DaemonState.create()
        s.record_wave_start(wave_num=1, mandate_name="momentum_aapl.json", path=state_path)

        assert s.current_wave == 1
        assert "momentum_aapl.json" in s.pending_mandates

        # Persisted
        loaded = DaemonState.load(state_path)
        assert loaded.current_wave == 1
        assert "momentum_aapl.json" in loaded.pending_mandates

    def test_record_wave_start_idempotent(self, state_path):
        s = DaemonState.create()
        s.record_wave_start(wave_num=1, mandate_name="x.json", path=state_path)
        s.record_wave_start(wave_num=2, mandate_name="x.json", path=state_path)

        assert s.pending_mandates.count("x.json") == 1

    def test_record_wave_completion_success(self, state_path):
        s = DaemonState.create()
        s.pending_mandates = ["momentum_aapl.json", "mean_reversion_spy.json"]
        s.record_wave_completion(
            mandate_name="momentum_aapl.json",
            status="success",
            sharpe=2.1,
            path=state_path,
        )

        assert s.total_mandates_run == 1
        assert "momentum_aapl.json" not in s.pending_mandates
        assert "momentum_aapl.json" in s.completed_mandates
        assert "momentum_aapl.json" not in s.failed_mandates
        assert s.last_wave_completed  # non-empty
        assert s.last_error is None
        assert s.pending_mandates == ["mean_reversion_spy.json"]

    def test_record_wave_completion_failure(self, state_path):
        s = DaemonState.create()
        s.pending_mandates = ["bad_mandate.json"]
        s.record_wave_completion(
            mandate_name="bad_mandate.json",
            status="failed",
            sharpe=-0.5,
            path=state_path,
        )

        assert s.total_mandates_run == 1
        assert "bad_mandate.json" in s.failed_mandates
        assert "bad_mandate.json" not in s.completed_mandates
        assert s.pending_mandates == []

    def test_record_wave_completion_promoted(self, state_path):
        s = DaemonState.create()
        s.pending_mandates = ["great.json"]
        s.record_wave_completion(
            mandate_name="great.json",
            status="success",
            sharpe=2.5,  # above 1.5 threshold
            path=state_path,
        )

        assert s.total_strategies_promoted == 1

    def test_record_wave_completion_not_promoted(self, state_path):
        s = DaemonState.create()
        s.pending_mandates = ["ok.json"]
        s.record_wave_completion(
            mandate_name="ok.json",
            status="success",
            sharpe=0.8,  # below 1.5 threshold
            path=state_path,
        )

        assert s.total_strategies_promoted == 0


# ── queries ────────────────────────────────────────────────────────────────


class TestQueries:
    def test_get_resume_point(self):
        s = DaemonState.create()
        s.pending_mandates = ["a.json", "b.json"]
        assert s.get_resume_point() == "a.json"

    def test_get_resume_point_empty(self):
        s = DaemonState.create()
        assert s.get_resume_point() is None


# ── shutdown ───────────────────────────────────────────────────────────────


class TestShutdown:
    def test_mark_shutdown(self, state_path):
        s = DaemonState.create()
        s.mark_shutdown(state_path)

        assert s.shutdown_requested is True
        loaded = DaemonState.load(state_path)
        assert loaded.shutdown_requested is True


# ── heartbeat ──────────────────────────────────────────────────────────────


class TestHeartbeat:
    def test_heartbeat_updates_timestamp(self, state_path):
        s = DaemonState.create()
        original = s.last_heartbeat
        time.sleep(0.01)  # ensure time passes
        s.heartbeat(state_path)

        assert s.last_heartbeat != original
        assert s.last_heartbeat > original

        # Persisted
        loaded = DaemonState.load(state_path)
        assert loaded.last_heartbeat == s.last_heartbeat
