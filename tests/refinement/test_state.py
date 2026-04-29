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


# ── additional edge cases ─────────────────────────────────────────────────


class TestLoadEdgeCases:
    """Edge cases for DaemonState.load()."""

    def test_load_with_type_error_returns_none(self, state_path):
        """Non-dict JSON (e.g., a list) returns None."""
        with open(state_path, "w") as f:
            json.dump(["not", "a", "dict"], f)
        assert DaemonState.load(state_path) is None

    def test_load_with_missing_fields_uses_defaults(self, state_path):
        """JSON with only daemon_id uses defaults for remaining fields."""
        with open(state_path, "w") as f:
            json.dump({"daemon_id": "abc"}, f)
        loaded = DaemonState.load(state_path)
        assert loaded is not None
        assert loaded.daemon_id == "abc"
        assert loaded.current_wave == 0  # default
        assert loaded.pending_mandates == []  # default

    def test_load_with_extra_fields_returns_none(self, state_path):
        """Extra fields in JSON cause TypeError, returning None."""
        s = DaemonState.create()
        s.current_wave = 5
        s.save(state_path)

        # Inject extra field
        data = json.loads(open(state_path).read())
        data["extra_field"] = "should be ignored"
        with open(state_path, "w") as f:
            json.dump(data, f)

        loaded = DaemonState.load(state_path)
        # cls(**data) raises TypeError for unknown kwargs
        assert loaded is None

    def test_load_with_wrong_field_type_succeeds(self, state_path):
        """Wrong type for daemon_id still loads (Python doesn't enforce types)."""
        with open(state_path, "w") as f:
            json.dump({"daemon_id": 12345}, f)
        loaded = DaemonState.load(state_path)
        assert loaded is not None
        assert loaded.daemon_id == 12345


class TestSaveEdgeCases:
    """Edge cases for DaemonState.save()."""

    def test_save_no_parent_dir_raises_error(self, tmp_path):
        """Save raises FileNotFoundError when parent dirs don't exist."""
        deep_path = str(tmp_path / "sub" / "dir" / "state.json")
        s = DaemonState.create()
        import pytest as _pytest
        with _pytest.raises(FileNotFoundError):
            s.save(deep_path)

    def test_multiple_save_load_cycles(self, state_path):
        """Multiple save/load cycles preserve state correctly."""
        s = DaemonState.create()

        # Cycle 1
        s.current_wave = 1
        s.save(state_path)
        s = DaemonState.load(state_path)
        assert s.current_wave == 1

        # Cycle 2
        s.current_wave = 2
        s.total_mandates_run = 5
        s.save(state_path)
        s = DaemonState.load(state_path)
        assert s.current_wave == 2
        assert s.total_mandates_run == 5

    def test_save_overwrites_previous(self, state_path):
        """Subsequent saves overwrite previous file content."""
        s1 = DaemonState.create()
        s1.current_wave = 1
        s1.save(state_path)

        s2 = DaemonState.create()
        s2.current_wave = 99
        s2.save(state_path)

        loaded = DaemonState.load(state_path)
        assert loaded.current_wave == 99


class TestMutationEdgeCases:
    """Edge cases for mutation methods."""

    def test_record_wave_completion_nonexistent_mandate(self, state_path):
        """Completing a mandate not in pending doesn't crash."""
        s = DaemonState.create()
        s.pending_mandates = ["a.json"]
        s.record_wave_completion(
            mandate_name="nonexistent.json",
            status="success",
            sharpe=2.0,
            path=state_path,
        )
        # Nonexistent is still added to completed
        assert "nonexistent.json" in s.completed_mandates
        assert s.total_mandates_run == 1
        # Original pending unchanged
        assert "a.json" in s.pending_mandates

    def test_record_wave_completion_clears_last_error(self, state_path):
        """Successful completion clears previous last_error."""
        s = DaemonState.create()
        s.last_error = "previous error"
        s.pending_mandates = ["a.json"]
        s.record_wave_completion(
            mandate_name="a.json",
            status="success",
            sharpe=1.0,
            path=state_path,
        )
        assert s.last_error is None

    def test_record_wave_start_updates_wave_number(self, state_path):
        """record_wave_start updates current_wave on each call."""
        s = DaemonState.create()
        s.record_wave_start(1, "m1.json", state_path)
        assert s.current_wave == 1
        s.record_wave_start(5, "m2.json", state_path)
        assert s.current_wave == 5

    def test_multiple_wave_completions_accumulate(self, state_path):
        """Multiple completions correctly accumulate counters."""
        s = DaemonState.create()
        s.pending_mandates = ["a.json", "b.json", "c.json"]

        s.record_wave_completion("a.json", "success", 2.0, state_path)
        s.record_wave_completion("b.json", "failed", -0.5, state_path)
        s.record_wave_completion("c.json", "success", 0.8, state_path)

        assert s.total_mandates_run == 3
        assert len(s.completed_mandates) == 2
        assert len(s.failed_mandates) == 1
        assert s.total_strategies_promoted == 1  # only a.json with sharpe >= 1.5
        assert s.pending_mandates == []

    def test_mark_shutdown_persists(self, state_path):
        """mark_shutdown persists across load."""
        s = DaemonState.create()
        s.mark_shutdown(state_path)

        loaded = DaemonState.load(state_path)
        assert loaded.shutdown_requested is True
        assert loaded.daemon_id == s.daemon_id

    def test_sharpe_boundary_exactly_1_5_promotes(self, state_path):
        """Sharpe exactly 1.5 should promote (boundary)."""
        s = DaemonState.create()
        s.pending_mandates = ["boundary.json"]
        s.record_wave_completion(
            mandate_name="boundary.json",
            status="success",
            sharpe=1.5,
            path=state_path,
        )
        assert s.total_strategies_promoted == 1

    def test_sharpe_just_below_1_5_does_not_promote(self, state_path):
        """Sharpe 1.4999 should not promote (boundary)."""
        s = DaemonState.create()
        s.pending_mandates = ["just_below.json"]
        s.record_wave_completion(
            mandate_name="just_below.json",
            status="success",
            sharpe=1.499,
            path=state_path,
        )
        assert s.total_strategies_promoted == 0

    def test_get_resume_point_returns_first(self):
        """get_resume_point always returns the first pending mandate."""
        s = DaemonState.create()
        s.pending_mandates = ["first.json", "second.json", "third.json"]
        assert s.get_resume_point() == "first.json"

    def test_construction_with_explicit_args(self):
        """DaemonState can be constructed with explicit arguments."""
        s = DaemonState(
            daemon_id="custom-id",
            current_wave=10,
            total_mandates_run=50,
            shutdown_requested=True,
        )
        assert s.daemon_id == "custom-id"
        assert s.current_wave == 10
        assert s.total_mandates_run == 50
        assert s.shutdown_requested is True
