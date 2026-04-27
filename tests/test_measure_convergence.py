"""Tests for scripts/measure_convergence.py"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.measure_convergence import (
    KNOWN_ARCHETYPES,
    collect_failure_modes,
    compute_report,
    extract_archetype,
    extract_tickers,
    format_report,
    load_runs,
    main,
    parse_args,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_run(tmp: Path, name: str, **overrides) -> dict:
    """Create a fake refinement run directory with a state.json."""
    defaults = dict(
        run_id=name,
        mandate_name=name,
        created_at="2026-04-27T06:00:00+00:00",
        max_turns=7,
        sharpe_target=1.5,
        tickers=["SPY"],
        period="2y",
        current_turn=3,
        status="max_turns_exhausted",
        best_sharpe=0.8,
        best_turn=2,
        best_code_path="",
        history=[],
        lock_pid=None,
        lock_timestamp=None,
    )
    defaults.update(overrides)
    run_dir = tmp / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(json.dumps(defaults, indent=2))
    return defaults


def _make_history_entry(turn, status="code_generation_failed", sharpe=None, failure_mode=None):
    entry = {"turn": turn, "status": status}
    if sharpe is not None:
        entry["sharpe"] = sharpe
    if failure_mode is not None:
        entry["failure_mode"] = failure_mode
    return entry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadRuns:
    def test_loads_all_state_files(self, tmp_path):
        _make_run(tmp_path, "run_a", status="success", best_sharpe=2.0, best_turn=1)
        _make_run(tmp_path, "run_b", status="abandoned")
        runs = load_runs(tmp_path, since=None)
        assert len(runs) == 2

    def test_skips_non_json_dirs(self, tmp_path):
        _make_run(tmp_path, "run_a", status="success", best_sharpe=2.0)
        # A directory without state.json
        (tmp_path / "empty_dir").mkdir()
        runs = load_runs(tmp_path, since=None)
        assert len(runs) == 1

    def test_since_filter(self, tmp_path):
        _make_run(tmp_path, "old_run", created_at="2026-01-01T00:00:00+00:00")
        _make_run(tmp_path, "new_run", created_at="2026-06-01T00:00:00+00:00")
        since = datetime(2026, 4, 1, tzinfo=timezone.utc)
        runs = load_runs(tmp_path, since=since)
        assert len(runs) == 1
        assert runs[0]["mandate_name"] == "new_run"

    def test_empty_directory(self, tmp_path):
        runs = load_runs(tmp_path, since=None)
        assert runs == []


class TestExtractArchetype:
    def test_standard_archetypes(self):
        for arch in KNOWN_ARCHETYPES:
            assert extract_archetype(f"{arch}_aapl") == arch

    def test_unknown_mandate(self):
        assert extract_archetype("e2e_stress_test") is None

    def test_multi_rsi(self):
        # multi_rsi doesn't match known archetypes
        assert extract_archetype("multi_rsi_googl") is None

    def test_spaces_in_name(self):
        assert extract_archetype("Momentum Strategy for SPY") == "momentum"


class TestExtractTickers:
    def test_uppercases_tickers(self):
        data = {"tickers": ["aapl", "spy"]}
        assert extract_tickers(data) == ["AAPL", "SPY"]

    def test_empty_tickers(self):
        assert extract_tickers({}) == []

    def test_string_tickers_ignored(self):
        data = {"tickers": "not_a_list"}
        assert extract_tickers(data) == []


class TestCollectFailureModes:
    def test_counts_failure_statuses(self):
        runs = [
            {
                "history": [
                    _make_history_entry(1, "code_generation_failed"),
                    _make_history_entry(2, "circuit_breaker_open"),
                ]
            },
            {
                "history": [
                    _make_history_entry(1, "code_generation_failed"),
                ]
            },
        ]
        modes = collect_failure_modes(runs)
        assert modes["code_generation_failed"] == 2
        assert modes["circuit_breaker_open"] == 1


class TestComputeReport:
    def test_basic_counts(self, tmp_path):
        _make_run(tmp_path, "momentum_aapl", status="success", best_sharpe=2.0, best_turn=1,
                  mandate_name="momentum_aapl", tickers=["AAPL"],
                  history=[_make_history_entry(1, "success", sharpe=2.0)])
        _make_run(tmp_path, "momentum_spy", status="abandoned", best_sharpe=-999.0,
                  mandate_name="momentum_spy", tickers=["SPY"],
                  history=[_make_history_entry(1, "code_generation_failed")])
        runs = load_runs(tmp_path, since=None)
        report = compute_report(runs)
        assert report["total_mandates"] == 2
        assert report["converged"] == 1
        assert report["convergence_rate"] == 50.0
        assert report["abandoned"] == 1
        assert report["avg_turns_to_converge"] == 1.0

    def test_per_archetype(self, tmp_path):
        _make_run(tmp_path, "momentum_aapl", status="success", best_sharpe=2.0, best_turn=2,
                  mandate_name="momentum_aapl", tickers=["AAPL"],
                  history=[_make_history_entry(2, "success", sharpe=2.0)])
        _make_run(tmp_path, "momentum_spy", status="max_turns_exhausted", best_sharpe=0.5,
                  mandate_name="momentum_spy", tickers=["SPY"],
                  history=[_make_history_entry(1, "code_generation_failed")])
        _make_run(tmp_path, "breakout_tsla", status="abandoned", best_sharpe=-999.0,
                  mandate_name="breakout_tsla", tickers=["TSLA"],
                  history=[_make_history_entry(1, "circuit_breaker_open")])
        runs = load_runs(tmp_path, since=None)
        report = compute_report(runs)
        arch = report["by_archetype"]
        assert arch["momentum"]["total"] == 2
        assert arch["momentum"]["success"] == 1
        assert arch["breakout"]["total"] == 1
        assert arch["breakout"]["success"] == 0

    def test_per_ticker(self, tmp_path):
        _make_run(tmp_path, "momentum_aapl", status="success", best_sharpe=2.0, best_turn=1,
                  mandate_name="momentum_aapl", tickers=["AAPL", "NVDA"],
                  history=[])
        _make_run(tmp_path, "momentum_aapl_2", status="abandoned", best_sharpe=-999.0,
                  mandate_name="momentum_aapl_2", tickers=["AAPL"],
                  history=[])
        runs = load_runs(tmp_path, since=None)
        report = compute_report(runs)
        assert report["by_ticker"]["AAPL"]["total"] == 2
        assert report["by_ticker"]["AAPL"]["success"] == 1
        assert report["by_ticker"]["NVDA"]["total"] == 1
        assert report["by_ticker"]["NVDA"]["success"] == 1

    def test_promotion_rate(self, tmp_path):
        _make_run(tmp_path, "momentum_aapl", status="success", best_sharpe=2.0, best_turn=1,
                  mandate_name="momentum_aapl", tickers=["AAPL"],
                  best_code_path="/some/path/momentum_aapl/strategy_v1.py",
                  history=[])
        runs = load_runs(tmp_path, since=None)
        report = compute_report(runs, promoted_codes={"momentum_aapl"})
        assert report["promotion_rate"] == 100.0
        assert report["promoted_from_converged"] == 1

    def test_promotion_rate_none_promoted(self, tmp_path):
        _make_run(tmp_path, "momentum_aapl", status="success", best_sharpe=2.0, best_turn=1,
                  mandate_name="momentum_aapl", tickers=["AAPL"],
                  best_code_path="/some/path/momentum_aapl/strategy_v1.py",
                  history=[])
        runs = load_runs(tmp_path, since=None)
        report = compute_report(runs, promoted_codes=set())
        assert report["promotion_rate"] == 0.0

    def test_time_trends(self, tmp_path):
        _make_run(tmp_path, "run_1", status="success", best_sharpe=2.0, best_turn=1,
                  created_at="2026-04-26T10:00:00+00:00", history=[])
        _make_run(tmp_path, "run_2", status="abandoned", best_sharpe=-999.0,
                  created_at="2026-04-26T10:00:00+00:00", history=[])
        _make_run(tmp_path, "run_3", status="success", best_sharpe=1.8, best_turn=2,
                  created_at="2026-04-27T10:00:00+00:00", history=[])
        runs = load_runs(tmp_path, since=None)
        report = compute_report(runs)
        trends = report["time_trends"]
        assert "2026-04-26" in trends
        assert trends["2026-04-26"]["total"] == 2
        assert trends["2026-04-26"]["success"] == 1
        assert "2026-04-27" in trends
        assert trends["2026-04-27"]["total"] == 1
        assert trends["2026-04-27"]["success"] == 1

    def test_empty_runs(self):
        report = compute_report([])
        assert report["total_mandates"] == 0
        assert report["convergence_rate"] == 0.0


class TestFormatReport:
    def test_contains_key_headers(self, tmp_path):
        _make_run(tmp_path, "momentum_aapl", status="success", best_sharpe=2.0, best_turn=1,
                  mandate_name="momentum_aapl", tickers=["AAPL"], history=[])
        runs = load_runs(tmp_path, since=None)
        report = compute_report(runs)
        text = format_report(report)
        assert "CrabQuant Convergence Report" in text
        assert "By Archetype:" in text
        assert "By Ticker:" in text
        assert "Common Failure Modes:" in text
        assert "Time Trends:" in text

    def test_json_flag(self, tmp_path):
        _make_run(tmp_path, "momentum_aapl", status="success", best_sharpe=2.0, best_turn=1,
                  mandate_name="momentum_aapl", tickers=["AAPL"], history=[])
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            main(["--runs-dir", str(tmp_path), "--json"])
        output = f.getvalue()
        data = json.loads(output)
        assert data["total_mandates"] == 1


class TestIntegration:
    """Integration test: create a multi-run scenario, verify end-to-end."""

    def test_full_scenario(self, tmp_path):
        # 3 successes, 2 abandoned, 1 exhausted
        _make_run(tmp_path, "momentum_aapl_1", status="success", best_sharpe=2.0, best_turn=1,
                  mandate_name="momentum_aapl_1", tickers=["AAPL"],
                  created_at="2026-04-27T06:00:00+00:00",
                  history=[_make_history_entry(1, "success", sharpe=2.0)])
        _make_run(tmp_path, "momentum_aapl_2", status="success", best_sharpe=1.8, best_turn=2,
                  mandate_name="momentum_aapl_2", tickers=["AAPL"],
                  created_at="2026-04-27T07:00:00+00:00",
                  history=[_make_history_entry(1, "code_generation_failed"),
                           _make_history_entry(2, "success", sharpe=1.8)])
        _make_run(tmp_path, "breakout_tsla", status="success", best_sharpe=1.6, best_turn=3,
                  mandate_name="breakout_tsla", tickers=["TSLA"],
                  created_at="2026-04-27T08:00:00+00:00",
                  history=[_make_history_entry(1, "code_generation_failed"),
                           _make_history_entry(2, "code_generation_failed"),
                           _make_history_entry(3, "success", sharpe=1.6)])
        _make_run(tmp_path, "mean_reversion_nvda", status="abandoned", best_sharpe=-999.0,
                  mandate_name="mean_reversion_nvda", tickers=["NVDA"],
                  created_at="2026-04-27T09:00:00+00:00",
                  history=[_make_history_entry(1, "code_generation_failed"),
                           _make_history_entry(2, "circuit_breaker_open")])
        _make_run(tmp_path, "volume_msft", status="abandoned", best_sharpe=-999.0,
                  mandate_name="volume_msft", tickers=["MSFT"],
                  created_at="2026-04-27T10:00:00+00:00",
                  history=[_make_history_entry(1, "module_load_failed")])
        _make_run(tmp_path, "trend_amzn", status="max_turns_exhausted", best_sharpe=0.4,
                  mandate_name="trend_amzn", tickers=["AMZN"],
                  created_at="2026-04-27T11:00:00+00:00",
                  history=[_make_history_entry(1, "code_generation_failed")] * 7)

        runs = load_runs(tmp_path, since=None)
        report = compute_report(runs)

        # Totals
        assert report["total_mandates"] == 6
        assert report["converged"] == 3
        assert report["abandoned"] == 2
        assert report["max_turns_exhausted"] == 1

        # Convergence rate
        assert report["convergence_rate"] == 50.0

        # Avg turns: (1 + 2 + 3) / 3 = 2.0
        assert report["avg_turns_to_converge"] == 2.0

        # Archetypes
        assert report["by_archetype"]["momentum"]["total"] == 2
        assert report["by_archetype"]["momentum"]["success"] == 2
        assert report["by_archetype"]["breakout"]["total"] == 1
        assert report["by_archetype"]["breakout"]["success"] == 1
        assert report["by_archetype"]["mean_reversion"]["total"] == 1
        assert report["by_archetype"]["mean_reversion"]["success"] == 0

        # Failure modes
        modes = report["failure_modes"]
        assert modes["code_generation_failed"] >= 4  # at least 4 code gen failures

        # Time trends — all on 2026-04-27
        trends = report["time_trends"]
        assert trends["2026-04-27"]["total"] == 6
        assert trends["2026-04-27"]["success"] == 3

        # Human-readable output doesn't crash
        text = format_report(report)
        assert "50.0%" in text
