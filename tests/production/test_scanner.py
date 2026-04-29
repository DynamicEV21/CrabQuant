"""Tests for crabquant.production.scanner"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from crabquant.production.scanner import (
    _load_json,
    _get_promoted_keys,
    _make_winner_key,
    scan_and_promote,
)


class TestLoadJson:
    def test_loads_existing_file(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('[{"a": 1}]')

        result = _load_json(f)
        assert result == [{"a": 1}]

    def test_returns_default_for_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.json"
        result = _load_json(f, default=[])
        assert result == []

    def test_returns_none_default(self, tmp_path):
        f = tmp_path / "nonexistent.json"
        # _load_json has `return default or []` — None becomes []
        result = _load_json(f)
        assert result == []

    def test_returns_empty_list_default_when_default_is_none(self, tmp_path):
        f = tmp_path / "nonexistent.json"
        result = _load_json(f, None)
        # `None or []` evaluates to []
        assert result == []

    def test_handles_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json")

        # Should raise since _load_json doesn't catch JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            _load_json(f)


class TestMakeWinnerKey:
    def test_key_format(self):
        key = _make_winner_key({"strategy": "ema_cross", "ticker": "SPY", "params": {"fast": 10}})
        assert key.startswith("ema_cross|SPY|")
        parts = key.split("|")
        assert len(parts) == 3
        assert len(parts[2]) == 12  # SHA256 truncated

    def test_deterministic(self):
        params = {"fast": 10, "slow": 20}
        k1 = _make_winner_key({"strategy": "s", "ticker": "SPY", "params": params})
        k2 = _make_winner_key({"strategy": "s", "ticker": "SPY", "params": {"slow": 20, "fast": 10}})
        assert k1 == k2

    def test_different_params_different_key(self):
        k1 = _make_winner_key({"strategy": "s", "ticker": "SPY", "params": {"a": 1}})
        k2 = _make_winner_key({"strategy": "s", "ticker": "SPY", "params": {"a": 2}})
        assert k1 != k2

    def test_different_ticker_different_key(self):
        k1 = _make_winner_key({"strategy": "s", "ticker": "SPY", "params": {"a": 1}})
        k2 = _make_winner_key({"strategy": "s", "ticker": "QQQ", "params": {"a": 1}})
        assert k1 != k2

    def test_empty_params(self):
        key = _make_winner_key({"strategy": "s", "ticker": "SPY", "params": {}})
        assert "|" in key


class TestGetPromotedKeys:
    def test_empty_registry(self, tmp_path):
        with patch("crabquant.production.scanner.REGISTRY_FILE", tmp_path / "nonexistent.json"):
            keys = _get_promoted_keys()
            assert keys == set()

    def test_loads_keys_from_registry(self, tmp_path):
        f = tmp_path / "registry.json"
        f.write_text(json.dumps([
            {"key": "s1|SPY|abc123"},
            {"key": "s2|QQQ|def456"},
        ]))

        with patch("crabquant.production.scanner.REGISTRY_FILE", f):
            keys = _get_promoted_keys()
            assert keys == {"s1|SPY|abc123", "s2|QQQ|def456"}

    def test_handles_corrupt_registry(self, tmp_path):
        f = tmp_path / "registry.json"
        f.write_text("not json")

        with patch("crabquant.production.scanner.REGISTRY_FILE", f):
            keys = _get_promoted_keys()
            assert keys == set()


class TestScanAndPromote:
    @patch("crabquant.production.scanner._get_promoted_keys", return_value=set())
    @patch("crabquant.production.scanner._load_json")
    def test_no_robust_strategies(self, mock_load, mock_keys):
        mock_load.side_effect = lambda path, default=None: (
            [{"verdict": "FAILED", "strategy": "s", "ticker": "SPY"}] if "confirmed" in str(path) else []
        )

        result = scan_and_promote()
        assert result == []

    @patch("crabquant.production.scanner._get_promoted_keys", return_value=set())
    @patch("crabquant.production.scanner._load_json")
    def test_no_matching_winner_found(self, mock_load, mock_keys):
        confirmed = [{"verdict": "ROBUST", "strategy": "new_strat", "ticker": "SPY", "params": {}}]
        mock_load.side_effect = lambda path, default=None: (
            confirmed if "confirmed" in str(path) else []
        )

        result = scan_and_promote()
        assert result == []

    @patch("crabquant.production.scanner._get_promoted_keys", return_value=set())
    @patch("crabquant.production.scanner._load_json")
    def test_promotes_robust_strategy(self, mock_load, mock_keys):
        confirmed = [{
            "verdict": "ROBUST",
            "strategy": "trend_strat",
            "ticker": "SPY",
            "params": {"fast": 10},
            "key": "trend_strat|SPY|abc123",
        }]
        winners = [{"key": "trend_strat|SPY|abc123", "sharpe": 1.5, "strategy": "trend_strat", "ticker": "SPY", "params": {"fast": 10}}]

        mock_load.side_effect = lambda path, default=None: (
            confirmed if "confirmed" in str(path) else winners
        )

        mock_report = MagicMock()
        mock_report.date_promoted = "2026-04-28"

        with patch("crabquant.production.promoter.promote_strategy", return_value=mock_report):
            result = scan_and_promote()

        assert len(result) == 1
        assert result[0]["strategy_name"] == "trend_strat"
        assert result[0]["ticker"] == "SPY"

    @patch("crabquant.production.scanner._get_promoted_keys", return_value=set())
    @patch("crabquant.production.scanner._load_json")
    def test_skips_already_promoted(self, mock_load, mock_keys):
        confirmed = [{
            "verdict": "ROBUST",
            "strategy": "trend_strat",
            "ticker": "SPY",
            "params": {"fast": 10},
            "key": "trend_strat|SPY|abc123",
        }]
        winners = [{"key": "trend_strat|SPY|abc123", "sharpe": 1.5, "strategy": "trend_strat", "ticker": "SPY", "params": {"fast": 10}}]

        mock_load.side_effect = lambda path, default=None: (
            confirmed if "confirmed" in str(path) else winners
        )
        # _promoter_make_key computes the same format as _make_winner_key
        # The scanner checks promoter_key against promoted_keys
        from crabquant.production.promoter import _make_key
        promoter_key = _make_key("trend_strat", "SPY", {"fast": 10})
        mock_keys.return_value = {promoter_key}

        with patch("crabquant.production.promoter.promote_strategy") as mock_promote:
            result = scan_and_promote()

        assert result == []
        mock_promote.assert_not_called()

    @patch("crabquant.production.scanner._get_promoted_keys", return_value=set())
    @patch("crabquant.production.scanner._load_json")
    def test_promotion_error_caught(self, mock_load, mock_keys):
        confirmed = [{
            "verdict": "ROBUST",
            "strategy": "bad_strat",
            "ticker": "SPY",
            "params": {},
            "key": "bad_strat|SPY|abc",
        }]
        winners = [{"key": "bad_strat|SPY|abc", "sharpe": 1.0, "strategy": "bad_strat", "ticker": "SPY", "params": {}}]

        mock_load.side_effect = lambda path, default=None: (
            confirmed if "confirmed" in str(path) else winners
        )

        with patch("crabquant.production.promoter.promote_strategy", side_effect=ValueError("Already promoted")):
            result = scan_and_promote()

        # Error should be caught, not raised
        assert result == []

    @patch("crabquant.production.scanner._get_promoted_keys", return_value=set())
    @patch("crabquant.production.scanner._load_json")
    def test_promotes_multiple_robust_strategies(self, mock_load, mock_keys):
        confirmed = [
            {"verdict": "ROBUST", "strategy": "s1", "ticker": "SPY", "params": {}, "key": "s1|SPY|a"},
            {"verdict": "ROBUST", "strategy": "s2", "ticker": "QQQ", "params": {}, "key": "s2|QQQ|b"},
            {"verdict": "FAILED", "strategy": "s3", "ticker": "SPY", "params": {}, "key": "s3|SPY|c"},
        ]
        winners = [
            {"key": "s1|SPY|a", "sharpe": 1.0, "strategy": "s1", "ticker": "SPY", "params": {}},
            {"key": "s2|QQQ|b", "sharpe": 1.5, "strategy": "s2", "ticker": "QQQ", "params": {}},
        ]

        mock_load.side_effect = lambda path, default=None: (
            confirmed if "confirmed" in str(path) else winners
        )

        mock_report = MagicMock()
        mock_report.date_promoted = "2026-04-28"

        with patch("crabquant.production.promoter.promote_strategy", return_value=mock_report) as mock_promote:
            result = scan_and_promote()

        assert len(result) == 2
        assert mock_promote.call_count == 2

    @patch("crabquant.production.scanner._get_promoted_keys", return_value=set())
    @patch("crabquant.production.scanner._load_json")
    def test_matches_winner_by_params_when_key_differs(self, mock_load, mock_keys):
        """When confirmed key doesn't match winner key, falls back to params lookup."""
        confirmed = [{
            "verdict": "ROBUST",
            "strategy": "s",
            "ticker": "SPY",
            "params": {"fast": 10},
            "key": "s|SPY|different_key",
        }]
        winners = [{
            "key": "s|SPY|abc123",
            "sharpe": 1.0,
            "strategy": "s",
            "ticker": "SPY",
            "params": {"fast": 10},
        }]

        mock_load.side_effect = lambda path, default=None: (
            confirmed if "confirmed" in str(path) else winners
        )

        mock_report = MagicMock()
        mock_report.date_promoted = "2026-04-28"

        with patch("crabquant.production.promoter.promote_strategy", return_value=mock_report) as mock_promote:
            result = scan_and_promote()

        assert len(result) == 1
        mock_promote.assert_called_once()
