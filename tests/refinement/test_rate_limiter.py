"""Tests for the rate limiter in llm_api."""

import time
import pytest
from unittest.mock import patch

from crabquant.refinement.llm_api import _RateLimiter, set_rate_limit, get_rate_limiter


class TestRateLimiter:
    def test_allows_within_limit(self):
        """Calls within the per-minute limit should not sleep."""
        limiter = _RateLimiter(max_calls_per_minute=10)
        slept = limiter.acquire()
        assert slept == 0.0

    def test_blocks_when_full(self):
        """Once the limit is reached, acquire() should sleep."""
        limiter = _RateLimiter(max_calls_per_minute=2)

        # Fill the bucket
        limiter.acquire()  # call 1
        limiter.acquire()  # call 2

        # Third call should sleep (but we won't wait the full 60s — we
        # patch time.sleep to make it instant).
        with patch("time.sleep") as mock_sleep:
            limiter.acquire()
            mock_sleep.assert_called_once()
            sleep_arg = mock_sleep.call_args[0][0]
            assert sleep_arg > 0

    def test_prunes_old_timestamps(self):
        """Timestamps older than 60s should be pruned, freeing capacity."""
        limiter = _RateLimiter(max_calls_per_minute=2)

        # Fill the bucket
        limiter.acquire()
        limiter.acquire()

        # Manually age the timestamps so they fall outside the window
        now = time.monotonic()
        limiter._timestamps = [now - 61.0, now - 62.0]

        # Should not sleep — old timestamps are pruned
        slept = limiter.acquire()
        assert slept == 0.0

    def test_thread_safety(self):
        """Multiple threads calling acquire() should not corrupt state."""
        import threading

        limiter = _RateLimiter(max_calls_per_minute=100)  # high limit
        errors = []

        def worker():
            try:
                for _ in range(20):
                    limiter.acquire()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"

    def test_set_rate_limit_changes_global(self):
        """set_rate_limit should replace the global rate limiter."""
        original = get_rate_limiter()
        set_rate_limit(5)
        new_limiter = get_rate_limiter()
        assert new_limiter._max == 5
        # Restore
        set_rate_limit(20)
        restored = get_rate_limiter()
        assert restored._max == 20


class TestCallZaiLlm429:
    """Test that call_zai_llm retries on HTTP 429 with longer backoff."""

    def _make_mock_response(self, content="{}", status_code=200, headers=None):
        """Create a mock httpx response."""
        mock_resp = pytest.importorskip("unittest.mock").MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_resp.headers = headers or {}
        mock_resp.raise_for_status.side_effect = None
        if status_code >= 400:
            import httpx
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "rate limited", request=pytest.importorskip("unittest.mock").MagicMock(), response=mock_resp,
            )
        return mock_resp

    def test_retries_on_429(self):
        """Should retry up to 3 times on 429 before raising."""
        import httpx
        from unittest.mock import MagicMock, patch
        from crabquant.refinement.llm_api import call_zai_llm

        call_count = 0
        responses_429 = self._make_mock_response(status_code=429)
        responses_200 = self._make_mock_response('{"action": "novel"}')

        def mock_post(url, json=None, headers=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return responses_429
            return responses_200

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                with patch("crabquant.refinement.llm_api._global_rate_limiter") as mock_rl:
                    mock_rl.acquire.return_value = 0
                    with patch("time.sleep"):
                        result = call_zai_llm(
                            messages=[{"role": "user", "content": "test"}],
                            timeout=10,
                        )
                        assert call_count == 3
                        assert "action" in result

    def test_respects_retry_after_header(self):
        """Should use Retry-After header value for backoff duration."""
        import httpx
        from unittest.mock import MagicMock, patch
        from crabquant.refinement.llm_api import call_zai_llm

        resp_429 = self._make_mock_response(
            status_code=429,
            headers={"Retry-After": "7"},
        )
        resp_200 = self._make_mock_response('{"action": "novel"}')

        call_count = 0
        def mock_post(url, json=None, headers=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return resp_429
            return resp_200

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = mock_post

        sleep_calls = []
        def track_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                with patch("crabquant.refinement.llm_api._global_rate_limiter") as mock_rl:
                    mock_rl.acquire.return_value = 0
                    with patch("time.sleep", side_effect=track_sleep):
                        call_zai_llm(
                            messages=[{"role": "user", "content": "test"}],
                            timeout=10,
                        )
                        # First 429 should use Retry-After header (7s)
                        assert sleep_calls[0] >= 7.0

    def test_raises_after_max_retries_on_429(self):
        """Should raise HTTPStatusError after exhausting retries."""
        import httpx
        from unittest.mock import MagicMock, patch
        from crabquant.refinement.llm_api import call_zai_llm

        resp_429 = self._make_mock_response(status_code=429)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = resp_429

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                with patch("crabquant.refinement.llm_api._global_rate_limiter") as mock_rl:
                    mock_rl.acquire.return_value = 0
                    with patch("time.sleep"):
                        with pytest.raises(httpx.HTTPStatusError):
                            call_zai_llm(
                                messages=[{"role": "user", "content": "test"}],
                                timeout=10,
                            )
