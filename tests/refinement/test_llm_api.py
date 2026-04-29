"""Tests for crabquant.refinement.llm_api"""

import json
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path


class TestLoadApiConfig:
    """Test loading z.ai API config from OpenClaw config."""

    def test_loads_base_url_and_api_key(self):
        from crabquant.refinement.llm_api import load_api_config

        mock_config = {
            "models": {
                "providers": {
                    "zai": {
                        "baseUrl": "https://api.z.ai/api/coding/paas/v4",
                        "apiKey": "test-key-123",
                    }
                }
            }
        }
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value=mock_config):
                config = load_api_config()
                assert config["base_url"] == "https://api.z.ai/api/coding/paas/v4"
                assert config["api_key"] == "test-key-123"

    def test_raises_on_missing_provider(self):
        from crabquant.refinement.llm_api import load_api_config

        mock_config = {"models": {"providers": {}}}
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value=mock_config):
                with pytest.raises(KeyError):
                    load_api_config()

    def test_raises_on_missing_file(self):
        from crabquant.refinement.llm_api import load_api_config

        with patch("builtins.open", side_effect=FileNotFoundError("not found")):
            with pytest.raises(FileNotFoundError):
                load_api_config()

    def test_raises_on_missing_models_key(self):
        from crabquant.refinement.llm_api import load_api_config

        mock_config = {"other_key": {}}
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value=mock_config):
                with pytest.raises(KeyError):
                    load_api_config()

    def test_raises_on_missing_providers_key(self):
        from crabquant.refinement.llm_api import load_api_config

        mock_config = {"models": {"other_key": {}}}
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value=mock_config):
                with pytest.raises(KeyError):
                    load_api_config()


class TestExtractJsonFromLlm:
    """Test JSON extraction from various LLM response formats."""

    def test_direct_json(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        text = '{"action": "modify_params", "hypothesis": "test"}'
        result = extract_json_from_llm(text)
        assert result["action"] == "modify_params"

    def test_json_in_code_block(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        text = 'Here is the strategy:\n```json\n{"action": "full_rewrite", "hypothesis": "redo"}\n```'
        result = extract_json_from_llm(text)
        assert result["action"] == "full_rewrite"

    def test_json_in_code_block_no_language(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        text = 'Result:\n```\n{"action": "novel", "hypothesis": "new idea"}\n```'
        result = extract_json_from_llm(text)
        assert result["action"] == "novel"

    def test_json_with_surrounding_text(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        text = 'Sure, here you go:\n{"action": "replace_indicator", "params": {"rsi_period": 20}}\nHope that helps!'
        result = extract_json_from_llm(text)
        assert result["action"] == "replace_indicator"
        assert result["params"]["rsi_period"] == 20

    def test_nested_json(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        text = '```json\n{"action": "full_rewrite", "params": {"nested": {"deep": true}}}\n```'
        result = extract_json_from_llm(text)
        assert result["params"]["nested"]["deep"] is True

    def test_json_with_code_string_inside(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        strategy_json = {
            "action": "full_rewrite",
            "new_strategy_code": "def generate_signals(df, params):\n    return df['close'] > 0, df['close'] < 0",
            "params": {},
        }
        text = f"```json\n{json.dumps(strategy_json)}\n```"
        result = extract_json_from_llm(text)
        assert "generate_signals" in result["new_strategy_code"]

    def test_raises_on_no_json(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        with pytest.raises(ValueError, match="Could not extract JSON"):
            extract_json_from_llm("No JSON here, just plain text.")

    def test_raises_on_empty_string(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        with pytest.raises(ValueError):
            extract_json_from_llm("")

    def test_brace_matching_fallback(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        # No code blocks, just raw JSON with surrounding text
        text = 'The answer is {"action": "add_filter", "hypothesis": "test"} ok?'
        result = extract_json_from_llm(text)
        assert result["action"] == "add_filter"

    def test_handles_multiline_code_in_json(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        code = "def foo():\n    return 1\n\ndef bar():\n    return 2"
        payload = {"action": "full_rewrite", "new_strategy_code": code}
        text = f"```json\n{json.dumps(payload)}\n```"
        result = extract_json_from_llm(text)
        assert "def foo" in result["new_strategy_code"]
        assert "def bar" in result["new_strategy_code"]

    def test_empty_json_object(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        result = extract_json_from_llm("{}")
        assert result == {}

    def test_none_input_raises(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        with pytest.raises((ValueError, TypeError, AttributeError)):
            extract_json_from_llm(None)

    def test_json_with_unicode_characters(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        text = '{"action": "modify", "description": "使用RSI指标策略"}'
        result = extract_json_from_llm(text)
        assert result["description"] == "使用RSI指标策略"

    def test_json_with_escaped_quotes(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        text = '{"action": "novel", "hypothesis": "He said \\"hello\\""}'
        result = extract_json_from_llm(text)
        assert 'He said "hello"' in result["hypothesis"]

    def test_json_with_arrays(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        text = '{"action": "novel", "params": [1, 2, 3], "tags": ["a", "b"]}'
        result = extract_json_from_llm(text)
        assert result["params"] == [1, 2, 3]
        assert result["tags"] == ["a", "b"]

    def test_multiple_code_blocks_picks_first_valid(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        text = '```json\n{"action": "first"}\n```\nSome text\n```json\n{"action": "second"}\n```'
        result = extract_json_from_llm(text)
        assert result["action"] == "first"

    def test_code_block_with_non_json_content_uses_brace_fallback(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        # Code block has non-JSON, but JSON exists after it
        text = '```python\nx = 1\n```\n{"action": "fallback", "value": 42}'
        result = extract_json_from_llm(text)
        assert result["action"] == "fallback"
        assert result["value"] == 42

    def test_json_with_newlines_in_strings(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        payload = {"action": "full_rewrite", "new_strategy_code": "line1\nline2\nline3"}
        text = f"```json\n{json.dumps(payload)}\n```"
        result = extract_json_from_llm(text)
        assert result["new_strategy_code"] == "line1\nline2\nline3"

    def test_error_message_includes_text_length(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        with pytest.raises(ValueError, match="len=12"):
            extract_json_from_llm("no json here")

    def test_json_with_deeply_nested_structure(self):
        from crabquant.refinement.llm_api import extract_json_from_llm

        text = '{"a": {"b": {"c": {"d": {"e": "deep"}}}}}'
        result = extract_json_from_llm(text)
        assert result["a"]["b"]["c"]["d"]["e"] == "deep"


class TestExtractJsonByBraces:
    """Direct tests for the _extract_json_by_braces helper."""

    def test_returns_none_for_no_braces(self):
        from crabquant.refinement.llm_api import _extract_json_by_braces

        assert _extract_json_by_braces("no braces here") is None

    def test_extracts_simple_object(self):
        from crabquant.refinement.llm_api import _extract_json_by_braces

        result = _extract_json_by_braces('{"key": "value"}')
        assert result == {"key": "value"}

    def test_extracts_from_text_with_braces(self):
        from crabquant.refinement.llm_api import _extract_json_by_braces

        text = 'before {"key": "value"} after'
        result = _extract_json_by_braces(text)
        assert result == {"key": "value"}

    def test_handles_nested_braces_in_strings(self):
        from crabquant.refinement.llm_api import _extract_json_by_braces

        # Braces inside a JSON string value should not confuse the parser
        text = '{"code": "if x { return 1 }", "action": "test"}'
        result = _extract_json_by_braces(text)
        assert result["action"] == "test"
        assert result["code"] == "if x { return 1 }"

    def test_handles_single_quotes_around_json(self):
        from crabquant.refinement.llm_api import _extract_json_by_braces

        # Single quotes inside JSON values
        text = '{"desc": "it\'s a test"}'
        result = _extract_json_by_braces(text)
        assert "it's a test" in result["desc"]

    def test_returns_none_for_unbalanced_braces(self):
        from crabquant.refinement.llm_api import _extract_json_by_braces

        text = '{"key": "value"'
        result = _extract_json_by_braces(text)
        assert result is None

    def test_skips_invalid_json_and_tries_next_brace(self):
        from crabquant.refinement.llm_api import _extract_json_by_braces

        # First { pair is not valid JSON; second is
        text = '{not valid json}{"action": "valid"}'
        result = _extract_json_by_braces(text)
        assert result["action"] == "valid"

    def test_handles_escaped_backslash_in_string(self):
        from crabquant.refinement.llm_api import _extract_json_by_braces

        text = '{"path": "C:\\\\Users\\\\test"}'
        result = _extract_json_by_braces(text)
        assert "Users" in result["path"]

    def test_empty_string_returns_none(self):
        from crabquant.refinement.llm_api import _extract_json_by_braces

        assert _extract_json_by_braces("") is None


class TestCallZaiLlm:
    """Test the z.ai API call function (mocked HTTP)."""

    def _make_mock_response(self, content="test response", status_code=200):
        """Create a mock httpx response."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_returns_content(self):
        from crabquant.refinement.llm_api import call_zai_llm

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = self._make_mock_response("Hello from LLM")

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                result = call_zai_llm([{"role": "user", "content": "hi"}])
                assert result == "Hello from LLM"

    def test_passes_correct_model(self):
        from crabquant.refinement.llm_api import call_zai_llm

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["model"] = json["model"]
            return self._make_mock_response()

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_zai_llm([{"role": "user", "content": "hi"}], model="glm-5.1")
                assert captured["model"] == "glm-5.1"

    def test_passes_temperature(self):
        from crabquant.refinement.llm_api import call_zai_llm

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["temperature"] = json["temperature"]
            return self._make_mock_response()

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_zai_llm([{"role": "user", "content": "hi"}], temperature=0.3)
                assert captured["temperature"] == 0.3

    def test_raises_on_empty_choices(self):
        from crabquant.refinement.llm_api import call_zai_llm

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": []}
        mock_resp.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                with pytest.raises(ValueError, match="Unexpected API response"):
                    call_zai_llm([{"role": "user", "content": "hi"}])

    def test_retries_on_server_error(self):
        from crabquant.refinement.llm_api import call_zai_llm
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        # First 2 calls fail with 500, 3rd succeeds
        err_resp = MagicMock()
        err_resp.status_code = 500
        err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=err_resp
        )
        mock_client.post.side_effect = [
            err_resp,
            err_resp,
            self._make_mock_response("OK after retries"),
        ]

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.time.sleep") as mock_sleep:
                with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                    result = call_zai_llm([{"role": "user", "content": "hi"}])
                    assert result == "OK after retries"
                    assert mock_client.post.call_count == 3
                    assert mock_sleep.call_count == 2

    def test_raises_on_4xx_no_retry(self):
        from crabquant.refinement.llm_api import call_zai_llm
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        err_resp = MagicMock()
        err_resp.status_code = 401
        err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=err_resp
        )
        mock_client.post.return_value = err_resp

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.time.sleep") as mock_sleep:
                with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                    with pytest.raises(httpx.HTTPStatusError):
                        call_zai_llm([{"role": "user", "content": "hi"}])
                    # 4xx should NOT trigger retries
                    assert mock_client.post.call_count == 1
                    assert mock_sleep.call_count == 0

    def test_retries_on_connect_timeout_then_raises(self):
        from crabquant.refinement.llm_api import call_zai_llm
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectTimeout("timed out")

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.time.sleep") as mock_sleep:
                with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                    with pytest.raises(httpx.ConnectTimeout):
                        call_zai_llm([{"role": "user", "content": "hi"}])
                    # Should retry 3 times (attempt 0,1,2 + final raise)
                    assert mock_client.post.call_count == 4
                    assert mock_sleep.call_count == 3

    def test_retries_on_read_timeout(self):
        from crabquant.refinement.llm_api import call_zai_llm
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        # First 2 calls timeout, 3rd succeeds
        mock_client.post.side_effect = [
            httpx.ReadTimeout("read timed out"),
            httpx.ReadTimeout("read timed out"),
            self._make_mock_response("recovered"),
        ]

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.time.sleep") as mock_sleep:
                with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                    result = call_zai_llm([{"role": "user", "content": "hi"}])
                    assert result == "recovered"
                    assert mock_sleep.call_count == 2

    def test_retries_on_connect_error(self):
        from crabquant.refinement.llm_api import call_zai_llm
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        # 2 connect errors then success
        mock_client.post.side_effect = [
            httpx.ConnectError("connection refused"),
            httpx.ConnectError("connection refused"),
            self._make_mock_response("connected"),
        ]

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.time.sleep") as mock_sleep:
                with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                    result = call_zai_llm([{"role": "user", "content": "hi"}])
                    assert result == "connected"
                    assert mock_sleep.call_count == 2

    def test_raises_on_missing_choices_key(self):
        from crabquant.refinement.llm_api import call_zai_llm

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": "no choices key"}
        mock_resp.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                with pytest.raises(ValueError, match="Unexpected API response"):
                    call_zai_llm([{"role": "user", "content": "hi"}])

    def test_constructs_correct_url(self):
        from crabquant.refinement.llm_api import call_zai_llm

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["url"] = url
            return self._make_mock_response()

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.example.com/v4", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_zai_llm([{"role": "user", "content": "hi"}])
                assert captured["url"] == "https://api.example.com/v4/chat/completions"

    def test_passes_authorization_header(self):
        from crabquant.refinement.llm_api import call_zai_llm

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["headers"] = headers
            return self._make_mock_response()

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "my-secret-key"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_zai_llm([{"role": "user", "content": "hi"}])
                assert captured["headers"]["Authorization"] == "Bearer my-secret-key"
                assert captured["headers"]["Content-Type"] == "application/json"

    def test_passes_max_tokens(self):
        from crabquant.refinement.llm_api import call_zai_llm

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["max_tokens"] = json["max_tokens"]
            return self._make_mock_response()

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_zai_llm([{"role": "user", "content": "hi"}], max_tokens=4096)
                assert captured["max_tokens"] == 4096

    def test_passes_messages_intact(self):
        from crabquant.refinement.llm_api import call_zai_llm

        captured = {}
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._make_mock_response()

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_zai_llm(messages)
                assert captured["messages"] == messages

    def test_backoff_delays_are_exponential(self):
        from crabquant.refinement.llm_api import call_zai_llm
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        err_resp = MagicMock()
        err_resp.status_code = 502
        err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Gateway", request=MagicMock(), response=err_resp
        )
        # All calls fail, exhausting retries
        mock_client.post.side_effect = [err_resp] * 4

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.time.sleep") as mock_sleep:
                with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                    with pytest.raises(httpx.HTTPStatusError):
                        call_zai_llm([{"role": "user", "content": "hi"}])
                    # Should have slept with backoff: 2, 4, 8
                    assert mock_sleep.call_count == 3
                    mock_sleep.assert_any_call(2)
                    mock_sleep.assert_any_call(4)
                    mock_sleep.assert_any_call(8)

    def test_budget_tracking_called_on_success(self):
        from crabquant.refinement.llm_api import call_zai_llm

        mock_resp = self._make_mock_response("test")
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "test"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp

        mock_tracker = MagicMock()

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                with patch("crabquant.refinement.api_budget.get_global_tracker", return_value=mock_tracker):
                    result = call_zai_llm([{"role": "user", "content": "hi"}])
                    assert result == "test"
                    mock_tracker.record_call.assert_called_once()
                    call_kwargs = mock_tracker.record_call.call_args[1]
                    assert call_kwargs["prompt_tokens"] == 100
                    assert call_kwargs["completion_tokens"] == 50
                    assert call_kwargs["model"] == "glm-5-turbo"
                    assert call_kwargs["success"] is True

    def test_budget_tracking_failure_does_not_crash(self):
        from crabquant.refinement.llm_api import call_zai_llm

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = self._make_mock_response("test")

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                with patch("crabquant.refinement.api_budget.get_global_tracker", side_effect=ImportError("no module")):
                    # Should NOT raise even if budget tracker fails
                    result = call_zai_llm([{"role": "user", "content": "hi"}])
                    assert result == "test"

    def test_server_error_503_retries(self):
        from crabquant.refinement.llm_api import call_zai_llm
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        err_resp = MagicMock()
        err_resp.status_code = 503
        err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service Unavailable", request=MagicMock(), response=err_resp
        )
        mock_client.post.side_effect = [
            err_resp,
            self._make_mock_response("recovered after 503"),
        ]

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.time.sleep") as mock_sleep:
                with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                    result = call_zai_llm([{"role": "user", "content": "hi"}])
                    assert result == "recovered after 503"
                    assert mock_sleep.call_count == 1


class TestCallLlmInventor:
    """Test the high-level inventor call function."""

    def _mock_llm_response(self, json_dict):
        """Create a mock that returns JSON-wrapped LLM response via httpx."""
        content = f"```json\n{json.dumps(json_dict)}\n```"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def _mock_client(self, response):
        """Create a mock httpx Client that returns the given response."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = response
        return mock_client

    def _default_patches(self):
        """Return standard patches for call_llm_inventor tests."""
        return (
            patch("crabquant.refinement.llm_api.load_api_config",
                  return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}),
            patch("crabquant.refinement.llm_api.httpx.Client"),
        )

    def test_returns_parsed_dict(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        expected = {
            "action": "modify_params",
            "hypothesis": "Increase RSI period",
            "new_strategy_code": "def generate_signals(df, params): pass",
            "params": {"rsi_period": 20},
            "expected_impact": "higher",
        }

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client",
                       return_value=self._mock_client(self._mock_llm_response(expected))):
                result = call_llm_inventor({
                    "backtest_report": {"sharpe": 0.8},
                    "failure_mode": "low_sharpe",
                })
                assert result["action"] == "modify_params"
                assert result["params"]["rsi_period"] == 20

    def test_returns_none_on_api_error(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("Network error")

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                result = call_llm_inventor({"failure_mode": "low_sharpe"})
                assert result is None

    def test_returns_none_on_bad_json(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "No JSON here at all!"}}]
        }
        mock_resp.raise_for_status.return_value = None

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client",
                       return_value=self._mock_client(mock_resp)):
                result = call_llm_inventor({"failure_mode": "low_sharpe"})
                assert result is None

    def test_uses_prompt_key_directly(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({"prompt": "Invent a strategy using RSI"})
                user_msg = captured["messages"][1]["content"]
                assert "Invent a strategy using RSI" in user_msg

    def test_saves_context_to_file(self, tmp_path):
        from crabquant.refinement.llm_api import call_llm_inventor

        context_path = str(tmp_path / "context.json")
        context = {"failure_mode": "low_sharpe", "backtest_report": {"sharpe": 0.5}}

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client",
                       return_value=self._mock_client(self._mock_llm_response({"action": "novel"}))):
                call_llm_inventor(context, context_path=context_path)

        saved = json.loads(Path(context_path).read_text())
        assert saved["failure_mode"] == "low_sharpe"

    def test_saves_raw_response_to_file(self, tmp_path):
        from crabquant.refinement.llm_api import call_llm_inventor

        context_path = str(tmp_path / "debug" / "ctx.json")

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client",
                       return_value=self._mock_client(self._mock_llm_response({"action": "test"}))):
                call_llm_inventor({"failure_mode": "test"}, context_path=context_path)

        raw_path = Path(context_path).parent / "ctx_raw_response.txt"
        assert raw_path.exists()
        content = raw_path.read_text()
        assert "test" in content

    def test_context_with_current_strategy_code(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "current_strategy_code": "def generate_signals(df, params):\n    pass",
                })
                user_msg = captured["messages"][1]["content"]
                assert "generate_signals" in user_msg
                assert "Current Strategy Code" in user_msg

    def test_context_with_failure_mode_and_reasoning(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "failure_mode": "low_sharpe",
                    "failure_reasoning": "RSI period too short",
                })
                user_msg = captured["messages"][1]["content"]
                assert "low_sharpe" in user_msg
                assert "RSI period too short" in user_msg

    def test_context_with_strategy_examples(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "strategy_examples": [
                        {"name": "RSI Strategy", "description": "Uses RSI", "default_params": {"period": 14}, "source_code": "pass"},
                    ]
                })
                user_msg = captured["messages"][1]["content"]
                assert "RSI Strategy" in user_msg
                assert "Strategy Examples" in user_msg

    def test_context_with_strategy_examples_non_dict(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "strategy_examples": ["just a string example"],
                })
                user_msg = captured["messages"][1]["content"]
                assert "just a string example" in user_msg

    def test_context_with_winner_examples(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "winner_examples": [
                        {"name": "Winner1", "sharpe": 2.5, "trades": 100, "ticker": "AAPL", "source_code": "pass"},
                    ]
                })
                user_msg = captured["messages"][1]["content"]
                assert "Winner1" in user_msg
                assert "AAPL" in user_msg
                assert "2.50" in user_msg
                assert "Proven Strategies" in user_msg

    def test_context_with_indicator_reference(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "indicator_reference": "RSI(close, length=14) -> Series",
                })
                system_msg = captured["messages"][0]["content"]
                assert "RSI(close, length=14)" in system_msg

    def test_context_with_archetype_section(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "archetype_section": "Use mean reversion with Bollinger Bands",
                })
                user_msg = captured["messages"][1]["content"]
                assert "Strategy Archetype Template" in user_msg
                assert "mean reversion" in user_msg

    def test_context_with_stagnation_recovery(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "stagnation_recovery": "Try a completely different approach!",
                })
                user_msg = captured["messages"][1]["content"]
                assert "completely different approach" in user_msg

    def test_context_with_multi_ticker_feedback(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "multi_ticker_feedback": "Strategy failed on 3 of 5 tickers",
                })
                user_msg = captured["messages"][1]["content"]
                assert "failed on 3 of 5 tickers" in user_msg

    def test_context_with_indicator_quick_ref(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "indicator_quick_ref": "rsi(close, length=14)",
                })
                user_msg = captured["messages"][1]["content"]
                assert "Indicator Quick Reference" in user_msg
                assert "rsi(close, length=14)" in user_msg

    def test_context_with_mandate(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "mandate": {"focus": "momentum", "max_trades": 50},
                })
                user_msg = captured["messages"][1]["content"]
                assert "Mandate" in user_msg
                assert "momentum" in user_msg

    def test_context_with_strategy_catalog(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({
                    "strategy_catalog": {"strategies": ["rsi", "macd"]},
                })
                user_msg = captured["messages"][1]["content"]
                assert "Available Strategies" in user_msg

    def test_parallel_variant_bias_injected(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                with patch("crabquant.refinement.prompts.get_variant_bias_text", return_value="VARIANT BIAS: be aggressive"):
                    call_llm_inventor({
                        "parallel_variant_index": 2,
                        "parallel_variant_count": 4,
                    })
                    user_msg = captured["messages"][1]["content"]
                    assert "VARIANT BIAS: be aggressive" in user_msg

    def test_parallel_variant_bias_skipped_when_count_is_one(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                with patch("crabquant.refinement.prompts.get_variant_bias_text") as mock_bias:
                    call_llm_inventor({
                        "parallel_variant_index": 0,
                        "parallel_variant_count": 1,
                    })
                    # Should not call get_variant_bias_text when count == 1
                    mock_bias.assert_not_called()

    def test_parallel_variant_bias_import_error_handled(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = self._mock_llm_response({"action": "novel"})

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                with patch("crabquant.refinement.prompts.get_variant_bias_text", side_effect=ImportError):
                    # Should not crash if get_variant_bias_text raises ImportError
                    result = call_llm_inventor({
                        "parallel_variant_index": 0,
                        "parallel_variant_count": 3,
                    })
                    assert result is not None

    def test_custom_model_and_max_tokens(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["model"] = json["model"]
            captured["max_tokens"] = json["max_tokens"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({"failure_mode": "test"}, model="glm-5.1", max_tokens=4096)
                assert captured["model"] == "glm-5.1"
                assert captured["max_tokens"] == 4096

    def test_system_message_contains_indicator_mistakes(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({})
                system_msg = captured["messages"][0]["content"]
                assert "atr(close, length=14)" in system_msg
                assert "stoch(close, k=14, d=3)" in system_msg
                assert "adx(close, length=14)" in system_msg
                assert "TOP 3 INDICATOR MISTAKES" in system_msg

    def test_context_with_previous_attempts(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        captured = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def mock_post(url, json=None, headers=None):
            captured["messages"] = json["messages"]
            return self._mock_llm_response({"action": "novel"})

        mock_client.post.side_effect = mock_post

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                with patch("crabquant.refinement.prompts.format_previous_attempts_section", return_value="Formatted attempts"):
                    call_llm_inventor({
                        "previous_attempts": [{"action": "modify"}],
                    })
                    user_msg = captured["messages"][1]["content"]
                    assert "Previous Attempts" in user_msg
                    assert "Formatted attempts" in user_msg

    def test_empty_context_still_works(self):
        from crabquant.refinement.llm_api import call_llm_inventor

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "test-key-123"}):
            with patch("crabquant.refinement.llm_api.httpx.Client",
                       return_value=self._mock_client(self._mock_llm_response({"action": "novel"}))):
                result = call_llm_inventor({})
                assert result["action"] == "novel"
