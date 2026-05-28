"""Tests for crabquant.refinement.llm_api"""

import json
import pytest
from unittest.mock import patch, MagicMock
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
                   return_value={"base_url": "https://api.test.com", "api_key": "key"}):
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
                   return_value={"base_url": "https://api.test.com", "api_key": "key"}):
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
                   return_value={"base_url": "https://api.test.com", "api_key": "key"}):
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
                   return_value={"base_url": "https://api.test.com", "api_key": "key"}):
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
                   return_value={"base_url": "https://api.test.com", "api_key": "key"}):
            with patch("crabquant.refinement.llm_api.time.sleep") as mock_sleep:
                with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                    result = call_zai_llm([{"role": "user", "content": "hi"}])
                    assert result == "OK after retries"
                    assert mock_client.post.call_count == 3
                    assert mock_sleep.call_count == 2


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
                   return_value={"base_url": "https://api.test.com", "api_key": "key"}):
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
                   return_value={"base_url": "https://api.test.com", "api_key": "key"}):
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
                   return_value={"base_url": "https://api.test.com", "api_key": "key"}):
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
                   return_value={"base_url": "https://api.test.com", "api_key": "key"}):
            with patch("crabquant.refinement.llm_api.httpx.Client", return_value=mock_client):
                call_llm_inventor({"prompt": "Invent a strategy using RSI"})
                user_msg = captured["messages"][1]["content"]
                assert "Invent a strategy using RSI" in user_msg

    def test_saves_context_to_file(self, tmp_path):
        from crabquant.refinement.llm_api import call_llm_inventor

        context_path = str(tmp_path / "context.json")
        context = {"failure_mode": "low_sharpe", "backtest_report": {"sharpe": 0.5}}

        with patch("crabquant.refinement.llm_api.load_api_config",
                   return_value={"base_url": "https://api.test.com", "api_key": "key"}):
            with patch("crabquant.refinement.llm_api.httpx.Client",
                       return_value=self._mock_client(self._mock_llm_response({"action": "novel"}))):
                call_llm_inventor(context, context_path=context_path)

        saved = json.loads(Path(context_path).read_text())
        assert saved["failure_mode"] == "low_sharpe"
