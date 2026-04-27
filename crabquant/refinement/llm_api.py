"""
CrabQuant Refinement Pipeline — LLM API Integration

Handles all communication with z.ai (GLM-5) for strategy generation and refinement.
"""

import json
import logging
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_api_config() -> dict:
    """Load z.ai credentials from OpenClaw config.
    
    Returns dict with 'base_url' and 'api_key'.
    Raises FileNotFoundError if config missing.
    Raises KeyError if z.ai provider not configured.
    """
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    with open(config_path) as f:
        cfg = json.load(f)
    p = cfg["models"]["providers"]["zai"]
    return {"base_url": p["baseUrl"], "api_key": p["apiKey"]}


def call_zai_llm(
    messages: list[dict],
    model: str = "glm-5-turbo",
    max_tokens: int = 8192,
    temperature: float = 0.7,
    timeout: int = 180,
) -> str:
    """Call z.ai API. Returns raw text response.
    
    Args:
        messages: OpenAI-format messages list
        model: model name (without zai/ prefix)
        max_tokens: max response tokens
        temperature: sampling temperature
        timeout: request timeout in seconds
    
    Returns:
        Raw text content from the API response.
    
    Raises:
        urllib.error.URLError: on network errors
        ValueError: on unexpected response format
    """
    cfg = load_api_config()
    url = f"{cfg['base_url']}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
    )
    # Retry loop for transient errors
    max_retries = 2
    backoff_times = [5, 10]
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            print(f"  Calling LLM...", flush=True)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
            elapsed = time.time() - t0
            print(f"  LLM responded in {elapsed:.1f}s", flush=True)
            break
        except urllib.error.HTTPError as e:
            # Only retry on 5xx server errors
            if e.code >= 500 and attempt < max_retries:
                last_error = e
                backoff = backoff_times[attempt]
                print(f"  LLM call failed (HTTP {e.code}), retry {attempt+1}/{max_retries} in {backoff}s...", flush=True)
                time.sleep(backoff)
                continue
            raise
        except urllib.error.URLError as e:
            if attempt < max_retries:
                last_error = e
                backoff = backoff_times[attempt]
                print(f"  LLM call timed out, retry {attempt+1}/{max_retries} in {backoff}s...", flush=True)
                time.sleep(backoff)
                continue
            raise
    else:
        raise last_error  # Should not reach here, but safety net
    
    if "choices" not in result or not result["choices"]:
        raise ValueError(f"Unexpected API response: {json.dumps(result)[:200]}")
    
    return result["choices"][0]["message"]["content"]


def extract_json_from_llm(text: str) -> dict:
    """Extract JSON from LLM response text.
    
    GLM-5-turbo wraps JSON in ```json ``` code blocks.
    This function tries multiple extraction strategies:
    1. Direct json.loads (rarely works with GLM-5)
    2. Code block extraction with brace matching
    3. Brace matching fallback (anywhere in text)
    
    Args:
        text: Raw LLM response text.
    
    Returns:
        Parsed dict from the JSON.
    
    Raises:
        ValueError: if no JSON can be extracted.
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Strategy 2: Extract code block content, then use brace matching on it
    # This handles the common case where GLM-5 wraps JSON in ```json ``` blocks.
    # We can't use simple regex on the JSON because Python code inside
    # new_strategy_code contains braces that confuse greedy/non-greedy matching.
    cb_match = re.search(r'```(?:json)?\s*([\\s\\S]*?)```', text)
    if cb_match:
        cb_content = cb_match.group(1).strip()
        extracted = _extract_json_by_braces(cb_content)
        if extracted is not None:
            return extracted
    
    # Strategy 3: Brace matching on full text
    extracted = _extract_json_by_braces(text)
    if extracted is not None:
        return extracted
    
    raise ValueError(f"Could not extract JSON from LLM response (first 200: {text[:200]}, len={len(text)})")


def _extract_json_by_braces(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from text using balanced brace matching.
    
    Correctly handles nested braces inside Python code strings.
    Returns None if no valid JSON object found.
    """
    first_brace = text.find('{')
    if first_brace < 0:
        return None
    
    # Find all brace positions for balanced matching
    depth = 0
    in_string = False
    string_char = None
    escaped = False
    
    for i in range(first_brace, len(text)):
        ch = text[i]
        
        if escaped:
            escaped = False
            continue
        
        if ch == '\\' and in_string:
            escaped = True
            continue
        
        if ch in ('"', "'") and not in_string:
            in_string = True
            string_char = ch
            continue
        
        if ch == string_char and in_string:
            in_string = False
            string_char = None
            continue
        
        if in_string:
            continue
        
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[first_brace:i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    # This brace balance didn't yield valid JSON;
                    # try starting from the next '{'
                    rest = _extract_json_by_braces(text[i + 1:])
                    return rest
    
    return None


def call_llm_inventor(
    context: dict,
    context_path: Optional[str] = None,
    model: str = "glm-5-turbo",
    max_tokens: int = 8192,
) -> Optional[dict]:
    """Call LLM to invent/modify a strategy. Returns parsed StrategyModification dict.
    
    Args:
        context: Dict with prompt context (built by context_builder).
        context_path: Optional path to save context JSON for debugging.
        model: LLM model name.
        max_tokens: Max response tokens.
    
    Returns:
        Parsed dict with strategy modification, or None on failure.
    """
    # Save context for debugging if path provided
    if context_path:
        Path(context_path).parent.mkdir(parents=True, exist_ok=True)
        Path(context_path).write_text(json.dumps(context, indent=2))
    
    # Build messages
    system_msg = {
        "role": "system",
        "content": (
            "You are a quantitative strategy researcher. You design trading strategies "
            "using technical indicators. You output valid JSON with strategy code. "
            "Always respond with a JSON object containing 'action', 'hypothesis', "
            "'new_strategy_code', 'params', and 'expected_impact' fields.\n\n"
            "CRITICAL RULES for new_strategy_code:\n"
            "1. Include DESCRIPTION (string variable), generate_signals(df, params), and DEFAULT_PARAMS.\n"
            "   Do NOT include generate_signals_matrix or PARAM_GRID.\n"
            "   Example: DESCRIPTION = 'Momentum strategy using RSI and MACD'\n"
            "2. Keep strategy code under 80 lines. Be concise — no excessive comments.\n"
            "3. Use from crabquant.indicator_cache import cached_indicator for indicators.\n"
            "4. generate_signals MUST return (entries: pd.Series[bool], exits: pd.Series[bool]).\n"
            "5. Ensure the JSON is COMPLETE — never truncate your response."
        ),
    }
    
    # Build user message from context
    user_parts = []
    if "prompt" in context:
        user_parts.append(context["prompt"])
    else:
        # Construct prompt from context fields
        if context.get("current_strategy_code"):
            user_parts.append(f"## Current Strategy Code\n```python\n{context['current_strategy_code']}\n```\n")
        
        if context.get("backtest_report"):
            user_parts.append(f"## Backtest Report\n{json.dumps(context['backtest_report'], indent=2)}\n")
        
        if context.get("failure_mode"):
            user_parts.append(f"## Failure Classification: {context['failure_mode']}\n")
            user_parts.append(f"## Failure Reasoning: {context.get('failure_reasoning', 'N/A')}\n")
        
        if context.get("previous_attempts"):
            user_parts.append(f"## Previous Attempts\n{json.dumps(context['previous_attempts'], indent=2)}\n")
        
        if context.get("strategy_examples"):
            user_parts.append(f"## Strategy Examples\n{context['strategy_examples']}\n")
        
        if context.get("strategy_catalog"):
            user_parts.append(f"## Available Strategies\n{json.dumps(context['strategy_catalog'], indent=2)}\n")
        
        if context.get("mandate"):
            user_parts.append(f"## Mandate\n{json.dumps(context['mandate'], indent=2)}\n")
    
    user_msg = {"role": "user", "content": "\n".join(user_parts)}
    
    try:
        raw_response = call_zai_llm(
            messages=[system_msg, user_msg],
            model=model,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        # Save raw response for debugging
        if context_path:
            Path(context_path).parent.mkdir(parents=True, exist_ok=True)
            raw_path = Path(context_path).parent / (Path(context_path).stem + "_raw_response.txt")
            raw_path.write_text(raw_response)
        result = extract_json_from_llm(raw_response)
        if result is None:
            logger.warning("LLM returned empty/unparseable response (len=%d)", len(raw_response) if raw_response else 0)
        return result
    except Exception as e:
        # Log but don't crash — return None so caller can retry
        logger.warning("LLM inventor call failed: %s", e)
        return None
