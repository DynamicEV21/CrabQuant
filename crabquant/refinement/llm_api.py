"""
CrabQuant Refinement Pipeline — LLM API Integration

Handles all communication with z.ai (GLM-5) for strategy generation and refinement.
"""

import json
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


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
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout: int = 120,
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    
    if "choices" not in result or not result["choices"]:
        raise ValueError(f"Unexpected API response: {json.dumps(result)[:200]}")
    
    return result["choices"][0]["message"]["content"]


def extract_json_from_llm(text: str) -> dict:
    """Extract JSON from LLM response text.
    
    GLM-5-turbo wraps JSON in ```json ``` code blocks.
    This function tries multiple extraction strategies:
    1. Direct json.loads (rarely works with GLM-5)
    2. Code block extraction (common case)
    3. Brace matching fallback
    
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
    
    # Strategy 2: Code block extraction (```json ... ``` or ``` ... ```)
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 2b: Greedy code block (for large JSON objects)
    match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Brace matching fallback
    first_brace = text.find('{')
    if first_brace >= 0:
        depth = 0
        for i in range(first_brace, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[first_brace:i + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        continue
    
    raise ValueError(f"Could not extract JSON from LLM response (first 200: {text[:200]})")


def call_llm_inventor(
    context: dict,
    context_path: Optional[str] = None,
    model: str = "glm-5-turbo",
    max_tokens: int = 4096,
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
            "'new_strategy_code', 'params', and 'expected_impact' fields."
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
        return extract_json_from_llm(raw_response)
    except Exception as e:
        # Log but don't crash — return None so caller can retry
        import logging
        logging.getLogger(__name__).warning("LLM inventor call failed: %s", e)
        return None
