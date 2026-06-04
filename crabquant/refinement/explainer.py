"""Explainer Agent — LLM-powered economic rationale for winning strategies.

Enhancement 4: After a strategy passes validation, ask the LLM to explain
WHY it works — the market inefficiency, failure conditions, and key risks.
Stored alongside the winner entry for human review.
"""

import json
import logging

logger = logging.getLogger(__name__)


# ── Prompt Template ──────────────────────────────────────────────────────────

EXPLAINER_SYSTEM_PROMPT = (
    "You are a concise quantitative trading analyst. "
    "Explain the economic rationale of a trading strategy in plain English. "
    "Be specific and avoid vague generalizations."
)

EXPLAINER_USER_TEMPLATE = """Analyze this trading strategy and explain its economic rationale in under 200 words.

## Strategy Code
```python
{strategy_code}
```

## Backtest Metrics
- Ticker: {ticker}
- Sharpe Ratio: {sharpe}
- Total Return: {total_return}
- Max Drawdown: {max_drawdown}
- Number of Trades: {num_trades}
- Win Rate: {win_rate}
- Calmar Ratio: {calmar_ratio}
- Sortino Ratio: {sortino_ratio}
- Profit Factor: {profit_factor}

Address these three points specifically:
1. **Market Inefficiency**: What market inefficiency or structural edge does this strategy exploit?
2. **Failure Conditions**: Under what market conditions would this strategy break down or underperform?
3. **Key Risk Factors**: What are the primary risks (regime change, liquidity, slippage, overfitting, etc.)?

Keep the total explanation under 200 words. Be concise and specific."""


# ── Core Function ────────────────────────────────────────────────────────────

def explain_strategy(
    strategy_code: str,
    backtest_result: dict,
    ticker: str,
    *,
    _llm_call=None,
) -> str:
    """Ask the LLM to explain why a winning strategy works.

    Args:
        strategy_code: Full Python source code of the strategy.
        backtest_result: Dict of backtest metrics (sharpe, total_return,
            max_drawdown, num_trades, win_rate, calmar_ratio, sortino_ratio,
            profit_factor, etc.).
        ticker: Ticker symbol the strategy was tested on.
        _llm_call: Optional callable for testing (mock the LLM call).
            Signature: (messages: list[dict]) -> str.

    Returns:
        The LLM's explanation string, capped at ~250 words.
        Returns a fallback message if the LLM call fails.
    """
    # Build the user prompt
    metrics = {
        "sharpe": backtest_result.get("sharpe", "N/A"),
        "total_return": backtest_result.get("total_return", "N/A"),
        "max_drawdown": backtest_result.get("max_drawdown", "N/A"),
        "num_trades": backtest_result.get("num_trades", "N/A"),
        "win_rate": backtest_result.get("win_rate", "N/A"),
        "calmar_ratio": backtest_result.get("calmar_ratio", "N/A"),
        "sortino_ratio": backtest_result.get("sortino_ratio", "N/A"),
        "profit_factor": backtest_result.get("profit_factor", "N/A"),
    }

    user_content = EXPLAINER_USER_TEMPLATE.format(
        strategy_code=strategy_code,
        ticker=ticker,
        **metrics,
    )

    messages = [
        {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Allow injecting a mock for testing
    try:
        if _llm_call is not None:
            raw = _llm_call(messages)
        else:
            from crabquant.refinement.llm_api import call_zai_llm
            raw = call_zai_llm(
                messages=messages,
                model="glm-5-turbo",
                max_tokens=512,
                temperature=0.4,
            )
    except Exception as e:
        logger.warning("Explainer LLM call failed: %s", e)
        return _fallback_explanation(backtest_result, ticker)

    # Cap at ~250 words (generous buffer over the 200-word prompt instruction)
    return _cap_words(raw, max_words=250)


def _cap_words(text: str, max_words: int = 250) -> str:
    """Truncate text to max_words, breaking at the last complete sentence."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    truncated = " ".join(words[:max_words])

    # Try to break at the last sentence boundary
    for punct in (". ", "! ", "? "):
        last_boundary = truncated.rfind(punct)
        if last_boundary > 0:
            truncated = truncated[: last_boundary + 1]
            break

    return truncated.strip()


def _fallback_explanation(backtest_result: dict, ticker: str) -> str:
    """Return a generic explanation when the LLM call fails."""
    sharpe = backtest_result.get("sharpe", "N/A")
    trades = backtest_result.get("num_trades", "N/A")
    return (
        f"Strategy for {ticker} achieved a Sharpe ratio of {sharpe} "
        f"across {trades} trades. "
        "Detailed economic rationale unavailable (LLM call failed). "
        "Review the strategy code and backtest metrics manually."
    )


def build_explainer_prompt(
    strategy_code: str,
    backtest_result: dict,
    ticker: str,
) -> str:
    """Build the explainer prompt without calling the LLM.

    Useful for testing prompt construction and for logging.
    """
    metrics = {
        "sharpe": backtest_result.get("sharpe", "N/A"),
        "total_return": backtest_result.get("total_return", "N/A"),
        "max_drawdown": backtest_result.get("max_drawdown", "N/A"),
        "num_trades": backtest_result.get("num_trades", "N/A"),
        "win_rate": backtest_result.get("win_rate", "N/A"),
        "calmar_ratio": backtest_result.get("calmar_ratio", "N/A"),
        "sortino_ratio": backtest_result.get("sortino_ratio", "N/A"),
        "profit_factor": backtest_result.get("profit_factor", "N/A"),
    }

    return EXPLAINER_USER_TEMPLATE.format(
        strategy_code=strategy_code,
        ticker=ticker,
        **metrics,
    )
