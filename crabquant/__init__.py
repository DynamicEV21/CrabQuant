"""
CrabQuant — CodeCrab's Autonomous Strategy Engine

Clean, fast, direct. No LangGraph. No agent overhead.
Just Python + VectorBT + judgment.
"""

__version__ = "0.1.0"

from crabquant.guardrails import GuardrailConfig, GuardrailReport, check_guardrails, OverfittingDetector
from crabquant.regime import MarketRegime, detect_regime, get_strategy_ranking
