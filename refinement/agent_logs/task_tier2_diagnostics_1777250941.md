# Build Task: tier2_diagnostics

## What to build
Tier 2 diagnostics: regime decomposition (bull/bear/sideways segments), top N drawdowns with dates, portfolio correlation analysis, benchmark comparison (buy-and-hold). Add compute_tier2_diagnostics() function.

## PRD Reference
Read /home/Zev/development/clawd/CRABQUANT_REFINEMENT_PIPELINE_PRD.md - specifically §Phase 3 - Tier 2 diagnostics

## Files to create/modify
Source: crabquant/refinement/diagnostics.py
Tests:  tests/refinement/test_diagnostics.py

## Rules
- TDD: write tests FIRST, then implement
- Use existing CrabQuant imports (crabquant.engine.backtest, crabquant.strategies, etc.)
- Activate venv: Use /home/Zev/development/QuantFactory/.venv/bin/python directly
- Run tests: cd /home/Zev/development/CrabQuant && /home/Zev/development/QuantFactory/.venv/bin/python -m pytest tests/refinement/test_diagnostics.py -v --tb=short
- Keep fixing until ALL tests pass
- Do NOT modify files outside the listed source/test files unless necessary for imports
- Do NOT modify BacktestEngine directly - document patches needed in a comment
- Use mock objects for LLM calls - do NOT make real API calls in tests
