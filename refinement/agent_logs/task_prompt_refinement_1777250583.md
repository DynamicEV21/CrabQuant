# Build Task: prompt_refinement

## What to build
Refine LLM prompt based on Phase 1 observations. Include failure mode examples, better structure, explicit instruction to avoid common mistakes (empty signals, infinite loops). Create build_refinement_prompt() function.

## PRD Reference
Read /home/Zev/development/clawd/CRABQUANT_REFINEMENT_PIPELINE_PRD.md - specifically §Phase 2 - Better prompts

## Files to create/modify
Source: crabquant/refinement/prompts.py
Tests:  tests/refinement/test_prompts.py

## Rules
- TDD: write tests FIRST, then implement
- Use existing CrabQuant imports (crabquant.engine.backtest, crabquant.strategies, etc.)
- Activate venv: Use /home/Zev/development/QuantFactory/.venv/bin/python directly
- Run tests: cd /home/Zev/development/CrabQuant && /home/Zev/development/QuantFactory/.venv/bin/python -m pytest tests/refinement/test_prompts.py -v --tb=short
- Keep fixing until ALL tests pass
- Do NOT modify files outside the listed source/test files unless necessary for imports
- Do NOT modify BacktestEngine directly - document patches needed in a comment
- Use mock objects for LLM calls - do NOT make real API calls in tests
