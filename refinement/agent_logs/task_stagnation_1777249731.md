# Build Task: stagnation

## What to build
Stagnation detection: scoring formula based on consecutive failed turns, Sharpe plateau, and lack of improvement. Response protocol — pivot (change strategy archetype) or abandon (mark as infeasible). Must integrate with orchestrator refinement_loop.

## PRD Reference
Read /home/Zev/development/clawd/CRABQUANT_REFINEMENT_PIPELINE_PRD.md — specifically §Phase 2 — Stagnation detection

## Files to create/modify
Source: crabquant/refinement/stagnation.py
Tests:  tests/refinement/test_stagnation.py

## Rules
- TDD: write tests FIRST, then implement
- Use existing CrabQuant imports (crabquant.engine.backtest, crabquant.strategies, etc.)
- Activate venv: source /home/Zev/development/QuantFactory/.venv/bin/activate
- Run tests: cd /home/Zev/development/CrabQuant && python -m pytest tests/refinement/test_stagnation.py -v --tb=short
- Keep fixing until ALL tests pass
- Do NOT modify files outside the listed source/test files unless necessary for imports
- Do NOT modify BacktestEngine directly — document patches needed in a comment
- Use mock objects for LLM calls — do NOT make real API calls in tests
