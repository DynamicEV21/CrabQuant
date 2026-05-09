# CrabQuant Cleanup Sprint

## Goal
Clean up the CrabQuant codebase: remove dead files, fix the failing test, organize accumulated artifacts, and update documentation to reflect the current state of the project (Phase 5.6 complete, Phase 6 next).

## Success Metrics
- All tests pass (0 failures) — fix the 1 failing test in test_backtest.py
- Root-level dead scripts removed (detailed_debug.py, fix_columns.py, simple_test_volume_roc_atr_trend.py, hello.py, probe_results.md, verify_results.md, validation_diagnosis.md, diagnose_validation.py)
- Root __pycache__/ removed
- refinement_runs/ bad_code directories archived or cleaned (keep last 10 per date, document the rest)
- README.md updated to reflect current architecture (loops system, 99 strategies, Phase 5.6 status)
- No git history lost — all changes committed on a feature branch

## Constraints
- Work on branch self-driving/cleanup-sprint, NOT phase5.6-overnight or master
- Do NOT modify any files under crabquant/ EXCEPT to fix the failing test in crabquant/engine/backtest.py
- Do NOT touch .gc/ directory (old GC city config)
- Do NOT run mandates or refinement loops — this is cleanup only
- Keep all strategy .py files in crabquant/strategies/ untouched
- The venv at .venv/ has all dependencies — activate it before running tests
- Test command: cd /home/Zev/development/CrabQuant && source .venv/bin/activate && python -m pytest tests/ -x -q

## Deadline
Complete within 2 hours of wall-clock time.
