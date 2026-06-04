# CrabQuant Overnight Decision Log

## 2026-04-29 Session

### Commit 2789a2d: Fix 3 HIGH-severity refinement loop bugs

**Bug 1 — action_analytics timing (HIGH)**
- Problem: `action_analytics` was computed in `refinement_loop.py` lines 690-697 AFTER `build_llm_context()` returned, so it was never available during prompt assembly. The append_sections in context_builder checked `context.get("action_analytics")` but it was always None at that point.
- Fix: Moved computation inside `context_builder.py` `build_llm_context()` near the other Phase 6 computations (~line 467). Removed redundant post-hoc code from refinement_loop.py.

**Bug 2 — stagnation_recovery not appended (HIGH)**
- Fixed in prior commit (a5b7231): added `stagnation_recovery` to the `append_sections` list.

**Bug 3 — validation_failed override not persisted (HIGH)**
- Problem: In `refinement_loop.py`, when `validation_failed` overrides the history entry's `failure_mode` at line 1224, `save_state()` was never called again. The state file on disk still had the old failure_mode, so the LLM got stale feedback on the next turn.
- Fix: Added `save_state(run_dir, state)` after the override.

**Bonus fix — append_sections on exception path**
- Problem: When `build_turn1_prompt` raised an exception, `context["prompt"]` was never set, but `append_sections` still ran and created a prompt from just `action_analytics` text (e.g., "No historical action data available."). Test `test_turn1_prompt_graceful_on_exception` caught this.
- Fix: Wrapped append_sections in `if prompt:` guard so it only runs when a prompt was actually built.

**Test results**: 4568 passed, 1 skipped, 0 failures.

---

## 2026-04-30 07:15 UTC — Cycle 20: Wire Dead Code Modules (code_quality_check + positive_feedback)

### Commit 354070e: Wire code_quality_check into refinement loop before backtesting
- **Problem**: `code_quality_check.py` module existed (491 lines, 786 tests) with `check_code_quality()` and `format_code_quality_for_prompt()` but was never imported or called in `scripts/refinement_loop.py` or `context_builder.py`. Dead code.
- **Fix**: Added import in `refinement_loop.py`. Inserted code quality pre-check block after validation gates pass but before backtesting:
  - Reject verdict (score < 0.50): skips backtesting, stores feedback on `state.code_quality_feedback`, records `code_quality_rejected` in history
  - Warning verdict (score < 0.75): proceeds to backtesting but stores feedback for LLM
  - Good verdict: proceeds, clears previous feedback
- Added `code_quality_feedback: str = ""` field to `RunState` in `schemas.py`
- Wired feedback into `context_builder.py` append_sections — appears as "## ⛔ CODE QUALITY PRE-CHECK FAILED" in prompt
- 24 integration tests in `test_code_quality_integration.py`

### Commit e881a9e: Wire positive_feedback analyzer into refinement loop context
- **Problem**: `positive_feedback.py` module existed with `analyze_positive_feedback()` and `format_positive_feedback_for_prompt()` but was never imported or called in `context_builder.py`. Dead code.
- **Fix**: Wired into `context_builder.py`'s `build_llm_context()`:
  - Calls `analyze_positive_feedback()` on latest backtest report
  - Scans up to 3 historical successful turns (where sharpe_target_hit=True)
  - Sets `positive_feedback_section` in context dict
  - Only emits section when strengths are non-empty (avoids noise on early iterations)
  - Placed in append_sections between param_optimization and gate_validation (positive-before-negative framing)
- 21 integration tests in `test_positive_feedback_integration.py`

### Impact
- Both modules were built in prior cycles but never wired — classic "build but not integrate" gap
- Code quality pre-check catches anti-patterns (flat signals, no generate_signals, over-complexity) BEFORE wasting a backtest
- Positive feedback prevents LLM regression by reinforcing what worked
- Test suite: 4867 → 4912 passed (+45 new integration tests)

### Live Mandate
- Running `momentum_aapl` mandate (sharpe_target=1.5, max_turns=5) to verify wiring end-to-end
