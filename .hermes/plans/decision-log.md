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
