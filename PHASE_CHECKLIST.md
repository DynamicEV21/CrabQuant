# Phase Completion Checklist

**This MUST be completed before declaring ANY phase done.**
**No exceptions. "Done" means tested.**

---

## After Every Phase

### 1. Unit Tests (automated — pre-commit hook)
```bash
cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/ -q --ignore=tests/refinement/test_e2e.py --ignore=tests/refinement/test_e2e_phase2.py --ignore=tests/refinement/test_e2e_phase3.py
```
- [ ] 0 failures
- [ ] Report pass count

### 2. E2E Integration Tests (fast, mocked)
```bash
cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_e2e.py tests/refinement/test_e2e_phase2.py tests/refinement/test_e2e_phase3.py -v --tb=short
```
- [ ] 0 failures
- [ ] Report pass count

### 3. Real LLM E2E (short — 3 turns, 1 mandate)
```bash
cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python scripts/refinement_loop.py --mandate mandates/momentum_aapl.json --max-turns 3 --sharpe-target 1.0
```
- [ ] Pipeline completes without crash
- [ ] Report final status and best Sharpe
- [ ] If crashes: fix, re-run, confirm fixed

### 4. Commit & Push
```bash
cd ~/development/CrabQuant && git add -A && git commit -m "phaseN: description" && git push
```
- [ ] Pre-commit hook passes (unit tests)
- [ ] Committed and pushed to GitHub

### 5. Report to Tristan
- [ ] Test counts (unit + E2E + real LLM)
- [ ] Commit hash
- [ ] Any issues found and fixed
- [ ] What's next

---

## Phase-Specific Checks

### Phase 4+ (daemon exists)
```bash
cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python scripts/run_pipeline.py --status
cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m crabquant.production.health
```
- [ ] Daemon status checked
- [ ] Health check passes

### Phase 5B+ (API budget exists)
- [ ] API call count within budget
- [ ] No rate limit errors in daemon log

---

## Rules
1. **Run steps 1-3 IN ORDER.** Unit tests first (fast), then E2E (medium), then real LLM (slow).
2. **If any step fails, FIX IT before moving on.** Do not skip.
3. **Report results to Tristan AFTER all 3 pass.** Never before.
4. **This checklist is non-negotiable.** Compaction, session reset, tired, doesn't matter.
