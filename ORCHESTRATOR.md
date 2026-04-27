# ORCHESTRATOR.md — Execution Plan

**Last Updated:** 2026-04-27
**Current Phase:** phase_5b
**Status:** not_started

---

## Overview

This document orchestrates the execution of Phases 5B, 6, and 7 of CrabQuant. Each phase has clearly defined steps with dependencies, agent allocations, deliverables, and failure handling.

**Execution model:** Steps are executed by spawning sub-agents (max 4 parallel). Steps with no dependencies on each other CAN run in parallel. Within a phase, steps are ordered by dependency.

**Current state:**
- 688 unit tests passing
- 27 refinement modules built
- Daemon running (PID 3759494, wave 7, 13 mandates run, 0 promoted)
- Phase 5A (daemon core) complete
- Phase 4.5 (convergence tuning) in progress

---

### Phase 5B: Intelligence & Reliability
**PRD:** PHASE5B_PRD.md
**Description:** API budget tracker, resource limiter, auto-mandate generation, status reporting

**Done Criteria:**
- [ ] Follow PHASE_CHECKLIST.md (unit tests + E2E + real LLM + commit + report)
- [ ] All 4 components built and tested
- [ ] Daemon restarted with new features
- [ ] Status report successfully sent to Telegram

**Steps:**

1. **step_01_api_budget_tracker** (Tier 2)
   - Create `crabquant/refinement/api_budget.py`
   - Track prompt count per day/week, throttle to glm-4.7 at 80% budget, alert at 90%
   - Implement `ApiBudgetTracker` class with `record_prompt()`, `should_throttle()`, `get_recommended_model()`, persistence
   - Agent allocation: 1 sub-agent
   - Deliverables: `crabquant/refinement/api_budget.py`, `tests/refinement/test_api_budget.py` (10+ tests)
   - Dependencies: None
   - On failure: retry × 2, then escalate to Tristan
   - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_api_budget.py -v`

2. **step_02_resource_limiter** (Tier 2)
   - Create `crabquant/refinement/resource_limiter.py`
   - Monitor CPU/RAM/disk, adjust parallel count dynamically, pause if RAM < 2GB
   - Implement `ResourceLimiter` class with `check_resources()`, `get_recommended_parallel()`, `should_pause()`
   - Use psutil with /proc/meminfo fallback
   - Agent allocation: 1 sub-agent
   - Deliverables: `crabquant/refinement/resource_limiter.py`, `tests/refinement/test_resource_limiter.py` (10+ tests)
   - Dependencies: None (can run in parallel with step_01)
   - On failure: retry × 2, then escalate to Tristan
   - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_resource_limiter.py -v`

3. **step_03_auto_mandate_enhancement** (Tier 2)
   - Enhance `crabquant/refinement/mandate_generator.py`
   - Add `generate_smart_mandates()` using market regime, completed mandate tracking, portfolio gap analysis
   - Add `get_portfolio_gaps()` to identify underrepresented archetypes
   - Integrate with `crabquant/regime.py` for current market conditions
   - Agent allocation: 1 sub-agent
   - Deliverables: Enhanced `mandate_generator.py`, `tests/refinement/test_mandate_generator_enhanced.py` (8+ tests)
   - Dependencies: None (can run in parallel with steps 01-02)
   - On failure: retry × 2, then escalate to Tristan
   - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_mandate_generator_enhanced.py -v`

4. **step_05_integration_wiring** (Tier 2)
   - Wire all Phase 5B components into the daemon pipeline
   - In `crabquant/refinement/llm_api.py`: call `budget_tracker.record_prompt()` after each LLM call; use `get_recommended_model()` for model selection
   - In `scripts/run_pipeline.py`: call `resource_limiter.check_resources()` at wave start; use `recommended_parallel`; call `generate_smart_mandates()` when queue empty
   - In `crabquant/refinement/state.py`: add `api_budget_used_today` and `api_budget_throttled` fields to DaemonState
   - Agent allocation: 1 sub-agent
   - Deliverables: Updated `llm_api.py`, `run_pipeline.py`, `state.py`; no new test files but integration must not break existing tests
   - Dependencies: step_01_api_budget_tracker, step_02_resource_limiter, step_03_auto_mandate_enhancement
   - On failure: review integration points, fix conflicts, retry × 1, escalate to Tristan
   - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/ -q`

5. **step_06_status_reporter** (Tier 2)
   - Create `crabquant/refinement/status_reporter.py`
   - Generate daily status reports: daemon health, mandate stats, API budget, resources, convergence
   - Format as Telegram-friendly markdown (< 4096 chars)
   - Send via OpenClaw Telegram channel
   - Agent allocation: 1 sub-agent
   - Deliverables: `crabquant/refinement/status_reporter.py`, `tests/refinement/test_status_reporter.py` (8+ tests)
   - Dependencies: step_01_api_budget_tracker (reads budget status), step_02_resource_limiter (reads resource status)
   - On failure: retry × 2, then escalate to Tristan
   - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_status_reporter.py -v`

6. **step_07_phase5b_e2e_and_checklist** (Tier 3)
   - Run PHASE_CHECKLIST.md for Phase 5B
   - Unit tests: `python -m pytest tests/refinement/ -q --ignore=tests/refinement/test_e2e.py --ignore=tests/refinement/test_e2e_phase2.py --ignore=tests/refinement/test_e2e_phase3.py`
   - E2E tests: `python -m pytest tests/refinement/test_e2e.py tests/refinement/test_e2e_phase2.py tests/refinement/test_e2e_phase3.py -v --tb=short`
   - Real LLM E2E: `python scripts/refinement_loop.py --mandate mandates/momentum_aapl.json --max-turns 3 --sharpe-target 1.0`
   - Commit and push: `git add -A && git commit -m "phase5b: intelligence and reliability" && git push`
   - Verify status report sends to Telegram
   - Agent allocation: 1 sub-agent
   - Deliverables: All tests passing, commit pushed, status report sent
   - Dependencies: step_05_integration_wiring, step_06_status_reporter
   - On failure: fix failing tests, re-run, escalate if persistent

---

### Phase 6: Intelligence Layer
**PRD:** PHASE6_PRD.md
**Description:** Action analytics feedback, adaptive prompts, strategy decay, portfolio correlation, mandate prioritization

**Done Criteria:**
- [ ] Follow PHASE_CHECKLIST.md (unit tests + E2E + real LLM + commit + report)
- [ ] Action analytics data flows into LLM context (verify by inspection)
- [ ] Adaptive prompts include regime + portfolio gap context
- [ ] Portfolio correlation blocks highly correlated promotions
- [ ] Strategy decay detection runs and flags decaying strategies
- [ ] Mandate prioritization produces different orderings than FIFO

**Steps:**

7. **step_08_action_analytics_feedback** (Tier 2)
   - Enhance `crabquant/refinement/action_analytics.py`
   - Add `get_failure_mode_action_stats()` with failure-mode-specific success rates
   - Add `format_action_feedback_for_context()` for LLM context injection
   - Wire into `crabquant/refinement/context_builder.py` for Turn 2+ context
   - Agent allocation: 1 sub-agent
   - Deliverables: Enhanced `action_analytics.py`, `context_builder.py`, `tests/refinement/test_action_analytics_enhanced.py` (8+ tests)
   - Dependencies: Phase 5B complete
   - On failure: retry × 2, then escalate to Tristan
   - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_action_analytics_enhanced.py -v`

8. **step_09_failure_pattern_analysis** (Tier 2)
   - Enhance `crabquant/refinement/classifier.py`
   - Add `analyze_failure_patterns()` for failure mode distribution across runs
   - Add `get_auto_adjustments()` to generate threshold/prompt adjustments
   - Add `correlation_reject` failure mode
   - Agent allocation: 1 sub-agent
   - Deliverables: Enhanced `classifier.py`, `tests/refinement/test_failure_patterns.py` (8+ tests)
   - Dependencies: step_08_action_analytics_feedback (uses analytics data)
   - On failure: retry × 2, then escalate to Tristan
   - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_failure_patterns.py -v`

9. **step_10_adaptive_prompts** (Tier 2)
   - Enhance `crabquant/refinement/prompts.py`
   - Add `build_adaptive_invention_prompt()` with regime context, portfolio gaps, action stats
   - 20% control group (base prompts) for comparison
   - Cap adaptive additions at 500 tokens
   - Agent allocation: 1 sub-agent
   - Deliverables: Enhanced `prompts.py`, `tests/refinement/test_adaptive_prompts.py` (8+ tests)
   - Dependencies: step_08_action_analytics_feedback, step_09_failure_pattern_analysis
   - On failure: retry × 2, then escalate to Tristan
   - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_adaptive_prompts.py -v`

10. **step_11_portfolio_correlation_promotion** (Tier 2)
    - Enhance `crabquant/refinement/portfolio_correlation.py`
    - Add `compute_strategy_correlation()` with rolling 90-day window
    - Add `should_promote_with_correlation()` as promotion gate (reject >0.8 correlation unless Sharpe 50% higher)
    - Add `compute_portfolio_diversification_score()`
    - Enhance `crabquant/refinement/diagnostics.py` to save equity curves during backtest
    - Wire into `crabquant/refinement/promotion.py` — check correlation before registering
    - Agent allocation: 1 sub-agent
    - Deliverables: Enhanced `portfolio_correlation.py`, `diagnostics.py`, `promotion.py`, `tests/refinement/test_portfolio_correlation_enhanced.py` (8+ tests)
    - Dependencies: None within Phase 6 (can run in parallel with steps 08-10)
    - On failure: retry × 2, then escalate to Tristan
    - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_portfolio_correlation_enhanced.py -v`

11. **step_12_strategy_decay_detection** (Tier 2)
    - Enhance `crabquant/production/scanner.py`
    - Add `check_strategy_decay()` — backtest promoted strategies on recent data, compare to promotion Sharpe
    - Add `check_all_strategies_decay()` — check all promoted strategies, require 3 consecutive below-threshold checks
    - Add retirement function to `crabquant/production/promoter.py`
    - Agent allocation: 1 sub-agent
    - Deliverables: Enhanced `scanner.py`, `promoter.py`, `tests/refinement/test_strategy_decay.py` (8+ tests)
    - Dependencies: step_11_portfolio_correlation_promotion (needs promotion pipeline)
    - On failure: retry × 2, then escalate to Tristan
    - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_strategy_decay.py -v`

12. **step_13_mandate_prioritization** (Tier 2)
    - Enhance `crabquant/refinement/mandate_generator.py`
    - Add `score_mandate()` with convergence probability + portfolio gap + regime match
    - Add `prioritize_mandates()` for sorted mandate selection
    - Wire into `scripts/run_pipeline.py` — pick next mandates by score, not FIFO
    - Agent allocation: 1 sub-agent
    - Deliverables: Enhanced `mandate_generator.py`, `run_pipeline.py`, `tests/refinement/test_mandate_prioritization.py` (8+ tests)
    - Dependencies: step_11_portfolio_correlation_promotion (uses portfolio gap data)
    - On failure: retry × 2, then escalate to Tristan
    - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_mandate_prioritization.py -v`

13. **step_14_phase6_e2e_and_checklist** (Tier 3)
    - Run PHASE_CHECKLIST.md for Phase 6
    - Unit tests: `python -m pytest tests/refinement/ -q --ignore=tests/refinement/test_e2e.py --ignore=tests/refinement/test_e2e_phase2.py --ignore=tests/refinement/test_e2e_phase3.py`
    - E2E tests: `python -m pytest tests/refinement/test_e2e.py tests/refinement/test_e2e_phase2.py tests/refinement/test_e2e_phase3.py -v --tb=short`
    - Real LLM E2E: `python scripts/refinement_loop.py --mandate mandates/momentum_aapl.json --max-turns 3 --sharpe-target 1.0`
    - Verify action analytics context appears in Turn 2+ LLM calls (inspect logs)
    - Verify adaptive prompt includes regime context (inspect logs)
    - Commit and push: `git add -A && git commit -m "phase6: intelligence layer" && git push`
    - Agent allocation: 1 sub-agent
    - Deliverables: All tests passing, commit pushed, intelligence features verified
    - Dependencies: All Phase 6 steps (08-13)
    - On failure: fix failing tests, re-run, escalate if persistent

---

### Phase 7: Deployment Readiness
**PRD:** PHASE7_PRD.md
**Description:** Slippage integration, walk-forward in loop, regime validation, multi-timeframe, paper trading, dashboard

**Done Criteria:**
- [ ] Follow PHASE_CHECKLIST.md (unit tests + E2E + real LLM + commit + report)
- [ ] Slippage check integrated into refinement loop (at least 1 strategy fails slippage)
- [ ] Walk-forward check runs at Turn 4+
- [ ] Regime-aware validation runs at promotion time
- [ ] Paper trading engine runs with ≥3 strategies
- [ ] Dashboard sends daily report to Telegram

**Steps:**

14. **step_15_slippage_integration** (Tier 2)
    - Create slippage check function in `crabquant/refinement/diagnostics.py`
    - Implement `quick_slippage_check()` — apply 5bp commission + 1-tick slippage post-hoc
    - Wire into `crabquant/refinement/promotion.py` — require slippage pass before promotion
    - Wire into `crabquant/refinement/orchestrator.py` — include slippage degradation in Turn 2+ context
    - Agent allocation: 1 sub-agent
    - Deliverables: Enhanced `diagnostics.py`, `promotion.py`, `orchestrator.py`, `tests/refinement/test_slippage_integration.py` (8+ tests)
    - Dependencies: Phase 6 complete
    - On failure: retry × 2, then escalate to Tristan
    - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_slippage_integration.py -v`

15. **step_16_walkforward_and_regime_validation** (Tier 2)
    - Add `quick_walk_forward_check()` to `crabquant/refinement/diagnostics.py`
    - Wire into `crabquant/refinement/orchestrator.py` — trigger at Turn 4+ if Sharpe ≥80% target
    - Add `regime_aware_validation()` to `crabquant/validation/__init__.py`
    - Wire into `crabquant/refinement/promotion.py` — require ≥2 regimes with positive Sharpe
    - Agent allocation: 1 sub-agent
    - Deliverables: Enhanced `diagnostics.py`, `orchestrator.py`, `validation/__init__.py`, `promotion.py`, `tests/refinement/test_quick_walkforward.py` (6+ tests), `tests/refinement/test_regime_validation.py` (6+ tests)
    - Dependencies: step_15_slippage_integration (both modify promotion.py — do after)
    - On failure: retry × 2, then escalate to Tristan
    - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_quick_walkforward.py tests/refinement/test_regime_validation.py -v`

16. **step_17_multi_timeframe_support** (Tier 2)
    - Add `load_multi_timeframe_data()` to `crabquant/data/__init__.py`
    - Update strategy interface to accept `dict[str, DataFrame]` for multi-timeframe
    - Ensure backward compatibility with existing strategies (single DataFrame still works)
    - Add multi-timeframe strategy examples to `crabquant/refinement/prompts.py`
    - Create 1 multi-timeframe mandate in `mandates/`
    - Agent allocation: 1 sub-agent
    - Deliverables: Enhanced `data/__init__.py`, `prompts.py`, `tests/refinement/test_multi_timeframe.py` (8+ tests), 1 multi-timeframe mandate JSON
    - Dependencies: None within Phase 7 (can run in parallel with steps 15-16)
    - On failure: retry × 2, then escalate to Tristan
    - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_multi_timeframe.py -v`

17. **step_18_paper_trading_engine** (Tier 2)
    - Create `crabquant/paper_trading/` directory
    - Implement `PaperTradingEngine` with virtual portfolio, signal generation, cash management
    - Implement `PaperPortfolio` and `PaperPosition` dataclasses
    - Use Yahoo Finance delayed data as default source
    - Max 10% per position, hold on data fetch failure
    - Agent allocation: 1 sub-agent
    - Deliverables: `crabquant/paper_trading/__init__.py`, `crabquant/paper_trading/engine.py`, `tests/refinement/test_paper_trading.py` (8+ tests)
    - Dependencies: step_15_slippage_integration (paper trading uses slippage model)
    - On failure: retry × 2, then escalate to Tristan
    - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_paper_trading.py -v`

18. **step_19_telegram_dashboard** (Tier 2)
    - Create `crabquant/dashboard/` directory
    - Implement `TelegramDashboard` with daily performance, text charts, position tables, alerts
    - Include drawdown alerts, decay alerts, signal alerts
    - Integrate with paper trading engine and status reporter
    - Agent allocation: 1 sub-agent
    - Deliverables: `crabquant/dashboard/__init__.py`, `crabquant/dashboard/telegram_dashboard.py`, `tests/refinement/test_telegram_dashboard.py` (8+ tests)
    - Dependencies: step_18_paper_trading_engine (dashboard displays paper portfolio)
    - On failure: retry × 2, then escalate to Tristan
    - Test command: `cd ~/development/CrabQuant && source ~/development/QuantFactory/.venv/bin/activate && python -m pytest tests/refinement/test_telegram_dashboard.py -v`

19. **step_20_phase7_e2e_and_checklist** (Tier 3)
    - Run PHASE_CHECKLIST.md for Phase 7
    - Unit tests: `python -m pytest tests/refinement/ -q --ignore=tests/refinement/test_e2e.py --ignore=tests/refinement/test_e2e_phase2.py --ignore=tests/refinement/test_e2e_phase3.py`
    - E2E tests: `python -m pytest tests/refinement/test_e2e.py tests/refinement/test_e2e_phase2.py tests/refinement/test_e2e_phase3.py -v --tb=short`
    - Real LLM E2E: `python scripts/refinement_loop.py --mandate mandates/momentum_aapl.json --max-turns 3 --sharpe-target 1.0`
    - Verify slippage check runs and at least 1 strategy fails (inspect logs)
    - Verify walk-forward check triggers at Turn 4+ (inspect logs)
    - Verify paper trading engine starts and generates signals
    - Verify dashboard sends report to Telegram
    - Commit and push: `git add -A && git commit -m "phase7: deployment readiness" && git push`
    - Agent allocation: 1 sub-agent
    - Deliverables: All tests passing, commit pushed, all Phase 7 features verified
    - Dependencies: All Phase 7 steps (15-19)
    - On failure: fix failing tests, re-run, escalate if persistent

---

## Execution Summary

| Step | Phase | Component | Parallel Group | Dependencies |
|------|-------|-----------|---------------|-------------|
| 01 | 5B | API Budget Tracker | A | None |
| 02 | 5B | Resource Limiter | A | None |
| 03 | 5B | Auto-Mandate Enhancement | A | None |
| 05 | 5B | Integration Wiring | B | 01, 02, 03 |
| 06 | 5B | Status Reporter | B | 01, 02 |
| 07 | 5B | Phase 5B E2E + Checklist | C | 05, 06 |
| 08 | 6 | Action Analytics Feedback | D | Phase 5B |
| 09 | 6 | Failure Pattern Analysis | E | 08 |
| 10 | 6 | Adaptive Prompts | E | 08, 09 |
| 11 | 6 | Portfolio Correlation Promotion | D | Phase 5B |
| 12 | 6 | Strategy Decay Detection | F | 11 |
| 13 | 6 | Mandate Prioritization | F | 11 |
| 14 | 6 | Phase 6 E2E + Checklist | G | 08-13 |
| 15 | 7 | Slippage Integration | H | Phase 6 |
| 16 | 7 | Walk-Forward + Regime Validation | I | 15 |
| 17 | 7 | Multi-Timeframe Support | H | Phase 6 |
| 18 | 7 | Paper Trading Engine | J | 15 |
| 19 | 7 | Telegram Dashboard | J | 18 |
| 20 | 7 | Phase 7 E2E + Checklist | K | 15-19 |

**Parallel groups** (steps that CAN run simultaneously):
- **Group A**: steps 01, 02, 03 (3 parallel agents)
- **Group D**: steps 08, 11 (2 parallel agents)
- **Group H**: steps 15, 17 (2 parallel agents)

**Total steps: 17** (across 3 phases)
**Estimated new tests: 135+** (across all 3 phases)
**Estimated effort: 13-19 sessions** (3 phases combined)

---

## Global Rules

1. **Venv activation**: Every command must start with `source ~/development/QuantFactory/.venv/bin/activate`
2. **Test command**: `cd ~/development/CrabQuant && python -m pytest tests/refinement/ -q`
3. **Current baseline**: 688 tests passing — every step must maintain this
4. **Commit after each phase** (not each step) — use PHASE_CHECKLIST.md
5. **No breaking changes**: Existing strategies, mandates, and pipeline must continue working
6. **Sub-agent tier**: Most steps are Tier 2 (new function/enhancement). Phase E2E steps are Tier 3 (milestone).
7. **Max 3 concurrent sub-agents** (leave API headroom for running crons)
8. **Escalation**: If any step fails 2 retries, log to `results/orchestrator_failures.json` and continue. Only escalate to Tristan if: (a) phase E2E gate fails, or (b) 3+ steps fail in same phase
9. **Progress persistence:** Write `results/orchestrator_progress.json` after each step with: `{phase, step, status, timestamp, test_results, commit_hash}`
10. **Continuous execution:** After a phase passes PHASE_CHECKLIST, immediately start the next phase. Do not stop between phases.
11. **Night mode:** When running unattended, minimize Telegram messages. Only send: phase completion summary, critical failures, and the final "all phases done" report.
