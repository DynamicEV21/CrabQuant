# CrabQuant — Roadmap

**Last Updated:** 2026-04-26  
**Status:** Phase 1-3 refinement pipeline built (31/31 components passing), moving to Phase 4 integration

---

## 1. Current State Summary

CrabQuant has a proven backtest engine (28 strategies, vectorized sweep, walk-forward validation) and a complete refinement pipeline with 31 individually-tested components — including orchestrator, LLM interface, failure classifier, stagnation detection, wave manager, mandate generator, auto-promotion, and action analytics. However, these components are **not yet wired into a working end-to-end loop**. The system currently runs via four legacy cron agents (`crabquant-wave`, `crabquant-improve`, `crabquant-validate`, `crabquant-meta`) that do periodic parameter sweeps and invention, while the refinement pipeline sits unused. The immediate priority is Phase 4: connecting the refinement pipeline components into a continuous autonomous loop, running real mandates with actual LLM calls, and measuring convergence before moving to always-on production.

---

## 2. Phase 4: Integration & Wire-Up

**Goal:** Make the refinement pipeline run end-to-end continuously — real mandates, real LLM calls, real backtests — and measure whether it actually converges.

### Key Deliverables

| # | Deliverable | File(s) | Description |
|---|-------------|---------|-------------|
| 1 | **Fix orchestrator imports** | `crabquant/refinement/orchestrator.py` | The `refinement_loop.py` script imports from `context_builder`, `llm_api`, `config` etc. but there's no `orchestrator.py` module — the loop logic lives in the script. Extract into a proper `Orchestrator` class. |
| 2 | **Wire mandate → orchestrator → promotion** | `scripts/refinement_loop.py` → `crabquant/refinement/auto_promotion.py` → `crabquant/strategies/__init__.py` | After orchestrator exits (success or max-turns), call `auto_promotion.promote_if_worthy()` to register winners. Currently `refinement_loop.py` doesn't import or call promotion. |
| 3 | **Wire wave_manager → refinement_loop** | `crabquant/refinement/wave_manager.py` | `run_waves()` spawns subprocesses calling `refinement_loop.py`. Verify subprocess entry point works with `python scripts/refinement_loop.py --mandate <path>`. Test with a single mandate first. |
| 4 | **Wire mandate_generator → wave_manager** | `scripts/wave_runner.py` | `wave_runner.py` already takes a `--mandates` dir. Add a `--auto-generate` flag that calls `mandate_generator.generate_mandates()` to populate the dir before running waves. |
| 5 | **Create continuous loop entry point** | `scripts/run_pipeline.py` | New script: generate mandates → run wave → promote winners → sleep → repeat. This is the main entry point that runs as a persistent process. |
| 6 | **Fix test collection errors** | `tests/` | 4 test collection errors exist (noted in VISION.md). Fix import paths and naming conflicts so `pytest tests/` passes cleanly. |
| 7 | **E2E smoke test with real LLM** | `tests/e2e_real_llm.py` | Run 1 mandate through the full pipeline (invent → backtest → classify → refine → promote). Use GLM-4.7 (cheaper) for initial test. Verify `state.json` is written, promotion fires if Sharpe target hit. |
| 8 | **Mandate template library** | `mandates/` | Create 10-15 starter mandates covering momentum, mean-reversion, breakout, trend archetypes across major tickers (SPY, AAPL, NVDA, TSLA). Currently only `e2e_stress_test.json` and `e2e_test_momentum.json` exist. |
| 9 | **Convergence measurement** | `scripts/measure_convergence.py` | Parse `refinement_runs/` output dirs, compute convergence rate (% mandates hitting Sharpe target), average turns to converge, per-archetype success rates. |

### Success Criteria
- `python scripts/run_pipeline.py --max-waves 2 --mandates-dir mandates/ --parallel 2` completes without errors
- At least 1 strategy promoted to `STRATEGY_REGISTRY` from a real LLM-driven mandate
- Convergence rate measured (even if low — we need the baseline)
- `pytest tests/` passes with 0 collection errors

### Dependencies
- None (all 31 components already built and unit-tested)

### Effort Estimate: **M** (2-3 focused sessions)

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM generates invalid Python despite 3-gate validation | Blocks pipeline | Gate 3 smoke backtest already catches this; add retry with stricter prompt if gate 3 fails |
| `register_strategy()` doesn't work for dynamically invented strategies | Promotion fails | Test `register_strategy()` with an invented strategy file before wiring promotion |
| Subprocess isolation breaks on WSL2 | Wave manager fails | Test `subprocess.run()` call to `refinement_loop.py` manually first |
| API budget burn rate too high | Financial | Use GLM-4.7 for refinement turns (cheaper), GLM-5-Turbo only for Turn 1 invention |
| Convergence rate is 0% | Pipeline useless | If <5% after 20 mandates, diagnose: are Sharpe targets too aggressive? Are prompts too constraining? Adjust before moving to Phase 5 |

---

## 3. Phase 5: Always-On Production

**Goal:** Convert the refinement pipeline from a manual script into a persistent, self-supervising daemon that runs 24/7 without human intervention.

### Key Deliverables

| # | Deliverable | File(s) | Description |
|---|-------------|---------|-------------|
| 1 | **Persistent daemon process** | `scripts/run_pipeline.py` (enhanced) | Add `--daemon` mode: runs the continuous loop, writes PID to `crabquant.pid`, handles SIGTERM/SIGINT for graceful shutdown. No more "run and forget" scripts. |
| 2 | **Supervisor cron job** | `openclaw cron` (replace existing 4 agents) | Single cron job `crabquant-supervisor` (every 5 min): check `crabquant.pid` alive, restart if dead, report status to Telegram. Replace `crabquant-wave`, `crabquant-improve`, `crabquant-validate`, `crabquant-meta`. |
| 3 | **Health check endpoint** | `crabquant/production/health.py` | Checks: daemon alive, last wave timestamp, API budget remaining, data cache freshness, disk space. Returns JSON status. Called by supervisor cron. |
| 4 | **State persistence & resume** | `crabquant/refinement/state.py` (new) | Track: current wave number, pending mandates, last completed mandate, total API calls this session. On restart, pick up where we left off instead of starting from scratch. |
| 5 | **API budget tracker** | `crabquant/refinement/api_budget.py` (new) | Track z.ai prompt count per day/week. Throttle to GLM-4.7 only when approaching limit. Alert via Telegram at 80% budget. |
| 6 | **Resource limiter** | `scripts/run_pipeline.py` | Monitor CPU and RAM usage. Reduce `--parallel` from 5→3→1 as available resources drop. Pause if RAM <2GB free. |
| 7 | **Auto-mandate generation with market data** | `crabquant/refinement/mandate_generator.py` (enhance) | Current generator scans strategy files. Enhance to: pull current SPY/VIX regime, generate mandates targeting underexplored regime-indicator combos, rotate tickers based on recent volatility. |
| 8 | **Status reporting** | `scripts/status_report.py` | Cron-triggered (or heartbeat): summarize last 24h of pipeline activity — mandates completed, winners promoted, convergence rate, API budget used. Post to Telegram. |
| 9 | **Graceful shutdown** | `scripts/run_pipeline.py` | On SIGTERM: finish current mandate, write state, save partial results, exit cleanly. On restart, resume from saved state. |

### Success Criteria
- Pipeline runs unattended for 48+ hours without crash or intervention
- Supervisor cron successfully restarts daemon if killed
- API budget tracked and enforced (no unexpected overages)
- Telegram receives daily status reports
- State survives daemon restart (resumes from where it stopped)

### Dependencies
- Phase 4 complete (pipeline runs end-to-end)

### Effort Estimate: **L** (4-5 sessions, includes testing 48h uptime)

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Daemon OOM kills (pandas DataFrames accumulate) | Crashes after hours | Add explicit `del` + `gc.collect()` after each mandate; limit concurrent subprocesses based on RAM |
| Yahoo Finance rate limiting | Data fetch failures | Cache is 20h TTL — verify freshness; add retry with exponential backoff; fallback to cached data if fetch fails |
| WSL2 sleep/hibernate kills daemon | Overnight gaps | Supervisor cron detects stale PID (last heartbeat >30min ago) and restarts |
| API budget exhaustion mid-wave | Wasted LLM calls | Check budget before each LLM call, not just at wave start |
| Disk space from run artifacts | Fills up | Auto-cleanup: keep last 50 run dirs, archive older ones to `refinement_runs/archive/` |

---

## 4. Phase 6: Intelligence Layer

**Goal:** Make the system learn from its own experience — not just run mandates, but get smarter about which mandates to run, what prompts work, and when strategies are decaying.

### Key Deliverables

| # | Deliverable | File(s) | Description |
|---|-------------|---------|-------------|
| 1 | **Action analytics feedback loop** | `crabquant/refinement/action_analytics.py` (enhance) | Already exists (tracks action success/fail). Wire into `context_builder.py`: when building LLM context for Turn 2+, include "actions that worked historically for this failure mode" and "actions that failed". |
| 2 | **Adaptive invention prompts** | `crabquant/refinement/prompts.py` (enhance) | Based on action analytics and regime data, dynamically adjust Turn 1 prompts: emphasize indicator families that have high convergence rates for current regime, discourage patterns that keep failing. |
| 3 | **Strategy decay detection** | `crabquant/production/scanner.py` (enhance) | Existing `scanner.py` exists in production/. Enhance: for each promoted strategy, run periodic (daily) backtests on recent data. If Sharpe drops >30% from promotion value, flag for retirement. |
| 4 | **Portfolio correlation matrix** | `crabquant/refinement/portfolio_correlation.py` (enhance) | Already exists. Wire into promotion: don't promote strategies with >0.8 equity curve correlation to existing winners unless Sharpe is significantly higher. |
| 5 | **Portfolio-level scoring** | `crabquant/production/portfolio.py` (new) | Beyond individual Sharpe: compute portfolio-level metrics (combined Sharpe, max drawdown, diversification ratio) across all promoted strategies. Use this to guide which archetypes need more research. |
| 6 | **Mandate prioritization** | `crabquant/refinement/mandate_generator.py` (enhance) | Score potential mandates by expected value: (convergence_probability × expected_Sharpe) × (1 - portfolio_correlation). Prioritize mandates that fill gaps in the current portfolio. |
| 7 | **Failure pattern analysis** | `crabquant/refinement/classifier.py` (enhance) | Aggregate failure mode statistics across all runs. If "too_few_trades" is 60% of failures, auto-adjust validation thresholds or add entry/exit constraints to invention prompts. |

### Success Criteria
- Action analytics data flows back into LLM prompts (verify by inspecting generated context)
- Convergence rate improves by ≥5 percentage points vs Phase 4 baseline (measured over 100 mandates)
- Strategy decay detector flags at least 1 decaying strategy (if any exist)
- Mandate prioritization produces different mandate orderings than random

### Dependencies
- Phase 5 complete (48h+ of run data to analyze)

### Effort Estimate: **L** (5-6 sessions)

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Feedback loop causes prompt drift | Prompts become too specialized | Keep a "control group" — 20% of mandates use base prompts without adaptation |
| Portfolio optimization overfits to historical correlation | Bad generalization | Use rolling 90-day correlation windows, not full-history |
| Decay detection has too many false positives | Premature strategy retirement | Require 3 consecutive below-threshold checks before flagging |
| Adaptive prompts become overly complex | LLM confused by long context | Cap adaptive additions to 500 tokens; keep base prompt clean |

---

## 5. Phase 7: Deployment Readiness

**Goal:** Validate that refined strategies would survive real market conditions — not just backtests with perfect fills.

### Key Deliverables

| # | Deliverable | File(s) | Description |
|---|-------------|---------|-------------|
| 1 | **Slippage + commission modeling** | `crabquant/confirm/` (existing) → integrate into `refinement/diagnostics.py` | The `confirm/` module exists but isn't wired into the refinement loop. Add post-backtest slippage simulation: apply 5bp commission per trade, 1-tick slippage on market orders. Strategies must pass with slippage to be promoted. |
| 2 | **Paper trading engine** | `crabquant/paper_trading/` (new directory) | `engine.py`: subscribe to real-time price feed (Yahoo Finance streaming or Alpaca paper API), execute signals from promoted strategies, track P&L. `portfolio.py`: virtual portfolio with cash management. |
| 3 | **Multi-timeframe strategies** | `crabquant/refinement/prompts.py` (enhance) + new strategy examples | Support strategies that combine daily signals with weekly trend filters or intraday entries. Update data loading to fetch multiple timeframes. Add multi-timeframe strategy examples to LLM context. |
| 4 | **Performance dashboard** | `crabquant/dashboard/` (new directory) | Simple HTML dashboard (or Telegram bot): daily P&L curve, live strategy signals, Sharpe tracking, drawdown monitoring. Served via `python -m http.server` or pushed as Telegram images. |
| 5 | **Walk-forward in refinement loop** | `crabquant/refinement/diagnostics.py` (enhance) | Currently walk-forward only runs at promotion time. Add optional Tier 2 walk-forward diagnostic at Turn 4+ (after initial convergence) to catch in-sample overfit early. |
| 6 | **Regime-aware validation** | `crabquant/validation/__init__.py` (enhance) | Enhance `walk_forward_test()` to report per-regime Sharpe. Reject strategies that only work in bull markets. |

### Success Criteria
- Slippage/commission integrated: at least 1 strategy that passes raw Sharpe target but fails with slippage (proving the filter works)
- Paper trading runs for 1 week with ≥3 strategies generating signals
- Dashboard accessible and showing live data
- Multi-timeframe mandate completes successfully

### Dependencies
- Phase 6 complete (portfolio of promoted strategies to paper trade)

### Effort Estimate: **L** (5-6 sessions)

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Slippage kills all strategies | No survivors | Expected — this is the point. Strategies that survive slippage are the real edge. If 0% survive, reduce slippage model or focus on longer holding periods. |
| Paper trading API rate limits | Can't get real-time data | Use 15-min delayed data initially; Alpaca paper API has generous limits |
| Multi-timeframe increases data requirements | Slower pipeline | Only use multi-timeframe for final validation, not every iteration |
| Dashboard becomes maintenance burden | Distraction from research | Keep it minimal — single HTML file with inline JS, no framework |

---

## 6. Phase 8: Live Trading (Far Future)

**Goal:** Deploy validated strategies to a real brokerage with proper risk management.

### Key Deliverables

| # | Deliverable | File(s) | Description |
|---|-------------|---------|-------------|
| 1 | **Broker integration** | `crabquant/broker/` (new directory) | Abstract broker interface with Alpaca implementation (free tier supports paper + live). `order.py`: market/limit orders, position tracking. `account.py`: balance, buying power, P&L. |
| 2 | **Risk management layer** | `crabquant/risk/` (new directory) | `position_sizer.py`: Kelly criterion or fixed-fractional sizing. `portfolio_risk.py`: max drawdown limit, max sector exposure, max correlation. `circuit_breaker.py`: halt all trading if daily loss exceeds threshold. |
| 3 | **Order execution engine** | `crabquant/broker/executor.py` | Signal → order pipeline with retry logic, partial fill handling, order status tracking. Handle market close (don't submit orders after 3:50 PM ET). |
| 4 | **Live monitoring & alerts** | `crabquant/monitoring/` (new directory) | Real-time P&L alerts to Telegram. Daily performance report. Anomaly detection (sudden drawdown, stuck orders). Auto-pause if risk limits breached. |
| 5 | **Deployment pipeline** | `scripts/deploy_strategy.py` | Promote a paper-traded strategy to live: validate against current market, set position limits, deploy to broker, monitor for 1 week before full sizing. |

### Success Criteria
- At least 1 strategy trading live with real (small) capital
- Risk limits enforced (no position exceeds configured max)
- 1 month of live trading data for performance review
- Automated alerts fire correctly on drawdown events

### Dependencies
- Phase 7 complete (strategies proven in paper trading with slippage)
- Broker account funded (Alpaca or Interactive Brokers)
- Legal/regulatory compliance verified

### Effort Estimate: **XL** (8+ sessions, significant testing)

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Real money losses | Financial | Start with minimum capital ($100-500). Hard daily loss limits. Auto-liquidate at market close. |
| Broker API goes down during trading hours | Missed signals / stuck positions | Implement order timeouts, daily reconciliation, manual override capability |
| Strategy works in paper but fails live | Capital loss | Require 3-month paper track record before live deployment. Gradual position sizing (start at 10% target). |
| Regulatory issues | Legal | Alpaca handles most compliance; ensure no pattern day trading violations; keep positions overnight only if strategy is designed for it |

---

## 7. PRDs Needed

| # | Document | Covers | Priority | Complexity |
|---|----------|--------|----------|------------|
| 1 | **PRD-4: Integration & Wire-Up** | Phase 4 detail: orchestrator class extraction, subprocess wiring, E2E test plan, mandate template spec, convergence measurement methodology | **Immediate** | M |
| 2 | **PRD-5: Always-On Daemon** | Phase 5 detail: daemon architecture (PID management, signal handling), supervisor cron spec, state persistence format, API budget tracking logic, resource limiter thresholds | **Next Sprint** | L |
| 3 | **PRD-6: Intelligence Layer** | Phase 6 detail: action analytics→prompt feedback format, adaptive prompt template spec, decay detection algorithm, portfolio scoring methodology, mandate scoring formula | **Next Sprint** | L |
| 4 | **PRD-7: Paper Trading & Slippage** | Phase 7 detail: slippage model spec (commission rates, slippage assumptions), paper trading architecture, data feed selection, dashboard wireframes, multi-timeframe data requirements | **Future** | L |
| 5 | **PRD-8: Broker Integration** | Phase 8 detail: broker abstraction interface, Alpaca API mapping, order lifecycle, risk management rules, deployment checklist, monitoring/alerting spec | **Future** | XL |
| 6 | **PRD-Data: Data Pipeline Reliability** | Yahoo Finance cache management, fallback data sources (Polygon.io free tier, Alpha Vantage), data quality checks, cache invalidation, stale data detection | **Immediate** | S |
| 7 | **PRD-Testing: Integration Test Strategy** | E2E test matrix (real LLM vs mocked), stress test plan, 48h soak test spec, test data fixtures, CI pipeline (pytest on push) | **Immediate** | M |
| 8 | **PRD-Ops: Operational Runbook** | How to start/stop/restart the daemon, how to check status, how to manually intervene, how to add a new strategy, how to interpret results, troubleshooting guide | **Next Sprint** | S |

---

## Appendix: File Tree (Current)

```
~/development/CrabQuant/
├── VISION.md                          # This document's vision
├── ROADMAP.md                         # THIS FILE
├── BUILD_STATUS.json                  # Component test status
├── requirements.txt
├── mandates/                          # Mandate JSON configs (only 2 E2E test files)
├── strategies/                        # Registered strategy .py files
├── refinement_runs/                   # Output from refinement pipeline runs
├── scripts/
│   ├── refinement_loop.py             # Main orchestrator script (needs extraction)
│   ├── wave_runner.py                 # Parallel wave CLI
│   ├── crabquant_cron.py              # Legacy cron entry point
│   └── sweep_task.py, improve_task.py, validate_task.py, meta_task.py  # Legacy cron tasks
├── crabquant/
│   ├── engine/                        # BacktestEngine, parallel_backtest
│   ├── data/                          # Yahoo Finance data loading + cache
│   ├── strategies/                    # STRATEGY_REGISTRY, 28 strategies
│   ├── validation/                    # walk_forward, cross_ticker, full_validation
│   ├── refinement/                    # 31 pipeline components (all unit-tested)
│   │   ├── orchestrator.py            # ← MISSING (logic is in scripts/refinement_loop.py)
│   │   ├── wave_manager.py            # ✅
│   │   ├── mandate_generator.py       # ✅
│   │   ├── auto_promotion.py          # ✅
│   │   ├── action_analytics.py        # ✅
│   │   ├── classifier.py              # ✅
│   │   ├── context_builder.py         # ✅
│   │   ├── llm_api.py                 # ✅
│   │   ├── prompts.py                 # ✅
│   │   ├── schemas.py                 # ✅
│   │   ├── config.py                  # ✅
│   │   ├── stagnation.py              # ✅
│   │   ├── validation_gates.py        # ✅
│   │   ├── diagnostics.py             # ✅
│   │   ├── module_loader.py           # ✅
│   │   ├── circuit_breaker.py         # ✅
│   │   ├── cosmetic_guard.py          # ✅
│   │   ├── hypothesis_enforcement.py  # ✅
│   │   ├── guardrails_integration.py  # ✅
│   │   ├── gate3_smoke.py             # ✅
│   │   ├── tier1_diagnostics.py       # ✅
│   │   ├── tier2_diagnostics.py       # ✅
│   │   ├── prompt_refinement.py       # ✅
│   │   ├── wave_scaling.py            # ✅
│   │   ├── per_wave_metrics.py        # ✅
│   │   ├── regime_sharpe.py           # ✅
│   │   ├── portfolio_correlation.py   # ✅
│   │   ├── wave_dashboard.py          # ✅
│   │   ├── cron_integration.py        # ✅
│   │   └── promotion.py               # ✅
│   ├── production/                    # scanner.py, promoter.py, report.py
│   ├── confirm/                       # Slippage/commission (exists, not wired)
│   ├── guardrails.py                  # GuardrailConfig, check_guardrails
│   ├── regime.py                      # Market regime detection
│   └── invention.py                   # Strategy invention module (legacy)
└── tests/                             # Unit + E2E tests (4 collection errors)
```

---

*This roadmap lives alongside VISION.md. Vision is the north star; this is the path. Update as we learn.*
