# Phase 5B PRD — Intelligence & Reliability

**Goal:** Make the daemon smarter and more reliable — track API budget, limit resource usage, auto-generate mandates from market conditions, and report status to Telegram.

**Scope:** API budget tracker, resource limiter, auto-mandate enhancement, status reporting.
**Out of scope:** Strategy decay detection, adaptive prompts, portfolio optimization (Phase 6). Slippage integration, paper trading, multi-timeframe (Phase 7).

**Dependencies:** Phase 5A complete (daemon running, PID 3759494, wave 7, 13 mandates run).

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              run_pipeline.py (DAEMON)                    │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ API Budget   │  │ Resource     │  │ Auto-Mandate │   │
│  │ Tracker      │  │ Limiter      │  │ Generator    │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │            │
│         ▼                 ▼                 ▼            │
│  ┌──────────────────────────────────────────────────┐    │
│  │           Wave Runner (existing)                  │    │
│  │     parallel mandates, subprocess isolation       │    │
│  └──────────────────────┬───────────────────────────┘    │
│                          │                               │
│  ┌──────────────────────▼───────────────────────────┐    │
│  │           Status Reporter                         │    │
│  │     daily summary → Telegram                     │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  State: results/daemon_state.json (enhanced)             │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Component 1: API Budget Tracker

**File:** `~/development/CrabQuant/crabquant/refinement/api_budget.py` (new)

### 2.1 Requirements

Track z.ai prompt usage per day/week. The GLM Coding Plan limits by prompt count (not tokens). When approaching the daily or weekly limit, throttle the model to a cheaper option (GLM-4.7) and alert via Telegram.

**Smart Model Routing (Task-Aware):** Not all LLM calls need the same model. The budget tracker should provide a `get_model_for_task(task_type)` method that routes calls based on reasoning requirements:

| Task Type | Model | Rationale |
|-----------|-------|-----------|
| `invention` (Turn 1, novel strategy) | GLM-5-Turbo | Needs creative reasoning, no prior code to iterate on |
| `retry_with_feedback` (gate failure fix) | GLM-4.7 | Simple error repair — specific error message provided, minimal creativity |
| `refinement` (Turns 2-5, improving existing code) | GLM-4.7 | Iterative improvement on existing code, not novel invention |
| `classification` (failure mode analysis) | GLM-4.7 | Structured output, pattern matching, no creativity needed |
| `analytics` (action analytics, regime detection) | GLM-4.7 | Data analysis, structured JSON output |

**Expected savings:** ~60-70% reduction in API cost. Only 1 out of every ~5 calls uses GLM-5-Turbo.

**Agent Budget Reservation:** Reserve 30% of the daily budget for interactive/coding agents (OpenClaw sub-agents). The daemon should never consume more than 70% of the budget. When daemon usage hits 70%, pause until the rolling window resets.

```python
# Budget allocation example (400 prompts / 5hr rolling window)
daemon_limit = 280          # 70% for daemon
interactive_reserve = 120   # 30% for coding agents, user interactions
daemon_throttle = 224       # 80% of daemon limit → switch to GLM-4.7
daemon_alert = 252          # 90% of daemon limit → send alert
```

### 2.2 Interface

```python
```python
from enum import Enum

class TaskType(Enum):
    INVENTION = "invention"                    # Turn 1, novel strategy generation
    RETRY_WITH_FEEDBACK = "retry_with_feedback"  # Gate failure, fixing specific error
    REFINEMENT = "refinement"                    # Turns 2-5, improving existing code
    CLASSIFICATION = "classification"            # Failure mode analysis
    ANALYTICS = "analytics"                      # Action analytics, regime detection

@dataclass
class ApiBudgetConfig:
    daily_limit: int = 400         # Max prompts per 5hr rolling window (GLM Coding Plan)
    weekly_limit: int = 2000       # Max prompts per week
    daemon_budget_pct: float = 0.70  # Daemon gets 70% of budget
    throttle_threshold: float = 0.80  # Switch to cheap model at 80% of daemon budget
    alert_threshold: float = 0.90    # Alert at 90% of daemon budget
    state_file: str = "results/api_budget_state.json"
    premium_model: str = "glm-5-turbo"    # Creative tasks
    throttle_model: str = "glm-4.7"        # Routine tasks, retries
    
    # Task-based model routing (task_type → model)
    task_model_map: dict = field(default_factory=lambda: {
        TaskType.INVENTION: "glm-5-turbo",       # Needs creative reasoning
        TaskType.RETRY_WITH_FEEDBACK: "glm-4.7",  # Simple error repair
        TaskType.REFINEMENT: "glm-4.7",            # Iterative improvement
        TaskType.CLASSIFICATION: "glm-4.7",        # Structured output
        TaskType.ANALYTICS: "glm-4.7",             # Data analysis
    })

@dataclass
class ApiBudgetState:
    date: str                       # YYYY-MM-DD
    daily_count: int
    weekly_count: int
    weekly_start: str               # YYYY-MM-DD (Monday)
    last_prompt_time: str           # ISO timestamp
    is_throttled: bool = False
    is_paused: bool = False         # True when daemon budget exhausted

@dataclass 
class PromptRecord:
    timestamp: str
    model: str
    task_type: str
    mandate: str
    turn: int
    success: bool

class ApiBudgetTracker:
    def __init__(self, config: ApiBudgetConfig | None = None): ...
    
    def get_model_for_task(self, task_type: TaskType) -> str:
        """Route to appropriate model based on task reasoning requirements.
        If budget is tight (over throttle_threshold), force throttle_model regardless."""
    
    def record_prompt(self, model: str, task_type: TaskType, mandate: str = "", turn: int = 0) -> dict:
        """Record an API call with full context. Returns {should_throttle, recommended_model, remaining, daemon_remaining}.
        Automatically checks daemon budget allocation."""
    
    def should_pause(self) -> bool:
        """Return True if daemon has consumed its 70% allocation. Caller should pause until window resets."""
    
    def get_status(self) -> dict:
        """Return full budget status: daily_count, daily_limit, weekly_count, weekly_limit, 
        usage_pct, is_throttled, is_paused, daemon_remaining, interactive_remaining."""
    
    def should_throttle(self) -> bool:
        """Return True if daemon usage is over throttle_threshold (80% of 70% allocation)."""
    
    def get_recommended_model(self) -> str:
        """Return premium_model if budget OK, else throttle_model."""
    
    def reset_if_new_day(self) -> None:
        """Check if date rolled over, reset daily counter if so."""
    
    def reset_if_new_week(self) -> None:
        """Check if week rolled over (Monday), reset weekly counter if so."""
    
    def get_recent_history(self, hours: int = 1) -> list[dict]:
        """Return prompt records from the last N hours for debugging."""
    
    def save(self) -> None:
        """Persist state + recent history to JSON atomically."""
    
    def load(self) -> None:
        """Load state from JSON. Start fresh if corrupted or missing."""
```

### 2.3 Integration Points

- `crabquant/refinement/llm_api.py` — call `budget_tracker.record_prompt(model, task_type, ...)` after each LLM call. Use `budget_tracker.get_model_for_task(task_type)` to select model BEFORE making the call. The `call_llm_inventor()` function should accept an optional `task_type` parameter.
- `scripts/refinement_loop.py` — pass `task_type=INVENTION` for Turn 1 calls, `task_type=RETRY_WITH_FEEDBACK` for retry calls, `task_type=REFINEMENT` for Turn 2+ calls.
- `scripts/run_pipeline.py` — check `budget_tracker.should_pause()` before each wave. If paused, sleep until window resets. Check `budget_tracker.should_throttle()` to reduce parallel count.
- `crabquant/refinement/state.py` — add `api_budget_used_today`, `api_budget_throttled`, `api_budget_paused` fields to `DaemonState`.
- Status reporter reads `budget_tracker.get_status()` for the daily report.

### 2.4 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_api_budget.py` (15+ tests)

- `test_record_prompt_increments_counter`
- `test_task_routing_invention_uses_premium`
- `test_task_routing_retry_uses_throttle`
- `test_task_routing_refinement_uses_throttle`
- `test_throttle_at_threshold`
- `test_throttle_forces_all_tasks_to_cheap_model`
- `test_alert_at_threshold`
- `test_daemon_pause_at_budget_limit`
- `test_daemon_budget_is_70pct_of_total`
- `test_reset_on_new_day`
- `test_reset_on_new_week`
- `test_persistence_roundtrip`
- `test_corrupted_state_starts_fresh`
- `test_recommended_model_before_threshold`
- `test_recommended_model_after_threshold`
- `test_get_status_fields`
- `test_recent_history_tracking`

### 2.5 Effort Estimate
**S-M (1-1.5 sessions)** — counter with persistence + task routing + budget allocation.

---

## 3. Component 2: Resource Limiter

**File:** `~/development/CrabQuant/crabquant/refinement/resource_limiter.py` (new)

### 3.1 Requirements

Monitor CPU and RAM usage. Dynamically adjust `--parallel` to prevent system overload. Pause the daemon if resources are critically low.

### 3.2 Interface

```python
@dataclass
class ResourceConfig:
    max_parallel: int = 3           # Starting parallel count
    min_parallel: int = 1           # Never go below this
    ram_threshold_high_gb: float = 4.0   # Reduce parallel if RAM below this
    ram_threshold_pause_gb: float = 2.0  # Pause if RAM below this
    cpu_threshold_high: float = 90.0     # Reduce parallel if CPU above this
    check_interval_seconds: int = 60     # How often to recheck
    process_memory_mb: int = 200         # Approximate memory per mandate subprocess

@dataclass
class ResourceStatus:
    cpu_percent: float
    ram_free_gb: float
    ram_total_gb: float
    disk_free_gb: float
    recommended_parallel: int
    should_pause: bool

class ResourceLimiter:
    def __init__(self, config: ResourceConfig | None = None): ...
    
    def check_resources(self) -> ResourceStatus:
        """Read current CPU/RAM/disk. Return recommended_parallel and should_pause."""
    
    def get_recommended_parallel(self) -> int:
        """Return how many parallel mandates to run based on current resources."""
    
    def should_pause(self) -> bool:
        """Return True if system resources too low to continue."""
    
    def adjust_parallel(self, current_parallel: int) -> int:
        """Given current parallel count, return adjusted count based on resources."""
```

### 3.3 Integration Points

- `scripts/run_pipeline.py` — call `resource_limiter.check_resources()` at the start of each wave. Use `recommended_parallel` for `wave_manager.run_waves()`. If `should_pause()`, sleep and retry.
- `crabquant/production/health.py` — reuse resource checking logic (extract to shared utility if needed).
- State reporter includes resource status.

### 3.4 Implementation Notes

- Use `psutil` if available, fallback to `/proc/meminfo` + `os.getloadavg()` on Linux.
- Don't check on every mandate — only at wave boundaries (every ~60s).
- Track process count externally: count running subprocess PIDs, not just CPU.

### 3.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_resource_limiter.py` (10+ tests)

- `test_recommended_parallel_normal_resources`
- `test_reduce_parallel_low_ram`
- `test_pause_very_low_ram`
- `test_reduce_parallel_high_cpu`
- `test_never_below_min_parallel`
- `test_resource_status_fields`
- `test_psutil_fallback`
- `test_adjust_parallel_down`
- `test_adjust_parallel_up_recovery`
- `test_disk_space_reported`

### 3.6 Effort Estimate
**S (1 session)** — system stats reading + thresholds.

---

## 4. Component 3: Auto-Mandate Generation Enhancement

**File:** `~/development/CrabQuant/crabquant/refinement/mandate_generator.py` (enhance existing)

### 4.1 Requirements

The current `mandate_generator.py` scans strategy files and generates mandates. Enhance it to:
1. Pull current market regime (SPY/VIX) to target underexplored regime-indicator combos
2. Rotate tickers based on recent volatility (prioritize volatile tickers for breakout mandates, stable for mean-reversion)
3. Track which archetype×ticker combos have been attempted and avoid excessive re-runs
4. Generate mandates that fill gaps in the portfolio (e.g., if no breakout strategies promoted, prioritize breakout mandates)

### 4.2 New Interface Additions

```python
def generate_smart_mandates(
    strategies_dir: Path | str,
    mandates_dir: Path | str,
    completed_mandates: list[str] | None = None,
    max_mandates: int = 10,
) -> list[dict]:
    """Generate mandates intelligently based on market regime, completed work, and portfolio gaps.
    
    Steps:
    1. Get current market regime from crabquant.regime.detect_regime()
    2. Load completed mandates list to avoid re-running
    3. Identify archetype gaps (archetypes with few/no promoted strategies)
    4. Score potential mandate combos by: regime_match + gap_fill + diversity
    5. Generate top-scoring mandate configs
    6. Write to mandates_dir as JSON files
    
    Returns list of generated mandate dicts.
    """

def get_portfolio_gaps(promoted_strategies: list[dict]) -> dict[str, float]:
    """Analyze promoted strategies and return archetype coverage scores.
    
    Returns dict like: {"momentum": 0.8, "mean_reversion": 0.2, "breakout": 0.0, "trend": 0.5}
    Scores based on count and diversity of promoted strategies per archetype.
    """
```

### 4.3 Integration Points

- `scripts/run_pipeline.py` — when mandate queue is empty, call `generate_smart_mandates()` instead of just scanning `mandates/` dir.
- `crabquant/regime.py` — use `detect_regime()` for current market conditions.
- `crabquant/refinement/promotion.py` — read promoted strategy metadata for gap analysis.
- `crabquant/refinement/state.py` — pass `completed_mandates` list to generator.

### 4.4 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_mandate_generator_enhanced.py` (8+ tests)

- `test_generate_mandates_respects_completed_list`
- `test_regime_aware_ticker_selection`
- `test_portfolio_gap_detection`
- `test_no_duplicate_mandates`
- `test_max_mandates_limit`
- `test_mandate_json_valid_format`
- `test_archetype_diversity`
- `test_volatility_based_ticker_rotation`

### 4.5 Effort Estimate
**M (1-2 sessions)** — requires regime integration and gap analysis logic.

---

## 5. Component 4: Status Reporting

**File:** `~/development/CrabQuant/crabquant/refinement/status_reporter.py` (new)

### 5.1 Requirements

Generate and send a daily status summary to Telegram. This can be triggered by the supervisor cron or by the daemon's heartbeat logic.

### 5.2 Interface

```python
@dataclass
class StatusReport:
    timestamp: str
    period_hours: int               # Report covers last N hours (default 24)
    daemon: dict                    # status, wave, uptime
    mandates: dict                  # completed, failed, in-progress, promoted
    api_budget: dict                # used today, remaining, throttled
    resources: dict                 # CPU, RAM, disk
    convergence: dict               # convergence rate, avg turns, best sharpe
    portfolio: dict                 # total promoted, per-archetype counts

class StatusReporter:
    def __init__(self, telegram_chat_id: str | None = None): ...
    
    def generate_report(self, period_hours: int = 24) -> StatusReport:
        """Collect data from daemon_state.json, api_budget_state.json, 
        refinement_runs/, and strategies/ to build a comprehensive status."""
    
    def format_telegram(self, report: StatusReport) -> str:
        """Format report as Telegram-friendly markdown. Keep under 4096 chars."""
    
    def send_telegram(self, message: str) -> bool:
        """Send message via OpenClaw Telegram channel. Return True on success."""
    
    def report_and_send(self, period_hours: int = 24) -> bool:
        """Generate, format, and send. Return True if sent successfully."""
```

### 5.3 Integration Points

- Supervisor cron — call `report_and_send()` once per day (or on health check if >24h since last report).
- `scripts/run_pipeline.py` — optionally send report after each wave completion (configurable).
- `crabquant/refinement/api_budget.py` — read budget status.
- `crabquant/refinement/state.py` — read daemon state.
- `crabquant/refinement/resource_limiter.py` — read resource status.

### 5.4 Report Format (Telegram)

```
🦀 CrabQuant Daily Report
━━━━━━━━━━━━━━━━━━━━━━━
📊 Daemon: healthy | Wave 12 | 47 mandates run
⏱ Uptime: 23h 14m

📋 Last 24h:
  ✅ Completed: 18 mandates
  ❌ Failed: 3 mandates
  🏆 Promoted: 2 strategies
  📈 Best Sharpe: 1.87 (momentum_nvda)

💰 API Budget: 72/100 daily (72%) | 340/500 weekly
🔧 Resources: CPU 45% | RAM 12.3GB free

🎯 Convergence: 14% (5/36 hit target)
   Avg turns: 4.2
```

### 5.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_status_reporter.py` (8+ tests)

- `test_generate_report_fields`
- `test_format_telegram_under_char_limit`
- `test_report_empty_daemon_state`
- `test_report_with_failures`
- `test_send_telegram_success`
- `test_send_telegram_failure_graceful`
- `test_period_filtering`
- `test_portfolio_counts`

### 5.6 Effort Estimate
**S-M (1 session)** — data aggregation + formatting.

---

## 6. Dependencies Between Components

```
API Budget Tracker ──────────► Status Reporter (reads budget status)
Resource Limiter ────────────► Status Reporter (reads resource status)
Auto-Mandate Generator ──────► Status Reporter (reports generation stats)
API Budget Tracker ──────────► llm_api.py integration (throttle model)
Resource Limiter ────────────► run_pipeline.py (adjust parallel)
Auto-Mandate Generator ──────► run_pipeline.py (fill queue when empty)
```

**Build order:**
1. API Budget Tracker (no deps — build first)
2. Resource Limiter (no deps — can parallel with #1)
3. Auto-Mandate Enhancement (no deps on 1-2 — can parallel)
4. Status Reporter (depends on all three — build last)
5. Integration: Wire all components into `run_pipeline.py` + `llm_api.py`

---

## 7. Success Criteria

- [ ] API budget tracked and enforced: daemon switches to GLM-4.7 at 80% daily budget
- [ ] Resource limiter reduces parallel count when RAM < 4GB free
- [ ] Auto-mandate generates diverse mandates covering all archetypes
- [ ] Telegram receives daily status reports with all sections populated
- [ ] All unit tests pass (target: 36+ new tests across 4 test files)
- [ ] E2E integration test passes
- [ ] Daemon runs for 24h+ with new features without intervention
- [ ] PHASE_CHECKLIST.md completed (unit + E2E + real LLM + commit + report)

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| psutil not installed | Resource limiter falls back to /proc | Add psutil to requirements.txt; implement /proc fallback for Linux |
| Telegram send fails silently | Reports lost | Log failures; retry once; if persistent, report in daemon state |
| Budget state corrupted by crash | Lost count → over-billing | Atomic writes (tmp + rename); start fresh on corruption, err on conservative side |
| Auto-mandate generates bad combos | Wasted API calls | Validate generated mandates before queueing; require valid archetype + ticker |
| Market regime detection is slow | Delays mandate generation | Cache regime result for 4 hours; async if needed |

---

## 9. Effort Summary

| Component | Effort | Tests | Dependencies |
|-----------|--------|-------|-------------|
| API Budget Tracker | S (1 session) | 10+ | None |
| Resource Limiter | S (1 session) | 10+ | None |
| Auto-Mandate Enhancement | M (1-2 sessions) | 8+ | None |
| Status Reporter | S-M (1 session) | 8+ | Budget + Resource + Mandate |
| Integration + Wiring | S (0.5 session) | — | All components |
| **Total** | **M (3-5 sessions)** | **36+** | — |

---

## 10. File Structure

```
~/development/CrabQuant/
├── crabquant/refinement/
│   ├── api_budget.py            # NEW — API budget tracking
│   ├── resource_limiter.py      # NEW — CPU/RAM resource monitoring
│   ├── status_reporter.py       # NEW — Telegram status reports
│   ├── mandate_generator.py     # ENHANCED — smart mandate generation
│   ├── llm_api.py               # ENHANCED — budget-aware model selection
│   └── state.py                 # ENHANCED — add budget fields to DaemonState
├── results/
│   ├── api_budget_state.json    # NEW — budget tracker persistence
│   └── daemon_state.json        # ENHANCED — new fields
└── tests/refinement/
    ├── test_api_budget.py       # NEW
    ├── test_resource_limiter.py # NEW
    ├── test_mandate_generator_enhanced.py  # NEW
    └── test_status_reporter.py  # NEW
```
