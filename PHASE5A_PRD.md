# Phase 5A PRD — Always-On Daemon Core

**Goal:** Convert the refinement pipeline from a single-mandate script into a persistent, self-supervising daemon that runs 24/7.

**Scope:** Daemon loop, state persistence, health checks, graceful shutdown, supervisor cron.
**Out of scope (Phase 5B):** API budget tracker, resource limiter, auto-mandate with market data, status reporting.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────┐
│             run_pipeline.py (DAEMON)         │
│                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ State    │   │ Mandate  │   │ Wave     │ │
│  │ Manager  │   │ Queue    │   │ Runner   │ │
│  └──────────┘   └──────────┘   └──────────┘ │
│                                              │
│  Signal handlers: SIGTERM, SIGINT            │
│  PID file: crabquant.pid                     │
│  State file: results/daemon_state.json       │
│  Heartbeat: results/daemon_heartbeat.json    │
└─────────────────────────────────────────────┘
         ▲                    │
         │ health check       │ status
         │                    ▼
┌─────────────────┐   ┌──────────────┐
│ Supervisor Cron │   │ Telegram     │
│ (every 5 min)   │   │ Notifications│
└─────────────────┘   └──────────────┘
```

---

## 2. Deliverables

### 2.1 `scripts/run_pipeline.py` — Persistent Daemon

**File:** `~/development/CrabQuant/scripts/run_pipeline.py`

```python
#!/usr/bin/env python3
"""
CrabQuant Pipeline Daemon — Always-on strategy research engine.

Usage:
  python scripts/run_pipeline.py                    # Start daemon
  python scripts/run_pipeline.py --max-waves 10     # Run N waves then stop
  python scripts/run_pipeline.py --stop             # Signal running daemon to stop
  python scripts/run_pipeline.py --status           # Check daemon status
"""
```

**Core loop:**
```
while not shutdown_requested:
    1. Load state from results/daemon_state.json (or create fresh)
    2. Check pending mandates in queue
    3. If queue empty: generate new mandates from mandates/ dir
    4. Pick next mandate(s), run via wave_manager.run_waves()
    5. After wave completes: promote winners, update state
    6. Write heartbeat to results/daemon_heartbeat.json
    7. Sleep configurable interval (default: 60s between waves)
    8. GOTO 1
```

**Signal handling:**
- `SIGTERM`: Set `shutdown_requested = True`. Finish current mandate. Write state. Exit 0.
- `SIGINT`: Same as SIGTERM.
- If mid-wave: let current subprocess finish (they have their own timeouts), don't kill children.

**PID management:**
- On start: write `os.getpid()` to `crabquant.pid`
- On stop: delete `crabquant.pid`
- On start: if `crabquant.pid` exists, check if process alive. If alive and heartbeat recent (<5min), refuse to start. If stale, take over.

**CLI flags:**
- `--max-waves N`: Run N waves then exit (for testing). Default: unlimited.
- `--parallel N`: Max concurrent mandates. Default: 3 (conservative for 32GB RAM).
- `--sleep N`: Seconds between waves. Default: 60.
- `--mandates-dir PATH`: Directory of mandate JSONs. Default: `mandates/`.
- `--stop`: Signal daemon to stop (reads PID, sends SIGTERM).
- `--status`: Print daemon status and exit.
- `--daemon`: Run in background (fork/nohup). Default: foreground.

### 2.2 `crabquant/refinement/state.py` — State Persistence

**File:** `~/development/CrabQuant/crabquant/refinement/state.py`

```python
@dataclass
class DaemonState:
    daemon_id: str              # UUID
    started_at: str             # ISO timestamp
    last_heartbeat: str         # ISO timestamp
    last_wave_completed: str    # ISO timestamp
    current_wave: int           # Wave number (1-indexed)
    total_mandates_run: int     # Cumulative
    total_strategies_promoted: int
    total_api_calls: int        # Approximate (counted per LLM call)
    pending_mandates: list      # Mandate names not yet run
    completed_mandates: list    # Mandate names that finished
    failed_mandates: list       # Mandate names that errored
    last_error: Optional[str]
    shutdown_requested: bool    # For graceful stop
```

**Methods:**
- `save(path)`: Write to JSON file atomically (write to .tmp, rename).
- `load(path)`: Load from JSON, return DaemonState or None.
- `heartbeat()`: Update last_heartbeat to now, save.
- `record_wave_completion(mandate_name, status, sharpe)`: Update counters.
- `get_resume_point()`: Return next mandate to run based on pending list.
- `mark_shutdown()`: Set shutdown_requested=True, save.

**State file:** `results/daemon_state.json`
**Atomic writes:** Write to `daemon_state.json.tmp` then `os.rename()` — prevents corruption if daemon crashes mid-write.

### 2.3 `crabquant/production/health.py` — Health Checks

**File:** `~/development/CrabQuant/crabquant/production/health.py`

```python
def check_health() -> dict:
    """
    Returns:
    {
        "status": "healthy" | "degraded" | "down",
        "daemon": {
            "alive": bool,           # PID file exists, process running
            "last_heartbeat": str,   # ISO timestamp
            "heartbeat_age_seconds": float,
            "current_wave": int,
            "total_mandates_run": int,
        },
        "system": {
            "cpu_percent": float,
            "ram_free_gb": float,
            "disk_free_gb": float,
        },
        "data": {
            "cache_fresh": bool,     # Price data cache < 24h old
            "cache_age_hours": float,
        },
        "api": {
            "calls_today": int,      # From daemon state
        },
        "recommendations": list[str]  # e.g. "RAM low — reduce parallel to 1"
    }
    """
```

**Health status logic:**
- `"healthy"`: daemon alive, heartbeat < 5min old, RAM > 4GB free
- `"degraded"`: daemon alive but heartbeat 5-30min old, or RAM 2-4GB
- `"down"`: daemon dead, stale PID, or heartbeat > 30min

**CLI:** `python -m crabquant.production.health` — prints JSON to stdout.

### 2.4 Supervisor Cron — `crabquant-supervisor`

**Replace existing 4 crons with 1 supervisor:**

```
Name: crabquant-supervisor
Schedule: every 5 minutes
Model: glm-4.7 (lightweight checks only)
Task:
  1. Run health check: python -m crabquant.production.health
  2. If status == "down": restart daemon, notify Telegram
  3. If status == "degraded": log warning, reduce parallel if RAM low
  4. Every 30 minutes: post status summary to Telegram
  5. Every 6 hours: clean up old refinement_runs (keep last 100)
```

**Supervisor should NOT run the pipeline itself** — it only monitors and restarts.

### 2.5 Graceful Shutdown — Integration

In `run_pipeline.py`:

```python
import signal

shutdown_requested = False

def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logger.info("Shutdown signal received. Finishing current wave...")

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)
```

In the main loop, check `shutdown_requested` before starting each new wave. If set, finish the current subprocess, save state, clean up PID, exit 0.

---

## 3. File Structure

```
~/development/CrabQuant/
├── scripts/
│   ├── run_pipeline.py          # NEW — daemon entry point
│   ├── refinement_loop.py       # EXISTING — single mandate loop (no changes)
│   └── wave_runner.py           # EXISTING — wave CLI (no changes)
├── crabquant/
│   ├── refinement/
│   │   ├── state.py             # NEW — daemon state persistence
│   │   └── __init__.py          # UPDATE — add state exports
│   └── production/
│       ├── health.py            # NEW — health check endpoint
│       ├── __init__.py          # UPDATE — add health exports
│       └── scanner.py           # EXISTING — no changes
├── results/
│   ├── daemon_state.json        # NEW — written by daemon
│   ├── daemon_heartbeat.json    # NEW — written every wave
│   └── dashboard.json           # EXISTING
├── crabquant.pid                # NEW — PID file (in .gitignore)
└── tests/
    └── refinement/
        ├── test_state.py        # NEW — state persistence tests
        ├── test_health.py       # NEW — health check tests
        └── test_pipeline.py     # NEW — daemon integration tests
```

---

## 4. Integration Points

### How `run_pipeline.py` calls `refinement_loop.py`:
```python
# The daemon does NOT import refinement_loop directly.
# It uses subprocess isolation (same as wave_manager):
subprocess.run([
    sys.executable, 
    "scripts/refinement_loop.py",
    "--mandate", mandate_path,
    "--max-turns", "7",
    "--sharpe-target", "1.5",
], timeout=600, capture_output=True)
```

### How state persists across restarts:
1. Daemon starts → loads `results/daemon_state.json`
2. If state exists and `shutdown_requested == False` (crash): resume from `pending_mandates`
3. If state exists and `shutdown_requested == True` (clean stop): start fresh wave
4. Each wave completion → atomically save state

### How supervisor detects problems:
1. Supervisor cron fires every 5min
2. Runs `python -m crabquant.production.health`
3. Parses JSON output
4. If `daemon.alive == False` or `heartbeat_age_seconds > 300`: restart

---

## 5. Test Requirements

### test_state.py (unit tests)
- `test_create_fresh_state` — new state has correct defaults
- `test_save_and_load` — roundtrip serialization
- `test_atomic_write` — verify .tmp + rename pattern
- `test_record_wave_completion` — counters increment
- `test_get_resume_point` — returns next pending mandate
- `test_mark_shutdown` — sets flag, saves
- `test_load_corrupted_file` — returns None, doesn't crash

### test_health.py (unit tests)
- `test_healthy_status` — mock daemon alive, recent heartbeat, plenty RAM
- `test_degraded_status` — mock stale heartbeat or low RAM
- `test_down_status` — mock dead daemon
- `test_recommendations` — low RAM → "reduce parallel"
- `test_data_cache_fresh` — cache < 24h → fresh
- `test_data_cache_stale` — cache > 24h → stale

### test_pipeline.py (integration tests)
- `test_daemon_start_and_stop` — start daemon, verify PID, send stop, verify clean exit
- `test_daemon_resume_after_crash` — start, kill -9, restart, verify it resumes
- `test_max_waves_flag` — start with --max-waves 1, verify exits after 1 wave
- `test_status_command` — start daemon, run --status, verify output

---

## 6. Success Criteria

- [ ] `python scripts/run_pipeline.py --max-waves 1 --parallel 1` completes 1 mandate and exits cleanly
- [ ] `python scripts/run_pipeline.py --stop` gracefully stops a running daemon
- [ ] `python -m crabquant.production.health` returns valid JSON with all fields
- [ ] Daemon survives `kill -9` and resumes on next start
- [ ] Supervisor cron detects dead daemon and restarts it
- [ ] All unit tests pass (target: 50+ new tests)
- [ ] PID file created/deleted correctly
- [ ] State file survives daemon restart
- [ ] Pre-commit hook passes on all new code

---

## 7. Implementation Order

Build in this order (each can be a sub-agent task):

1. **state.py + test_state.py** — no dependencies, foundation for everything
2. **health.py + test_health.py** — depends on state.py for daemon state reading
3. **run_pipeline.py + test_pipeline.py** — depends on state.py, calls refinement_loop via subprocess
4. **Supervisor cron** — depends on health.py, configure via openclaw cron
5. **Integration test** — run full daemon cycle, verify end-to-end

---

## 8. Constraints

- **No new pip dependencies** — use only stdlib (signal, os, json, psutil if available, fallback to /proc)
- **No database** — JSON files on disk only
- **No daemonization library** — simple fork or nohup, not supervisord/systemd
- **Subprocess isolation** — daemon never imports refinement_loop, always subprocess
- **WSL2 compatible** — no Linux-specific syscalls beyond what WSL2 supports
- **Respect pre-commit hook** — all new code must pass 596 unit tests
- **32GB RAM, 12 threads** — max 3 parallel mandates, ~200MB per process
