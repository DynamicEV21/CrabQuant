#!/usr/bin/env bash
# hermes-watchdog.sh — HermGC factory health monitor
# Closes ISSUE-001 (token-tracking blind spot) by monitoring all factory subsystems.
#
# Usage:
#   hermes-watchdog.sh              — run health checks, print summary
#   hermes-watchdog.sh --verbose    — include detailed output for each check
#
# Exit codes:
#   0 = all healthy
#   1 = alert condition detected
#
# Dependencies: bash, jq, gc (~/go/bin), standard unix tools

set -uo pipefail

# --- Configuration ---
GC_CITY="${GC_CITY:-$HOME/gas-hermes}"
export PATH="$HOME/go/bin:$PATH"
BUDGET_FILE="$HOME/gas-hermes/.budget-daily.json"
CB_FILE="$HOME/gas-hermes/.circuit-breaker.json"
DEATH_LOG="/tmp/hermes-deaths.log"
MAX_PROCESSES=3
MAX_SESSION_AGE_MIN=30
MAX_CB_ENTRIES=20
DAILY_BUDGET=500000

VERBOSE=false
[[ "${1:-}" == "--verbose" ]] && VERBOSE=true

# --- Alert accumulator ---
ALERTS=()
DETAILS=""

alert() {
    ALERTS+=("$1")
    $VERBOSE && echo "[ALERT] $1" >&2
}

detail() {
    $VERBOSE && echo "[INFO] $1" >&2
    DETAILS="${DETAILS}  - $1
"
}

# ============================================================
# 1. Process monitoring — count running hermes chat processes
# ============================================================
# Count actual hermes processes by process name (not full cmdline).
# pgrep -f 'hermes chat' falsely matches tmux/bash wrappers that contain
# the string in their arguments. pgrep hermes matches only processes whose
# /proc/comm is "hermes" (the actual python hermes binary).
#
# Also exclude our own PID tree (watchdog/janitor may run inside a hermes session).
EXCLUDE_PIDS="$$"
CURRENT=$$
while [ "$CURRENT" -gt 1 ]; do
    PARENT=$(cat /proc/$CURRENT/status 2>/dev/null | grep PPid: | awk '{print $2}')
    [ -z "$PARENT" ] && break
    EXCLUDE_PIDS="${EXCLUDE_PIDS}|${PARENT}"
    CURRENT=$PARENT
done

PROCESS_COUNT=0
while IFS= read -r pid; do
    [ -n "$pid" ] && PROCESS_COUNT=$((PROCESS_COUNT + 1))
done < <(pgrep hermes 2>/dev/null | grep -v -E "^(${EXCLUDE_PIDS})$")

detail "hermes chat processes: $PROCESS_COUNT (max $MAX_PROCESSES)"

if [ "$PROCESS_COUNT" -gt "$MAX_PROCESSES" ]; then
    alert "Rogue agents: $PROCESS_COUNT hermes chat processes (max $MAX_PROCESSES)"
fi

# ============================================================
# 2. Session age check — find and kill stale sessions (>30 min)
# ============================================================
SESSIONS_JSON=$(gc --city "$GC_CITY" session list --json 2>/dev/null || echo '[]')
SESSION_COUNT=0
STALE_COUNT=0
FORMULA_SESSION_COUNT=0
KILLED_SESSIONS=""

if [ "$SESSIONS_JSON" != "[]" ] && [ -n "$SESSIONS_JSON" ]; then
    SESSION_COUNT=$(echo "$SESSIONS_JSON" | jq 'length')
    NOW_EPOCH=$(date +%s)

    # Iterate sessions: check age, identify formula sessions, kill stale ones
    while IFS= read -r sid; do
        [ -z "$sid" ] && continue
        session=$(echo "$SESSIONS_JSON" | jq -r --arg sid "$sid" '.[] | select(.id == $sid)')
        state=$(echo "$session" | jq -r '.state // "unknown"')
        template=$(echo "$session" | jq -r '.template // ""')
        created=$(echo "$session" | jq -r '.created_at // ""')

        # Skip non-active sessions
        if [ "$state" != "active" ]; then
            continue
        fi

        # Check if it's a formula session (no template = formula-spawned)
        if [ -z "$template" ] || [ "$template" = "null" ]; then
            FORMULA_SESSION_COUNT=$((FORMULA_SESSION_COUNT + 1))
        fi

        # Calculate age
        if [ -n "$created" ] && [ "$created" != "null" ]; then
            # Handle ISO timestamps and epoch
            if echo "$created" | grep -qE '^[0-9]{4}-'; then
                created_epoch=$(date -d "$created" +%s 2>/dev/null || echo 0)
            else
                created_epoch=$(echo "$created" | grep -oE '^[0-9]+' | head -1)
            fi

            if [ "$created_epoch" -gt 0 ]; then
                age_seconds=$((NOW_EPOCH - created_epoch))
                age_minutes=$((age_seconds / 60))

                if [ "$age_minutes" -gt "$MAX_SESSION_AGE_MIN" ]; then
                    STALE_COUNT=$((STALE_COUNT + 1))
                    detail "Killing stale session $sid (age: ${age_minutes}m, state: $state)"
                    gc --city "$GC_CITY" session kill "$sid" 2>/dev/null || true
                    KILLED_SESSIONS="${KILLED_SESSIONS}${sid},"
                fi
            fi
        fi
    done < <(echo "$SESSIONS_JSON" | jq -r '.[].id // empty')
fi

detail "Active sessions: $SESSION_COUNT, stale killed: $STALE_COUNT, formula sessions: $FORMULA_SESSION_COUNT"

if [ "$STALE_COUNT" -gt 0 ]; then
    alert "Killed $STALE_COUNT stale session(s) [${KILLED_SESSIONS%,}]"
fi

# ============================================================
# 3. Budget — actual Hermes token tracking (replaces blind estimate)
# ============================================================
if [ -f "$BUDGET_FILE" ]; then
    BUDGET_USED=$(jq -r '.tokens_used // 0' "$BUDGET_FILE" 2>/dev/null || echo 0)
    BUDGET_DAILY=$(jq -r '.daily_budget // 500000' "$BUDGET_FILE" 2>/dev/null || echo 500000)
else
    BUDGET_USED=0
    BUDGET_DAILY=$DAILY_BUDGET
fi

# Try to get actual token data from Hermes session logs
TOKEN_SAMPLE=$(bash "$GC_CITY/scripts/hermes-token-sampler.sh" 2>/dev/null || echo '{"total_tokens":0,"session_count":0}')
HERMES_TOKENS=$(echo "$TOKEN_SAMPLE" | jq -r '.total_tokens // 0')
HERMES_SESSIONS_SAMPLED=$(echo "$TOKEN_SAMPLE" | jq -r '.session_count // 0')

# The budget file already includes tokens recorded by budget-check.sh record.
# Hermes tokens represent what's currently in active sessions.
# If hermes-token-sampler --record is run periodically (e.g., by janitor),
# budget-check.sh already has the cumulative total. Use that directly.
# Only fall back to estimation if budget shows 0 (recorder not wired yet).
if [ "$BUDGET_USED" -gt 0 ]; then
    # Budget file has real data — use it directly
    TOTAL_ESTIMATE=$BUDGET_USED
    detail "Budget: $BUDGET_USED tokens recorded (hermes sessions sampled: $HERMES_TOKENS tokens in $HERMES_SESSIONS_SAMPLED active sessions)"
else
    # Fallback: estimate from active sessions (the old blind spot behavior)
    FORMULA_AVG_TOKENS=50000
    FORMULA_ESTIMATE=$((FORMULA_SESSION_COUNT * FORMULA_AVG_TOKENS))
    TOTAL_ESTIMATE=$((BUDGET_USED + FORMULA_ESTIMATE))
    detail "Budget: NO recorded data — estimating $FORMULA_ESTIMATE tokens ($FORMULA_SESSION_COUNT sessions × ~50K avg)"
fi
TOTAL_ESTIMATE_K=$((TOTAL_ESTIMATE / 1000))
BUDGET_K=$((BUDGET_DAILY / 1000))

if [ "$TOTAL_ESTIMATE" -gt "$BUDGET_DAILY" ]; then
    alert "Budget over: ~${TOTAL_ESTIMATE_K}K/${BUDGET_K}K tokens (estimated)"
fi

# ============================================================
# 4. Circuit breaker health — check for state bloat
# ============================================================
CB_ENTRIES=0
CB_PAUSED=0

if [ -f "$CB_FILE" ]; then
    # Count entries excluding the structural keys (threshold, max_backoff_seconds, auto_resume_seconds, failures)
    CB_ENTRIES=$(jq '[paths(type == "object") as $p | getpath($p) | objects | select(has("failures"))] | length' "$CB_FILE" 2>/dev/null || echo 0)
    CB_PAUSED=$(jq '[to_entries[] | select(.value.paused == true)] | length' "$CB_FILE" 2>/dev/null || echo 0)
fi

detail "Circuit breaker: $CB_ENTRIES entries, $CB_PAUSED paused (max $MAX_CB_ENTRIES)"

if [ "$CB_ENTRIES" -gt "$MAX_CB_ENTRIES" ]; then
    alert "Circuit breaker bloat: $CB_ENTRIES entries (max $MAX_CB_ENTRIES), $CB_PAUSED paused"
fi

# ============================================================
# 5. Death log — check for recent deaths (last hour)
# ============================================================
DEATH_COUNT=0
RECENT_DEATHS=""

if [ -f "$DEATH_LOG" ]; then
    CUTOFF=$(date -d '1 hour ago' +%s 2>/dev/null || echo 0)
    if [ "$CUTOFF" -gt 0 ]; then
        # Parse ISO timestamps from the log (or epoch)
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            ts=$(echo "$line" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
            if [ -n "$ts" ]; then
                line_epoch=$(date -d "$ts" +%s 2>/dev/null || echo 0)
                if [ "$line_epoch" -ge "$CUTOFF" ]; then
                    DEATH_COUNT=$((DEATH_COUNT + 1))
                    RECENT_DEATHS="${RECENT_DEATHS}${line}
"
                fi
            fi
        done < "$DEATH_LOG"
    fi
fi

detail "Deaths (last hour): $DEATH_COUNT"

if [ "$DEATH_COUNT" -gt 0 ]; then
    alert "$DEATH_COUNT recent death(s) in last hour"
fi

# ============================================================
# 6. Output summary
# ============================================================
if [ ${#ALERTS[@]} -eq 0 ]; then
    echo "WATCHDOG OK: $SESSION_COUNT sessions, ~${TOTAL_ESTIMATE_K}K/${BUDGET_K}K tokens, $DEATH_COUNT deaths"
    exit 0
else
    echo "WATCHDOG ALERT: ${ALERTS[*]}"
    if $VERBOSE; then
        echo ""
        echo "Details:"
        echo -n "$DETAILS"
    fi
    exit 1
fi
