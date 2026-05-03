#!/usr/bin/env bash
# factory-janitor.sh — HermGC Factory Janitor: lightweight health checks + auto-fixes
# Designed to be called by the Factory Janitor Hermes cron OR standalone.
#
# Usage:
#   factory-janitor.sh              — run checks, auto-fix, output JSON report
#   factory-janitor.sh --check-only — run checks only, no auto-fixes
#
# Output: JSON to stdout for Hermes cron consumption
# Exit codes: 0 = all healthy (or fixed), 1 = issues that need LLM attention

set -uo pipefail

# --- Configuration ---
GC_CITY="${GC_CITY:-$HOME/gas-hermes}"
export PATH="$HOME/go/bin:$PATH"
CB_FILE="$HOME/gas-hermes/.circuit-breaker.json"
CB_RESET_TEMPLATE='{"threshold":3,"max_backoff_seconds":1800,"auto_resume_seconds":3600,"failures":{}}'
MAX_CB_ENTRIES=20
JANITOR_LOG="$HOME/gas-hermes/.gc/runtime/janitor.log"
CHECK_ONLY=false

[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=true

# --- Helpers ---
mkdir -p "$(dirname "$JANITOR_LOG")"
log() { echo "[$(date -Iseconds)] $*" >> "$JANITOR_LOG"; }

# Accumulate results
RESULTS="{}"
add_result() {
    local key="$1" status="$2" message="$3" action="${4:-}"
    RESULTS=$(echo "$RESULTS" | jq --arg k "$key" --arg s "$status" --arg m "$message" --arg a "$action" \
        '. + {($k): {status: $s, message: $m, action: $a}}')
}

NEEDS_ATTENTION=false
mark_attention() { NEEDS_ATTENTION=true; }

# ============================================================
# 1. Run watchdog script
# ============================================================
WATCHDOG_OUTPUT=$(bash "$GC_CITY/scripts/hermes-watchdog.sh" 2>&1)
WATCHDOG_EXIT=$?

if [ $WATCHDOG_EXIT -eq 0 ]; then
    add_result "watchdog" "ok" "$WATCHDOG_OUTPUT"
else
    add_result "watchdog" "alert" "$WATCHDOG_OUTPUT"
    mark_attention
    log "WATCHDOG ALERT: $WATCHDOG_OUTPUT"
fi

# ============================================================
# 1b. Token tracking — sample Hermes sessions and record to budget
# ============================================================
TOKEN_SAMPLE_OUTPUT=$(bash "$GC_CITY/scripts/hermes-token-sampler.sh" --record 2>&1)
TOKEN_SAMPLE_EXIT=$?
if [ $TOKEN_SAMPLE_EXIT -eq 0 ]; then
    TOKENS_RECORDED=$(echo "$TOKEN_SAMPLE_OUTPUT" | jq -r '.total_tokens // 0')
    add_result "token_tracking" "ok" "Sampled $TOKENS_RECORDED tokens from active Hermes sessions"
    log "TOKEN_TRACK: $TOKEN_SAMPLE_OUTPUT"
else
    add_result "token_tracking" "info" "Token sampler unavailable (non-fatal)"
fi

# ============================================================
# 2. Circuit breaker health
# ============================================================
CB_ENTRIES=0
CB_PAUSED=0

if [ -f "$CB_FILE" ]; then
    # Count tracked beads (top-level keys minus structural ones)
    CB_ENTRIES=$(jq '[keys[] | select(. as $k | ["threshold","max_backoff_seconds","auto_resume_seconds","failures"] | index($k) | not)] | length' "$CB_FILE" 2>/dev/null || echo 0)
    CB_PAUSED=$(jq '[to_entries[] | select(.value.paused == true)] | length' "$CB_FILE" 2>/dev/null || echo 0)
fi

if [ "$CB_ENTRIES" -gt "$MAX_CB_ENTRIES" ]; then
    add_result "circuit_breaker" "alert" "$CB_ENTRIES entries, $CB_PAUSED paused (max $MAX_CB_ENTRIES)"
    mark_attention

    if ! $CHECK_ONLY; then
        log "CB_BLOAT: Pruning circuit breaker ($CB_ENTRIES entries)"
        bash "$GC_CITY/scripts/circuit-breaker.sh" prune 2>&1 >> "$JANITOR_LOG" || true
        CB_ENTRIES_AFTER=$(jq '[keys[] | select(. as $k | ["threshold","max_backoff_seconds","auto_resume_seconds","failures"] | index($k) | not)] | length' "$CB_FILE" 2>/dev/null || echo 0)
        if [ "$CB_ENTRIES_AFTER" -lt "$CB_ENTRIES" ]; then
            add_result "circuit_breaker_fix" "fixed" "Pruned circuit breaker from $CB_ENTRIES to $CB_ENTRIES_AFTER entries"
        else
            # Prune didn't help — fall back to nuclear reset
            log "CB_BLOAT: Prune insufficient, resetting circuit breaker"
            echo "$CB_RESET_TEMPLATE" | jq '.' > "$CB_FILE"
            add_result "circuit_breaker_fix" "fixed" "Reset circuit breaker from $CB_ENTRIES entries to clean state (prune insufficient)"
        fi
    fi
else
    add_result "circuit_breaker" "ok" "$CB_ENTRIES entries, $CB_PAUSED paused"
fi

# ============================================================
# 3. Dolt health
# ============================================================
DOLT_PID=$(pgrep -f "dolt sql-server" 2>/dev/null | head -1)

if [ -n "$DOLT_PID" ]; then
    # Verify it's actually responding
    DOLT_CHECK=$(gc --city "$GC_CITY" bd sql "SELECT 1" 2>/dev/null)
    if [ $? -eq 0 ]; then
        add_result "dolt" "ok" "Running (PID $DOLT_PID), responding to queries"
    else
        add_result "dolt" "alert" "Running (PID $DOLT_PID) but not responding to queries"
        mark_attention
        log "DOLT_UNRESPONSIVE: PID $DOLT_PID running but queries fail"
    fi
else
    add_result "dolt" "alert" "Not running"
    mark_attention

    if ! $CHECK_ONLY; then
        log "DOLT_DOWN: Attempting supervisor restart"
        # Try supervisor status first
        SUP_STATUS=$(gc --city "$GC_CITY" supervisor status 2>/dev/null)
        if echo "$SUP_STATUS" | grep -q "running"; then
            # Supervisor is up, Dolt should have been started by it
            # Try triggering a supervisor restart of the dolt pack
            gc --city "$GC_CITY" supervisor restart 2>/dev/null
            sleep 3
            DOLT_PID_NEW=$(pgrep -f "dolt sql-server" 2>/dev/null | head -1)
            if [ -n "$DOLT_PID_NEW" ]; then
                add_result "dolt_fix" "fixed" "Restarted via supervisor (new PID $DOLT_PID_NEW)"
            else
                add_result "dolt_fix" "failed" "Supervisor restart did not bring Dolt back up"
            fi
        else
            add_result "dolt_fix" "failed" "Supervisor not running, cannot auto-restart Dolt"
        fi
    fi
fi

# ============================================================
# 4. Supervisor health
# ============================================================
SUP_STATUS=$(gc --city "$GC_CITY" supervisor status 2>/dev/null)
SUP_EXIT=$?

if [ $SUP_EXIT -eq 0 ] && echo "$SUP_STATUS" | grep -q "running"; then
    SUP_PID=$(echo "$SUP_STATUS" | grep -oP 'PID \K\d+' | head -1)
    add_result "supervisor" "ok" "Running (PID $SUP_PID)"
else
    add_result "supervisor" "alert" "Not running or status check failed: $SUP_STATUS"
    mark_attention
    log "SUPERVISOR_DOWN: $SUP_STATUS"

    if ! $CHECK_ONLY; then
        log "SUPERVISOR_DOWN: Attempting restart"
        nohup bash -c "export PATH=\"$HOME/go/bin:$PATH\" && gc --city $GC_CITY supervisor run" > /dev/null 2>&1 &
        sleep 2
        SUP_STATUS_NEW=$(gc --city "$GC_CITY" supervisor status 2>/dev/null)
        if echo "$SUP_STATUS_NEW" | grep -q "running"; then
            add_result "supervisor_fix" "fixed" "Supervisor restarted successfully"
        else
            add_result "supervisor_fix" "failed" "Supervisor restart failed"
        fi
    fi
fi

# ============================================================
# 5. Fix registry staleness check
# ============================================================
FIX_REGISTRY="$GC_CITY/.gc/FIX_REGISTRY.md"
if [ -f "$FIX_REGISTRY" ]; then
    # Check last-updated date in the file
    LAST_UPDATED=$(grep -oP 'Last.updated.\*: \K[0-9-]+' "$FIX_REGISTRY" | head -1)
    TODAY=$(date +%Y-%m-%d)
    if [ "$LAST_UPDATED" = "$TODAY" ]; then
        add_result "fix_registry" "ok" "Last updated today ($LAST_UPDATED)"
    else
        add_result "fix_registry" "info" "Last updated $LAST_UPDATED (not today)"
        # Not an alert — just informational for the LLM
    fi
else
    add_result "fix_registry" "alert" "File not found at $FIX_REGISTRY"
fi

# ============================================================
# 6. Stale Hermes sessions cleanup
# ============================================================
SESSION_JSON=$(gc --city "$GC_CITY" session list --json 2>/dev/null || echo '[]')
if [ "$SESSION_JSON" != "[]" ] && [ -n "$SESSION_JSON" ]; then
    SESSION_COUNT=$(echo "$SESSION_JSON" | jq 'length')
    add_result "sessions" "ok" "$SESSION_COUNT active sessions"
else
    add_result "sessions" "ok" "0 active sessions"
fi

# ============================================================
# Output JSON report
# ============================================================
TIMESTAMP=$(date -Iseconds)
FINAL=$(echo "$RESULTS" | jq --arg ts "$TIMESTAMP" --arg needs_attention "$NEEDS_ATTENTION" \
    '{timestamp: $ts, needs_attention: ($needs_attention == "true"), results: .}')

echo "$FINAL"

if $NEEDS_ATTENTION; then
    exit 1
else
    exit 0
fi
