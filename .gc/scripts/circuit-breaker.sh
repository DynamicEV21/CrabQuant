#!/usr/bin/env bash
# circuit-breaker.sh — Consecutive failure circuit breaker
# §19.3: Track consecutive failures per bead. After 3, pause the pool.
#
# Usage:
#   circuit-breaker.sh check <bead-id>    — check if bead is circuit-broken
#   circuit-breaker.sh record <bead-id>   — record a failure
#   circuit-breaker.sh reset <bead-id>    — reset failure count (on success)
#   circuit-breaker.sh status              — show all circuit states
#   circuit-breaker.sh prune               — remove stale entries (safe Hermes-layer fix)
#
# Exit codes:
#   0: OK (can proceed)
#   1: PAUSED (circuit breaker tripped)
#
# State file: ~/gas-hermes/.circuit-breaker.json

set -euo pipefail

STATE_FILE="${CIRCUIT_BREAKER_STATE:-$HOME/gas-hermes/.circuit-breaker.json}"
CITY="${GC_CITY:-}"
CONSECUTIVE_LIMIT=3
BACKOFF_BASE=60  # seconds
MAX_BACKOFF=1800 # 30 minutes
AUTO_RESUME=3600 # 1 hour
PRUNE_STALE_AGE=$((AUTO_RESUME * 2)) # 2 hours — prune paused entries older than this
PRUNE_INACTIVE_AGE=$((AUTO_RESUME * 3)) # 3 hours — prune any inactive entry older than this

init_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo '{}' | jq '.' > "$STATE_FILE"
    fi
}

get_bead_state() {
    local bead_id="$1"
    jq -r --arg id "$bead_id" '.[$id] // {"failures":0,"first_failure":null,"paused":false,"paused_at":null}' "$STATE_FILE"
}

record_failure() {
    local bead_id="$1"
    init_state
    local now=$(date +%s)
    local state=$(get_bead_state "$bead_id")
    local failures=$(echo "$state" | jq '.failures')
    local first_failure=$(echo "$state" | jq --arg now "$now" 'if .first_failure == null then $now | tonumber else .first_failure end')
    
    failures=$((failures + 1))
    
    # Check if auto-resume should apply
    local paused_at=$(echo "$state" | jq '.paused_at // 0')
    if [ "$paused_at" -gt 0 ]; then
        local elapsed=$((now - paused_at))
        if [ "$elapsed" -ge "$AUTO_RESUME" ]; then
            echo "[$(date -Iseconds)] [circuit-breaker] Auto-resuming '$bead_id' after ${elapsed}s cooldown" >&2
            failures=1
            first_failure=$now
        fi
    fi
    
    local paused="false"
    local paused_at_val="null"
    local backoff=0
    
    if [ "$failures" -ge "$CONSECUTIVE_LIMIT" ]; then
        paused="true"
        paused_at_val="$now"
        backoff=$((BACKOFF_BASE * (2 ** (failures - CONSECUTIVE_LIMIT))))
        if [ "$backoff" -gt "$MAX_BACKOFF" ]; then
            backoff=$MAX_BACKOFF
        fi
        echo "[$(date -Iseconds)] [circuit-breaker] TRIPPED: '$bead_id' has $failures consecutive failures. PAUSED. Backoff: ${backoff}s" >&2
    else
        backoff=$((BACKOFF_BASE * (2 ** (failures - 1))))
        echo "[$(date -Iseconds)] [circuit-breaker] Failure #$failures for '$bead_id'. Next backoff: ${backoff}s" >&2
    fi
    
    jq --arg id "$bead_id" \
       --argjson failures "$failures" \
       --argjson first "$first_failure" \
       --arg paused "$paused" \
       --argjson paused_at "$paused_at_val" \
       --argjson backoff "$backoff" \
       '.[$id] = {"failures": $failures, "first_failure": $first, "paused": ($paused == "true"), "paused_at": $paused_at, "backoff_seconds": $backoff}' \
       "$STATE_FILE" > "${STATE_FILE}.tmp" && mv -f "${STATE_FILE}.tmp" "$STATE_FILE"
    
    if [ "$paused" = "true" ]; then
        # Send mail notification on trip
        if [ -n "$CITY" ]; then
            gc --city "$CITY" mail send human "CIRCUIT BREAKER: $bead_id tripped after $failures consecutive failures" 2>/dev/null || true
        fi
        # Note: gc pool pause is not a valid command; pool management
        # is handled externally by the GC controller.
        exit 1
    fi
    exit 0
}

reset() {
    local bead_id="$1"
    init_state
    echo "[$(date -Iseconds)] [circuit-breaker] Reset '$bead_id' (success)" >&2
    jq --arg id "$bead_id" 'del(.[$id])' "$STATE_FILE" > "${STATE_FILE}.tmp" && mv -f "${STATE_FILE}.tmp" "$STATE_FILE"
    exit 0
}

check() {
    local bead_id="$1"
    init_state
    local state=$(get_bead_state "$bead_id")
    local paused=$(echo "$state" | jq '.paused')
    
    if [ "$paused" = "true" ]; then
        local paused_at=$(echo "$state" | jq '.paused_at')
        local backoff=$(echo "$state" | jq '.backoff_seconds')
        local now=$(date +%s)
        local elapsed=$((now - paused_at))

        if [ "$elapsed" -ge "$AUTO_RESUME" ]; then
            echo "[$(date -Iseconds)] [circuit-breaker] Auto-resuming '$bead_id'" >&2
            # Reset failure count in state without calling reset() which exits
            jq --arg id "$bead_id" 'del(.[$id])' "$STATE_FILE" > "${STATE_FILE}.tmp" && mv -f "${STATE_FILE}.tmp" "$STATE_FILE"
            exit 0
        fi

        echo "PAUSED: '$bead_id' tripped circuit breaker. Elapsed: ${elapsed}s, backoff: ${backoff}s" >&2
        exit 1
    fi
    
    exit 0
}

status() {
    init_state
    if [ "$(jq 'length' "$STATE_FILE")" -eq 0 ]; then
        echo "No circuit breaker state."
        exit 0
    fi
    jq -r 'to_entries[] | select(.value | type == "object" and has("backoff_seconds")) | "\(.key): failures=\(.value.failures) paused=\(.value.paused) backoff=\(.value.backoff_seconds // 0)s"' "$STATE_FILE"
    exit 0
}

# prune removes stale entries that accumulate from normal formula session failures.
# - Removes paused entries older than 2×AUTO_RESUME (they would auto-resume anyway)
# - Removes non-paused entries whose first_failure is older than 3×AUTO_RESUME
# - Resets any paused entry that has exceeded AUTO_RESUME (same logic as check)
# Safe to run from cron/janitor — only removes entries, never blocks dispatch.
prune() {
    init_state
    local now=$(date +%s)
    local before_count
    before_count=$(jq '[keys[] | select(. as $k | ["threshold","max_backoff_seconds","auto_resume_seconds","failures"] | index($k) | not)] | length' "$STATE_FILE" 2>/dev/null || echo 0)

    if [ "$before_count" -eq 0 ]; then
        echo "[$(date -Iseconds)] [circuit-breaker] Prune: nothing to prune" >&2
        exit 0
    fi

    # Build list of keys to remove
    local keys_to_remove=""
    local keys_to_resume=""

    while IFS= read -r key; do
        [ -z "$key" ] && continue
        # Skip structural keys (they are scalars, not bead entries)
        case "$key" in threshold|max_backoff_seconds|auto_resume_seconds|failures) continue ;; esac
        local state
        state=$(jq -r --arg k "$key" '.[$k]' "$STATE_FILE")

        # Verify it's an actual CB entry (has "failures" field)
        local has_failures
        has_failures=$(echo "$state" | jq 'has("failures")')
        [ "$has_failures" = "false" ] && continue

        local paused=$(echo "$state" | jq '.paused')
        local first_failure=$(echo "$state" | jq '.first_failure // 0')

        if [ "$paused" = "true" ]; then
            local paused_at=$(echo "$state" | jq '.paused_at // 0')
            local elapsed=$((now - paused_at))
            if [ "$elapsed" -ge "$PRUNE_STALE_AGE" ]; then
                keys_to_remove="${keys_to_remove}${key} "
                echo "[$(date -Iseconds)] [circuit-breaker] Prune: removing paused '$key' (elapsed=${elapsed}s, prune threshold=${PRUNE_STALE_AGE}s)" >&2
            elif [ "$elapsed" -ge "$AUTO_RESUME" ]; then
                # Should have auto-resumed; reset it
                keys_to_resume="${keys_to_resume}${key} "
            fi
        else
            # Not paused — check if the failure is ancient (entry never reset)
            if [ "$first_failure" -gt 0 ]; then
                local age=$((now - first_failure))
                if [ "$age" -ge "$PRUNE_INACTIVE_AGE" ]; then
                    keys_to_remove="${keys_to_remove}${key} "
                    echo "[$(date -Iseconds)] [circuit-breaker] Prune: removing stale '$key' (age=${age}s, failures=$(echo "$state" | jq '.failures'))" >&2
                fi
            fi
        fi
    done < <(jq -r 'keys[]' "$STATE_FILE")

    local removed=0

    # Remove stale keys
    for key in $keys_to_remove; do
        jq --arg k "$key" 'del(.[$k])' "$STATE_FILE" > "${STATE_FILE}.tmp" && mv -f "${STATE_FILE}.tmp" "$STATE_FILE"
        removed=$((removed + 1))
    done

    # Resume expired paused keys
    for key in $keys_to_resume; do
        jq --arg k "$key" 'del(.[$k])' "$STATE_FILE" > "${STATE_FILE}.tmp" && mv -f "${STATE_FILE}.tmp" "$STATE_FILE"
        removed=$((removed + 1))
    done

    local after_count
    after_count=$(jq '[keys[] | select(. as $k | ["threshold","max_backoff_seconds","auto_resume_seconds","failures"] | index($k) | not)] | length' "$STATE_FILE" 2>/dev/null || echo 0)

    echo "[$(date -Iseconds)] [circuit-breaker] Prune: removed/resumed $removed entries ($before_count → $after_count)" >&2
    exit 0
}

case "${1:-}" in
    check) check "${2:?Usage: circuit-breaker.sh check <bead-id>}" ;;
    record) record_failure "${2:?Usage: circuit-breaker.sh record <bead-id>}" ;;
    reset) reset "${2:?Usage: circuit-breaker.sh reset <bead-id>}" ;;
    status) status ;;
    prune) prune ;;
    *) echo "Usage: circuit-breaker.sh {check|record|reset|status|prune} [bead-id]" >&2; exit 1 ;;
esac
