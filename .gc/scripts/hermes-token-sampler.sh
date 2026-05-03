#!/usr/bin/env bash
# hermes-token-sampler.sh — Extract actual token usage from Hermes session logs
# Fixes ISSUE-001 (token-tracking blind spot) by reading session JSONL files
# instead of estimating from session count × average.
#
# Usage:
#   hermes-token-sampler.sh              — scan all Hermes sessions, output JSON
#   hermes-token-sampler.sh --record     — scan + record to budget file via budget-check.sh
#   hermes-token-sampler.sh --sessions   — output per-session breakdown
#
# Output: JSON to stdout
# Exit codes: 0 = success, 1 = error

set -uo pipefail

GC_CITY="${GC_CITY:-$HOME/gas-hermes}"
export PATH="$HOME/go/bin:$PATH"
BUDGET_FILE="$HOME/gas-hermes/.budget-daily.json"
HERMES_SESSIONS_DIR="$HOME/.hermes/sessions"

RECORD=false
SHOW_SESSIONS=false
[[ "${1:-}" == "--record" ]] && RECORD=true
[[ "${1:-}" == "--sessions" ]] && SHOW_SESSIONS=true

# --- Hermes session log format ---
# Hermes stores sessions as JSONL files. Each line with type "assistant"
# contains a "message" with "usage" having input_tokens, output_tokens, etc.
# The session files are typically at ~/.hermes/sessions/ or referenced by gc session list.

# --- Collect sessions from gc ---
SESSIONS_JSON=$(gc --city "$GC_CITY" session list --json 2>/dev/null || echo '[]')

if [ "$SESSIONS_JSON" = "[]" ] || [ -z "$SESSIONS_JSON" ]; then
    echo '{"total_input_tokens":0,"total_output_tokens":0,"total_tokens":0,"session_count":0,"sessions":[]}'
    exit 0
fi

TOTAL_INPUT=0
TOTAL_OUTPUT=0
TOTAL_CACHE_READ=0
TOTAL_CACHE_CREATION=0
SESSION_COUNT=0
SESSION_DETAILS="[]"

NOW_EPOCH=$(date +%s)
TODAY=$(date +%Y-%m-%d)

while IFS= read -r sid; do
    [ -z "$sid" ] && continue
    session=$(echo "$SESSIONS_JSON" | jq -r --arg sid "$sid" '.[] | select(.id == $sid)')
    state=$(echo "$session" | jq -r '.state // "unknown"')
    created=$(echo "$session" | jq -r '.created_at // ""')

    # Only count active sessions (they're the ones burning tokens)
    if [ "$state" != "active" ]; then
        continue
    fi

    # Get the session log file path
    log_path=$(echo "$session" | jq -r '.log_path // .path // ""')

    # If no explicit path, try common locations
    if [ -z "$log_path" ] || [ ! -f "$log_path" ]; then
        # Try to find session log by ID in hermes sessions dir
        if [ -d "$HERMES_SESSIONS_DIR" ]; then
            # Search for files containing the session ID
            log_path=$(find "$HERMES_SESSIONS_DIR" -name "*.jsonl" -type f 2>/dev/null | while read -r f; do
                if head -1 "$f" 2>/dev/null | grep -q "$sid"; then
                    echo "$f"
                    break
                fi
            done)
        fi
    fi

    # Also try gc session show for a log path
    if [ -z "$log_path" ] || [ ! -f "$log_path" ]; then
        log_path=$(gc --city "$GC_CITY" session show "$sid" 2>/dev/null | grep -oP '(?<=log:|path:)\s*\S+' | head -1 | tr -d ' ')
    fi

    SESSION_INPUT=0
    SESSION_OUTPUT=0
    SESSION_CACHE_READ=0
    SESSION_CACHE_CREATION=0

    if [ -n "$log_path" ] && [ -f "$log_path" ]; then
        # Parse JSONL: extract all usage fields from assistant messages
        # Hermes format: {"type":"assistant","message":{"usage":{"input_tokens":N,"output_tokens":N,...}}}
        # Or wrapped: {"type":"assistant","message":"<JSON string>"}
        token_data=$(jq -s '[.[] | 
            select(.type == "assistant") | 
            .message | 
            if type == "string" then fromjson else . end |
            select(.usage != null) |
            .usage
        ] | {
            input: ([.[].input_tokens // 0] | add),
            output: ([.[].output_tokens // 0] | add),
            cache_read: ([.[].cache_read_input_tokens // 0] | add),
            cache_creation: ([.[].cache_creation_input_tokens // 0] | add)
        }' "$log_path" 2>/dev/null || echo '{"input":0,"output":0,"cache_read":0,"cache_creation":0}')

        SESSION_INPUT=$(echo "$token_data" | jq '.input // 0')
        SESSION_OUTPUT=$(echo "$token_data" | jq '.output // 0')
        SESSION_CACHE_READ=$(echo "$token_data" | jq '.cache_read // 0')
        SESSION_CACHE_CREATION=$(echo "$token_data" | jq '.cache_creation // 0')
    fi

    TOTAL_INPUT=$((TOTAL_INPUT + SESSION_INPUT))
    TOTAL_OUTPUT=$((TOTAL_OUTPUT + SESSION_OUTPUT))
    TOTAL_CACHE_READ=$((TOTAL_CACHE_READ + SESSION_CACHE_READ))
    TOTAL_CACHE_CREATION=$((TOTAL_CACHE_CREATION + SESSION_CACHE_CREATION))
    SESSION_COUNT=$((SESSION_COUNT + 1))

    if $SHOW_SESSIONS; then
        SESSION_DETAILS=$(echo "$SESSION_DETAILS" | jq --arg sid "$sid" \
            --argjson input "$SESSION_INPUT" \
            --argjson output "$SESSION_OUTPUT" \
            --arg state "$state" \
            '. + [{"id": $sid, "input_tokens": $input, "output_tokens": $output, "state": $state}]')
    fi
done < <(echo "$SESSIONS_JSON" | jq -r '.[].id // empty')

TOTAL_ALL=$((TOTAL_INPUT + TOTAL_OUTPUT))

# Output JSON
RESULT=$(jq -n \
    --argjson input "$TOTAL_INPUT" \
    --argjson output "$TOTAL_OUTPUT" \
    --argjson cache_read "$TOTAL_CACHE_READ" \
    --argjson cache_creation "$TOTAL_CACHE_CREATION" \
    --argjson total "$TOTAL_ALL" \
    --argjson count "$SESSION_COUNT" \
    --argjson sessions "$SESSION_DETAILS" \
    '{
        total_input_tokens: $input,
        total_output_tokens: $output,
        total_cache_read_tokens: $cache_read,
        total_cache_creation_tokens: $cache_creation,
        total_tokens: $total,
        session_count: $count,
        sessions: $sessions
    }')

echo "$RESULT"

# Optionally record to budget file
if $RECORD && [ "$TOTAL_ALL" -gt 0 ]; then
    # We only record NEW tokens not already counted.
    # The approach: record the delta. We store a snapshot of last-seen totals
    # and only record the difference.
    SNAPSHOT_FILE="$HOME/gas-hermes/.token-snapshot.json"

    if [ -f "$SNAPSHOT_FILE" ]; then
        PREV_TOTAL=$(jq -r '.total_tokens // 0' "$SNAPSHOT_FILE" 2>/dev/null || echo 0)
        PREV_DATE=$(jq -r '.date // ""' "$SNAPSHOT_FILE" 2>/dev/null || echo "")
    else
        PREV_TOTAL=0
        PREV_DATE=""
    fi

    # Reset on new day
    if [ "$PREV_DATE" != "$TODAY" ]; then
        PREV_TOTAL=0
    fi

    DELTA=$((TOTAL_ALL - PREV_TOTAL))
    if [ "$DELTA" -gt 0 ]; then
        bash "$GC_CITY/scripts/budget-check.sh" record "$DELTA" 2>/dev/null || true
        echo "$RESULT" | jq --arg date "$TODAY" --arg now "$(date -Iseconds)" '. + {recorded_delta: $DELTA, snapshot_date: $date, snapshot_time: $now}' > "$SNAPSHOT_FILE"
        echo "[$(date -Iseconds)] [token-sampler] Recorded $DELTA new tokens (total now: $TOTAL_ALL, sessions: $SESSION_COUNT)" >&2
    else
        echo "[$(date -Iseconds)] [token-sampler] No new tokens to record (total: $TOTAL_ALL, prev: $PREV_TOTAL, sessions: $SESSION_COUNT)" >&2
        echo "$RESULT" | jq --arg date "$TODAY" --arg now "$(date -Iseconds)" '. + {recorded_delta: 0, snapshot_date: $date, snapshot_time: $now}' > "$SNAPSHOT_FILE"
    fi
fi

exit 0
