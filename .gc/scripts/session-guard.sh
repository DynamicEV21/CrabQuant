#!/usr/bin/env bash
# session-guard.sh — Per-session turn limit for Gas City polecats
# §19.2: Kill tmux sessions after N turns to prevent runaway agents
#
# Usage: session-guard.sh <session-name> <max-turns>
#   session-name: tmux session name (e.g., CrabQuant-hermes-1)
#   max-turns: maximum turns before kill (test=5, prod=30)
#
# Environment:
#   HERMES_SESSION_TURNS_DIR: directory for turn counter files (default: /tmp/gc-turns)
#   GC_CITY: city directory for gc commands (set by GC when running orders)
#
# Turn counting: the calling script (work-dispatch.sh or order exec) writes
# to a counter file at $HERMES_SESSION_TURNS_DIR/<session-name>. This guard
# monitors that file and enforces the limit.

set -euo pipefail

SESSION_NAME="${1:?Usage: session-guard.sh <session-name> <max-turns>}"
MAX_TURNS="${2:?Usage: session-guard.sh <session-name> <max-turns>}"
CITY="${GC_CITY:-}"
TURNS_DIR="${HERMES_SESSION_TURNS_DIR:-/tmp/gc-turns}"
TURNS_FILE="$TURNS_DIR/$SESSION_NAME"

mkdir -p "$TURNS_DIR"

# Only initialize the turn counter file if it doesn't already exist
# (another process or a previous run may have created it)
if [ ! -f "$TURNS_FILE" ]; then
    echo "0" > "$TURNS_FILE"
fi

log() { echo "[$(date -Iseconds)] [session-guard] $*" >&2; }

log "Watching session '$SESSION_NAME' with max_turns=$MAX_TURNS"

# File-based turn counting: read the counter file written by the Hermes
# agent's shell wrapper, clamp to 0, and enforce the limit.
while tmux -L gas-hermes has-session -t "$SESSION_NAME" 2>/dev/null; do
    sleep 5

    CURRENT_TURNS=$(cat "$TURNS_FILE" 2>/dev/null || echo "0")
    # Clamp negative values to 0
    if [ "$CURRENT_TURNS" -lt 0 ] 2>/dev/null; then
        CURRENT_TURNS=0
    fi
    # Validate it's actually a number
    if ! [[ "$CURRENT_TURNS" =~ ^[0-9]+$ ]]; then
        CURRENT_TURNS=0
    fi

    if [ "$CURRENT_TURNS" -ge "$MAX_TURNS" ]; then
        log "TURN LIMIT REACHED: session '$SESSION_NAME' hit $CURRENT_TURNS turns (max=$MAX_TURNS)"
        log "Killing session '$SESSION_NAME'"
        tmux -L gas-hermes kill-session -t "$SESSION_NAME" 2>/dev/null || true
        # Write sentinel for monitoring
        echo "{\"session\":\"$SESSION_NAME\",\"killed_at\":$(date +%s),\"turns\":$CURRENT_TURNS,\"reason\":\"turn_limit\"}" >> "$TURNS_DIR/killed.jsonl"
        # Send notification
        if [ -n "$CITY" ]; then
            gc --city "$CITY" mail send human "SESSION KILLED: '$SESSION_NAME' hit $CURRENT_TURNS turns (max=$MAX_TURNS)" 2>/dev/null || true
        fi
        exit 0
    fi
done

log "Session '$SESSION_NAME' ended naturally"
rm -f "$TURNS_FILE"
