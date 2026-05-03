#!/usr/bin/env bash
# budget-check.sh — Daily token spend cap
# §19.4: Check daily budget before LLM invocations. Hard stop at 100%.
#
# Usage:
#   budget-check.sh preflight              — check if budget allows (exit 0=OK, 1=blocked)
#   budget-check.sh record <tokens>        — record token usage
#   budget-check.sh status                 — show current budget state
#   budget-check.sh init <budget>          — initialize/reset daily budget
#
# State file: ~/gas-hermes/.budget-daily.json
# Format: {"date":"2026-05-01","daily_budget":50000,"tokens_used":0,"last_updated":"...","sessions_today":0,"sessions_killed_by_budget":0,"circuit_breaks_today":0,"hard_stops_today":0}

set -euo pipefail

CITY="${GC_CITY:-}"
STATE_FILE="${BUDGET_STATE_FILE:-$HOME/gas-hermes/.budget-daily.json}"
WARNING_PCT="${BUDGET_WARNING_PCT:-50}"
TODAY=$(date +%Y-%m-%d)

init_budget() {
    local budget="${1:-50000}"
    cat > "$STATE_FILE" << EOF
{
  "date": "$TODAY",
  "daily_budget": $budget,
  "tokens_used": 0,
  "last_updated": "$(date -Iseconds)",
  "sessions_today": 0,
  "sessions_killed_by_budget": 0,
  "circuit_breaks_today": 0,
  "hard_stops_today": 0
}
EOF
    echo "[$(date -Iseconds)] [budget] Initialized daily budget: $budget tokens" >&2
}

check_date_reset() {
    if [ ! -f "$STATE_FILE" ]; then
        init_budget 50000
        return
    fi
    local stored_date=$(jq -r '.date' "$STATE_FILE")
    if [ "$stored_date" != "$TODAY" ]; then
        local prev_budget=$(jq -r '.daily_budget' "$STATE_FILE")
        echo "[$(date -Iseconds)] [budget] New day ($TODAY). Resetting daily counter. Previous: used $(jq -r '.tokens_used' "$STATE_FILE") / $prev_budget" >> "${STATE_FILE}.history"
        init_budget "$prev_budget"
    fi
}

preflight() {
    check_date_reset
    local used=$(jq -r '.tokens_used' "$STATE_FILE")
    local budget=$(jq -r '.daily_budget' "$STATE_FILE")
    local pct=0
    if [ "$budget" -gt 0 ]; then
        pct=$((used * 100 / budget))
    fi
    
    if [ "$pct" -ge 100 ]; then
        echo "[$(date -Iseconds)] [budget] HARD STOP: daily budget exceeded ($used / $budget tokens = ${pct}%)" >&2
        jq '.hard_stops_today += 1' "$STATE_FILE" > "${STATE_FILE}.tmp" && mv -f "${STATE_FILE}.tmp" "$STATE_FILE"
        # Kill all tmux sessions on the gas-hermes socket
        tmux -L gas-hermes list-sessions 2>/dev/null | while read -r sname; do
            echo "[$(date -Iseconds)] [budget] Killing tmux session '$sname'" >&2
            tmux -L gas-hermes kill-session -t "$sname" 2>/dev/null || true
        done
        # Send mail alert
        if [ -n "${CITY:-}" ]; then
            gc --city "$CITY" mail send human "BUDGET HARD STOP: daily budget exceeded ($used / $budget tokens = ${pct}%). All tmux sessions killed." 2>/dev/null || true
        fi
        exit 1
    fi

    if [ "$pct" -ge "$WARNING_PCT" ]; then
        echo "[$(date -Iseconds)] [budget] WARNING: daily budget at ${pct}% ($used / $budget tokens)" >&2
        if [ -n "${CITY:-}" ]; then
            gc --city "$CITY" mail send human "BUDGET WARNING: daily budget at ${pct}% ($used / $budget tokens)" 2>/dev/null || true
        fi
    fi
    
    exit 0
}

record_usage() {
    check_date_reset
    local tokens="${1:?Usage: budget-check.sh record <tokens>}"
    jq --argjson tokens "$tokens" --arg now "$(date -Iseconds)" \
       '.tokens_used += $tokens | .last_updated = $now' \
       "$STATE_FILE" > "${STATE_FILE}.tmp" && mv -f "${STATE_FILE}.tmp" "$STATE_FILE"
    echo "[$(date -Iseconds)] [budget] Recorded $tokens tokens. Total: $(jq -r '.tokens_used' "$STATE_FILE") / $(jq -r '.daily_budget' "$STATE_FILE")" >&2
}

status() {
    check_date_reset
    jq '.' "$STATE_FILE"
    local used=$(jq -r '.tokens_used' "$STATE_FILE")
    local budget=$(jq -r '.daily_budget' "$STATE_FILE")
    local pct=0
    if [ "$budget" -gt 0 ]; then
        pct=$((used * 100 / budget))
    fi
    echo "Budget utilization: ${pct}% ($used / $budget tokens)" >&2
}

case "${1:-}" in
    preflight) preflight ;;
    record) record_usage "${2:?Usage: budget-check.sh record <tokens>}" ;;
    status) status ;;
    init) init_budget "${2:-50000}" ;;
    *) echo "Usage: budget-check.sh {preflight|record|status|init [budget]}" >&2; exit 1 ;;
esac
