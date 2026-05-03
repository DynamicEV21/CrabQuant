#!/usr/bin/env bash
# dry-run-wrapper.sh — Dry-run mode for Gas City operations
# §19.5: Validate pipeline without spending tokens
#
# Usage:
#   dry-run-wrapper.sh doctor          — run gc doctor, report findings
#   dry-run-wrapper.sh orders          — show what orders would fire
#   dry-run-wrapper.sh sessions        — show what sessions would be managed
#   dry-run-wrapper.sh dispatch        — show what beads would be dispatched
#   dry-run-wrapper.sh full            — run all checks
#
# Set DRY_RUN=1 to enable dry-run mode in the environment

set -euo pipefail

CITY="${GC_CITY:-$HOME/gas-hermes}"
LOG_DIR="$CITY/.dry-run-logs"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d-%H%M%S).jsonl"

mkdir -p "$LOG_DIR"

export PATH="$HOME/go/bin:$PATH"

log_action() {
    local action="$1"
    local detail="$2"
    local entry
    entry=$(jq -n --arg action "$action" --arg detail "$detail" \
        '{timestamp:(now|strftime("%Y-%m-%dT%H:%M:%SZ")),action:$action,detail:$detail}')
    echo "$entry" >> "$LOG_FILE"
    echo "DRY-RUN: $action — $detail"
}

check_doctor() {
    log_action "doctor" "Running gc doctor"
    gc --city "$CITY" doctor 2>&1 | head -30
}

check_orders() {
    log_action "orders" "Checking order status"
    gc --city "$CITY" order list 2>&1
    log_action "orders" "Use 'gc --city $CITY order check --dry-run' to see what would fire (if supported)"
}

check_sessions() {
    log_action "sessions" "Checking session state"
    gc --city "$CITY" session list 2>&1 || true
    local zombie_count=$(tmux -L gas-hermes list-sessions 2>/dev/null | grep -c dead || echo 0)
    log_action "sessions" "Zombie sessions: $zombie_count"
}

check_dispatch() {
    log_action "dispatch" "Checking ready beads"
    for rig_dir in /home/Zev/development/CrabQuant /home/Zev/development/alpha-lab; do
        if [ -d "$rig_dir/.beads" ]; then
            local ready=$(cd "$rig_dir" && bd ready --json 2>/dev/null || echo "[]")
            local count=$(echo "$ready" | jq 'length')
            log_action "dispatch" "$rig_dir: $count ready beads"
        fi
    done
}

check_budget() {
    if [ -f "$CITY/.budget-daily.json" ]; then
        local budget_info=$(jq '.' "$CITY/.budget-daily.json")
        log_action "budget" "$budget_info"
    else
        log_action "budget" "No budget file found"
    fi
}

check_circuit_breaker() {
    if [ -f "$CITY/.circuit-breaker.json" ]; then
        "$CITY/scripts/circuit-breaker.sh" status 2>&1 || true
    else
        log_action "circuit_breaker" "No circuit breaker state found"
    fi
}

case "${1:-full}" in
    doctor) check_doctor ;;
    orders) check_orders ;;
    sessions) check_sessions ;;
    dispatch) check_dispatch ;;
    budget) check_budget ;;
    circuit) check_circuit_breaker ;;
    full)
        echo "=== DRY-RUN FULL CHECK ==="
        echo "City: $CITY"
        echo "Log: $LOG_FILE"
        echo ""
        check_doctor
        echo ""
        check_budget
        echo ""
        check_circuit_breaker
        echo ""
        check_sessions
        echo ""
        check_orders
        echo ""
        check_dispatch
        echo ""
        echo "=== DRY-RUN COMPLETE ==="
        echo "Zero tokens spent. Review log at: $LOG_FILE"
        ;;
    *) echo "Usage: dry-run-wrapper.sh {doctor|orders|sessions|dispatch|budget|circuit|full}" >&2; exit 1 ;;
esac
