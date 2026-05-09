#!/usr/bin/env bash
# health-check-qrm.sh — quant-research-mas health monitoring
# §19: Guardrails: budget preflight before sling (health check itself is free)
set -euo pipefail

export PATH="$HOME/go/bin:$PATH"
CITY="${GC_CITY:-$HOME/gas-hermes}"
RIG_DIR="/home/Zev/development/quant-research-mas"
RIG_NAME="quant-research-mas"

OUTPUT=$(python3 ~/.hermes/scripts/qrm-health-check.py 2>&1)
if echo "$OUTPUT" | grep -q '\[SILENT\]'; then
  exit 0
fi

# Health check found an issue — about to sling (token spend). Guardrails first.
if ! "$CITY/scripts/budget-check.sh" preflight 2>&1; then
    echo "[$(date -Iseconds)] [health-check-qrm] ABORTED: budget preflight failed" >&2
    exit 1
fi

BEAD_ID=$(cd "$RIG_DIR" && unset BEADS_DIR && bd create --type=bug --title="Health Alert: quant-research-mas" --labels=qrm,escalation,health 2>/dev/null | grep -oP '(?<=issue: )\S+')
if [ -n "$BEAD_ID" ]; then
    if ! "$CITY/scripts/circuit-breaker.sh" check "$BEAD_ID" 2>&1; then
        echo "[$(date -Iseconds)] [health-check-qrm] SKIPPING: circuit breaker tripped on '$BEAD_ID'" >&2
        exit 0
    fi

    echo "[$(date -Iseconds)] [health-check-qrm] Slinging health alert $BEAD_ID" >&2
    gc --city "$CITY" sling "$RIG_NAME/hermes" "$BEAD_ID" --nudge 2>/dev/null || {
        echo "[$(date -Iseconds)] [health-check-qrm] SLING FAILED for $BEAD_ID, recording failure" >&2
        "$CITY/scripts/circuit-breaker.sh" record "$BEAD_ID" 2>&1 || true
        exit 1
    }
    "$CITY/scripts/circuit-breaker.sh" reset "$BEAD_ID" 2>&1 || true
fi
