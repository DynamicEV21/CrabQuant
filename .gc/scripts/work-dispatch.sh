#!/usr/bin/env bash
# work-dispatch.sh — Dispatch ready beads to Gas Hermes sessions
# §19: Guardrails: budget preflight + circuit breaker before any sling
set -euo pipefail

export PATH="$HOME/go/bin:$PATH"
CITY="${GC_CITY:-$HOME/gas-hermes}"
RIGS=(/home/Zev/development/CrabQuant /home/Zev/development/alpha-lab)

# Guardrail: budget preflight — must pass before any token spend
if ! "$CITY/scripts/budget-check.sh" preflight 2>&1; then
    echo "[$(date -Iseconds)] [work-dispatch] ABORTED: budget preflight failed" >&2
    exit 1
fi

for rig_dir in "${RIGS[@]}"; do
  if [ -d "$rig_dir/.beads" ]; then
    READY=$(cd "$rig_dir" && bd ready --json 2>/dev/null || echo "[]")
    if [ -n "$READY" ] && [ "$READY" != "[]" ]; then
      RIG_NAME=$(basename "$rig_dir")
      echo "$READY" | jq -r '.[] | .id' 2>/dev/null | while read -r BEAD_ID; do
        # Guardrail: circuit breaker check per bead
        if ! "$CITY/scripts/circuit-breaker.sh" check "$BEAD_ID" 2>&1; then
            echo "[$(date -Iseconds)] [work-dispatch] SKIPPING '$BEAD_ID': circuit breaker tripped" >&2
            continue
        fi

        # Guardrail: re-check budget before each sling (another dispatch may have consumed tokens)
        if ! "$CITY/scripts/budget-check.sh" preflight 2>&1; then
            echo "[$(date -Iseconds)] [work-dispatch] ABORTED: budget exhausted mid-dispatch" >&2
            exit 1
        fi

        echo "[$(date -Iseconds)] [work-dispatch] Slinging $RIG_NAME/hermes $BEAD_ID" >&2
        gc --city "$CITY" sling "$RIG_NAME/hermes" "$BEAD_ID" --nudge 2>/dev/null || {
            echo "[$(date -Iseconds)] [work-dispatch] SLING FAILED for $BEAD_ID, recording failure" >&2
            "$CITY/scripts/circuit-breaker.sh" record "$BEAD_ID" 2>&1 || true
            continue
        }
        # Record success — reset circuit breaker for this bead
        "$CITY/scripts/circuit-breaker.sh" reset "$BEAD_ID" 2>&1 || true
      done
    fi
  fi
done
