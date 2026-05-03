#!/usr/bin/env bash
# auto-qa.sh — Automatically create QA beads for recently closed work
# §19: Guardrails: budget preflight + circuit breaker before sling
set -euo pipefail

export PATH="$HOME/go/bin:$PATH"
CITY="${GC_CITY:-$HOME/gas-hermes}"
RIGS=(/home/Zev/development/CrabQuant /home/Zev/development/alpha-lab)

SINCE=$(date -u -d '30 minutes ago' +%Y-%m-%dT%H:%M:%SZ)

# Guardrail: budget preflight before creating QA beads (sling = token spend)
if ! "$CITY/scripts/budget-check.sh" preflight 2>&1; then
    echo "[$(date -Iseconds)] [auto-qa] ABORTED: budget preflight failed" >&2
    exit 1
fi

for rig_dir in "${RIGS[@]}"; do
  RIG_NAME=$(basename "$rig_dir")
  CLOSED=$(cd "$rig_dir" && bd list --status=closed --since="$SINCE" --json 2>/dev/null || echo "[]")
  echo "$CLOSED" | jq -r '.[] | .id + " " + (.labels // [] | join(","))' 2>/dev/null | while read -r BEAD_ID LABELS; do
    case "$LABELS" in
      *feature*|*bug*|*refactor*)
        QA_BEAD=$(cd "$rig_dir" && bd create --type=task --title="QA: Verify $BEAD_ID" --labels=qa,automated 2>/dev/null | grep -oP '(?<=issue: )\S+')
        if [ -n "$QA_BEAD" ]; then
            # Guardrail: re-check budget before each sling
            if ! "$CITY/scripts/budget-check.sh" preflight 2>&1; then
                echo "[$(date -Iseconds)] [auto-qa] ABORTED: budget exhausted mid-run" >&2
                exit 1
            fi

            # Guardrail: circuit breaker check for the QA bead
            if ! "$CITY/scripts/circuit-breaker.sh" check "$QA_BEAD" 2>&1; then
                echo "[$(date -Iseconds)] [auto-qa] SKIPPING QA for '$BEAD_ID': circuit breaker tripped on '$QA_BEAD'" >&2
                continue
            fi

            echo "[$(date -Iseconds)] [auto-qa] Slinging QA bead $QA_BEAD for $BEAD_ID" >&2
            gc --city "$CITY" sling "$RIG_NAME/hermes" "$QA_BEAD" --nudge 2>/dev/null || {
                echo "[$(date -Iseconds)] [auto-qa] SLING FAILED for $QA_BEAD, recording failure" >&2
                "$CITY/scripts/circuit-breaker.sh" record "$QA_BEAD" 2>&1 || true
                continue
            }
            "$CITY/scripts/circuit-breaker.sh" reset "$QA_BEAD" 2>&1 || true
        fi
        ;;
    esac
  done
done
