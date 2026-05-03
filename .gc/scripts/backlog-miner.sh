#!/usr/bin/env bash
# backlog-miner.sh — Scan KPI alerts and create specific maintenance beads
# Reads KPI JSON, checks for existing open beads, creates beads for new alerts.
# Runs as exec order (no LLM needed — pure data processing).
set -euo pipefail

export PATH="$HOME/go/bin:$PATH"
CITY="${GC_CITY:-$HOME/gas-hermes}"
LOG_TS="[$(date -Iseconds)] [backlog-miner]"

# Rig definitions: rig_name|rig_path|kpi_path
RIGS=(
  "alpha-lab|/home/Zev/development/alpha-lab|/home/Zev/.hermes/scripts/alpha-lab-ops-kpis.json"
  "crabquant|/home/Zev/development/CrabQuant|/home/Zev/.hermes/scripts/crabquant-ops-kpis.json"
)

created=0
skipped=0

for rig_def in "${RIGS[@]}"; do
  IFS='|' read -r RIG_NAME RIG_DIR KPI_FILE <<< "$rig_def"

  if [ ! -f "$KPI_FILE" ]; then
    echo "$LOG_TS SKIP $RIG_NAME: KPI file not found ($KPI_FILE)" >&2
    continue
  fi

  echo "$LOG_TS Processing $RIG_NAME..." >&2

  # Get existing bead titles (open + closed) to avoid duplicates
  # Use a file to avoid subshell issues
  OPEN_TITLES_FILE=$(mktemp)
  trap "rm -f '$OPEN_TITLES_FILE'" EXIT
  {
    (cd "$RIG_DIR" && bd list --status=open --json 2>/dev/null \
      | jq -r '.[].title // empty' 2>/dev/null || true)
    (cd "$RIG_DIR" && bd list --status=closed --json 2>/dev/null \
      | jq -r '.[].title // empty' 2>/dev/null || true)
  } | sort -u > "$OPEN_TITLES_FILE"

  # Extract warn+ alerts and create beads (process substitution avoids subshell counter bug)
  while IFS=$'\t' read -r SIGNAL LEVEL MESSAGE; do
    [ -z "$SIGNAL" ] && continue

    # Skip signals that are too broad — let directors handle strategy
    case "$SIGNAL" in
      git.uncommitted_files) continue ;;  # Too noisy, everyone has uncommitted files
      quality.complexity)    continue ;;  # Directors create specific refactor beads
    esac

    # Generate a deterministic bead title from the signal
    BEAD_TITLE="KPI Fix: ${SIGNAL} (${LEVEL})"

    # Check for duplicate
    if grep -qF "$BEAD_TITLE" "$OPEN_TITLES_FILE" 2>/dev/null; then
      echo "$LOG_TS SKIP $RIG_NAME: already open — $BEAD_TITLE" >&2
      skipped=$((skipped + 1))
      continue
    fi

    # Build a specific description from the KPI data
    DESC=$(jq -r --arg sig "$SIGNAL" '
      .signals[$sig] // "See KPI file for details."
    ' "$KPI_FILE" 2>/dev/null)

    # Map signal to bead type and priority
    case "$LEVEL" in
      critical) TYPE="bug"; PRIORITY="1" ;;
      warn)     TYPE="task"; PRIORITY="2" ;;
      *)        TYPE="task"; PRIORITY="3" ;;
    esac

    # Add actionable context to description
    case "$SIGNAL" in
      tests.report_age)
        ACTION="Run tests and regenerate QA report: cd $RIG_DIR && pytest --cov"
        DESC="${DESC}\n\nAction: ${ACTION}"
        ;;
      tests.coverage_below_target)
        TARGET=$(jq -r '.signals["tests.coverage_below_target"].target // "80%"' "$KPI_FILE" 2>/dev/null)
        ACTION="Increase test coverage to ${TARGET}. Focus on lowest-coverage modules."
        DESC="${DESC}\n\nAction: ${ACTION}"
        ;;
      deps.vulnerabilities)
        ACTION="Run: cd $RIG_DIR && pip-audit --fix"
        DESC="${DESC}\n\nAction: ${ACTION}"
        ;;
      deps.outdated)
        COUNT=$(jq -r '.signals["deps.outdated"].count // "?"' "$KPI_FILE" 2>/dev/null)
        ACTION="Review and update ${COUNT} outdated dependencies: pip list --outdated"
        DESC="${DESC}\n\nAction: ${ACTION}"
        ;;
      quality.lint_errors)
        COUNT=$(jq -r '.signals["quality.lint_errors"].count // "?"' "$KPI_FILE" 2>/dev/null)
        ACTION="Fix ${COUNT} lint errors: cd $RIG_DIR && ruff check --fix ."
        DESC="${DESC}\n\nAction: ${ACTION}"
        ;;
      quality.dead_code)
        COUNT=$(jq -r '.signals["quality.dead_code"].count // "?"' "$KPI_FILE" 2>/dev/null)
        ACTION="Review ${COUNT} dead code items: cd $RIG_DIR && vulture . --min-confidence 80"
        DESC="${DESC}\n\nAction: ${ACTION}"
        ;;
      git.stale_branches)
        ACTION="Clean up stale branches: cd $RIG_DIR && git branch -vv | grep ': gone]' | awk '{print $1}' | xargs git branch -D"
        DESC="${DESC}\n\nAction: ${ACTION}"
        ;;
      *)
        DESC="${DESC}\n\nAction: Investigate and resolve this ${LEVEL} alert."
        ;;
    esac

    # Create the bead
    RESULT=$(cd "$RIG_DIR" && unset BEADS_DIR && echo -e "$DESC" | bd create "$BEAD_TITLE" \
      --type="$TYPE" \
      --priority="$PRIORITY" \
      --labels="kpi,automated,${SIGNAL}" \
      --description=- \
      --json 2>/dev/null || echo "")

    if [ -n "$RESULT" ]; then
      BEAD_ID=$(echo "$RESULT" | jq -r '.id // empty' 2>/dev/null)
      echo "$LOG_TS CREATED $RIG_NAME: $BEAD_ID — $BEAD_TITLE" >&2
      created=$((created + 1))
    else
      echo "$LOG_TS FAILED $RIG_NAME: could not create bead — $BEAD_TITLE" >&2
    fi
  done < <(jq -r '.alerts[] | select(.level == "critical" or .level == "warn") | [.signal, .level, .message] | @tsv' "$KPI_FILE" 2>/dev/null)
done

echo "$LOG_TS Done. Created=$created, Skipped=$skipped" >&2
echo "{\"created\": $created, \"skipped\": $skipped}"
