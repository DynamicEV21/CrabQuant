#!/usr/bin/env bash
# dolt-recovery.sh — Recover a beads rig after Dolt database corruption or WSL reboot
# Usage: dolt-recovery.sh <rig-path> [--dry-run]
#   <rig-path>  Absolute or tilde path to the rig directory (e.g., ~/development/CrabQuant)
#   --dry-run   Show what would be done without making changes
#
# Steps:
#   1. Kill orphaned Dolt server processes
#   2. Remove corrupted .beads/dolt or .beads/embeddeddolt directories
#   3. Run bd init --non-interactive to recreate the database
#   4. Attempt bd bootstrap if .beads/issues.jsonl exists

set -euo pipefail

RIG_PATH=""
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) RIG_PATH="$arg" ;;
    esac
done

if [[ -z "$RIG_PATH" ]]; then
    echo "Usage: dolt-recovery.sh <rig-path> [--dry-run]"
    exit 1
fi
RIG_PATH="$(eval echo "$RIG_PATH")"  # Expand ~

if [[ ! -d "$RIG_PATH" ]]; then
    echo "ERROR: Rig path does not exist: $RIG_PATH"
    exit 1
fi

if [[ ! -d "$RIG_PATH/.beads" ]]; then
    echo "ERROR: No .beads directory found at: $RIG_PATH/.beads"
    exit 1
fi

export PATH="$HOME/go/bin:$PATH"

RIG_NAME="$(basename "$RIG_PATH")"
BEADS_DIR="$RIG_PATH/.beads"
HAS_JSONL=false
if [[ -f "$BEADS_DIR/issues.jsonl" ]]; then
    HAS_JSONL=true
    JSONL_LINES=$(wc -l < "$BEADS_DIR/issues.jsonl")
fi

echo "========================================"
echo " Dolt Recovery: $RIG_NAME"
echo " Path:    $RIG_PATH"
echo " Dry run: $DRY_RUN"
echo " JSONL:   $([ "$HAS_JSONL" = true ] && echo "yes ($JSONL_LINES lines)" || echo "no")"
echo "========================================"
echo ""

# Step 1: Kill orphaned Dolt server processes
echo "[1/4] Killing orphaned Dolt server processes..."
DOLT_PIDS=$(pgrep -f "dolt.*sql-server" || true)
if [[ -n "$DOLT_PIDS" ]]; then
    echo "  Found Dolt processes: $DOLT_PIDS"
    if [[ "$DRY_RUN" = false ]]; then
        kill $DOLT_PIDS 2>/dev/null || true
        sleep 1
        # Force kill if still running
        DOLT_REMAIN=$(pgrep -f "dolt.*sql-server" || true)
        if [[ -n "$DOLT_REMAIN" ]]; then
            kill -9 $DOLT_REMAIN 2>/dev/null || true
        fi
        echo "  Killed."
    else
        echo "  (dry-run: would kill these PIDs)"
    fi
else
    echo "  No orphaned Dolt processes found."
fi
echo ""

# Step 2: Remove corrupted Dolt data directories
DOLT_DIRS=()
if [[ -d "$BEADS_DIR/dolt" ]]; then
    DOLT_DIRS+=("$BEADS_DIR/dolt")
fi
if [[ -d "$BEADS_DIR/embeddeddolt" ]]; then
    DOLT_DIRS+=("$BEADS_DIR/embeddeddolt")
fi

echo "[2/4] Removing Dolt data directories..."
if [[ ${#DOLT_DIRS[@]} -eq 0 ]]; then
    echo "  No Dolt data directories found to remove."
else
    for dir in "${DOLT_DIRS[@]}"; do
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  Found: $dir ($SIZE)"
        if [[ "$DRY_RUN" = false ]]; then
            rm -rf "$dir"
            echo "  Removed."
        else
            echo "  (dry-run: would rm -rf $dir)"
        fi
    done
fi
echo ""

# Step 3: Re-initialize the database
echo "[3/4] Running bd init --non-interactive..."
if [[ "$DRY_RUN" = false ]]; then
    cd "$RIG_PATH"
    if bd init --non-interactive 2>&1; then
        echo "  bd init succeeded."
    else
        echo "  WARNING: bd init returned non-zero exit code."
        echo "  The rig may need manual intervention."
    fi
else
    echo "  (dry-run: would run: cd $RIG_PATH && bd init --non-interactive)"
fi
echo ""

# Step 4: Bootstrap from JSONL if available
echo "[4/4] Bootstrapping from JSONL..."
if [[ "$HAS_JSONL" = true ]]; then
    if [[ "$DRY_RUN" = false ]]; then
        cd "$RIG_PATH"
        if bd import 2>&1; then
            echo "  bd import succeeded. Imported from issues.jsonl."
        else
            echo "  WARNING: bd import returned non-zero exit code."
            echo "  Issues may need manual recovery from .beads/issues.jsonl"
        fi
    else
        echo "  (dry-run: would run: cd $RIG_PATH && bd import)"
        echo "  Would restore $JSONL_LINES issues from .beads/issues.jsonl"
    fi
else
    echo "  No .beads/issues.jsonl found. Skipping import."
    echo "  If issues were previously exported, restore issues.jsonl manually first."
fi
echo ""

echo "========================================"
echo " Recovery complete for: $RIG_NAME"
if [[ "$DRY_RUN" = true ]]; then
    echo " (DRY RUN — no changes were made)"
fi
echo "========================================"
