#!/usr/bin/env bash
# populate-kpis.sh — Populate project health KPI files for both rigs
#
# Extracts signals from git, pytest, coverage, ruff, radon, vulture, pip-audit,
# and existing qa-report.json files. Writes enriched JSON that director agents
# read to make strategic decisions.
#
# Usage:
#   populate-kpis.sh              # run once, verbose
#   populate-kpis.sh --cron       # silent unless errors
#
# Cron example (every 30 min):
#   */30 * * * * /home/Zev/gas-hermes/scripts/populate-kpis.sh --cron

set -euo pipefail

ALPHA_LAB_DIR="/home/Zev/development/alpha-lab"
CRABQUANT_DIR="/home/Zev/development/CrabQuant"
BUDGET_FILE="${HOME}/gas-hermes/.budget-daily.json"
KPI_DIR="${HOME}/.hermes/scripts"
ALPHA_KPI="${KPI_DIR}/alpha-lab-ops-kpis.json"
CRABQUANT_KPI="${KPI_DIR}/crabquant-ops-kpis.json"
TRENDS_DIR="${KPI_DIR}/trends"
ALPHA_TRENDS="${TRENDS_DIR}/alpha-lab"
CRABQUANT_TRENDS="${TRENDS_DIR}/crabquant"
SILENT=false

[[ "${1:-}" == "--cron" ]] && SILENT=true

log() { $SILENT || echo "[$(date -Iseconds)] $*" >&2; }
err() { echo "[$(date -Iseconds)] ERROR: $*" >&2; }

# Keep 7 days of trend snapshots
TREND_RETENTION_DAYS=7

# ── Git Intelligence Pipeline ────────────────────────────────────────
extract_git() {
    local project_dir="$1"
    cd "$project_dir"

    local commits_7d commits_24h stale_branches uncommitted_files
    local branch current_branch

    current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

    commits_7d=$(git log --oneline --since="7 days ago" 2>/dev/null | wc -l | tr -d ' ')
    commits_24h=$(git log --oneline --since="24 hours ago" 2>/dev/null | wc -l | tr -d ' ')
    uncommitted_files=$(git status --short 2>/dev/null | wc -l | tr -d ' ')

    # Stale branches: merged branches that still exist locally
    stale_branches=$(git branch --merged main 2>/dev/null | grep -v '^\*\|main\|master' | wc -l | tr -d ' ')
    # If no main, use default branch
    if [ "$stale_branches" -eq 0 ] && git rev-parse --verify main >/dev/null 2>&1; then
        :
    elif git rev-parse --verify master >/dev/null 2>&1; then
        stale_branches=$(git branch --merged master 2>/dev/null | grep -v '^\*\|master' | wc -l | tr -d ' ')
    fi

    # Open PRs (if gh CLI available)
    local prs_open=0
    if command -v gh >/dev/null 2>&1; then
        prs_open=$(gh pr list --state open --json number --jq 'length' 2>/dev/null || echo 0)
    fi

    # Lines changed in last 7 days
    local lines_added lines_removed
    lines_added=$(git log --since="7 days ago" --format='' --numstat 2>/dev/null | awk '{s+=$1} END {print s+0}')
    lines_removed=$(git log --since="7 days ago" --format='' --numstat 2>/dev/null | awk '{s+=$2} END {print s+0}')

    jq -n \
        --arg branch "$current_branch" \
        --argjson commits_7d "${commits_7d:-0}" \
        --argjson commits_24h "${commits_24h:-0}" \
        --argjson stale_branches "${stale_branches:-0}" \
        --argjson prs_open "${prs_open:-0}" \
        --argjson uncommitted_files "${uncommitted_files:-0}" \
        --argjson lines_added_7d "${lines_added:-0}" \
        --argjson lines_removed_7d "${lines_removed:-0}" \
        '{
            branch: $branch,
            commits_7d: $commits_7d,
            commits_24h: $commits_24h,
            stale_branches: $stale_branches,
            prs_open: $prs_open,
            uncommitted_files: $uncommitted_files,
            lines_added_7d: $lines_added_7d,
            lines_removed_7d: $lines_removed_7d
        }'
}

# ── Test Health Pipeline (from existing qa-report.json) ──────────────
extract_tests() {
    local project_dir="$1"
    local qa_report="$2"  # path to qa-report.json or qa_report.json

    if [ ! -f "$qa_report" ]; then
        jq -n '{status: "no_report", total: 0, pass: 0, fail: 0, skipped: 0, coverage_pct: 0, duration_s: 0}'
        return
    fi

    local report
    report=$(cat "$qa_report")

    # Parse — handle both formats (alpha-lab uses tests.pass, CrabQuant uses tests.pass too)
    local total pass fail skipped coverage_pct duration_s warnings_count
    total=$(echo "$report" | jq '.tests.total // 0')
    pass=$(echo "$report" | jq '.tests.pass // 0')
    fail=$(echo "$report" | jq '.tests.fail // 0')
    skipped=$(echo "$report" | jq '.tests.skipped // 0')
    coverage_pct=$(echo "$report" | jq '.coverage.pct // 0')
    duration_s=$(echo "$report" | jq '.tests.duration_seconds // 0')
    warnings_count=$(echo "$report" | jq '.tests.warnings // (.warnings.count // 0)')

    # Compute pass rate
    local pass_rate
    if [ "$total" -gt 0 ]; then
        pass_rate=$(echo "$report" | jq --argjson pass "$pass" --argjson total "$total" \
            '($pass / $total * 100) | . * 10 | round / 10')
    else
        pass_rate=0
    fi

    # Extract notable issues from report if available
    local issues_summary
    issues_summary=$(echo "$report" | jq -r \
        '[.issues[]? // empty | .description // .] | join(" | ")' 2>/dev/null | head -c 500)

    # Extract lowest coverage modules
    local lowest_modules
    lowest_modules=$(echo "$report" | jq -r \
        '[.coverage.lowest_modules[]? // empty | "\(.module): \(.pct)%"] | join(", ")' 2>/dev/null | head -c 300)

    # Report age (seconds since report was generated)
    local report_ts report_age_s
    report_ts=$(echo "$report" | jq -r '.timestamp // empty')
    if [ -n "$report_ts" ] && [ "$report_ts" != "null" ]; then
        report_age_s=$(( $(date +%s) - $(date -d "$report_ts" +%s 2>/dev/null || echo 0) ))
    else
        report_age_s=-1
    fi

    jq -n \
        --argjson total "$total" \
        --argjson pass "$pass" \
        --argjson fail "$fail" \
        --argjson skipped "$skipped" \
        --argjson pass_rate "$pass_rate" \
        --argjson coverage_pct "$coverage_pct" \
        --argjson duration_s "$duration_s" \
        --argjson warnings "$warnings_count" \
        --argjson report_age_s "$report_age_s" \
        --arg issues "$issues_summary" \
        --arg lowest_modules "$lowest_modules" \
        '{
            status: "ok",
            total: $total,
            pass: $pass,
            fail: $fail,
            skipped: $skipped,
            pass_rate_pct: $pass_rate,
            coverage_pct: $coverage_pct,
            duration_s: $duration_s,
            warnings: $warnings,
            report_age_s: $report_age_s,
            issues: $issues,
            lowest_modules: $lowest_modules
        }'
}

# ── Code Quality Pipeline ────────────────────────────────────────────
extract_quality() {
    local project_dir="$1"
    local venv_python="$2"
    local src_dir="$3"

    cd "$project_dir"

    # Ruff lint
    local lint_errors=0 lint_files=0
    local lint_top_codes="[]"
    if $venv_python -m ruff check "$src_dir" --output-format=json >/dev/null 2>&1; then
        local ruff_json
        ruff_json=$($venv_python -m ruff check "$src_dir" --output-format=json 2>&1 || echo "[]")
        lint_errors=$(echo "$ruff_json" | jq 'length')
        if [ "$lint_errors" -gt 0 ]; then
            lint_files=$(echo "$ruff_json" | jq '[.[].filepath] | unique | length')
            lint_top_codes=$(echo "$ruff_json" | jq \
                '[group_by(.code) | sort_by(-length) | .[0:5] | .[] | {code: .[0].code, count: length}]')
        fi
    fi

    # Radon complexity (average)
    local complexity_avg=0 complexity_high=0
    local radon_output
    radon_output=$($venv_python -m radon cc "$src_dir" -a -s 2>/dev/null || echo "")
    if [ -n "$radon_output" ]; then
        complexity_avg=$(echo "$radon_output" | grep "Average complexity" | grep -oP '[\d.]+' || echo "0")
        complexity_high=$(echo "$radon_output" | grep -c " F\| E " || echo "0")
    fi

    # TODO/FIXME/HACK comments
    local todo_count=0
    todo_count=$(grep -rn 'TODO\|FIXME\|HACK' --include="*.py" "$src_dir" 2>/dev/null | wc -l | tr -d ' ')

    # Vulture dead code
    local dead_code_items=0
    dead_code_items=$($venv_python -m vulture "$src_dir" --min-confidence 80 2>/dev/null | wc -l | tr -d ' ')

    jq -n \
        --argjson lint_errors "$lint_errors" \
        --argjson lint_files "$lint_files" \
        --argjson lint_top_codes "$lint_top_codes" \
        --argjson complexity_avg "$complexity_avg" \
        --argjson complexity_high "$complexity_high" \
        --argjson todo_count "$todo_count" \
        --argjson dead_code "$dead_code_items" \
        '{
            lint: {
                errors: $lint_errors,
                files_affected: $lint_files,
                top_codes: $lint_top_codes
            },
            complexity: {
                avg: $complexity_avg,
                high_risk_functions: $complexity_high
            },
            todo_count: $todo_count,
            dead_code_items: $dead_code
        }'
}

# ── Dependency Health Pipeline ───────────────────────────────────────
extract_deps() {
    local project_dir="$1"
    local venv_python="$2"

    cd "$project_dir"

    # pip-audit
    local vuln_critical=0 vuln_high=0 vuln_medium=0 vuln_low=0
    local vuln_details="[]"
    if $venv_python -m pip_audit --format=json 2>/dev/null >/dev/null; then
        local audit_json
        audit_json=$($venv_python -m pip_audit --format=json --desc 2>/dev/null || echo '{"dependencies":[]}')
        vuln_critical=$(echo "$audit_json" | jq '[.dependencies[].vulns[]? | select(.fix_versions == [])] | length')
        vuln_high=$(echo "$audit_json" | jq '[.dependencies[].vulns[]? | select(.aliases // [] | map(select(startswith("CVE"))) | length > 0)] | length')
        # Simpler: count deps with any vulns
        vuln_critical=$(echo "$audit_json" | jq '[.dependencies[] | select(.vulns | length > 0)] | length')
        vuln_details=$(echo "$audit_json" | jq '[.dependencies[] | select(.vulns | length > 0) | {name, version, vuln_count: (.vulns | length)}] | .[0:10]')
    fi

    # Outdated deps count
    local outdated=0
    outdated=$($venv_python -m pip list --outdated --format=json 2>/dev/null | jq 'length' || echo 0)

    # Total deps
    local total_deps=0
    total_deps=$($venv_python -m pip list --format=json 2>/dev/null | jq 'length' || echo 0)

    jq -n \
        --argjson vuln_deps "$vuln_critical" \
        --argjson outdated "$outdated" \
        --argjson total "$total_deps" \
        --argjson vuln_details "$vuln_details" \
        '{
            vulnerable_deps: $vuln_deps,
            outdated: $outdated,
            total: $total,
            top_vulns: $vuln_details
        }'
}

# ── Ops Pipeline (bead counts from bd) ───────────────────────────────
extract_ops() {
    local project_dir="$1"
    local rig_name="$2"
    local today
    today=$(date +%Y-%m-%d)

    local total=0 closed=0 in_progress=0 open=0 failed=0 beads_today=0 avg_time_min=0

    if cd "$project_dir" && command -v bd >/dev/null 2>&1; then
        local rig_json
        if rig_json=$(bd list --json 2>/dev/null); then
            total=$(echo "$rig_json" | jq 'length')
            closed=$(echo "$rig_json" | jq '[.[] | select(.status == "closed")] | length')
            in_progress=$(echo "$rig_json" | jq '[.[] | select(.status == "in_progress")] | length')
            open=$(echo "$rig_json" | jq '[.[] | select(.status == "open")] | length')
            failed=$(echo "$rig_json" | jq '[.[] | select(.status == "failed" or .status == "error")] | length')
            beads_today=$(echo "$rig_json" | jq --arg today "$today" \
                '[.[] | select(.status == "closed" and ((.updated_at // "x") | startswith($today)))] | length')
            avg_time_min=$(echo "$rig_json" | jq --arg today "$today" \
                '[.[] | select(.status == "closed" and .created_at and .updated_at)
                  | ((.updated_at | fromdateiso8601) - (.created_at | fromdateiso8601)) / 60]
                | if length == 0 then 0 else (add / length) | . * 100 | round / 100 end')
        fi
    fi

    # Budget data
    local sessions_today=0 tokens_today=0
    if [ -f "$BUDGET_FILE" ]; then
        sessions_today=$(jq -r '.sessions_today // 0' "$BUDGET_FILE")
        tokens_today=$(jq -r '.tokens_used // 0' "$BUDGET_FILE")
    fi

    jq -n \
        --arg rig "$rig_name" \
        --argjson beads_today "$beads_today" \
        --argjson sessions_today "$sessions_today" \
        --argjson tokens_today "$tokens_today" \
        --argjson avg_completion_time_min "$avg_time_min" \
        --argjson total "$total" \
        --argjson closed "$closed" \
        --argjson in_progress "$in_progress" \
        --argjson open "$open" \
        --argjson failed "$failed" \
        '{
            rig: $rig,
            beads_completed_today: $beads_today,
            sessions_today: $sessions_today,
            tokens_used_today: $tokens_today,
            avg_completion_time_min: $avg_completion_time_min,
            snapshot: {
                total: $total,
                closed: $closed,
                in_progress: $in_progress,
                open: $open,
                failed: $failed
            }
        }'
}

# ── Alert Generation ─────────────────────────────────────────────────
generate_alerts() {
    local git_json="$1" test_json="$2" quality_json="$3" deps_json="$4" ops_json="$5"

    local alerts="[]"

    # Git alerts
    local commits_24h
    commits_24h=$(echo "$git_json" | jq '.commits_24h')
    if [ "$commits_24h" -eq 0 ]; then
        # Check if there are open beads (work expected)
        local open_beads
        open_beads=$(echo "$ops_json" | jq '.snapshot.open')
        if [ "$open_beads" -gt 0 ]; then
            alerts=$(echo "$alerts" | jq -c '. + [{"level": "warn", "signal": "git.commits_24h", "message": "No commits in 24h but open beads exist — work may be stalled"}]')
        fi
    fi

    local stale
    stale=$(echo "$git_json" | jq '.stale_branches')
    if [ "$stale" -ge 3 ]; then
        alerts=$(echo "$alerts" | jq -c --argjson n "$stale" '. + [{"level": "info", "signal": "git.stale_branches", "message": "\($n) stale merged branches could be cleaned up"}]')
    fi

    local uncommitted
    uncommitted=$(echo "$git_json" | jq '.uncommitted_files')
    if [ "$uncommitted" -ge 20 ]; then
        alerts=$(echo "$alerts" | jq -c --argjson n "$uncommitted" '. + [{"level": "warn", "signal": "git.uncommitted_files", "message": "\($n) uncommitted files — consider committing or stashing"}]')
    fi

    # Test alerts
    local test_status pass_rate
    test_status=$(echo "$test_json" | jq -r '.status')
    pass_rate=$(echo "$test_json" | jq '.pass_rate_pct // 100')

    if [ "$test_status" = "ok" ]; then
        if (( $(echo "$pass_rate < 90" | bc -l) )); then
            local fail_count
            fail_count=$(echo "$test_json" | jq '.fail')
            alerts=$(echo "$alerts" | jq -c --argjson r "$pass_rate" --argjson f "$fail_count" \
                '. + [{"level": "critical", "signal": "tests.pass_rate", "message": "Test pass rate at \($r)% (\($f) failures) — below 90% threshold"}]')
        elif (( $(echo "$pass_rate < 95" | bc -l) )); then
            alerts=$(echo "$alerts" | jq -c --argjson r "$pass_rate" \
                '. + [{"level": "warn", "signal": "tests.pass_rate", "message": "Test pass rate at \($r)% — below 95% threshold"}]')
        fi

        # Report age check
        local report_age
        report_age=$(echo "$test_json" | jq '.report_age_s // -1')
        if [ "$report_age" -ge 86400 ]; then
            local hours=$(( report_age / 3600 ))
            alerts=$(echo "$alerts" | jq -c --argjson h "$hours" \
                '. + [{"level": "warn", "signal": "tests.report_age", "message": "QA report is \($h)h old — tests may need re-running"}]')
        fi

        # Warnings
        local warnings
        warnings=$(echo "$test_json" | jq '.warnings // 0')
        if [ "$warnings" -ge 50 ]; then
            alerts=$(echo "$alerts" | jq -c --argjson w "$warnings" \
                '. + [{"level": "info", "signal": "tests.warnings", "message": "\($w) test warnings — consider addressing"}]')
        fi
    fi

    # Quality alerts
    local lint_errors
    lint_errors=$(echo "$quality_json" | jq '.lint.errors')
    if [ "$lint_errors" -ge 50 ]; then
        alerts=$(echo "$alerts" | jq -c --argjson n "$lint_errors" \
            '. + [{"level": "warn", "signal": "quality.lint", "message": "\($n) lint errors across codebase"}]')
    fi

    local complexity_high
    complexity_high=$(echo "$quality_json" | jq '.complexity.high_risk_functions')
    if [ "$complexity_high" -ge 5 ]; then
        alerts=$(echo "$alerts" | jq -c --argjson n "$complexity_high" \
            '. + [{"level": "warn", "signal": "quality.complexity", "message": "\($n) functions with high cyclomatic complexity (D/E/F)"}]')
    fi

    local dead_code
    dead_code=$(echo "$quality_json" | jq '.dead_code_items')
    if [ "$dead_code" -ge 20 ]; then
        alerts=$(echo "$alerts" | jq -c --argjson n "$dead_code" \
            '. + [{"level": "info", "signal": "quality.dead_code", "message": "\($n) potential dead code items detected by vulture"}]')
    fi

    # Dependency alerts
    local vuln_deps
    vuln_deps=$(echo "$deps_json" | jq '.vulnerable_deps')
    if [ "$vuln_deps" -ge 1 ]; then
        alerts=$(echo "$alerts" | jq -c --argjson n "$vuln_deps" \
            '. + [{"level": "critical", "signal": "deps.vulnerabilities", "message": "\($n) dependencies with known vulnerabilities"}]')
    fi

    local outdated
    outdated=$(echo "$deps_json" | jq '.outdated')
    if [ "$outdated" -ge 10 ]; then
        alerts=$(echo "$alerts" | jq -c --argjson n "$outdated" \
            '. + [{"level": "info", "signal": "deps.outdated", "message": "\($n) outdated dependencies"}]')
    fi

    echo "$alerts"
}

# ── Trend Management ─────────────────────────────────────────────────
save_trend() {
    local trends_dir="$1"
    local today
    today=$(date +%Y-%m-%d)
    local trend_file="${trends_dir}/${today}.json"

    mkdir -p "$trends_dir"

    # Prune old trends
    find "$trends_dir" -name "*.json" -mtime +$TREND_RETENTION_DAYS -delete 2>/dev/null || true

    echo "$2" > "$trend_file"
}

build_trends() {
    local trends_dir="$1"

    if [ ! -d "$trends_dir" ]; then
        echo "[]"
        return
    fi

    # Collect recent trend files, sorted by date
    local files
    files=$(ls -1 "${trends_dir}/"*.json 2>/dev/null | sort | tail -7)

    if [ -z "$files" ]; then
        echo "[]"
        return
    fi

    # Build trends array from historical data
    local trends="[]"
    for f in $files; do
        local date_val
        date_val=$(basename "$f" .json)
        local pass_rate coverage
        pass_rate=$(jq -r '.tests.pass_rate_pct // null' "$f" 2>/dev/null)
        coverage=$(jq -r '.tests.coverage_pct // null' "$f" 2>/dev/null)
        trends=$(echo "$trends" | jq -c \
            --arg date "$date_val" \
            --argjson pass "$pass_rate" \
            --argjson cov "$coverage" \
            '. + [{"date": $date, "test_pass_rate": $pass, "coverage": $cov}]')
    done

    echo "$trends"
}

# ── Main: Build KPI for one rig ──────────────────────────────────────
build_kpi() {
    local project_dir="$1"
    local rig_name="$2"
    local venv_python="$3"
    local src_dir="$4"
    local qa_report="$5"
    local trends_dir="$6"

    local today
    today=$(date +%Y-%m-%d)

    log "  [${rig_name}] Extracting git signals..."
    local git_json
    git_json=$(extract_git "$project_dir")

    log "  [${rig_name}] Extracting test health..."
    local test_json
    test_json=$(extract_tests "$project_dir" "$qa_report")

    log "  [${rig_name}] Extracting code quality..."
    local quality_json
    quality_json=$(extract_quality "$project_dir" "$venv_python" "$src_dir")

    log "  [${rig_name}] Extracting dependency health..."
    local deps_json
    deps_json=$(extract_deps "$project_dir" "$venv_python")

    log "  [${rig_name}] Extracting ops metrics..."
    local ops_json
    ops_json=$(extract_ops "$project_dir" "$rig_name")

    log "  [${rig_name}] Generating alerts..."
    local alerts
    alerts=$(generate_alerts "$git_json" "$test_json" "$quality_json" "$deps_json" "$ops_json")

    # Save trend snapshot
    save_trend "$trends_dir" "$(jq -n \
        --arg date "$today" \
        --argjson tests "$test_json" \
        --argjson quality "$quality_json" \
        '{date: $date, tests: {pass_rate_pct: $tests.pass_rate_pct, coverage_pct: $tests.coverage_pct}, quality: {lint_errors: $quality.lint.errors, complexity_avg: $quality.complexity.avg}}')"

    # Build trends from history
    local trends
    trends=$(build_trends "$trends_dir")

    # Assemble final KPI document
    jq -n \
        --arg date "$today" \
        --arg rig "$rig_name" \
        --argjson git "$git_json" \
        --argjson tests "$test_json" \
        --argjson quality "$quality_json" \
        --argjson deps "$deps_json" \
        --argjson ops "$ops_json" \
        --argjson alerts "$alerts" \
        --argjson trends "$trends" \
        '{
            date: $date,
            rig: $rig,
            git: $git,
            tests: $tests,
            quality: $quality,
            deps: $deps,
            ops: $ops,
            alerts: $alerts,
            trends: $trends
        }'
}

# ── Run ──────────────────────────────────────────────────────────────
mkdir -p "$KPI_DIR" "$TRENDS_DIR"

log "=== populate-kpis.sh ==="

log "Building alpha-lab KPIs..."
ALPHA_RESULT=$(build_kpi \
    "$ALPHA_LAB_DIR" "alpha-lab" \
    "${ALPHA_LAB_DIR}/.venv/bin/python" "src/" \
    "${ALPHA_LAB_DIR}/qa_report.json" \
    "$ALPHA_TRENDS")

echo "$ALPHA_RESULT" > "${ALPHA_KPI}.tmp"
mv -f "${ALPHA_KPI}.tmp" "$ALPHA_KPI"

log "Building CrabQuant KPIs..."
CRABQUANT_RESULT=$(build_kpi \
    "$CRABQUANT_DIR" "CrabQuant" \
    "${CRABQUANT_DIR}/.venv/bin/python" "crabquant/" \
    "${CRABQUANT_DIR}/qa-report.json" \
    "$CRABQUANT_TRENDS")

echo "$CRABQUANT_RESULT" > "${CRABQUANT_KPI}.tmp"
mv -f "${CRABQUANT_KPI}.tmp" "$CRABQUANT_KPI"

log "Done."

$SILENT || echo ""
$SILENT || echo "=== alpha-lab ==="
$SILENT || jq '{date, rig, git, tests: {total, pass, fail, pass_rate_pct, coverage_pct}, quality: {lint: .quality.lint.errors, complexity: .quality.complexity.avg, todo_count, dead_code_items}, deps: {vulnerable_deps, outdated}, ops: {beads_completed_today, tokens_used_today}, alerts: (.alerts | length)}' "$ALPHA_KPI"

$SILENT || echo ""
$SILENT || echo "=== CrabQuant ==="
$SILENT || jq '{date, rig, git, tests: {total, pass, fail, pass_rate_pct, coverage_pct}, quality: {lint: .quality.lint.errors, complexity: .quality.complexity.avg, todo_count, dead_code_items}, deps: {vulnerable_deps, outdated}, ops: {beads_completed_today, tokens_used_today}, alerts: (.alerts | length)}' "$CRABQUANT_KPI"
