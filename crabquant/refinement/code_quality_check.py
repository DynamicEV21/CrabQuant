"""
Code quality pre-check — analyze LLM-generated strategy source code BEFORE
backtesting and flag common anti-patterns that cause poor performance.

Checks:
  1. Over-complex entry conditions (stacked & operators)
  2. Contradictory conditions (indicator > X AND indicator < Y where X >= Y)
  3. Very long lookback periods (indicator period > 100)
  4. No exit logic (constant-false exits)
  5. Missing time/holding stop
  6. Hardcoded extreme thresholds
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class CodeQualityIssue:
    severity: str       # "info", "warning", "critical"
    category: str       # "over_complex", "contradictory", "long_lookback",
                        # "no_exit", "no_time_stop", "extreme_threshold"
    line_range: str     # approximate line location
    description: str    # what's wrong
    fix_suggestion: str # how to fix it


@dataclass
class CodeQualityReport:
    score: float                          # 0.0-1.0, higher = better quality
    issues: list[CodeQualityIssue]        # all detected issues
    overall_verdict: str                  # "good", "warning", "reject"
    summary_for_llm: str                  # formatted feedback for the LLM prompt


# ── Severity weights for score calculation ─────────────────────────────────

_SEVERITY_PENALTY: dict[str, float] = {
    "info": 0.0,
    "warning": 0.10,
    "critical": 0.20,
}

# ── Thresholds for verdict ─────────────────────────────────────────────────

_REJECT_THRESHOLD = 0.50
_WARNING_THRESHOLD = 0.75


# ── Check implementations ─────────────────────────────────────────────────

def _collect_assignment_lines(
    lines: list[str],
    var_name: str,
    extra_stop_patterns: list[str] | None = None,
) -> tuple[int, str] | None:
    """Find and collect all lines of a variable assignment, handling multiline.

    Returns (start_line_idx, combined_text) or None if not found.
    """
    stop_pats = [rf"{var_name}\s*=", r"return\b"]
    if extra_stop_patterns:
        stop_pats.extend(extra_stop_patterns)
    stop_re = re.compile("|".join(stop_pats))

    start_idx = None
    for i, line in enumerate(lines):
        if re.match(rf"{var_name}\s*=", line.strip()):
            start_idx = i
            break

    if start_idx is None:
        return None

    collected: list[str] = [lines[start_idx]]
    for i in range(start_idx + 1, min(start_idx + 20, len(lines))):
        stripped = lines[i].strip()
        # Check stop conditions BEFORE appending
        if stop_re.match(stripped):
            break
        if stripped and not re.match(r"[&|)\],\s]", stripped):
            # Check balanced parens so far
            combined_so_far = " ".join(collected)
            if combined_so_far.count("(") <= combined_so_far.count(")"):
                break
        collected.append(lines[i])

    return start_idx, " ".join(collected)


def _check_over_complex_entry(source: str, lines: list[str]) -> list[CodeQualityIssue]:
    """Flag entries with 4+ stacked & operators (high overfit risk)."""
    issues: list[CodeQualityIssue] = []

    result = _collect_assignment_lines(lines, "entries", ["exits\\s*="])
    if result is None:
        return issues

    start_idx, combined = result
    # Count standalone & operators (not &=, not &&)
    amp_count = len(re.findall(r"(?<![&=])&(?![&=])", combined))
    if amp_count >= 4:
        severity = "critical" if amp_count >= 5 else "warning"
        issues.append(CodeQualityIssue(
            severity=severity,
            category="over_complex",
            line_range=f"L{start_idx + 1}",
            description=(
                f"Entry condition has {amp_count} stacked '&' operators, "
                f"indicating over-complex logic with high overfit risk."
            ),
            fix_suggestion=(
                "Simplify entry logic to at most 3 conditions. "
                "Combine related filters or remove the weakest signal."
            ),
        ))

    return issues


def _check_contradictory_conditions(source: str, lines: list[str]) -> list[CodeQualityIssue]:
    """Detect indicator > X AND indicator < Y where X >= Y."""
    issues: list[CodeQualityIssue] = []

    result = _collect_assignment_lines(lines, "entries", ["exits\\s*="])
    if result is None:
        return issues

    start_idx, combined = result

    # Find all comparisons: var op number
    comp_pattern = re.compile(r"(\w+)\s*(>|<)\s*(-?\d+\.?\d*)")
    matches = comp_pattern.findall(combined)

    # Group by variable
    var_comparisons: dict[str, list[tuple[str, float]]] = {}
    for var, op, val_str in matches:
        var_comparisons.setdefault(var, []).append((op, float(val_str)))

    # Check for contradictions within each variable
    for var, comparisons in var_comparisons.items():
        # Look for > X combined with < Y where X >= Y
        gt_vals = [v for op, v in comparisons if op == ">"]
        lt_vals = [v for op, v in comparisons if op == "<"]

        for gt_val in gt_vals:
            for lt_val in lt_vals:
                if gt_val >= lt_val:
                    issues.append(CodeQualityIssue(
                        severity="critical",
                        category="contradictory",
                        line_range=f"L{start_idx + 1}",
                        description=(
                            f"Contradictory condition: {var} > {gt_val} AND "
                            f"{var} < {lt_val}. Since {gt_val} >= {lt_val}, "
                            f"this condition can never be true."
                        ),
                        fix_suggestion=(
                            f"Fix the thresholds for {var} so the upper bound "
                            f"is strictly greater than the lower bound."
                        ),
                    ))

    return issues


def _check_long_lookback_periods(source: str, lines: list[str]) -> list[CodeQualityIssue]:
    """Detect indicator calls with period > 100."""
    issues: list[CodeQualityIssue] = []

    # Match cached_indicator calls with a length/period parameter > 100
    # Also match common parameter names: length, period, window, span
    indicator_pattern = re.compile(
        r"cached_indicator\s*\(\s*[\"'](\w+)[\"'].*?(?:length|period|window|span)\s*=\s*(\d+)",
        re.DOTALL,
    )

    for i, line in enumerate(lines):
        stripped = line.strip()
        for m in indicator_pattern.finditer(stripped):
            indicator_name = m.group(1).lower()
            period = int(m.group(2))
            if period > 100:
                issues.append(CodeQualityIssue(
                    severity="warning",
                    category="long_lookback",
                    line_range=f"L{i + 1}",
                    description=(
                        f"{indicator_name.upper()} called with period={period}, "
                        f"which is very long. This causes lag and reduces signal quality."
                    ),
                    fix_suggestion=(
                        f"Reduce {indicator_name} period to 50 or less. "
                        f"Long lookbacks add lag and reduce responsiveness."
                    ),
                ))

    return issues


def _check_no_exit_logic(source: str, lines: list[str]) -> list[CodeQualityIssue]:
    """Detect if exits is just a constant-false Series."""
    issues: list[CodeQualityIssue] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"exits\s*=", stripped):
            # Check for constant-false patterns
            constant_false_patterns = [
                r"pd\.Series\s*\(\s*False",
                r"pd\.Series\s*\(\s*0\s*,",
                r"exits\s*=\s*False",
                r"np\.zeros\s*\(",
                r"np\.full\s*\([^)]*,\s*False",
            ]
            for pat in constant_false_patterns:
                if re.search(pat, stripped):
                    issues.append(CodeQualityIssue(
                        severity="critical",
                        category="no_exit",
                        line_range=f"L{i + 1}",
                        description=(
                            "Exit logic is a constant-false value. Positions will "
                            "never be closed, leading to unrealistic backtests."
                        ),
                        fix_suggestion=(
                            "Implement proper exit logic (e.g., indicator-based exits, "
                            "trailing stops, or time-based exits)."
                        ),
                    ))
                    break
            break

    return issues


def _check_no_time_stop(source: str, lines: list[str]) -> list[CodeQualityIssue]:
    """Check if there's no time-based exit mechanism."""
    issues: list[CodeQualityIssue] = []

    # First, check if exits actually have logic (not constant-false)
    has_real_exits = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"exits\s*=", stripped):
            constant_false_patterns = [
                r"pd\.Series\s*\(\s*False",
                r"pd\.Series\s*\(\s*0\s*,",
                r"exits\s*=\s*False",
                r"np\.zeros\s*\(",
                r"np\.full\s*\([^)]*,\s*False",
            ]
            has_real_exits = not any(
                re.search(pat, stripped) for pat in constant_false_patterns
            )
            break

    if not has_real_exits:
        return issues  # Don't warn about time stop if there's no exit at all

    # Look for time-based exit patterns
    time_stop_patterns = [
        r"hold_periods?\b",
        r"bars_since_entry\b",
        r"bars_since\b",
        r"max_hold\b",
        r"max_bars\b",
        r"timedelta\b",
        r"\.diff\(\)\.dt\.days",
        r"entry_age\b",
        r"holding_period\b",
    ]

    has_time_stop = any(
        re.search(pat, source) for pat in time_stop_patterns
    )

    if not has_time_stop:
        issues.append(CodeQualityIssue(
            severity="warning",
            category="no_time_stop",
            line_range="function",
            description=(
                "No time-based exit mechanism detected. Without a maximum holding "
                "period, losing positions can be held indefinitely."
            ),
            fix_suggestion=(
                "Add a time-based stop (e.g., exit after N bars using hold_periods "
                "or bars_since_entry) to limit downside risk."
            ),
        ))

    return issues


def _check_extreme_thresholds(source: str, lines: list[str]) -> list[CodeQualityIssue]:
    """Detect hardcoded extreme thresholds."""
    issues: list[CodeQualityIssue] = []

    # RSI extremes
    rsi_patterns = [
        (r"rsi[^=]*>\s*(\d+)", lambda v: float(v) > 95, "RSI > {val} is too extreme"),
        (r"rsi[^=]*<\s*(\d+)", lambda v: float(v) < 5, "RSI < {val} is too extreme"),
    ]

    # Volume multiplier extremes
    volume_patterns = [
        (r"(?:volume|vol)[^=]*>\s*(\d+(?:\.\d+)?)", lambda v: float(v) > 5, "Volume > {val}x is too extreme"),
    ]

    # Generic extreme numeric thresholds in indicator comparisons
    # ATR multiplier > 3
    atr_patterns = [
        (r"atr[^=]*\*\s*(\d+(?:\.\d+)?)", lambda v: float(v) > 4, "ATR multiplier {val}x is too wide"),
    ]

    all_patterns = rsi_patterns + volume_patterns + atr_patterns

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip parameter defaults and non-signal lines
        if re.match(r"(DEFAULT_PARAMS|params\.get|threshold\s*=)", stripped):
            continue
        if re.match(r"#", stripped):
            continue

        for pat, check_fn, msg_template in all_patterns:
            m = re.search(pat, stripped, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if check_fn(str(val)):
                    issues.append(CodeQualityIssue(
                        severity="warning",
                        category="extreme_threshold",
                        line_range=f"L{i + 1}",
                        description=msg_template.format(val=val),
                        fix_suggestion=(
                            "Use more moderate thresholds (e.g., RSI 30-70 zone, "
                            "volume 1.5-3x) to avoid filtering out all signals."
                        ),
                    ))

    return issues


# ── Main check function ───────────────────────────────────────────────────

def check_code_quality(source_code: str) -> CodeQualityReport:
    """Analyze strategy source code and return a quality report.

    Parameters
    ----------
    source_code : str
        The full source code string of the strategy module.

    Returns
    -------
    CodeQualityReport
        Contains score, issues, verdict, and LLM-friendly summary.
    """
    lines = source_code.splitlines()
    all_issues: list[CodeQualityIssue] = []

    # Early exit: empty or trivially small code
    stripped_source = source_code.strip()
    if not stripped_source:
        return CodeQualityReport(
            score=0.0,
            issues=[CodeQualityIssue(
                severity="critical",
                category="no_exit",  # reuse category
                line_range="all",
                description="Source code is empty.",
                fix_suggestion="Provide a valid strategy with generate_signals function.",
            )],
            overall_verdict="reject",
            summary_for_llm="Source code is empty. Provide a complete strategy with a generate_signals function.",
        )

    # Check for generate_signals presence
    has_generate_signals = "def generate_signals" in source_code
    if not has_generate_signals:
        return CodeQualityReport(
            score=0.0,
            issues=[CodeQualityIssue(
                severity="critical",
                category="no_exit",
                line_range="all",
                description="No generate_signals function found.",
                fix_suggestion="Strategy must define a generate_signals(df, params) function.",
            )],
            overall_verdict="reject",
            summary_for_llm="No generate_signals function found. Strategy must define generate_signals(df, params).",
        )

    # Run all checks
    all_issues.extend(_check_over_complex_entry(source_code, lines))
    all_issues.extend(_check_contradictory_conditions(source_code, lines))
    all_issues.extend(_check_long_lookback_periods(source_code, lines))
    all_issues.extend(_check_no_exit_logic(source_code, lines))
    all_issues.extend(_check_no_time_stop(source_code, lines))
    all_issues.extend(_check_extreme_thresholds(source_code, lines))

    # Calculate score: start at 1.0, deduct per issue
    score = 1.0
    for issue in all_issues:
        score -= _SEVERITY_PENALTY.get(issue.severity, 0.05)
    score = max(0.0, min(1.0, score))

    # Determine verdict
    if score < _REJECT_THRESHOLD:
        verdict = "reject"
    elif score < _WARNING_THRESHOLD:
        verdict = "warning"
    else:
        verdict = "good"

    # Build summary for LLM
    if not all_issues:
        summary_for_llm = "Code quality check passed. No anti-patterns detected."
    else:
        criticals = [i for i in all_issues if i.severity == "critical"]
        warnings = [i for i in all_issues if i.severity == "warning"]
        infos = [i for i in all_issues if i.severity == "info"]

        parts: list[str] = []
        if criticals:
            cat_summary = ", ".join(
                f"{i.category} ({i.description})" for i in criticals[:3]
            )
            parts.append(f"Critical: {cat_summary}")
        if warnings:
            cat_summary = ", ".join(i.category for i in warnings[:3])
            parts.append(f"Warnings: {cat_summary}")
        if infos:
            parts.append(f"Info: {len(infos)} minor issue(s)")

        summary_for_llm = f"Code quality score: {score:.2f} ({verdict}). " + " | ".join(parts)

    return CodeQualityReport(
        score=score,
        issues=all_issues,
        overall_verdict=verdict,
        summary_for_llm=summary_for_llm,
    )


# ── Formatting for LLM prompt ─────────────────────────────────────────────

def format_code_quality_for_prompt(report: CodeQualityReport) -> str:
    """Format a CodeQualityReport as a concise prompt string for the LLM.

    Parameters
    ----------
    report : CodeQualityReport
        The quality report to format.

    Returns
    -------
    str
        Formatted string suitable for injection into the LLM prompt.
    """
    lines: list[str] = []
    lines.append(f"## Code Quality Pre-Check (Score: {report.score:.2f} — {report.overall_verdict.upper()})")

    if not report.issues:
        lines.append("No issues detected. Code quality looks good.")
        lines.append("")
        lines.append(report.summary_for_llm)
        return "\n".join(lines)

    # Group by severity
    for severity in ("critical", "warning", "info"):
        matching = [i for i in report.issues if i.severity == severity]
        if not matching:
            continue
        header = severity.upper()
        for issue in matching:
            lines.append(f"- [{header}] {issue.category} ({issue.line_range}): {issue.description}")
            lines.append(f"  Fix: {issue.fix_suggestion}")

    lines.append("")
    lines.append(report.summary_for_llm)

    return "\n".join(lines)
