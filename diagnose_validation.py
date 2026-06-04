#!/usr/bin/env python3
"""
Diagnostic Analysis: Why 0% of strategies pass validation.

Tests hand-crafted strategies against rolling_walk_forward, walk_forward_test,
and cross_ticker_validation to identify dominant failure modes.

READ-ONLY — does not modify any source files.
"""

import sys
import time
import logging
from datetime import datetime
from dataclasses import asdict

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
log = logging.getLogger("diagnose")

# ── Imports ──
from crabquant.strategies.ema_crossover import generate_signals as ema_cross, DEFAULT_PARAMS as ema_cross_params
from crabquant.strategies.macd_momentum import generate_signals as macd_mom, DEFAULT_PARAMS as macd_mom_params
from crabquant.strategies.bollinger_squeeze import generate_signals as bb_squeeze, DEFAULT_PARAMS as bb_squeeze_params
from crabquant.validation import (
    walk_forward_test,
    rolling_walk_forward,
    cross_ticker_validation,
)
from crabquant.engine import BacktestEngine

# ── Constants ──
TICKER = "SPY"
CROSS_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM"]

STRATEGIES = [
    ("ema_crossover", ema_cross, ema_cross_params),
    ("macd_momentum", macd_mom, macd_mom_params),
    ("bollinger_squeeze", bb_squeeze, bb_squeeze_params),
]

# ── Rolling walk-forward thresholds (from VALIDATION_CONFIG) ──
RWF_THRESHOLDS = {
    "min_avg_test_sharpe": 0.5,
    "min_windows_passed": 2,
    "per_window_min_test_sharpe": 0.3,  # hardcoded in rolling_walk_forward line 431
    "per_window_max_degradation": 0.7,   # hardcoded in rolling_walk_forward line 431
}

# ── Single-split walk-forward thresholds ──
WF_THRESHOLDS = {
    "min_test_sharpe": 0.3,
    "min_test_trades": 10,
    "max_degradation": 0.7,
}


def separator(title: str, char="=", width=80):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_walk_forward_detail(wf_result):
    """Print detailed walk_forward_test result."""
    r = wf_result
    print(f"  Train Sharpe:     {r.train_sharpe:+.3f}")
    print(f"  Train Return:     {r.train_return:+.2%}")
    print(f"  Test Sharpe:      {r.test_sharpe:+.3f}")
    print(f"  Test Return:      {r.test_return:+.2%}")
    print(f"  Test Max DD:      {r.test_max_dd:.2%}")
    print(f"  Degradation:      {r.degradation:.1%}")
    print(f"  Test Trades:      (see notes)")
    print(f"  Train Regime:     {r.train_regime}")
    print(f"  Test Regime:      {r.test_regime}")
    print(f"  Regime Shift:     {r.regime_shift}")
    print(f"  ROBUST:           {'✅ YES' if r.robust else '❌ NO'}")
    print(f"  Notes:            {r.notes}")


def print_rolling_detail(rwf_result):
    """Print detailed rolling_walk_forward result."""
    r = rwf_result
    print(f"  Num Windows:      {r.num_windows}")
    print(f"  Windows Passed:   {r.windows_passed}")
    print(f"  Avg Test Sharpe:  {r.avg_test_sharpe:+.3f}")
    print(f"  Min Test Sharpe:  {r.min_test_sharpe:+.3f}")
    print(f"  Avg Degradation:  {r.avg_degradation:.1%}")
    print(f"  ROBUST:           {'✅ YES' if r.robust else '❌ NO'}")
    print(f"  Notes:            {r.notes}")

    # Per-window breakdown
    print("\n  Per-Window Breakdown:")
    print(f"  {'Win':>3} | {'Train Sharpe':>12} | {'Test Sharpe':>11} | {'Degradation':>11} | {'Passed':>6}")
    print(f"  {'---':>3}-+-{'-'*12}-+-{'-'*11}-+-{'-'*11}-+-{'-'*6}")
    for w in r.window_results:
        win = w.get("window", "?")
        ts = w.get("train_sharpe", 0)
        tts = w.get("test_sharpe", 0)
        deg = w.get("degradation", 1.0)
        passed = "✅" if w.get("passed") else "❌"
        err = w.get("error", "")
        print(f"  {win:>3} | {ts:>+12.3f} | {tts:>+11.3f} | {deg:>10.1%} | {passed:>6}  {err}")


def print_cross_ticker_detail(ct_result):
    """Print detailed cross-ticker result."""
    r = ct_result
    print(f"  Tickers Tested:   {r.tickers_tested}")
    print(f"  Tickers Profit:   {r.tickers_profitable}")
    print(f"  Tickers Passed:   {r.tickers_passed}")
    print(f"  Avg Sharpe:       {r.avg_sharpe:+.3f}")
    print(f"  Median Sharpe:    {r.median_sharpe:+.3f}")
    print(f"  Sharpe Std:       {r.sharpe_std:.3f}")
    print(f"  Avg Return:       {r.avg_return:+.2%}")
    print(f"  Avg Max DD:       {r.avg_max_dd:.2%}")
    print(f"  Win Rate:         {r.win_rate_across_tickers:.0%}")
    print(f"  ROBUST:           {'✅ YES' if r.robust else '❌ NO'}")
    print(f"  Notes:            {r.notes}")


def main():
    lines = []  # Collect all output for writing to markdown

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# Validation Pipeline Diagnostic Report")
    emit(f"**Generated:** {datetime.now().isoformat()}")
    emit(f"**Ticker:** {TICKER}")
    emit()

    separator("CURRENT THRESHOLDS", char="-")
    emit("### Rolling Walk-Forward (rolling_walk_forward)")
    for k, v in RWF_THRESHOLDS.items():
        emit(f"  - `{k}`: {v}")
    emit()
    emit("### Single-Split Walk-Forward (walk_forward_test)")
    for k, v in WF_THRESHOLDS.items():
        emit(f"  - `{k}`: {v}")
    emit()
    emit("### Cross-Ticker Validation")
    emit("  - `profitable_rate > 0.4` AND `avg_sharpe > 0.5`")
    emit()

    engine = BacktestEngine()
    all_results = {}

    for strat_name, strat_fn, default_params in STRATEGIES:
        t0 = time.time()
        emit(separator(f"STRATEGY: {strat_name}"))
        emit(f"  Params: {default_params}")
        emit()

        result_entry = {
            "name": strat_name,
            "params": default_params,
        }

        # ── 1. Rolling Walk-Forward ──
        emit("### 1. Rolling Walk-Forward (5y, 18mo train / 6mo test / 6mo step)")
        try:
            rwf = rolling_walk_forward(
                strat_fn, TICKER, default_params,
                engine=engine,
            )
            print_rolling_detail(rwf)
            result_entry["rolling_wf"] = {
                "robust": rwf.robust,
                "num_windows": rwf.num_windows,
                "windows_passed": rwf.windows_passed,
                "avg_test_sharpe": rwf.avg_test_sharpe,
                "min_test_sharpe": rwf.min_test_sharpe,
                "avg_degradation": rwf.avg_degradation,
                "notes": rwf.notes,
            }
        except Exception as e:
            emit(f"  ❌ ERROR: {e}")
            result_entry["rolling_wf"] = {"error": str(e)}
        emit()

        # ── 2. Single-Split Walk-Forward ──
        emit("### 2. Single-Split Walk-Forward (3y, 18mo train / 6mo test)")
        try:
            wf = walk_forward_test(
                strat_fn, TICKER, default_params,
                engine=engine,
            )
            print_walk_forward_detail(wf)
            result_entry["single_wf"] = {
                "robust": wf.robust,
                "train_sharpe": wf.train_sharpe,
                "test_sharpe": wf.test_sharpe,
                "degradation": wf.degradation,
                "regime_shift": wf.regime_shift,
                "train_regime": wf.train_regime or "",
                "test_regime": wf.test_regime or "",
                "notes": wf.notes,
            }
        except Exception as e:
            emit(f"  ❌ ERROR: {e}")
            result_entry["single_wf"] = {"error": str(e)}
        emit()

        # ── 3. Cross-Ticker Validation ──
        emit(f"### 3. Cross-Ticker Validation ({len(CROSS_TICKERS)} tickers)")
        try:
            ct = cross_ticker_validation(
                strat_fn, default_params, CROSS_TICKERS,
                engine=engine,
            )
            print_cross_ticker_detail(ct)
            result_entry["cross_ticker"] = {
                "robust": ct.robust,
                "tickers_tested": ct.tickers_tested,
                "tickers_profitable": ct.tickers_profitable,
                "avg_sharpe": ct.avg_sharpe,
                "median_sharpe": ct.median_sharpe,
                "win_rate": ct.win_rate_across_tickers,
                "notes": ct.notes,
            }
        except Exception as e:
            emit(f"  ❌ ERROR: {e}")
            result_entry["cross_ticker"] = {"error": str(e)}

        elapsed = time.time() - t0
        emit(f"\n  ⏱️  Elapsed: {elapsed:.1f}s")
        emit()

        all_results[strat_name] = result_entry

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    separator("ANALYSIS: DOMINANT FAILURE MODES", char="#")

    # Count failures
    rwf_passes = sum(1 for r in all_results.values() if r.get("rolling_wf", {}).get("robust"))
    single_passes = sum(1 for r in all_results.values() if r.get("single_wf", {}).get("robust"))
    ct_passes = sum(1 for r in all_results.values() if r.get("cross_ticker", {}).get("robust"))

    emit(f"\n## Pass Rate Summary")
    emit(f"| Validation Type          | Passed | Total | Rate  |")
    emit(f"|--------------------------|--------|-------|-------|")
    emit(f"| Rolling Walk-Forward     | {rwf_passes}      | {len(STRATEGIES)}      | {rwf_passes/len(STRATEGIES):.0%}    |")
    emit(f"| Single-Split Walk-Forward| {single_passes}      | {len(STRATEGIES)}      | {single_passes/len(STRATEGIES):.0%}    |")
    emit(f"| Cross-Ticker             | {ct_passes}      | {len(STRATEGIES)}      | {ct_passes/len(STRATEGIES):.0%}    |")
    emit()

    # Analyze rolling WF failures
    emit(f"## Rolling Walk-Forward Failure Analysis")
    emit()
    for name, res in all_results.items():
        rwf = res.get("rolling_wf", {})
        if not rwf.get("robust"):
            emit(f"### {name}")
            emit(f"- Avg test Sharpe: {rwf.get('avg_test_sharpe', 0):.3f} (need >= {RWF_THRESHOLDS['min_avg_test_sharpe']})")
            emit(f"- Windows passed: {rwf.get('windows_passed', 0)}/{rwf.get('num_windows', 0)} (need >= {RWF_THRESHOLDS['min_windows_passed']})")
            emit(f"- Min test Sharpe: {rwf.get('min_test_sharpe', 0):.3f}")
            emit(f"- Avg degradation: {rwf.get('avg_degradation', 0):.1%}")
            notes = rwf.get("notes", "")
            emit(f"- Notes: {notes}")
            emit()

    # Analyze single-split failures
    emit(f"## Single-Split Walk-Forward Failure Analysis")
    emit()
    for name, res in all_results.items():
        wf = res.get("single_wf", {})
        if not wf.get("robust"):
            emit(f"### {name}")
            emit(f"- Train Sharpe: {wf.get('train_sharpe', 0):.3f}")
            emit(f"- Test Sharpe: {wf.get('test_sharpe', 0):.3f} (need >= {WF_THRESHOLDS['min_test_sharpe']})")
            emit(f"- Degradation: {wf.get('degradation', 0):.1%} (max {WF_THRESHOLDS['max_degradation']:.0%})")
            emit(f"- Regime shift: {wf.get('regime_shift', False)}")
            emit(f"- Train regime: {wf.get('train_regime', '?')}")
            emit(f"- Test regime: {wf.get('test_regime', '?')}")
            notes = wf.get("notes", "")
            emit(f"- Notes: {notes}")
            emit()

    # Analyze cross-ticker failures
    emit(f"## Cross-Ticker Failure Analysis")
    emit()
    for name, res in all_results.items():
        ct = res.get("cross_ticker", {})
        if not ct.get("robust"):
            emit(f"### {name}")
            emit(f"- Avg Sharpe: {ct.get('avg_sharpe', 0):.3f} (need > 0.5)")
            emit(f"- Win rate: {ct.get('win_rate', 0):.0%} (need > 40%)")
            emit(f"- Median Sharpe: {ct.get('median_sharpe', 0):.3f}")
            emit(f"- Notes: {ct.get('notes', '')}")
            emit()

    # ═══════════════════════════════════════════════════════════════════
    # THRESHOLD RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════
    separator("THRESHOLD RECOMMENDATIONS", char="#")
    emit()
    emit("Based on the observed metrics from hand-crafted strategies:")
    emit()

    # Collect observed values
    avg_sharpes = [r["rolling_wf"].get("avg_test_sharpe", 0) for r in all_results.values() if "rolling_wf" in r]
    min_sharpes = [r["rolling_wf"].get("min_test_sharpe", 0) for r in all_results.values() if "rolling_wf" in r]
    windows_passed = [r["rolling_wf"].get("windows_passed", 0) for r in all_results.values() if "rolling_wf" in r]
    degradations = [r["rolling_wf"].get("avg_degradation", 1) for r in all_results.values() if "rolling_wf" in r]

    best_avg_sharpe = max(avg_sharpes) if avg_sharpes else 0
    best_min_sharpe = max(min_sharpes) if min_sharpes else 0
    best_windows = max(windows_passed) if windows_passed else 0
    best_degradation = min(degradations) if degradations else 1.0

    emit(f"### Observed Best Metrics (from {len(STRATEGIES)} hand-crafted strategies)")
    emit(f"- Best avg test Sharpe:  {best_avg_sharpe:.3f}")
    emit(f"- Best min test Sharpe:  {best_min_sharpe:.3f}")
    emit(f"- Best windows passed:   {best_windows}")
    emit(f"- Best avg degradation:  {best_degradation:.1%}")
    emit()

    # Recommendations
    emit("### Recommended Thresholds for >50% Pass Rate")
    emit()
    emit("| Parameter | Current | Recommended | Rationale |")
    emit("|-----------|---------|-------------|-----------|")

    if best_avg_sharpe < RWF_THRESHOLDS["min_avg_test_sharpe"]:
        rec = round(best_avg_sharpe * 0.9, 2)  # 90% of best observed
        emit(f"| `min_avg_test_sharpe` | {RWF_THRESHOLDS['min_avg_test_sharpe']} | {rec} | Best observed: {best_avg_sharpe:.3f} — current threshold unreachable |")
    else:
        emit(f"| `min_avg_test_sharpe` | {RWF_THRESHOLDS['min_avg_test_sharpe']} | {RWF_THRESHOLDS['min_avg_test_sharpe']} | OK — best observed exceeds it |")

    if best_windows < RWF_THRESHOLDS["min_windows_passed"]:
        rec = max(1, best_windows)
        emit(f"| `min_windows_passed` | {RWF_THRESHOLDS['min_windows_passed']} | {rec} | Best observed: {best_windows} — current threshold unreachable |")
    else:
        emit(f"| `min_windows_passed` | {RWF_THRESHOLDS['min_windows_passed']} | {RWF_THRESHOLDS['min_windows_passed']} | OK |")

    # Per-window thresholds
    emit(f"| per-window `min_test_sharpe` | {RWF_THRESHOLDS['per_window_min_test_sharpe']} | 0.0 | Remove or lower — single window failures cascade |")
    emit(f"| per-window `max_degradation` | {RWF_THRESHOLDS['per_window_max_degradation']} | 1.0 | Remove or raise — too strict for individual windows |")
    emit()

    # ═══════════════════════════════════════════════════════════════════
    # BUG DETECTION
    # ═══════════════════════════════════════════════════════════════════
    separator("POTENTIAL BUGS IN VALIDATION LOGIC", char="#")
    emit()

    emit("### Bug 1: Hardcoded per-window thresholds in rolling_walk_forward (line 431)")
    emit("```python")
    emit("window_passed = test_result.sharpe >= 0.3 and degradation <= 0.7")
    emit("```")
    emit("These thresholds (0.3 Sharpe, 0.7 degradation) are **hardcoded** and NOT")
    emit("configurable via function parameters. The `min_avg_test_sharpe` and `min_windows_passed`")
    emit("params only control the FINAL aggregate check. Individual windows must pass these")
    emit("hardcoded gates to count as 'passed'. This means:")
    emit("- If a strategy has 3 windows with Sharpe [0.4, 0.2, 0.4], only 2 pass (need 2) — OK")
    emit("- If a strategy has 3 windows with Sharpe [0.4, 0.25, 0.4], only 2 pass — OK")
    emit("- But degradation can silently kill windows that have OK Sharpe")
    emit()

    emit("### Bug 2: Double-gating makes thresholds excessively strict")
    emit("The rolling walk-forward requires BOTH:")
    emit("1. `avg_test_sharpe >= min_avg_test_sharpe` (0.5)")
    emit("2. `windows_passed >= min_windows_passed` (2)")
    emit("AND each window must individually pass `sharpe >= 0.3 AND degradation <= 0.7`.")
    emit("This is **triple-gating**: per-window Sharpe, per-window degradation, AND aggregate.")
    emit("For a strategy to pass, it needs consistently positive OOS performance across ALL windows.")
    emit()

    emit("### Bug 3: No num_trades check in rolling_walk_forward")
    emit("Unlike walk_forward_test() which checks `min_test_trades >= 10`, rolling_walk_forward()")
    emit("does NOT check trade count. A strategy with 1 trade in a window that happens to be")
    emit("profitable can pass the Sharpe check, creating a false positive. Conversely, a strategy")
    emit("with many trades might fail on Sharpe noise.")
    emit()

    emit("### Bug 4: Train Sharpe = 0 edge case in degradation calculation")
    emit("When `train_result.sharpe == 0`, the degradation formula divides by zero.")
    emit("The code checks `train_result.sharpe > 0`, but a Sharpe of exactly 0.0 (not negative)")
    emit("falls through to the `elif test_result.sharpe > 0` branch, setting degradation=0.0.")
    emit("This means strategies with zero in-sample Sharpe but positive OOS Sharpe get")
    emit("degradation=0.0, which is incorrect — it should be undefined/inconclusive.")
    emit()

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    separator("SUMMARY", char="#")
    emit()
    emit("### Can ANY strategy pass current thresholds?")
    if rwf_passes > 0:
        emit(f"YES — {rwf_passes}/{len(STRATEGIES)} strategies passed rolling walk-forward.")
    else:
        emit("NO — 0/{} hand-crafted strategies passed rolling walk-forward.".format(len(STRATEGIES)))
        emit("These are well-known, historically profitable strategies (EMA crossover, MACD,")
        emit("Bollinger Squeeze) with DEFAULT parameters. If they can't pass, the thresholds")
        emit("are almost certainly too strict for the invented strategies.")
    emit()

    emit("### Dominant Failure Mode(s)")
    if best_avg_sharpe < RWF_THRESHOLDS["min_avg_test_sharpe"]:
        emit(f"1. **LOW TEST SHARPE**: Best avg test Sharpe = {best_avg_sharpe:.3f}, need >= {RWF_THRESHOLDS['min_avg_test_sharpe']}")
    if best_windows < RWF_THRESHOLDS["min_windows_passed"]:
        emit(f"2. **TOO FEW WINDOWS PASSING**: Best windows passed = {best_windows}, need >= {RWF_THRESHOLDS['min_windows_passed']}")
    if best_degradation > RWF_THRESHOLDS["per_window_max_degradation"]:
        emit(f"3. **HIGH DEGRADATION**: Best avg degradation = {best_degradation:.1%}, per-window max = {RWF_THRESHOLDS['per_window_max_degradation']:.0%}")
    emit()

    emit("### Specific Threshold Recommendations")
    emit("To achieve >50% pass rate with hand-crafted strategies:")
    emit()
    if best_avg_sharpe < 0.5:
        emit(f"- Lower `min_avg_test_sharpe` to **{max(0.0, best_avg_sharpe - 0.1):.1f}**")
    if best_windows < 2:
        emit(f"- Lower `min_windows_passed` to **{max(1, best_windows)}**")
    emit("- Make per-window thresholds configurable (currently hardcoded)")
    emit("- Consider removing degradation gate from per-window check (keep only aggregate)")
    emit("- Add min_trades check to rolling_walk_forward for consistency")
    emit()

    # Write to file
    report_path = "/home/Zev/development/CrabQuant-agents/worker-1/validation_diagnosis.md"
    with open(report_path, "w") as f:
        f.write("\n".join(str(line) if line is not None else "" for line in lines))
    print(f"\nReport written to: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
