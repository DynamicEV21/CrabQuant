#!/usr/bin/env python3
"""
Validation Diagnostic Probe
============================
Tests 5 hand-crafted strategies through both walk_forward_test() and
rolling_walk_forward() with current thresholds, then calculates what
threshold values WOULD allow each to pass.

Goal: Measure baseline validation passability and recommend threshold values.
"""

import sys
import os
import traceback
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crabquant.data import load_data
from crabquant.engine.backtest import BacktestEngine
from crabquant.validation import walk_forward_test, rolling_walk_forward
from crabquant.indicator_cache import cached_indicator, clear_cache


# ──────────────────────────────────────────────────────────────────────────────
# Strategy Definitions
# ──────────────────────────────────────────────────────────────────────────────

def ema_crossover(df, params=None):
    """Fast/slow EMA crossover."""
    p = params or {"fast": 9, "slow": 21}
    close = df["close"]
    ema_f = cached_indicator("ema", close, length=p["fast"])
    ema_s = cached_indicator("ema", close, length=p["slow"])
    entries = ((ema_f.shift(1) < ema_s.shift(1)) & (ema_f > ema_s)).fillna(False)
    exits = ((ema_f.shift(1) > ema_s.shift(1)) & (ema_f < ema_s)).fillna(False)
    return entries, exits


def rsi_mean_reversion(df, params=None):
    """RSI oversold/overbought mean reversion."""
    p = params or {"rsi_len": 14, "oversold": 30, "overbought": 70}
    close = df["close"]
    rsi = cached_indicator("rsi", close, length=p["rsi_len"])
    entries = (rsi.shift(1) < p["oversold"]) & (rsi >= p["oversold"]).fillna(False)
    exits = (rsi.shift(1) > p["overbought"]) & (rsi <= p["overbought"]).fillna(False)
    return entries, exits


def bollinger_squeeze(df, params=None):
    """Bollinger Band squeeze breakout."""
    p = params or {"bb_len": 20, "bb_std": 2.0, "squeeze_mult": 0.8}
    close = df["close"]
    volume = df["volume"]
    bb = cached_indicator("bbands", close, length=p["bb_len"], std=p["bb_std"])
    bbu_col = [c for c in bb.columns if c.startswith("BBU")][0]
    bbl_col = [c for c in bb.columns if c.startswith("BBL")][0]
    bbm_col = [c for c in bb.columns if c.startswith("BBM")][0]
    bbu = bb[bbu_col]
    bbl = bb[bbl_col]
    bbm = bb[bbm_col]
    bb_width = (bbu - bbl) / bbm
    bb_width_avg = bb_width.rolling(50).mean()
    squeeze = bb_width < bb_width_avg * p["squeeze_mult"]
    breakout_up = close > bbu
    vol_confirm = volume > volume.rolling(20).mean() * 1.2
    entries = (squeeze.shift(1) & breakout_up & vol_confirm).fillna(False)
    exits = (close < bbm).fillna(False)
    return entries, exits


def macd_momentum(df, params=None):
    """MACD signal crossover."""
    p = params or {"fast": 12, "slow": 26, "signal": 9}
    close = df["close"]
    macd = cached_indicator("macd", close, fast=p["fast"], slow=p["slow"], signal=p["signal"])
    hist_col = f"MACDh_{p['fast']}_{p['slow']}_{p['signal']}"
    hist = macd[hist_col]
    entries = ((hist.shift(1) < 0) & (hist >= 0)).fillna(False)
    exits = ((hist.shift(1) > 0) & (hist <= 0)).fillna(False)
    return entries, exits


def simple_momentum(df, params=None):
    """Price above N-day moving average."""
    p = params or {"sma_len": 50}
    close = df["close"]
    sma = cached_indicator("sma", close, length=p["sma_len"])
    entries = ((close.shift(1) <= sma.shift(1)) & (close > sma)).fillna(False)
    exits = ((close.shift(1) >= sma.shift(1)) & (close < sma)).fillna(False)
    return entries, exits


STRATEGIES = {
    "ema_crossover": (ema_crossover, {"fast": 9, "slow": 21}),
    "rsi_mean_reversion": (rsi_mean_reversion, {"rsi_len": 14, "oversold": 30, "overbought": 70}),
    "bollinger_squeeze": (bollinger_squeeze, {"bb_len": 20, "bb_std": 2.0, "squeeze_mult": 0.8}),
    "macd_momentum": (macd_momentum, {"fast": 12, "slow": 26, "signal": 9}),
    "simple_momentum": (simple_momentum, {"sma_len": 50}),
}

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run a single backtest and return key metrics
# ──────────────────────────────────────────────────────────────────────────────

def run_single_backtest(strategy_fn, ticker, params):
    """Run a single backtest on the full 3y period and return metrics."""
    try:
        df = load_data(ticker, period="3y")
        engine = BacktestEngine()
        entries, exits = strategy_fn(df, params)
        result = engine.run(df, entries, exits, strategy_fn.__name__, ticker, params=params)
        return result
    except Exception as e:
        print(f"  [ERROR] Backtest failed for {strategy_fn.__name__}/{ticker}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Threshold Analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_wf_thresholds(result):
    """Given a WalkForwardResult, compute what thresholds would allow it to pass."""
    info = {}
    # Current thresholds
    info["test_sharpe"] = result.test_sharpe
    info["degradation"] = result.degradation
    info["robust"] = result.robust

    # What min_test_sharpe would allow pass?
    # Need: test_sharpe >= min_test_sharpe AND degradation <= max_degradation
    # Also need test_trades >= min_test_trades
    info["min_test_sharpe_to_pass"] = round(result.test_sharpe - 0.05, 3) if result.test_sharpe > 0 else round(result.test_sharpe, 3)
    info["max_degradation_to_pass"] = round(result.degradation + 0.05, 3)

    # How close to passing with current thresholds (min_test_sharpe=0.3, max_degradation=0.8)?
    sharpe_gap = 0.3 - result.test_sharpe  # negative = already passing
    degrad_gap = 0.8 - result.degradation  # negative = already passing
    info["sharpe_gap"] = round(sharpe_gap, 3)
    info["degrad_gap"] = round(degrad_gap, 3)

    return info


def analyze_rwf_thresholds(result):
    """Given a RollingWalkForwardResult, compute what thresholds would allow it to pass."""
    info = {}
    info["avg_test_sharpe"] = result.avg_test_sharpe
    info["min_test_sharpe"] = result.min_test_sharpe
    info["avg_degradation"] = result.avg_degradation
    info["windows_passed"] = result.windows_passed
    info["num_windows"] = result.num_windows
    info["robust"] = result.robust

    # Per-window details
    info["window_sharpes"] = [w["test_sharpe"] for w in result.window_results]
    info["window_degradations"] = [w["degradation"] for w in result.window_results]

    # How close to passing with current thresholds?
    sharpe_gap = 0.3 - result.avg_test_sharpe
    info["sharpe_gap"] = round(sharpe_gap, 3)

    # Windows passed threshold: need >= min_windows_passed (default 1)
    windows_gap = 1 - result.windows_passed
    info["windows_gap"] = windows_gap

    return info


# ──────────────────────────────────────────────────────────────────────────────
# Main Probe
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("CrabQuant Validation Diagnostic Probe")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    print()

    all_results = {}

    for strat_name, (strat_fn, params) in STRATEGIES.items():
        print(f"\n{'─' * 70}")
        print(f"STRATEGY: {strat_name}")
        print(f"{'─' * 70}")
        strat_results = {"strategies": [], "wf": {}, "rwf": {}}

        # Run across multiple tickers for a comprehensive view
        ticker_wf_results = []
        ticker_rwf_results = []

        for ticker in TICKERS:
            clear_cache()
            print(f"\n  Ticker: {ticker}")
            print(f"  {'─' * 50}")

            # ── walk_forward_test with CURRENT thresholds ──
            try:
                wf = walk_forward_test(
                    strat_fn, ticker, params,
                    min_test_sharpe=0.3,
                    min_test_trades=10,
                    max_degradation=0.8,
                )
                wf_info = analyze_wf_thresholds(wf)
                wf_info["ticker"] = ticker
                ticker_wf_results.append(wf_info)

                print(f"  [WALK-FORWARD]")
                print(f"    Train Sharpe: {wf.train_sharpe:.3f}  |  Test Sharpe: {wf.test_sharpe:.3f}")
                print(f"    Degradation:  {wf.degradation:.1%}  |  Robust: {wf.robust}")
                print(f"    Sharpe gap to pass (need 0.3): {wf_info['sharpe_gap']:+.3f}")
                print(f"    Degrad gap to pass (need ≤0.8): {wf_info['degrad_gap']:+.3f}")
                print(f"    Notes: {wf.notes}")
            except Exception as e:
                print(f"  [WALK-FORWARD] ERROR: {e}")
                traceback.print_exc()

            # ── rolling_walk_forward with CURRENT thresholds ──
            try:
                rwf = rolling_walk_forward(
                    strat_fn, ticker, params,
                    min_avg_test_sharpe=0.3,
                    min_windows_passed=1,
                )
                rwf_info = analyze_rwf_thresholds(rwf)
                rwf_info["ticker"] = ticker
                ticker_rwf_results.append(rwf_info)

                print(f"  [ROLLING-WF]")
                print(f"    Windows: {rwf.num_windows}  |  Passed: {rwf.windows_passed}")
                print(f"    Avg Test Sharpe: {rwf.avg_test_sharpe:.3f}  |  Min: {rwf.min_test_sharpe:.3f}")
                print(f"    Avg Degradation: {rwf.avg_degradation:.1%}  |  Robust: {rwf.robust}")
                print(f"    Sharpe gap to pass (need 0.3): {rwf_info['sharpe_gap']:+.3f}")
                if rwf.window_results:
                    ws = [f"W{w['window']}:{w['test_sharpe']:.2f}" for w in rwf.window_results]
                    print(f"    Per-window Sharpes: {', '.join(ws)}")
                    wd = [f"W{w['window']}:{w['degradation']:.1%}" for w in rwf.window_results]
                    print(f"    Per-window Degrads:  {', '.join(wd)}")
                print(f"    Notes: {rwf.notes}")
            except Exception as e:
                print(f"  [ROLLING-WF] ERROR: {e}")
                traceback.print_exc()

        strat_results["wf"] = ticker_wf_results
        strat_results["rwf"] = ticker_rwf_results
        all_results[strat_name] = strat_results

    # ──────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print("\n### Walk-Forward Test Results (current thresholds: min_sharpe=0.3, max_degrad=0.8, min_trades=10)")
    print(f"{'Strategy':<22} {'Ticker':<8} {'TestSharpe':>11} {'Degradation':>12} {'SharpeGap':>11} {'DegradGap':>11} {'Robust':>8}")
    print("-" * 85)

    wf_pass_count = 0
    wf_total = 0
    for strat_name, strat_res in all_results.items():
        for wf in strat_res["wf"]:
            wf_total += 1
            if wf.get("robust"):
                wf_pass_count += 1
            print(f"{strat_name:<22} {wf['ticker']:<8} {wf['test_sharpe']:>11.3f} {wf['degradation']:>11.1%} {wf['sharpe_gap']:>+11.3f} {wf['degrad_gap']:>+11.3f} {'✓' if wf.get('robust') else '✗':>8}")

    print(f"\nWalk-Forward pass rate: {wf_pass_count}/{wf_total} ({100*wf_pass_count/max(wf_total,1):.0f}%)")

    print("\n### Rolling Walk-Forward Results (current thresholds: min_avg_sharpe=0.3, min_windows=1)")
    print(f"{'Strategy':<22} {'Ticker':<8} {'AvgTstShp':>11} {'MinTstShp':>11} {'WinPass':>8} {'TotWin':>8} {'SharpeGap':>11} {'Robust':>8}")
    print("-" * 89)

    rwf_pass_count = 0
    rwf_total = 0
    for strat_name, strat_res in all_results.items():
        for rwf in strat_res["rwf"]:
            rwf_total += 1
            if rwf.get("robust"):
                rwf_pass_count += 1
            print(f"{strat_name:<22} {rwf['ticker']:<8} {rwf['avg_test_sharpe']:>11.3f} {rwf['min_test_sharpe']:>11.3f} {rwf['windows_passed']:>8} {rwf['num_windows']:>8} {rwf['sharpe_gap']:>+11.3f} {'✓' if rwf.get('robust') else '✗':>8}")

    print(f"\nRolling-WF pass rate: {rwf_pass_count}/{rwf_total} ({100*rwf_pass_count/max(rwf_total,1):.0f}%)")

    # ──────────────────────────────────────────────────────────────────────────
    # THRESHOLD RECOMMENDATIONS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("THRESHOLD ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)

    # Collect all observed values across strategies and tickers
    all_wf_sharpes = []
    all_wf_degradations = []
    all_rwf_avg_sharpes = []
    all_rwf_min_sharpes = []

    for strat_name, strat_res in all_results.items():
        for wf in strat_res["wf"]:
            all_wf_sharpes.append(wf["test_sharpe"])
            all_wf_degradations.append(wf["degradation"])
        for rwf in strat_res["rwf"]:
            all_rwf_avg_sharpes.append(rwf["avg_test_sharpe"])
            all_rwf_min_sharpes.append(rwf["min_test_sharpe"])

    if all_wf_sharpes:
        print(f"\nWalk-Forward Test Sharpe Distribution:")
        print(f"  Min:  {min(all_wf_sharpes):.3f}")
        print(f"  Max:  {max(all_wf_sharpes):.3f}")
        print(f"  Mean: {np.mean(all_wf_sharpes):.3f}")
        print(f"  P25:  {np.percentile(all_wf_sharpes, 25):.3f}")
        print(f"  P50:  {np.percentile(all_wf_sharpes, 50):.3f}")
        print(f"  P75:  {np.percentile(all_wf_sharpes, 75):.3f}")
        print(f"  Positive: {sum(1 for s in all_wf_sharpes if s > 0)}/{len(all_wf_sharpes)}")

    if all_wf_degradations:
        print(f"\nWalk-Forward Degradation Distribution:")
        print(f"  Min:  {min(all_wf_degradations):.3f}")
        print(f"  Max:  {max(all_wf_degradations):.3f}")
        print(f"  Mean: {np.mean(all_wf_degradations):.3f}")
        print(f"  P50:  {np.percentile(all_wf_degradations, 50):.3f}")
        print(f"  P75:  {np.percentile(all_wf_degradations, 75):.3f}")
        print(f"  ≤0.8: {sum(1 for d in all_wf_degradations if d <= 0.8)}/{len(all_wf_degradations)}")
        print(f"  ≤1.0: {sum(1 for d in all_wf_degradations if d <= 1.0)}/{len(all_wf_degradations)}")
        print(f"  ≤1.5: {sum(1 for d in all_wf_degradations if d <= 1.5)}/{len(all_wf_degradations)}")

    if all_rwf_avg_sharpes:
        print(f"\nRolling-WF Avg Test Sharpe Distribution:")
        print(f"  Min:  {min(all_rwf_avg_sharpes):.3f}")
        print(f"  Max:  {max(all_rwf_avg_sharpes):.3f}")
        print(f"  Mean: {np.mean(all_rwf_avg_sharpes):.3f}")
        print(f"  P25:  {np.percentile(all_rwf_avg_sharpes, 25):.3f}")
        print(f"  P50:  {np.percentile(all_rwf_avg_sharpes, 50):.3f}")
        print(f"  P75:  {np.percentile(all_rwf_avg_sharpes, 75):.3f}")
        print(f"  Positive: {sum(1 for s in all_rwf_avg_sharpes if s > 0)}/{len(all_rwf_avg_sharpes)}")

    # Calculate what thresholds would achieve different pass rates
    print("\n\n### Simulated Pass Rates at Various Thresholds (Walk-Forward)")
    print(f"{'min_test_sharpe':>16} {'max_degradation':>16} {'pass_count':>12} {'pass_rate':>10}")
    print("-" * 58)
    for min_sh in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3]:
        for max_deg in [0.5, 0.8, 1.0, 1.5, 2.0]:
            passes = 0
            for strat_name, strat_res in all_results.items():
                for wf in strat_res["wf"]:
                    if wf["test_sharpe"] >= min_sh and wf["degradation"] <= max_deg:
                        passes += 1
            rate = 100 * passes / max(wf_total, 1)
            marker = " ◄" if rate >= 50 else ""
            print(f"{min_sh:>16.1f} {max_deg:>16.1f} {passes:>12} {rate:>9.0f}%{marker}")

    print("\n### Simulated Pass Rates at Various Thresholds (Rolling-WF)")
    print(f"{'min_avg_test_sharpe':>22} {'min_windows_passed':>20} {'pass_count':>12} {'pass_rate':>10}")
    print("-" * 68)
    for min_sh in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3]:
        for min_win in [1, 2, 3]:
            passes = 0
            for strat_name, strat_res in all_results.items():
                for rwf in strat_res["rwf"]:
                    if rwf["avg_test_sharpe"] >= min_sh and rwf["windows_passed"] >= min_win:
                        passes += 1
            rate = 100 * passes / max(rwf_total, 1)
            marker = " ◄" if rate >= 50 else ""
            print(f"{min_sh:>22.1f} {min_win:>20} {passes:>12} {rate:>9.0f}%{marker}")

    # ──────────────────────────────────────────────────────────────────────────
    # RECOMMENDED VALUES
    # ──────────────────────────────────────────────────────────────────────────
    print("\n\n### RECOMMENDED THRESHOLD VALUES")
    print("=" * 60)

    # Find the loosest thresholds that still maintain >50% pass rate
    # while being stricter than "everything passes"
    recommended_wf = None
    for min_sh in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3]:
        for max_deg in [0.5, 0.8, 1.0, 1.5, 2.0]:
            passes = 0
            for strat_name, strat_res in all_results.items():
                for wf in strat_res["wf"]:
                    if wf["test_sharpe"] >= min_sh and wf["degradation"] <= max_deg:
                        passes += 1
            rate = 100 * passes / max(wf_total, 1)
            if rate >= 50 and rate <= 85:
                recommended_wf = (min_sh, max_deg, passes, rate)
                break
        if recommended_wf:
            break

    if recommended_wf:
        print(f"\nWalk-Forward recommended:")
        print(f"  min_test_sharpe:   {recommended_wf[0]}")
        print(f"  max_degradation:   {recommended_wf[1]}")
        print(f"  Expected pass rate: {recommended_wf[2]}/{wf_total} ({recommended_wf[3]:.0f}%)")
    else:
        # Fall back to observed median
        print(f"\nWalk-Forward recommended (based on observed data):")
        median_sharpe = np.median(all_wf_sharpes) if all_wf_sharpes else 0
        median_degrad = np.median(all_wf_degradations) if all_wf_degradations else 0.8
        print(f"  min_test_sharpe:   {median_sharpe:.2f} (median of observed test Sharpes)")
        print(f"  max_degradation:   {median_degrad:.2f} (median of observed degradations)")

    recommended_rwf = None
    for min_sh in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3]:
        for min_win in [1, 2, 3]:
            passes = 0
            for strat_name, strat_res in all_results.items():
                for rwf in strat_res["rwf"]:
                    if rwf["avg_test_sharpe"] >= min_sh and rwf["windows_passed"] >= min_win:
                        passes += 1
            rate = 100 * passes / max(rwf_total, 1)
            if rate >= 50 and rate <= 85:
                recommended_rwf = (min_sh, min_win, passes, rate)
                break
        if recommended_rwf:
            break

    if recommended_rwf:
        print(f"\nRolling-WF recommended:")
        print(f"  min_avg_test_sharpe: {recommended_rwf[0]}")
        print(f"  min_windows_passed:  {recommended_rwf[1]}")
        print(f"  Expected pass rate:  {recommended_rwf[2]}/{rwf_total} ({recommended_rwf[3]:.0f}%)")
    else:
        print(f"\nRolling-WF recommended (based on observed data):")
        median_sharpe = np.median(all_rwf_avg_sharpes) if all_rwf_avg_sharpes else 0
        print(f"  min_avg_test_sharpe: {median_sharpe:.2f} (median of observed avg test Sharpes)")
        print(f"  min_windows_passed:  1")

    # ──────────────────────────────────────────────────────────────────────────
    # WHICH STRATEGIES ARE CLOSEST TO PASSING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n\n### Strategies Closest to Passing (Walk-Forward)")
    print("=" * 60)

    # Score by average closeness across tickers
    strategy_scores = []
    for strat_name, strat_res in all_results.items():
        wf_scores = []
        for wf in strat_res["wf"]:
            # Distance to passing: both sharpe_gap and degrad_gap must be ≤ 0
            # Score = max(sharpe_gap, degrad_gap) where lower = closer to passing
            score = max(wf["sharpe_gap"], wf["degrad_gap"])
            wf_scores.append(score)
        avg_score = np.mean(wf_scores) if wf_scores else 999
        best_ticker = strat_res["wf"][wf_scores.index(min(wf_scores))]["ticker"] if wf_scores else "N/A"
        strategy_scores.append((strat_name, avg_score, best_ticker))

    strategy_scores.sort(key=lambda x: x[1])
    for name, score, ticker in strategy_scores:
        bar = "█" * max(0, int(10 + score * 5)) + "░" * max(0, int(-score * 5))
        status = "PASSING" if score <= 0 else f"gap={score:.3f}"
        print(f"  {name:<22} {status:>15}  (best ticker: {ticker})")

    print("\n\n### Strategies Closest to Passing (Rolling-WF)")
    print("=" * 60)

    strategy_scores_rwf = []
    for strat_name, strat_res in all_results.items():
        rwf_scores = []
        for rwf in strat_res["rwf"]:
            score = rwf["sharpe_gap"]
            rwf_scores.append(score)
        avg_score = np.mean(rwf_scores) if rwf_scores else 999
        best_ticker = strat_res["rwf"][rwf_scores.index(min(rwf_scores))]["ticker"] if rwf_scores else "N/A"
        strategy_scores_rwf.append((strat_name, avg_score, best_ticker))

    strategy_scores_rwf.sort(key=lambda x: x[1])
    for name, score, ticker in strategy_scores_rwf:
        status = "PASSING" if score <= 0 else f"gap={score:.3f}"
        print(f"  {name:<22} {status:>15}  (best ticker: {ticker})")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    # Return raw data for the markdown report
    return all_results


if __name__ == "__main__":
    results = main()
