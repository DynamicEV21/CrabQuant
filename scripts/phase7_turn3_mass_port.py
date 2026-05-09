"""
phase7_turn3_mass_port.py v3
CrabQuant strategy mass-port to backtesting.py -- signal replay path.

Uses trade-timing similarity instead of raw signal correlation.
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

CRAB_DIR = Path("~/development/CrabQuant").expanduser()
sys.path.insert(0, str(CRAB_DIR))

from crabquant.data import load_data

TICKERS = ["AAPL", "MSFT", "SPY", "QQQ", "GLD"]
DATA_PERIOD = "6mo"
MIN_PRECISION = 0.80
OUTPUT_DIR = Path("~/development/quant-research-mas/server/backtesting_strategies").expanduser()

STRATEGY_MODULES = [
    "ema_crossover",
    "adx_pullback",
    "bollinger_squeeze",
    "bb_stoch_macd",
    "atr_channel_breakout",
    "ema_ribbon_reversal",
    "ichimoku_trend",
    "invented_momentum_confluence_cat",
    "invented_volume_breakout_adx_nvda",
    "invented_volatility_rsi_breakout",
    "invented_vwap_reversion_aapl",
    "invented_rsi_macd_volume_msft",
]


def safe_load_strategy(module_name: str):
    try:
        module = importlib.import_module(f"crabquant.strategies.{module_name}")
        gs = getattr(module, "generate_signals", None)
        dp = getattr(module, "DEFAULT_PARAMS", {})
        desc = getattr(module, "DESCRIPTION", "")
        return gs, dp, desc
    except Exception as e:
        return None, None, str(e)


def make_replay_strategy(entries: pd.Series, exits: pd.Series, name: str):
    e = entries.fillna(False).astype(bool)
    x = exits.fillna(False).astype(bool)

    class ReplayStrategy(Strategy):
        def init(self):
            pass

        def next(self):
            idx = len(self.data) - 1
            if idx < 0:
                return
            try:
                ts = self.data.index[idx]
            except Exception:
                return
            if e.get(ts, False) and not self.position:
                self.buy()
            elif x.get(ts, False) and self.position:
                self.position.close()

    ReplayStrategy.__name__ = name
    ReplayStrategy.__qualname__ = name
    return ReplayStrategy


def _precision(original_times: list, derived_times: list, tol_days: int = 1):
    """What fraction of original times have a derived match within ±tol_days?"""
    if not original_times:
        return 1.0 if not derived_times else 0.0
    derived_set = set(derived_times)
    matched = 0
    for ot in original_times:
        if any(abs((ot - dt).days) <= tol_days for dt in derived_set):
            matched += 1
    return matched / len(original_times)


def compare_trade_timing(orig_entries, orig_exits, bt_stats, ticker):
    orig_entry_times = orig_entries[orig_entries.fillna(False)].index.tolist()
    orig_exit_times = orig_exits[orig_exits.fillna(False)].index.tolist()

    trades = bt_stats._trades if bt_stats is not None else None
    derived_entry_times = []
    derived_exit_times = []
    if trades is not None and len(trades) > 0:
        for _, row in trades.iterrows():
            et = row.get("EntryTime")
            xt = row.get("ExitTime")
            if pd.notna(et) and isinstance(et, pd.Timestamp):
                derived_entry_times.append(et)
            if pd.notna(xt) and isinstance(xt, pd.Timestamp):
                derived_exit_times.append(xt)

    entry_prec = _precision(orig_entry_times, derived_entry_times)
    exit_prec = _precision(orig_exit_times, derived_exit_times)

    # Weight: 60% entry, 40% exit
    fidelity = entry_prec * 0.6 + exit_prec * 0.4
    passed = fidelity >= MIN_PRECISION
    return {
        "ticker": ticker,
        "fidelity": round(fidelity, 4),
        "entry_precision": round(entry_prec, 4),
        "exit_precision": round(exit_prec, 4),
        "orig_entries": len(orig_entry_times),
        "orig_exits": len(orig_exit_times),
        "derived_entries": len(derived_entry_times),
        "derived_exits": len(derived_exit_times),
        "passed": passed,
        "reason": None,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for mod_name in STRATEGY_MODULES:
        print(f"\n=== {mod_name} ===")
        gen, params, desc = safe_load_strategy(mod_name)
        if gen is None:
            print(f"  SKIP load failed: {desc}")
            results.append({"module": mod_name, "status": "SKIP", "reason": desc})
            continue

        ticker_results = []
        all_passed = True
        for ticker in TICKERS:
            print(f"  {ticker}...", end=" ")
            df = load_data(ticker, period=DATA_PERIOD)
            if df is None or len(df) < 50:
                print("NO_DATA")
                ticker_results.append({"ticker": ticker, "status": "NO_DATA"})
                all_passed = False
                continue

            try:
                orig_entries, orig_exits = gen(df, params)
            except Exception as e:
                print(f"ORIG_ERR: {e}")
                ticker_results.append({"ticker": ticker, "status": "ORIG_ERR", "reason": str(e)})
                all_passed = False
                continue

            bt_df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
            bt_df.index = pd.to_datetime(bt_df.index)
            bt_df = bt_df.dropna()

            replay_cls = make_replay_strategy(orig_entries, orig_exits, mod_name)

            try:
                bt = Backtest(bt_df, replay_cls, cash=100000, commission=0.001)
                stats = bt.run()
            except Exception as e:
                print(f"BT_ERR: {e}")
                ticker_results.append({"ticker": ticker, "status": "BT_ERR", "reason": str(e)})
                all_passed = False
                continue

            comp = compare_trade_timing(orig_entries, orig_exits, stats, ticker)
            ticker_results.append(comp)
            if not comp["passed"]:
                all_passed = False
            detail = f"{comp['orig_entries']}E/{comp['orig_exits']}X vs {comp['derived_entries']}E/{comp['derived_exits']}X fid={comp['fidelity']}"
            print(f"{'PASS' if comp['passed'] else 'FAIL'} {detail}")

        results.append({
            "module": mod_name,
            "status": "PASS" if all_passed else "DIVERGENCE",
            "ticker_results": ticker_results,
            "description": desc[:200] if desc else "",
        })

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n--- Summary: {passed}/{total} strategies passed ---")
    for r in results:
        print(f"  {r['module']}: {r['status']}")

    report_path = OUTPUT_DIR / "mass_port_report_v3.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
