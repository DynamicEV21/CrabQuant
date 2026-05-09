"""
phase7_turn3_cross_verify.py v6
Verify that CrabQuant signals replay correctly in backtesting.py.
Compares trade ENTRY/EXIT timestamps, not overall metrics (expected divergences from position sizing).

Flags divergences for investigation but does NOT gate on Sharpe parity.
"""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
from backtesting import Backtest, Strategy

CRAB_DIR = Path("~/development/CrabQuant").expanduser()
sys.path.insert(0, str(CRAB_DIR))

from crabquant.data import load_data

TICKERS = ["AAPL", "MSFT", "SPY", "QQQ", "GLD"]
DATA_PERIOD = "2y"
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
]


def safe_load_strategy(module_name):
    try:
        mod = importlib.import_module(f"crabquant.strategies.{module_name}")
        return getattr(mod, "generate_signals"), getattr(mod, "DEFAULT_PARAMS", {}), getattr(mod, "DESCRIPTION", "")
    except Exception as e:
        return None, None, str(e)


def _make_bt_df(df):
    return df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}).dropna()


def _build_replay_class(entries, exits, name):
    e = entries.fillna(False).astype(bool)
    x = exits.fillna(False).astype(bool)

    class ReplayStrategy(Strategy):
        def init(self): pass
        def next(self):
            idx = len(self.data) - 1
            if idx < 0: return
            ts = self.data.index[idx]
            if e.get(ts, False) and not self.position:
                self.buy()
            elif x.get(ts, False) and self.position:
                self.position.close()

    ReplayStrategy.__name__ = name
    return ReplayStrategy


def compare_timing(entries, exits, bt_stats):
    orig_e = set(entries[entries.fillna(False)].index)
    orig_x = set(exits[exits.fillna(False)].index)
    derived_e = set()
    derived_x = set()
    if bt_stats._trades is not None and len(bt_stats._trades) > 0:
        for _, row in bt_stats._trades.iterrows():
            et = row.get("EntryTime")
            xt = row.get("ExitTime")
            if pd.notna(et) and isinstance(et, pd.Timestamp):
                derived_e.add(et)
            if pd.notna(xt) and isinstance(xt, pd.Timestamp):
                derived_x.add(xt)

    # Precision: what % of original have a match within 1 day in derived
    def _prec(orig, derived):
        if not orig:
            return 1.0
        matched = 0
        for ts in orig:
            if any(abs((ts - d).days) <= 1 for d in derived):
                matched += 1
        return matched / len(orig)

    entry_prec = _prec(orig_e, derived_e)
    exit_prec = _prec(orig_x, derived_x)
    fidelity = entry_prec * 0.6 + exit_prec * 0.4

    return {
        "fidelity": round(fidelity, 4),
        "entry_precision": round(entry_prec, 4),
        "exit_precision": round(exit_prec, 4),
        "orig_entries": len(orig_e),
        "orig_exits": len(orig_x),
        "derived_entries": len(derived_e),
        "derived_exits": len(derived_x),
        "pass": fidelity >= 0.80,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for mod_name in STRATEGY_MODULES:
        print(f"\n=== {mod_name} ===")
        gen, params, desc = safe_load_strategy(mod_name)
        if gen is None:
            print(f"  SKIP")
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
                entries, exits = gen(df, params)
            except Exception as e:
                print(f"ERR {e}")
                ticker_results.append({"ticker": ticker, "status": "ERR", "reason": str(e)})
                all_passed = False
                continue

            bt_df = _make_bt_df(df)
            bt_df.index = pd.to_datetime(bt_df.index)
            replay_cls = _build_replay_class(entries, exits, mod_name)

            try:
                bt = Backtest(bt_df, replay_cls, cash=100000, commission=0.001)
                stats = bt.run()
            except Exception as e:
                print(f"BT_ERR {e}")
                ticker_results.append({"ticker": ticker, "status": "BT_ERR", "reason": str(e)})
                all_passed = False
                continue

            comp = compare_timing(entries, exits, stats)
            ticker_results.append({"ticker": ticker, **comp})
            if not comp["pass"]:
                all_passed = False
            detail = f"{comp['orig_entries']}E/{comp['orig_exits']}X vs {comp['derived_entries']}E/{comp['derived_exits']}X fid={comp['fidelity']}"
            print(f"{'PASS' if comp['pass'] else 'FAIL'} {detail}")

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

    report_path = OUTPUT_DIR / "cross_verify_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
