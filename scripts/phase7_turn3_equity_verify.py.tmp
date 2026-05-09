"""
phase7_turn3_equity_verify.py v7
Cross-engine verification via daily equity curve correlation.

1. Run CrabQuant generate_signals -> simple VBT-style equity curve.
2. Replay same signals in backtesting.py -> equity curve.
3. Compare daily equity correlation. >0.80 means signals replay correctly.
4. Flag strategies/tickers with divergence > 20%.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

CRAB_DIR = Path("~/development/CrabQuant").expanduser()
sys.path.insert(0, str(CRAB_DIR))

from crabquant.data import load_data

TICKERS = ["AAPL", "MSFT", "SPY", "QQQ", "GLD"]
DATA_PERIOD = "1y"
OUTPUT_DIR = Path("~/development/quant-research-mas/server/backtesting_strategies").expanduser()
MIN_CORR = 0.80
MAX_DIV = 0.20

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


def vbt_equity(gen, df, params, initial=100_000):
    entries, exits = gen(df, params)
    entries = entries.fillna(False)
    exits = exits.fillna(False)
    cash = float(initial)
    position = 0.0
    entry_price = 0.0
    equity = []
    for i in range(len(df)):
        price = float(df["close"].iloc[i])
        if entries.iloc[i] and position == 0:
            position = cash / price * 0.99
            cash = 0.0
            entry_price = price
        elif exits.iloc[i] and position > 0:
            cash = position * price * 0.999
            position = 0.0
        val = cash + position * price
        equity.append(val)
    eq = pd.Series(equity, index=df.index)
    return eq


def _build_replay(entries: pd.Series, exits: pd.Series, name: str):
    e = entries.fillna(False).astype(bool)
    x = exits.fillna(False).astype(bool)

    class ReplayStrategy(Strategy):
        def init(self): pass
        def next(self):
            idx = len(self.data) - 1
            if idx < 0:
                return
            ts = self.data.index[idx]
            if e.get(ts, False) and not self.position:
                self.buy()
            elif x.get(ts, False) and self.position:
                self.position.close()

    ReplayStrategy.__name__ = name
    return ReplayStrategy


def bt_equity(gen, df, params, initial=100_000):
    entries, exits = gen(df, params)
    bt_df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}).dropna()
    bt_df.index = pd.to_datetime(bt_df.index)
    cls = _build_replay(entries, exits, "Replay")
    bt = Backtest(bt_df, cls, cash=initial, commission=0.001)
    stats = bt.run()
    # Extract equity curve from stats
    equity = stats._equity_curve["Equity"] if hasattr(stats, "_equity_curve") and stats._equity_curve is not None else pd.Series(initial, index=bt_df.index)
    return equity


def compare_equity(vbt_eq, bt_eq):
    common = vbt_eq.index.intersection(bt_eq.index)
    v = vbt_eq.loc[common].astype(float).values
    b = bt_eq.loc[common].astype(float).values
    if len(v) < 10:
        return {"correlation": None, "divergence": None, "passed": False, "reason": "too_few"}
    v = np.nan_to_num(v, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    if np.std(v) < 1e-10 or np.std(b) < 1e-10:
        corr = 1.0 if np.allclose(v, b) else 0.0
    else:
        corr = float(np.corrcoef(v, b)[0, 1])
    divergence = abs(corr - 1.0)
    passed = corr >= MIN_CORR and divergence <= MAX_DIV
    return {"correlation": round(corr, 4), "divergence": round(divergence, 4), "passed": passed}


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
                vbt_eq = vbt_equity(gen, df, params)
                bt_eq = bt_equity(gen, df, params)
            except Exception as e:
                print(f"ERR {e}")
                ticker_results.append({"ticker": ticker, "status": "ERR", "reason": str(e)})
                all_passed = False
                continue

            comp = compare_equity(vbt_eq, bt_eq)
            ticker_results.append({"ticker": ticker, **comp})
            if not comp["passed"]:
                all_passed = False
            print(f"{'PASS' if comp['passed'] else 'FAIL'} corr={comp['correlation']} divergence={comp['divergence']}")

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

    report_path = OUTPUT_DIR / "phase7_turn3_equity_verify_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
