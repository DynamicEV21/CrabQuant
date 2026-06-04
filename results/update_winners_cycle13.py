"""
Add batch_cycle13 promoted entries to winners.json with proper JSON serialization.
"""
import json
from datetime import datetime, timezone

winners_path = "results/winners/winners.json"
winners = json.loads(open(winners_path).read())
print(f"Current winners: {len(winners)}")

# New promoted entries from Cycle 13 sweep
new_entries = [
    {"strategy": "adx_pullback_gld", "ticker": "GLD", "sharpe": 1.648, "return": 0, "max_drawdown": 0, "trades": 0,
     "params": {"adx_len": 14, "adx_threshold": 25, "atr_len": 14, "atr_mult": 2.0, "pullback_lookback": 5},
     "refinement_run": "batch_cycle13", "refinement_turns": 0,
     "validation": {"passed": True, "walk_forward_robust": True, "cross_ticker_robust": True,
                    "validation_method": "rolling", "sweep_wf_avg_test_sharpe": 1.648, "sweep_wf_windows": "5/6",
                    "sweep_ct_avg_sharpe": 0.966},
     "promoted_at": datetime.now(timezone.utc).isoformat(), "validation_status": "promoted", "regime_tags": None},
    {"strategy": "ema_crossover_gld", "ticker": "GLD", "sharpe": 1.836, "return": 0, "max_drawdown": 0, "trades": 0,
     "params": {"fast": 12, "slow": 26, "signal": 9},
     "refinement_run": "batch_cycle13", "refinement_turns": 0,
     "validation": {"passed": True, "walk_forward_robust": True, "cross_ticker_robust": True,
                    "validation_method": "rolling", "sweep_wf_avg_test_sharpe": 1.836, "sweep_wf_windows": "6/6",
                    "sweep_ct_avg_sharpe": 1.061},
     "promoted_at": datetime.now(timezone.utc).isoformat(), "validation_status": "promoted", "regime_tags": None},
    {"strategy": "informed_simple_adaptive_spy", "ticker": "SPY", "sharpe": 1.756, "return": 0, "max_drawdown": 0, "trades": 0,
     "params": {"adx_len": 14, "adx_threshold": 25, "rsi_len": 14, "rsi_oversold": 35, "rsi_overbought": 65, "volume_window": 20, "volume_mult": 1.3},
     "refinement_run": "batch_cycle13", "refinement_turns": 0,
     "validation": {"passed": True, "walk_forward_robust": True, "cross_ticker_robust": True,
                    "validation_method": "rolling", "sweep_wf_avg_test_sharpe": 1.756, "sweep_wf_windows": "4/6",
                    "sweep_ct_avg_sharpe": 0.415},
     "promoted_at": datetime.now(timezone.utc).isoformat(), "validation_status": "promoted",
     "regime_tags": {"preferred_regimes": ["trending_down"], "is_regime_specific": True}},
    {"strategy": "injected_momentum_atr_volume_gld", "ticker": "GLD", "sharpe": 2.166, "return": 0, "max_drawdown": 0, "trades": 0,
     "params": {"atr_len": 14, "ema_fast": 9, "ema_slow": 21, "volume_threshold": 1.5},
     "refinement_run": "batch_cycle13", "refinement_turns": 0,
     "validation": {"passed": True, "walk_forward_robust": True, "cross_ticker_robust": True,
                    "validation_method": "rolling", "sweep_wf_avg_test_sharpe": 2.166, "sweep_wf_windows": "6/6",
                    "sweep_ct_avg_sharpe": 0.939},
     "promoted_at": datetime.now(timezone.utc).isoformat(), "validation_status": "promoted", "regime_tags": None},
    {"strategy": "invented_momentum_confluence_nvda", "ticker": "NVDA", "sharpe": 2.121, "return": 0, "max_drawdown": 0, "trades": 0,
     "params": {"atr_len": 14, "ema_len": 20, "rsi_len": 14, "volume_window": 20},
     "refinement_run": "batch_cycle13", "refinement_turns": 0,
     "validation": {"passed": True, "walk_forward_robust": True, "cross_ticker_robust": True,
                    "validation_method": "rolling", "sweep_wf_avg_test_sharpe": 2.121, "sweep_wf_windows": "5/6",
                    "sweep_ct_avg_sharpe": 1.051},
     "promoted_at": datetime.now(timezone.utc).isoformat(), "validation_status": "promoted", "regime_tags": None},
    {"strategy": "invented_momentum_rsi_stoch_de", "ticker": "DE", "sharpe": 1.390, "return": 0, "max_drawdown": 0, "trades": 0,
     "params": {"momentum_len": 10, "rsi_len": 14, "stoch_k": 14, "stoch_d": 3},
     "refinement_run": "batch_cycle13", "refinement_turns": 0,
     "validation": {"passed": True, "walk_forward_robust": True, "cross_ticker_robust": True,
                    "validation_method": "rolling", "sweep_wf_avg_test_sharpe": 1.390, "sweep_wf_windows": "5/6",
                    "sweep_ct_avg_sharpe": 0.748},
     "promoted_at": datetime.now(timezone.utc).isoformat(), "validation_status": "promoted", "regime_tags": None},
    {"strategy": "invented_volume_adx_ema_nvda", "ticker": "NVDA", "sharpe": 0.836, "return": 0, "max_drawdown": 0, "trades": 0,
     "params": {"adx_len": 14, "ema_len": 20, "volume_sma": 20},
     "refinement_run": "batch_cycle13", "refinement_turns": 0,
     "validation": {"passed": True, "walk_forward_robust": True, "cross_ticker_robust": True,
                    "validation_method": "rolling", "sweep_wf_avg_test_sharpe": 0.836, "sweep_wf_windows": "5/6",
                    "sweep_ct_avg_sharpe": 0.755},
     "promoted_at": datetime.now(timezone.utc).isoformat(), "validation_status": "promoted", "regime_tags": None},
    {"strategy": "invented_volume_breakout_adx_cat", "ticker": "CAT", "sharpe": 1.363, "return": 0, "max_drawdown": 0, "trades": 0,
     "params": {"adx_threshold": 25, "atr_len": 14, "atr_mult": 2.0, "volume_mult": 1.5},
     "refinement_run": "batch_cycle13", "refinement_turns": 0,
     "validation": {"passed": True, "walk_forward_robust": True, "cross_ticker_robust": True,
                    "validation_method": "rolling", "sweep_wf_avg_test_sharpe": 1.363, "sweep_wf_windows": "5/6",
                    "sweep_ct_avg_sharpe": 0.716},
     "promoted_at": datetime.now(timezone.utc).isoformat(), "validation_status": "promoted", "regime_tags": None},
    {"strategy": "invented_volume_roc_atr_trend_jpm", "ticker": "JPM", "sharpe": 2.339, "return": 0, "max_drawdown": 0, "trades": 0,
     "params": {"atr_len": 14, "ema_len": 20, "roc_len": 10, "vol_sma_len": 20},
     "refinement_run": "batch_cycle13", "refinement_turns": 0,
     "validation": {"passed": True, "walk_forward_robust": True, "cross_ticker_robust": True,
                    "validation_method": "rolling", "sweep_wf_avg_test_sharpe": 2.339, "sweep_wf_windows": "5/6",
                    "sweep_ct_avg_sharpe": 1.167},
     "promoted_at": datetime.now(timezone.utc).isoformat(), "validation_status": "promoted", "regime_tags": None},
    {"strategy": "invented_vpt_roc_ema_gld", "ticker": "GLD", "sharpe": 1.339, "return": 0, "max_drawdown": 0, "trades": 0,
     "params": {"ema_len": 20, "roc_len": 10, "vpt_len": 20},
     "refinement_run": "batch_cycle13", "refinement_turns": 0,
     "validation": {"passed": True, "walk_forward_robust": True, "cross_ticker_robust": True,
                    "validation_method": "rolling", "sweep_wf_avg_test_sharpe": 1.339, "sweep_wf_windows": "5/6",
                    "sweep_ct_avg_sharpe": 0.861},
     "promoted_at": datetime.now(timezone.utc).isoformat(), "validation_status": "promoted", "regime_tags": None},
    {"strategy": "roc_ema_volume_cat", "ticker": "CAT", "sharpe": 1.626, "return": 0, "max_drawdown": 0, "trades": 0,
     "params": {"atr_len": 14, "atr_mult": 2.0, "ema_len": 20, "roc_len": 10, "trailing_len": 20, "vol_sma_len": 20},
     "refinement_run": "batch_cycle13", "refinement_turns": 0,
     "validation": {"passed": True, "walk_forward_robust": True, "cross_ticker_robust": True,
                    "validation_method": "rolling", "sweep_wf_avg_test_sharpe": 1.626, "sweep_wf_windows": "5/6",
                    "sweep_ct_avg_sharpe": 0.732},
     "promoted_at": datetime.now(timezone.utc).isoformat(), "validation_status": "promoted", "regime_tags": None},
]

# Add existing promoted entries from earlier cycles (4 already promoted)
for w in winners:
    if w.get("validation_status") == "promoted":
        print(f"  Already promoted: {w.get('strategy')}")

# Mark the 4 previously promoted entries
for w in winners:
    if w.get("strategy") in ("refined_roc_ema_volume_googl", "roc_ema_volume_googl",
                              "roc_ema_volume_googl_v2", "refined_e2e_test_momentum"):
        w["validation_status"] = w.get("validation_status", "promoted")

# Append new entries
winners.extend(new_entries)

# Count promoted
promoted_count = sum(1 for w in winners if w.get("validation_status") == "promoted")
print(f"\nTotal winners after update: {len(winners)}")
print(f"Total promoted: {promoted_count}")

# Write
open(winners_path, "w").write(json.dumps(winners, indent=2))
print(f"Written to {winners_path}")
