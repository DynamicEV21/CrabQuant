# CrabQuant Fix Priority List

**Generated:** 2026-04-28  
**Method:** Verification audit against AUDIT.md (2026-04-27)  
**Scope:** Only issues confirmed still present

---

## Verification Summary

| Audit Issue | Status | Notes |
|---|---|---|
| P1: invention.py stubbed with mock data | ✅ STILL PRESENT | Properly deprecated but still importable |
| P2: parallel.py syntax error | ✅ FIXED | Confirmed correct `import` syntax |
| P3: 5 of 22 strategies missing converters | ✅ FIXED | Now 23 converters for 22 strategies |
| P4: llm_api.py uses raw urllib | ✅ FIXED | Now uses `httpx` with retries/timeouts |
| P5: Pure-numpy rolling functions in converter | ✅ STILL PRESENT | ~228 lines of O(n×window) helpers |
| P6: No strategy retirement mechanism | ✅ STILL PRESENT | No retire/re-validate/remove functions |
| P7: discoveries.py parses openclaw output | NOT CHECKED | Requires runtime environment |
| P8: promoter.py hides missing slippage data | ✅ STILL PRESENT | Lines 262-268: dummy SlippageResults |
| P9: Hardcoded regime-strategy affinity | ✅ STILL PRESENT | Design issue, not a bug |
| P10: mandate_generator doesn't deduplicate | NOT CHECKED | Requires runtime verification |
| run.py tuple unpacking bug | ✅ STILL PRESENT | Lines 128 and 195: 4-tuple vs 5-tuple |
| requirements.txt incomplete | ✅ STILL PRESENT | Missing 5+ dependencies |
| pyproject.toml missing | ✅ STILL PRESENT | No pyproject.toml exists |
| Root debug scripts | ✅ FIXED | No debug_*.py files found at root |
| sys.path.insert boilerplate | ✅ STILL PRESENT | 24 occurrences across codebase |
| Hardcoded cache path | ✅ STILL PRESENT | `~/.cache/crabquant` not configurable |

---

## Fix Priority List

### P0 — CRITICAL: Will crash at runtime

#### 1. run.py tuple unpacking — 4-tuple vs 5-tuple
- **File:** `crabquant/run.py:128` and `crabquant/run.py:195`
- **Bug:** STRATEGY_REGISTRY stores 5-tuples `(fn, defaults, grid, desc, matrix_fn)` but run.py unpacks 4 values. This will raise `ValueError: too many values to unpack`.
- **Broken line 128:**
  ```python
  strategy_fn, defaults, param_grid, desc = STRATEGY_REGISTRY[strat_name]
  ```
- **Fix:**
  ```python
  strategy_fn, defaults, param_grid, desc, _matrix_fn = STRATEGY_REGISTRY[strat_name]
  ```
- **Broken line 195:**
  ```python
  strategy_fn, _, _, _ = STRATEGY_REGISTRY[strat_name]
  ```
- **Fix:**
  ```python
  strategy_fn, _, _, _, _ = STRATEGY_REGISTRY[strat_name]
  ```
- **Effort:** 5min
- **Safe:** Yes — purely additive, no logic change
- **Test impact:** None — run.py is a CLI runner, no unit tests cover this path

---

### P1 — HIGH: Misleading data or broken pipeline features

#### 2. promoter.py hides missing slippage data with dummy zeros
- **File:** `crabquant/production/promoter.py:262-268`
- **Bug:** When `_extract_slippage_results()` can't parse notes but verdict is ROBUST, it creates `SlippageResult(passed=True, sharpe=0, return=0, ...)` — looks like real data but is fabricated.
- **Broken code:**
  ```python
  if verdict == "ROBUST":
      return [
          SlippageResult(slippage_pct=0.0, passed=True, sharpe=0, total_return=0, max_drawdown=0, num_trades=0, win_rate=0),
          SlippageResult(slippage_pct=0.001, passed=True, sharpe=0, total_return=0, max_drawdown=0, num_trades=0, win_rate=0),
          SlippageResult(slippage_pct=0.002, passed=True, sharpe=0, total_return=0, max_drawdown=0, num_trades=0, win_rate=0),
      ]
  ```
- **Fix:** Return empty list and let the report show "slippage data unavailable" instead of fake zeros:
  ```python
  if verdict == "ROBUST":
      logger.warning("ROBUST verdict but no parseable slippage notes — returning empty slippage results")
      return []
  ```
- **Effort:** 15min
- **Safe:** Yes — downstream code already handles empty list (line 257-258)
- **Test impact:** Check `tests/test_production.py` — may need to update if tests assert on slippage count

#### 3. requirements.txt is severely incomplete
- **File:** `requirements.txt`
- **Bug:** Only lists 5 packages (`vectorbt`, `pandas`, `pandas_ta`, `numpy`, `yfinance`). Missing:
  - `httpx` (used by llm_api.py — will crash on import)
  - `backtesting` (used by strategy_converter.py — will crash on import)
  - `pytest` (test framework)
  - `python-dateutil` (likely needed by yfinance/pandas)
- **Fix:** Replace requirements.txt contents with:
  ```
  vectorbt>=0.26.0
  pandas>=2.0.0
  pandas_ta>=0.3.14b
  numpy>=1.24.0
  yfinance>=0.2.30
  httpx>=0.25.0
  backtesting>=0.3.3
  pytest>=7.0.0
  ```
- **Effort:** 5min
- **Safe:** Yes — only adds missing deps
- **Test impact:** None

---

### P2 — MEDIUM: Performance or maintainability issues

#### 4. strategy_converter.py re-implements rolling functions in pure numpy
- **File:** `crabquant/confirm/strategy_converter.py:23-246`
- **Bug:** 13 helper functions (`_rolling_max`, `_rolling_min`, `_rolling_mean`, `_rolling_sum`, `_ewm_mean`, `_rsi`, `_atr`, `_adx`, `_macd`, `_stoch`, `_bbands`, `_roc`, `_sma`) are O(n×window) pure-numpy loops. Batch confirm runs 9 backtests per strategy — this compounds.
- **Fix strategy:** Use `backtesting.lib` for available functions (SMA, EMA, RSI, ATR). For missing ones (ADX, Stoch, MACD, BBands, VPT, ROC), either use `talib` (already a pandas_ta dependency) or compute via pandas before passing to `self.I()`.
- **Quick win (1hr):** Replace `_sma`, `_ewm_mean`, `_rsi`, `_atr` with `backtesting.lib` equivalents. These are the most-called functions.
- **Full fix (2hr):** Replace all 13 functions.
- **Effort:** 1-2hr
- **Safe:** Requires careful testing — converter output must match exactly for regression tests
- **Test impact:** `tests/test_confirm.py` — all converter tests should pass if replacement is correct

#### 5. invention.py is stubbed with mock data
- **File:** `crabquant/invention.py:22-37`
- **Bug:** `analyze_market_data()` returns hardcoded ticker lists and regime classifications. Properly deprecated but still importable and referenced.
- **Note:** AUDIT says this is LOW priority since the refinement pipeline is the active path. The deprecation docstring is already present.
- **Fix:** Either:
  - (a) Wire `analyze_market_data()` to use `crabquant.regime.detect_regime()` + `crabquant.data.load_data()` — 2hr
  - (b) Make it raise `NotImplementedError` with clear redirect message — 5min
- **Effort:** 5min (option b) or 2hr (option a)
- **Safe:** Option b is safe. Option a requires testing.
- **Test impact:** Check if any test imports invention.py

#### 6. No strategy retirement mechanism
- **File:** `crabquant/production/` (no retirement code exists)
- **Bug:** Once promoted to `registry.json`, strategies stay forever. No re-validation, no expiry.
- **Fix:** Add `retire_strategy(key)` and `re_validate_all()` functions to `promoter.py`. Add `retired_at` field to registry entries.
- **Effort:** 1day (needs design + cron integration + brief updates)
- **Safe:** Additive only — doesn't change existing promotion logic
- **Test impact:** New tests needed

---

### P3 — LOW: Nice-to-have improvements

#### 7. No pyproject.toml
- **File:** N/A (missing)
- **Bug:** No modern Python packaging. Every script needs `sys.path.insert(0, ...)` boilerplate (24 occurrences).
- **Fix:** Create `pyproject.toml` with project metadata and all dependencies. Run `pip install -e .`.
- **Effort:** 15min for pyproject.toml, 1hr to remove all sys.path hacks
- **Safe:** Yes
- **Test impact:** Test imports may need adjustment

#### 8. Hardcoded cache path not configurable
- **File:** `crabquant/data/__init__.py:21`
- **Bug:** `CACHE_DIR = Path(os.path.expanduser("~/.cache/crabquant"))` — not configurable via env var or config.
- **Fix:**
  ```python
  CACHE_DIR = Path(os.environ.get("CRABQUANT_CACHE_DIR", 
                                  os.path.expanduser("~/.cache/crabquant")))
  ```
- **Effort:** 5min
- **Safe:** Yes — backward compatible
- **Test impact:** None

#### 9. invention.py test_strategy_code checks for generate_signals_matrix
- **File:** `crabquant/invention.py:100`
- **Bug:** `required_functions = ['generate_signals', 'generate_signals_matrix']` — but strategies now consistently have `generate_signals_matrix` (it's in the registry), so this may no longer be an issue. Still, the function itself is deprecated.
- **Fix:** If keeping invention.py, update to only check `generate_signals`. If deprecating (option 5b), this is moot.
- **Effort:** 5min
- **Safe:** Yes
- **Test impact:** None

#### 10. sys.path.insert boilerplate in scripts
- **Files:** 11 scripts + 4 test files = 15 occurrences (excluding test conftest)
- **Bug:** Every script manually adds project root to sys.path. Fragile, error-prone.
- **Fix:** After creating pyproject.toml (issue #7) and running `pip install -e .`, remove all `sys.path.insert(0, ...)` lines from scripts.
- **Effort:** 1hr
- **Safe:** Only after pyproject.toml is in place
- **Test impact:** Tests that use sys.path for scripts/ dir may need adjustment

---

## Quick Wins (< 15min total)

These can be done right now with zero risk:

1. **Fix run.py tuple unpacking** (5min) — P0 crash bug
2. **Fix promoter.py dummy slippage** (15min) — P1 data integrity
3. **Fix requirements.txt** (5min) — P1 missing deps
4. **Make cache path configurable** (5min) — P3 nicety
5. **Add pyproject.toml** (15min) — P3 foundation

**Total quick-win effort: ~45 minutes**
