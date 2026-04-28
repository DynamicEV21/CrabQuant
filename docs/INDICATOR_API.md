# CrabQuant Indicator API Reference

> **Purpose**: This document is injected into LLM prompts as reference material.
> It contains the exact, tested API signatures for every pandas_ta indicator
> used in CrabQuant strategies. Follow these patterns exactly — do NOT guess.

---

## 1. Import

```python
from crabquant.indicator_cache import cached_indicator
```

`cached_indicator(name, *args, **kwargs)` is a drop-in wrapper for `pandas_ta.<name>()`.
It caches results so identical calls across param combos are free.

---

## 2. DataFrame Columns

The `df` passed to `generate_signals()` has these **lowercase** columns:
- `df["open"]`, `df["high"]`, `df["low"]`, `df["close"]`, `df["volume"]`

Extract them once at the top of your function:

```python
close = df["close"]
high = df["high"]
low = df["low"]
volume = df["volume"]
```

---

## 3. Indicator Signatures (EXACT — copy these)

### Single-output indicators (return pd.Series)

```python
# RSI — takes close only
rsi = cached_indicator("rsi", close, length=14)

# EMA — takes close only
ema = cached_indicator("ema", close, length=20)

# SMA — takes close or volume (any Series)
sma = cached_indicator("sma", close, length=50)
vol_avg = cached_indicator("sma", volume, length=20)

# ROC (Rate of Change) — takes close only
roc = cached_indicator("roc", close, length=10)
```

### Multi-input indicators (return pd.Series) — ⚠️ MOST COMMON ERROR SOURCE

```python
# ATR — REQUIRES high, low, close as THREE SEPARATE SERIES
# WRONG: cached_indicator("atr", close, length=14)    ← CRASHES
# WRONG: cached_indicator("atr", df, length=14)       ← CRASHES
# CORRECT:
atr = cached_indicator("atr", high, low, close, length=14)
```

### Multi-output indicators (return pd.DataFrame)

These return DataFrames with named columns. Use **`.iloc[:, N]`** for positional
access (immune to naming quirks) or **named column strings**.

```python
# MACD — returns DataFrame with columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
macd = cached_indicator("macd", close, fast=12, slow=26, signal=9)
# Positional access (preferred — immune to naming changes):
macd_line = macd.iloc[:, 0]   # MACD line
macd_hist = macd.iloc[:, 1]   # Histogram
macd_signal = macd.iloc[:, 2]  # Signal line
# Named access (works but fragile):
macd_hist = macd[f"MACDh_{fast}_{slow}_{signal}"]

# Bollinger Bands — returns DataFrame
# Columns (positional): 0=BBL(lower), 1=BBM(mid), 2=BBU(upper), 3=BBB(bandwidth), 4=BBP(percent)
bb = cached_indicator("bbands", close, length=20, std=2.0)
bb_lower = bb.iloc[:, 0]
bb_mid = bb.iloc[:, 1]
bb_upper = bb.iloc[:, 2]

# Stochastic — REQUIRES high, low, close as THREE SEPARATE SERIES
# WRONG: cached_indicator("stoch", close, k=14, d=3)  ← CRASHES
# CORRECT:
stoch = cached_indicator("stoch", high, low, close, k=14, d=3)
# Columns (positional): 0=STOCHk, 1=STOCHd, 2=STOCHh
stoch_k = stoch.iloc[:, 0]
stoch_d = stoch.iloc[:, 1]

# ADX — REQUIRES high, low, close as THREE SEPARATE SERIES
# WRONG: cached_indicator("adx", close, length=14)  ← CRASHES
# CORRECT:
adx_df = cached_indicator("adx", high, low, close, length=14)
# Columns include: ADX_14, DMP_14, DMN_14 (names change with length)
# Safest: find the ADX column dynamically:
adx_col = [c for c in adx_df.columns if "ADX" in c and "DI" not in c][0]
adx_val = adx_df[adx_col]
# Or use named column if you know the length:
adx_val = adx_df[f"ADX_{length}"]
```

### Other available indicators (not yet used in core strategies but valid)

```python
# Supertrend — REQUIRES high, low, close
supertrend = cached_indicator("supertrend", high, low, close, length=10, multiplier=3.0)
# Returns DataFrame: SUPERT_10_3.0, SUPERTd_10_3.0, SUPERTl_10_3.0

# Ichimoku — REQUIRES high, low, close
ich = cached_indicator("ichimoku", high, low, close, tenkan=9, kijun=26, senkou=52)

# OBV (On Balance Volume) — takes close and volume
obv = cached_indicator("obv", close, volume)

# VWAP — takes high, low, close, volume
vwap = cached_indicator("vwap", high, low, close, volume)

# CCI (Commodity Channel Index) — takes high, low, close
cci = cached_indicator("cci", high, low, close, length=20)

# Williams %R — takes high, low, close
willr = cached_indicator("willr", high, low, close, length=14)

# WMA (Weighted Moving Average) — takes close only
wma = cached_indicator("wma", close, length=20)
```

---

## 4. Common Mistakes & How to Avoid Them

| ❌ WRONG | ✅ CORRECT | Why |
|---|---|---|
| `cached_indicator("atr", close, length=14)` | `cached_indicator("atr", high, low, close, length=14)` | ATR needs HLC, not just close |
| `cached_indicator("stoch", close, k=14, d=3)` | `cached_indicator("stoch", high, low, close, k=14, d=3)` | Stochastic needs HLC |
| `cached_indicator("adx", close, length=14)` | `cached_indicator("adx", high, low, close, length=14)` | ADX needs HLC |
| `macd["histogram"]` | `macd.iloc[:, 1]` | Column names include params (e.g. `MACDh_12_26_9`) |
| `bb["upper"]` | `bb.iloc[:, 2]` | Column names are `BBU_20_2.0`, not `upper` |
| `df["Close"]` or `df["HIGH"]` | `df["close"]` or `df["high"]` | All column names are lowercase |
| `entries = ...` (no fillna) | `entries = (...).fillna(False)` | NaN in boolean Series causes errors |
| `close[50:]` (label slicing) | `close.iloc[50:]` | Use .iloc for position-based slicing |

---

## 5. Mandatory Patterns

### Always .fillna(False) on signal Series
```python
entries = (condition1 & condition2 & condition3).fillna(False)
exits = (exit_condition).fillna(False)
```

### Always use .iloc for positional slicing
```python
# If you need to skip warmup rows, use iloc:
atr.iloc[20:]  # ✅ positional
atr["2024-01-01":]  # ❌ label-based, fragile
```

### Always extract DataFrame columns from indicators
```python
# Multi-output indicators return DataFrames — extract what you need:
macd = cached_indicator("macd", close, fast=12, slow=26, signal=9)
hist = macd.iloc[:, 1]  # Histogram column

# Then use the extracted Series in conditions:
entries = (hist > hist.shift(1)).fillna(False)
```

---

## 6. Complete Working Example

```python
"""
Example strategy showing correct indicator usage patterns.
"""
import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi_len": 14,
    "atr_len": 14,
}

DESCRIPTION = "Example strategy with correct indicator API calls."

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Single-output indicators (close only)
    ema_fast = cached_indicator("ema", close, length=p["ema_fast"])
    ema_slow = cached_indicator("ema", close, length=p["ema_slow"])
    rsi = cached_indicator("rsi", close, length=p["rsi_len"])

    # Multi-input indicator (high, low, close — NOT just close!)
    atr = cached_indicator("atr", high, low, close, length=p["atr_len"])

    # Multi-output indicator — use .iloc for column access
    macd = cached_indicator("macd", close, fast=12, slow=26, signal=9)
    macd_hist = macd.iloc[:, 1]  # Histogram

    # Volume SMA
    vol_avg = cached_indicator("sma", volume, length=20)

    # Entry: EMA crossover + RSI not overbought + volume confirmation
    entries = (
        (ema_fast.shift(1) < ema_slow.shift(1))
        & (ema_fast > ema_slow)
        & (rsi < 70)
        & (volume > vol_avg)
    ).fillna(False)

    # Exit: MACD histogram drops below zero
    exits = (macd_hist < 0).fillna(False)

    return entries, exits
```

---

## 7. Quick Reference Card

| Indicator | Args | Returns | Key Columns |
|---|---|---|---|
| `rsi` | `(close, length=N)` | Series | — |
| `ema` | `(close, length=N)` | Series | — |
| `sma` | `(series, length=N)` | Series | — |
| `roc` | `(close, length=N)` | Series | — |
| `wma` | `(close, length=N)` | Series | — |
| `atr` | `(high, low, close, length=N)` | Series | — |
| `macd` | `(close, fast=N, slow=N, signal=N)` | DataFrame | `[:,0]`=line `[:,1]`=hist `[:,2]`=signal |
| `bbands` | `(close, length=N, std=N)` | DataFrame | `[:,0]`=lower `[:,1]`=mid `[:,2]`=upper |
| `stoch` | `(high, low, close, k=N, d=N)` | DataFrame | `[:,0]`=K `[:,1]`=D |
| `adx` | `(high, low, close, length=N)` | DataFrame | `ADX_N` col (find dynamically) |
| `supertrend` | `(high, low, close, length=N, mult=N)` | DataFrame | direction, trend, long, short |
| `obv` | `(close, volume)` | Series | — |
| `vwap` | `(high, low, close, volume)` | Series | — |
| `cci` | `(high, low, close, length=N)` | Series | — |
| `willr` | `(high, low, close, length=N)` | Series | — |
