# CrabQuant — Agent Context

**Project:** `/home/Zev/development/CrabQuant/`
**Branch:** `phase5.6-overnight` (active crons)

## Project Overview

Quantitative trading strategy research platform. Auto-discovers, validates, and ranks trading strategies using walk-forward optimization. Hybrid data loading: yfinance base + massive.com parquet overlay.

## Shared Financial Data

**Location:** `/home/Zev/development/quant-projects/financial-data/`

CrabQuant uses the **hybrid data strategy** (implemented in `crabquant/data/__init__.py`): yfinance for full history + massive.com parquet overlay for higher-quality data on overlapping dates.

### Data Config

```python
# crabquant/data/__init__.py
MASSIVE_DATA_DIR = "/home/Zev/development/quant-projects/financial-data/stocks/daily/"
# Override via env var: CRABQUANT_DATA_DIR
```

### Data Tier Awareness

| Tier | Directory | Quality | Use |
|------|-----------|---------|-----|
| **TIER 1** | `stocks/daily/` | Real VWAP, real n_trades | Daily overlay (1yr, 33 tickers) |
| **TIER 2** | yfinance (API) | Approx VWAP, no n_trades | Deep history base (configurable period) |

### Hybrid Loading (`load_data()`)

```python
from crabquant.data import load_data
# Hybrid: yfinance base + massive.com overlay
df = load_data("AAPL", period="2y")  # massive.com takes priority on overlap
```

### Download Scripts

```bash
# TIER 1: massive.com daily
POLYGON_API_KEY=$KEY python /home/Zev/development/quant-projects/financial-data/shared-scripts/scripts/download_ohlcv.py \
  --tickers AAPL,MSFT,... --bar-size day --output-dir stocks/daily/

# TIER 2: refresh yfinance cache (automatic via load_data with use_cache=False)
from crabquant.data import clear_cache
clear_cache()  # then next load_data() fetches fresh from yfinance
```

### Known Data Gaps

- `stocks/daily/` ends 2025-04-30 — needs refresh via download_ohlcv.py
- `stocks/ohlcv_1min/` — MAG7 has ~1yr; other tickers only ~2 weeks
- `stocks/yfinance_1h/` — only AAPL; needs expansion

See full inventory: `/home/Zev/development/quant-projects/financial-data/DATA.md`

## Key Files

- `crabquant/data/__init__.py` — hybrid data loader
- `VISION.md` — project north star
- `.hermes/plans/overnight-tasks.md` — active task queue
- `.hermes/plans/build-status.json` — current build status
