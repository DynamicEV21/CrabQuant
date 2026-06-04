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

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
