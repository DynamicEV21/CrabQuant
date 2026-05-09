# PRD: Tournament Mode + Regime Detection (Quant Factory Integration)

> **Source:** AgentQuant (https://github.com/OnePunchMonk/AgentQuant) — regime detection concepts
> **Integration target:** Quant Factory (`quant-research-mas`) — TypeScript pipeline + React dashboard
> **Phase:** 8.1 (inserted into Phase 8 infinite loop priorities)
> **Status:** 🔨 Planned — awaiting implementation
> **Created:** 2026-05-04
> **Last reviewed:** 2026-05-04 (v2 — architectural pivot to QF-integrated tournament)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [Architecture](#3-architecture)
4. [Tournament Runner](#4-tournament-runner-servertournamentrunner-ts)
5. [LLM Semaphore](#5-llm-semaphore-serverllmllmsemaphore-ts)
6. [Regime Detector — Outline Only](#6-regime-detector--outline-only)
7. [Regime Archetype Map](#7-regime-archetype-map-serverregimearchetypemapts-or-yaml)
8. [Tournament Config](#8-tournament-config)
9. [API Endpoints](#9-api-endpoints)
10. [Dashboard Tournament Tab](#10-dashboard-tournament-tab)
11. [Data Requirements](#11-data-requirements)
12. [Testing Strategy](#12-testing-strategy)
13. [What This Does NOT Replace](#13-what-this-does-not-replace)
14. [Implementation Phases](#14-implementation-phases)
15. [Risk Register & Mitigations](#15-risk-register--mitigations)
16. [Acceptance Criteria](#16-acceptance-criteria)
17. [Changelog](#17-changelog)

---

## 1. Executive Summary

This PRD replaces the original "Regime Advisor Loop" concept with a **Quant Factory-integrated Tournament Mode**. Instead of a separate CrabQuant Python loop, this system adds two new capabilities directly into the existing QF pipeline:

1. **Tournament Mode** — A new `/api/tournament` endpoint that runs N style pack archetypes through the full QF pipeline in parallel, ranks them by score, and auto-promotes the winner. This answers the question: *"Given my hypothesis, which style pack archetype performs best right now?"*

2. **Regime Detection** (outline only) — An optional early pipeline stage that classifies the current market regime (VIX percentile + momentum + SMA trend) and uses that to select which archetypes enter the tournament. This is off by default and is a code structure + placeholder implementation only; actual VIX fetching and classification comes later.

**LLM-Gated Parallel Execution:** The tournament runs N pipeline instances concurrently, but gates all LLM calls through a semaphore (default: max 3 concurrent). Non-LLM stages (backtest, validation, overfit detection) run fully parallel with no semaphore. This prevents API rate limit violations while maximizing CPU utilization.

**What gets built (Tier 1, ~5 hrs):**
- `server/llm/llmSemaphore.ts` — LLM concurrency limiter (~30 lines)
- `server/tournamentRunner.ts` — Tournament coordinator (~80 lines)
- `/api/tournament` endpoint — ~30 lines
- Config toggles in `.env` / `runConfig`
- Sequential mode only (parallel=false default initially)

**What does NOT get built (this revision):**
- No Python code — everything is TypeScript in Quant Factory
- No `loops/regime-advisor/` directory
- No `program.md` loop document
- No HMM regime detection (deferred to later phase)
- No standalone backtest engine — reuses existing QF pipeline
- No Gas City / HermGC integration

---

## 2. Background & Motivation

### 2.1 The Problem

Quant Factory currently runs one experiment at a time. Each experiment uses a single style pack (e.g., `momentum_ignition`, `vwap_reversion`) chosen by the user. This means:

- **No archetype comparison:** You can't answer "would this hypothesis work better as a breakout or a mean reversion?" without manually running N experiments.
- **No market awareness:** The pipeline operates identically whether VIX is at 12 or 45. A momentum strategy hypothesis may produce very different results depending on the current regime.
- **Wasted LLM budget:** Running 5 sequential experiments to compare archetypes takes 5× the wall-clock time, even though 3 of the 5 backtests could run simultaneously on CPU.

### 2.2 AgentQuant's Best Ideas

AgentQuant introduces several concepts we port into QF:

1. **Percentile-based regime detection** — VIX percentile relative to trailing 252 days (not absolute thresholds)
2. **Multi-signal regime labels** — compound labels like "Crisis-Bear", "LowVol-Bull", "MidVol-Neutral"
3. **Proposal tournaments** — generate N competing proposals, ranked by expected performance
4. **Regime-aware archetype selection** — map detected regime to best-fit style packs

We port these concepts but implement them natively in the QF TypeScript stack, reusing the existing `runExperiment()` function, style pack system, and SQLite database.

### 2.3 Why Integrate Into QF (Not a Separate Loop)

| Aspect | Separate CrabQuant Loop | QF Tournament Integration |
|--------|------------------------|--------------------------|
| Language | Python | TypeScript (matches QF) |
| Backtest engine | Need to duplicate | Reuses existing `runExperiment()` |
| Style packs | Need to mirror | Uses existing `stylePacks.ts` (10+ packs) |
| Database | New TSV files | Uses existing `factory.db` SQLite |
| Dashboard | Separate API bridge | Same Express server |
| LLM calls | Separate provider | Shares existing `LLMProvider` |
| Config | `params.yaml` | `.env` + `runConfig` |

**Verdict:** Integrate directly into QF. Reuse everything, build only the tournament orchestration layer.

---

## 3. Architecture

### 3.1 Tournament Flow

```
                    ┌─────────────────────────────────┐
                    │       POST /api/tournament       │
                    │   { prompt, mode, assetClass }   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     1. REGIME DETECT (optional)  │
                    │     regime_detection.enabled=false│
                    │     → skipped, uses all archetypes│
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     2. ARCHETYPE SELECT          │
                    │     Map regime → stylePackIds     │
                    │     OR use all (up to N)          │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
     ┌────────▼────────┐  ┌───────▼───────┐  ┌────────▼────────┐
     │  Archetype A    │  │ Archetype B   │  │  Archetype C    │
     │  stylePackId=X  │  │ stylePackId=Y │  │  stylePackId=Z  │
     └────────┬────────┘  └───────┬───────┘  └────────┬────────┘
              │                    │                     │
     ┌────────▼────────┐  ┌───────▼───────┐  ┌────────▼────────┐
     │ runExperiment() │  │ runExperiment()│  │ runExperiment() │
     │                 │  │               │  │                 │
     │ LLM stages ─────┼──┼── SEMAPHORE ──┼──┼─ max 3 concur. │
     │ CPU stages ─────┼──┼── FULL PARALLEL            │
     └────────┬────────┘  └───────┬───────┘  └────────┬────────┘
              │                    │                     │
              └────────────────────┼────────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     3. RANK & PROMOTE           │
                    │     Sort by score, top N above  │
                    │     promotion_threshold → PROMOTED│
                    │     Winner fed to Ralph Loop     │
                    └─────────────────────────────────┘
```

### 3.2 LLM Semaphore vs CPU-Parallel

```
                    LLM SEMAPHORE (max 3 concurrent)
                    ┌─────────────────────────────┐
                    │  Slot 1: Arch A — playbook  │
                    │  Slot 2: Arch B — playbook  │
                    │  Slot 3: Arch C — playbook  │
                    │  Queue:  Arch A — setup (waiting) │
                    └─────────────────────────────┘

                    CPU PARALLEL (unlimited)
                    ┌─────────────────────────────┐
                    │  Arch A — backtest          │
                    │  Arch B — backtest          │
                    │  Arch C — validation        │
                    │  Arch D — overfit           │
                    └─────────────────────────────┘
```

**LLM stages** (gated by semaphore): `playbook`, `setup`, `trade_plan`, `execution`, `implementation`, `validation`, `overfit`, `rank`

**Non-LLM stages** (fully parallel): `backtest`

Note: In the current QF architecture, `validation`, `overfit`, and `rank` are LLM-powered stages (they call `generateJson`/`generateText`). Only `backtest` is CPU-only (calls the Python adapter). The semaphore wraps the LLM provider, so any stage that makes an LLM call automatically acquires a semaphore slot.

### 3.3 Config Toggles

```typescript
// From .env or runConfig
tournament.enabled: true                    // master switch
tournament.parallel: false                  // false = sequential (default for Tier 1)
tournament.max_concurrent_llm: 3           // semaphore limit
tournament.archetypes_per_tournament: 5    // how many style packs to test
tournament.promotion_threshold: 70         // score threshold for auto-promotion

regime_detection.enabled: false             // off by default, outline only
regime_detection.vix_data_source: "yfinance"
regime_detection.vix_refresh_cron: "0 6 * * *"
regime_detection.min_data_points: 60
```

### 3.4 Sequential vs Parallel Mode

**Sequential mode** (`tournament.parallel: false`, default in Tier 1):
- Runs archetypes one at a time through the full pipeline
- No semaphore needed (only 1 pipeline active at a time)
- Simpler debugging, predictable resource usage
- Still uses existing `runExperiment()` per archetype

**Parallel mode** (`tournament.parallel: true`, Tier 2):
- Spawns all archetype pipelines concurrently via `Promise.all()`
- LLM calls gated by semaphore, backtests run freely
- Faster wall-clock time but requires parallel safety (no SQLite write collisions)
- Each `runExperiment()` call gets its own `expId` and writes to separate rows

---

## 4. Tournament Runner

**File:** `server/tournamentRunner.ts` (~80 lines)

### 4.1 Interface

```typescript
// ── Types ───────────────────────────────────────────────────────────────────

export interface TournamentConfig {
  prompt: string;
  mode: string;
  assetClass?: string;
  stylePackIds?: string[];       // explicit list (overrides regime selection)
  maxArchetypes?: number;        // cap on archetypes to run
  promotionThreshold?: number;   // score >= this → PROMOTED
  parallel?: boolean;            // false = sequential
  maxConcurrentLlm?: number;     // semaphore limit
}

export interface ArchetypeResult {
  experimentId: string;
  stylePackId: string;
  stylePackName: string;
  score: number;
  state: 'PROMOTED' | 'RANKED' | 'KILLED' | 'FAILED';
  error?: string;
  iterationCount: number;
  stopReason?: string;
  completedAt: number;
}

export interface TournamentResult {
  tournamentId: string;
  status: 'RUNNING' | 'COMPLETED' | 'CANCELLED' | 'FAILED';
  config: TournamentConfig;
  regime?: RegimeClassification;      // if regime detection was enabled
  results: ArchetypeResult[];
  winner?: ArchetypeResult;
  startedAt: number;
  completedAt?: number;
  rankedByScore: ArchetypeResult[];   // sorted descending
}
```

### 4.2 Core Function

```typescript
/**
 * Run a tournament: N archetypes through the pipeline, ranked by score.
 *
 * @param config - Tournament configuration
 * @param onProgress - Optional callback for SSE broadcasting
 * @returns TournamentResult with all archetype results, ranked
 */
export async function runTournament(
  config: TournamentConfig,
  onProgress?: (event: TournamentProgressEvent) => void,
): Promise<TournamentResult>;
```

### 4.3 Implementation Logic

```
1. Generate tournamentId: "TNM-XXXXXXXX"
2. Insert tournament row into tournaments table (status: RUNNING)
3. Broadcast SSE: { type: 'tournament_start', tournamentId }

4. REGIME SELECT (if regime_detection.enabled):
   a. Call detectRegime() → RegimeClassification
   b. Call selectArchetypes(regime) → stylePackIds[]
   c. Broadcast SSE: { type: 'tournament_regime', regime, archetypes }
   ELSE:
   a. Use config.stylePackIds OR getAllStylePacks(config.assetClass)
   b. Slice to config.maxArchetypes (default: 5)

5. FOR EACH archetype (parallel or sequential based on config.parallel):
   a. Enrich prompt with regime context (if available):
      - Append regime summary to prompt string
      - e.g., "Current regime: MidVol-Bull (VIX 42nd %ile, +6.3% 63d momentum)"
   b. Call runExperiment(enrichedPrompt, stylePackId, mode, runConfig, 1, false, assetClass)
   c. On completion, extract final score and state from experiments table
   d. Broadcast SSE: { type: 'tournament_archetype_complete', result }
   e. Store ArchetypeResult

6. RANK results:
   a. Sort by score descending
   b. Mark score >= promotionThreshold as PROMOTED
   c. Select top result as winner

7. Update tournament row: status: COMPLETED, results JSON, winner
8. Broadcast SSE: { type: 'tournament_complete', tournamentResult }
9. Return TournamentResult
```

### 4.4 Parallel Execution Detail

When `tournament.parallel: true`:

```typescript
// Simplified — actual implementation uses the LLM semaphore
const promises = stylePackIds.map(async (packId) => {
  const enrichedPrompt = regimeContext
    ? `${config.prompt}\n\n[REGIME CONTEXT: ${regimeContext.toPromptString()}]`
    : config.prompt;
  return runExperiment(enrichedPrompt, packId, config.mode, runConfig, 1, false, assetClass);
});

// In parallel mode, runExperiment calls share the LLM semaphore
// The semaphore is injected into the LLMProvider, not into runExperiment
const results = await Promise.allSettled(promises);
```

**Key design decision:** The semaphore wraps the `LLMProvider`, not `runExperiment()`. This means existing single-experiment runs are unaffected — the semaphore only activates when multiple `runExperiment()` calls are concurrent (i.e., during a tournament). When `isRunning === true` for a single experiment, the semaphore has 1 concurrent slot and behaves identically to no semaphore.

### 4.5 Sequential Fallback

When `tournament.parallel: false`:

```typescript
const results: ArchetypeResult[] = [];
for (const packId of stylePackIds) {
  const enrichedPrompt = regimeContext
    ? `${config.prompt}\n\n[REGIME CONTEXT: ${regimeContext.toPromptString()}]`
    : config.prompt;
  try {
    await runExperiment(enrichedPrompt, packId, config.mode, runConfig, 1, false, assetClass);
    // Query experiments table for latest result
    results.push(extractResult(packId));
  } catch (err) {
    results.push({ stylePackId: packId, state: 'FAILED', error: String(err) });
  }
}
```

### 4.6 Database Schema

Add to `server/db.ts` migrations:

```sql
CREATE TABLE IF NOT EXISTS tournaments (
  id TEXT PRIMARY KEY,
  created_at INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'RUNNING',
  config_json TEXT NOT NULL,
  regime_json TEXT,
  results_json TEXT,
  winner_id TEXT,
  completed_at INTEGER
);

CREATE INDEX IF NOT EXISTS idx_tournaments_status ON tournaments(status);
CREATE INDEX IF NOT EXISTS idx_tournaments_created ON tournaments(created_at);
```

---

## 5. LLM Semaphore

**File:** `server/llm/llmSemaphore.ts` (~30 lines)

### 5.1 Purpose

Wraps all LLM calls with a max-concurrent limit to prevent API rate limit violations during tournament parallel execution. When a single experiment runs, the semaphore is a no-op (1 concurrent slot, never blocks). During tournaments, it queues excess LLM calls until a slot opens.

### 5.2 Interface

```typescript
/**
 * Creates a semaphore-gated wrapper around an LLM provider.
 * Only active when concurrent calls exceed maxConcurrent.
 *
 * @param provider - The underlying LLMProvider to wrap
 * @param maxConcurrent - Maximum concurrent LLM calls (default: 3)
 * @returns A new LLMProvider that gates calls through the semaphore
 */
export function withLLMSemaphore(
  provider: LLMProvider,
  maxConcurrent: number = 3,
): LLMProvider;
```

### 5.3 Implementation

```typescript
import { LLMProvider, LLMCallOptions } from './index';
import { z } from 'zod';

export function withLLMSemaphore(
  provider: LLMProvider,
  maxConcurrent: number = 3,
): LLMProvider {
  let active = 0;
  const queue: Array<{ resolve: () => void; timer: NodeJS.Timeout }> = [];

  async function acquire(timeoutMs: number = 60_000): Promise<void> {
    if (active < maxConcurrent) {
      active++;
      return;
    }
    // Queue the call with a timeout to prevent deadlocks
    return new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        // Remove from queue on timeout
        const idx = queue.findIndex(q => q.resolve === resolve);
        if (idx !== -1) queue.splice(idx, 1);
        reject(new Error(`LLM semaphore: queued call timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      queue.push({ resolve, timer });
    });
  }

  function release(): void {
    active--;
    if (queue.length > 0) {
      const next = queue.shift()!;
      clearTimeout(next.timer);
      active++;
      next.resolve();
    }
  }

  return {
    async generateJson<T>(role: string, schema: z.ZodType<T>, prompt: string, options?: LLMCallOptions): Promise<T> {
      await acquire();
      try {
        return await provider.generateJson(role, schema, prompt, options);
      } finally {
        release();
      }
    },
    async generateText(role: string, prompt: string, options?: LLMCallOptions): Promise<string> {
      await acquire();
      try {
        return await provider.generateText(role, prompt, options);
      } finally {
        release();
      }
    },
  };
}
```

### 5.4 Key Properties

- **Queue-based:** 4th+ concurrent calls wait until a slot opens. First-in-first-out.
- **Deadlock prevention:** Every queued call has a 60-second timeout. If it expires, the call is removed from the queue and throws an error. This ensures the system never hangs.
- **No-op for single experiments:** When only 1 experiment runs, `active` never exceeds 1, so `acquire()` returns immediately.
- **Configurable:** `maxConcurrent` is read from `tournament.max_concurrent_llm` in config (default: 3).
- **Logging:** When a call is queued, log at `debug` level: `"LLM semaphore: queued (active: 3, queued: 1)"`. When it starts executing: `"LLM semaphore: acquired slot (active: 3, queued: 0)"`.
- **Rate limit awareness:** The existing `SqliteQuotaScheduler` in `server/llm/sqliteQuotaScheduler.ts` already handles 429 backoff with exponential delays. The semaphore is a complementary layer — it prevents *sending* too many concurrent requests, while `SqliteQuotaScheduler` handles *rate limiting* between requests. They compose: semaphore gates concurrency, scheduler gates rate.

### 5.5 Integration Point

In `server.ts` or a provider factory, wrap the LLM provider at startup:

```typescript
// server.ts (or server/llm/providerFactory.ts)
import { withLLMSemaphore } from './server/llm/llmSemaphore';

const baseProvider = createLLMProvider(); // existing provider creation
const maxConcurrent = parseInt(process.env.TOURNAMENT_MAX_CONCURRENT_LLM || '3', 10);
export const llmProvider = withLLMSemaphore(baseProvider, maxConcurrent);
```

---

## 6. Regime Detector — Outline Only

> ⚠️ **This section is an OUTLINE.** The code structure and interfaces go in now (Tier 3). The actual VIX fetching and classification logic is a later phase. The regime detector starts as a placeholder that returns `Unknown` / neutral.

**File:** `server/regimeDetector.ts` (~50 lines, mostly interfaces + placeholder)

### 6.1 Interface

```typescript
// ── Types ───────────────────────────────────────────────────────────────────

export type VolRegime = 'low' | 'mid' | 'high' | 'crisis';
export type TrendLabel = 'Bull' | 'Bear' | 'Neutral';

export interface RegimeClassification {
  regime_label: string;         // e.g. "MidVol-Bull"
  confidence: number;           // 0.0-1.0
  vol_regime: VolRegime;
  trend_label: TrendLabel;
  vix_level: number;
  vix_percentile: number;       // 0-100
  momentum_63d: number;         // decimal, e.g. 0.063 = 6.3%
  above_200sma: boolean;
  above_50sma: boolean;
  detected_at: number;          // unix ms
  data_points: number;          // how many VIX data points were used
}

// ── Main Function ───────────────────────────────────────────────────────────

/**
 * Classify the current market regime.
 *
 * When regime_detection.enabled is false (default), this returns a neutral
 * placeholder. When enabled, it fetches VIX data from yfinance, computes
 * percentile, and classifies.
 *
 * @returns RegimeClassification
 */
export async function detectRegime(): Promise<RegimeClassification>;
```

### 6.2 Classification Logic (Placeholder — Implement Later)

When `regime_detection.enabled: true`, the actual classification logic:

1. **Fetch VIX data** from yfinance: `^VIX` ticker, daily close, trailing 252+ days
2. **Save to `data/vix_daily.parquet`** in the QF project directory
3. **Compute VIX percentile:** `percentileofscore(vix_series, latest_vix)`
4. **Map percentile → vol regime:**
   - `>85` → `crisis`
   - `>65` → `high`
   - `>35` → `mid`
   - `≤35` → `low`
5. **Fetch SPY data** from yfinance: `SPY` ticker, daily, trailing 252+ days (already exists in financial-data/)
6. **Compute 63-day momentum:** `(close_today / close_63d_ago) - 1`
7. **Map momentum → trend:**
   - `>5%` → `Bull`
   - `<-5%` → `Bear`
   - else → `Neutral`
8. **SMA trend signals:**
   - `above_200sma`: `close > SMA(200)`
   - `above_50sma`: `close > SMA(50)`
9. **Combine:** `regime_label = "{vol_regime}-{trend_label}"` (e.g., "MidVol-Bull")
10. **Confidence:** `(vol_confidence + mom_confidence) / 2`
    - `vol_confidence = 2 * |vix_percentile/100 - 0.5|`
    - `mom_confidence = min(|momentum_63d| / 0.10, 1.0)`

### 6.3 Data Source: yfinance

- **Package:** `yfinance` (Python) — already available in CrabQuant environment
- **No API key needed** — free, confirmed working
- **VIX ticker:** `^VIX` — returns ~332 days of daily data
- **SPY ticker:** `SPY` — already used in financial-data/
- **Storage:** `data/vix_daily.parquet` (QF project root)
- **Refresh:** Cron job daily at 6am (`0 6 * * *`)
- **Minimum data points:** 60 (for meaningful percentile calculation)

### 6.4 VIX Download Script (Tier 3)

**File:** `scripts/download_vix.py` (~20 lines)

```python
#!/usr/bin/env python3
"""Download VIX daily data from yfinance and save as parquet."""
import yfinance as yf
import pandas as pd
from pathlib import Path

def download_vix(output_path: str = "data/vix_daily.parquet", lookback: str = "1y"):
    vix = yf.download("^VIX", period=lookback, progress=False)
    vix.columns = [c.lower() for c in vix.columns]  # normalize
    vix = vix[['close']].rename(columns={'close': 'vix_close'})
    vix.index.name = 'date'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vix.to_parquet(output_path)
    print(f"Downloaded {len(vix)} VIX data points to {output_path}")
    return vix

if __name__ == '__main__':
    download_vix()
```

### 6.5 Placeholder Return (Tier 3 Default)

When `regime_detection.enabled: false` or when VIX data is insufficient:

```typescript
export async function detectRegime(): Promise<RegimeClassification> {
  return {
    regime_label: 'Unknown',
    confidence: 0.0,
    vol_regime: 'mid',
    trend_label: 'Neutral',
    vix_level: 0,
    vix_percentile: 50,
    momentum_63d: 0,
    above_200sma: true,    // assume neutral
    above_50sma: true,
    detected_at: Date.now(),
    data_points: 0,
  };
}
```

---

## 7. Regime Archetype Map

**File:** `server/regimeArchetypeMap.ts` (or `config/regimeArchetypeMap.yaml`)

### 7.1 Purpose

Maps 12 compound regime labels to QF style pack IDs. Used by the tournament runner to select which archetypes to test when regime detection is enabled. Based on QF's existing 10+ style packs.

### 7.2 Map

```typescript
// server/regimeArchetypeMap.ts

import { getAllStylePacks } from './stylePacks';

export const REGIME_ARCHETYPE_MAP: Record<string, string[]> = {
  'LowVol-Bull':       ['momentum_ignition', 'momentum_breakout', 'momentum_swing', 'earnings_catalyst', 'sector_rotation'],
  'LowVol-Neutral':    ['vwap_reversion', 'mean_reversion_daily', 'smb_capital', 'sector_rotation', 'momentum_swing'],
  'MidVol-Bull':       ['momentum_ignition', 'momentum_swing', 'smb_capital', 'earnings_catalyst', 'momentum_breakout'],
  'MidVol-Bear':       ['vwap_reversion', 'mean_reversion_daily', 'sector_rotation', 'smb_capital', 'earnings_catalyst'],
  'MidVol-Neutral':    ['smb_capital', 'vwap_reversion', 'sector_rotation', 'mean_reversion_daily', 'momentum_swing'],
  'HighVol-Bull':      ['momentum_breakout', 'momentum_ignition', 'earnings_catalyst', 'smb_capital', 'sector_rotation'],
  'HighVol-Bear':      ['vwap_reversion', 'mean_reversion_daily', 'smb_capital', 'earnings_catalyst', 'sector_rotation'],
  'HighVol-Neutral':   ['smb_capital', 'vwap_reversion', 'mean_reversion_daily', 'sector_rotation', 'earnings_catalyst'],
  'Crisis-Bull':       ['momentum_breakout', 'earnings_catalyst', 'smb_capital', 'sector_rotation', 'vwap_reversion'],
  'Crisis-Bear':       ['vwap_reversion', 'mean_reversion_daily', 'smb_capital', 'sector_rotation', 'earnings_catalyst'],
  'Crisis-Neutral':    ['smb_capital', 'vwap_reversion', 'mean_reversion_daily', 'sector_rotation', 'vwap_reversion'],
  'Unknown':           [],  // Falls through to getAllStylePacks()
};

/**
 * Get archetype style pack IDs for a given regime label.
 * Filters to only include style packs that actually exist in the system.
 * Falls back to all available packs if regime is unknown or map is empty.
 */
export function getArchetypesForRegime(
  regimeLabel: string,
  maxArchetypes: number = 5,
): string[] {
  const mapped = REGIME_ARCHETYPE_MAP[regimeLabel] || REGIME_ARCHETYPE_MAP['Unknown'];
  const allPacks = getAllStylePacks();
  const allIds = allPacks.map(p => p.style_id);

  if (mapped.length === 0) {
    // Fall back to all packs, limited by maxArchetypes
    return allIds.slice(0, maxArchetypes);
  }

  // Filter to existing packs only
  const valid = mapped.filter(id => allIds.includes(id));
  return valid.slice(0, maxArchetypes);
}
```

### 7.3 Rationale for Mappings

- **LowVol-Bull:** Favors momentum and breakout — tight spreads, trending markets, low noise
- **LowVol-Neutral:** Favors mean reversion and swing — range-bound, low vol = mean reversion works
- **MidVol-Bull:** Balanced — momentum + institutional swing + earnings catalysts
- **MidVol-Bear:** Defensive — reversion + rotation + swing with tight risk
- **HighVol-Bull:** Opportunistic — breakout on vol expansion, earnings plays
- **HighVol-Bear:** Defensive — reversion, institutional risk management
- **Crisis:** Maximum defense — swing, reversion, sector rotation for hedging
- **Unknown:** All archetypes (no bias)

---

## 8. Tournament Config

### 8.1 Environment Variables (.env)

```bash
# ── Tournament ──────────────────────────────────────────────────────────────
TOURNAMENT_ENABLED=true
TOURNAMENT_PARALLEL=false                  # false = sequential (default for Tier 1)
TOURNAMENT_MAX_CONCURRENT_LLM=3            # LLM semaphore limit
TOURNAMENT_ARCHETYPES_PER_TOURNAMENT=5     # how many style packs to test
TOURNAMENT_PROMOTION_THRESHOLD=70          # score >= this → auto-promote

# ── Regime Detection (outline only, off by default) ─────────────────────────
REGIME_DETECTION_ENABLED=false
REGIME_DETECTION_VIX_SOURCE=yfinance       # free, no API key
REGIME_DETECTION_VIX_REFRESH_CRON=0 6 * * *  # daily at 6am
REGIME_DETECTION_MIN_DATA_POINTS=60        # minimum VIX history for percentile
```

### 8.2 Config Schema (for runConfig)

```typescript
// Parsed from .env + request body
interface TournamentRunConfig {
  // Tournament settings
  enabled: boolean;                        // TOURNAMENT_ENABLED (default: true)
  parallel: boolean;                       // TOURNAMENT_PARALLEL (default: false)
  max_concurrent_llm: number;              // TOURNAMENT_MAX_CONCURRENT_LLM (default: 3)
  archetypes_per_tournament: number;       // TOURNAMENT_ARCHETYPES_PER_TOURNAMENT (default: 5)
  promotion_threshold: number;             // TOURNAMENT_PROMOTION_THRESHOLD (default: 70)

  // Regime detection settings
  regime_detection: {
    enabled: boolean;                      // REGIME_DETECTION_ENABLED (default: false)
    vix_data_source: string;               // REGIME_DETECTION_VIX_SOURCE (default: "yfinance")
    vix_refresh_cron: string;              // REGIME_DETECTION_VIX_REFRESH_CRON (default: "0 6 * * *")
    min_data_points: number;               // REGIME_DETECTION_MIN_DATA_POINTS (default: 60)
  };
}
```

### 8.3 YAML Reference (documentation only)

```yaml
# config/tournament.yaml — Reference config, actual values from .env

tournament:
  enabled: true
  parallel: true                    # false = sequential
  max_concurrent_llm: 3            # semaphore limit
  archetypes_per_tournament: 5     # how many style packs to test
  promotion_threshold: 70          # score threshold for auto-promotion
  max_archetypes: 5                # max number of archetypes to run

regime_detection:
  enabled: false                   # off by default, outline only
  vix_data_source: "yfinance"      # free, no API key
  vix_refresh_cron: "0 6 * * *"   # daily at 6am
  min_data_points: 60              # minimum VIX history for percentile calc
```

---

## 9. API Endpoints

**Location:** `server.ts` (add alongside existing `/api/run` endpoint)

### 9.1 `POST /api/tournament` — Start a tournament run

**Request body:**
```json
{
  "prompt": "Find an edge in S&P 500 earnings season using options flow data",
  "mode": "DayTrade",
  "assetClass": "equities",
  "stylePackIds": ["momentum_ignition", "vwap_reversion", "smb_capital"],
  "maxArchetypes": 5,
  "parallel": false,
  "promotionThreshold": 70
}
```

All fields except `prompt` are optional. If `stylePackIds` is omitted and regime detection is enabled, archetypes are selected from the regime map. If both are omitted, all available style packs are used (up to `maxArchetypes`).

**Response (immediate):**
```json
{
  "status": "started",
  "tournamentId": "TNM-A3F8K2M1"
}
```

**Implementation:**
1. Validate request body
2. Resolve archetype list (explicit → regime map → all packs)
3. Spawn tournament runner asynchronously (don't await — return immediately)
4. Use existing `isRunning` guard? **No** — tournament runs alongside the existing single-strategy pipeline. Add a separate `isTournamentRunning` flag.

### 9.2 `GET /api/tournament/:id` — Get tournament status + results

**Response:**
```json
{
  "tournamentId": "TNM-A3F8K2M1",
  "status": "COMPLETED",
  "config": { ... },
  "regime": {
    "regime_label": "MidVol-Bull",
    "confidence": 0.72,
    "vix_percentile": 42
  },
  "results": [
    {
      "experimentId": "EXP-X7K2M1",
      "stylePackId": "smb_capital",
      "stylePackName": "SMB Capital Swing",
      "score": 82,
      "state": "PROMOTED",
      "iterationCount": 1,
      "completedAt": 1714800000000
    },
    {
      "experimentId": "EXP-Y3F8K2",
      "stylePackId": "momentum_ignition",
      "stylePackName": "Momentum Ignition",
      "score": 71,
      "state": "PROMOTED",
      "iterationCount": 1,
      "completedAt": 1714800060000
    },
    {
      "experimentId": "EXP-Z9P4N3",
      "stylePackId": "vwap_reversion",
      "stylePackName": "VWAP Reversion",
      "score": 55,
      "state": "RANKED",
      "iterationCount": 1,
      "completedAt": 1714800120000
    }
  ],
  "winner": {
    "experimentId": "EXP-X7K2M1",
    "stylePackId": "smb_capital",
    "score": 82,
    "state": "PROMOTED"
  },
  "startedAt": 1714799900000,
  "completedAt": 1714800150000
}
```

**Implementation:** Query `tournaments` table by ID, parse `results_json`.

### 9.3 `GET /api/tournament` — List tournament history

**Query params:**
- `limit` (default: 20) — max tournaments to return
- `status` (optional) — filter by status

**Response:**
```json
{
  "tournaments": [
    {
      "id": "TNM-A3F8K2M1",
      "status": "COMPLETED",
      "winnerStylePack": "smb_capital",
      "winnerScore": 82,
      "archetypeCount": 5,
      "createdAt": 1714799900000,
      "completedAt": 1714800150000
    }
  ]
}
```

**Implementation:** Query `tournaments` table, order by `created_at DESC`, limit.

### 9.4 `POST /api/tournament/:id/cancel` — Cancel running tournament

**Response:**
```json
{
  "status": "cancelled",
  "tournamentId": "TNM-A3F8K2M1",
  "completedResults": 2,
  "pendingResults": 3
}
```

**Implementation:**
1. Set tournament status to `CANCELLED` in DB
2. Set an `AbortController` signal (passed to `runTournament`)
3. `runTournament` checks `signal.aborted` before starting each archetype
4. Already-running experiments complete (no mid-experiment abort)

### 9.5 `GET /api/regime` — Get current regime classification

**Response (when `regime_detection.enabled: false`):**
```json
{
  "enabled": false,
  "regime": {
    "regime_label": "Unknown",
    "confidence": 0.0,
    "vol_regime": "mid",
    "trend_label": "Neutral",
    "vix_level": 0,
    "vix_percentile": 50,
    "momentum_63d": 0,
    "detected_at": 1714799900000
  }
}
```

**Response (when `regime_detection.enabled: true`):**
```json
{
  "enabled": true,
  "regime": {
    "regime_label": "MidVol-Bull",
    "confidence": 0.72,
    "vol_regime": "mid",
    "trend_label": "Bull",
    "vix_level": 18.5,
    "vix_percentile": 42,
    "momentum_63d": 0.063,
    "above_200sma": true,
    "above_50sma": true,
    "detected_at": 1714799900000,
    "data_points": 252
  },
  "suggested_archetypes": ["momentum_ignition", "momentum_swing", "smb_capital", "earnings_catalyst", "momentum_breakout"]
}
```

**Implementation:** Call `detectRegime()`, then `getArchetypesForRegime()`. Lightweight — no LLM calls, just data fetch + math.

---

## 10. Dashboard Tournament Tab

**File:** `src/pages/Tournament.tsx` (new)

### 10.1 Navigation Integration

Update `src/App.tsx` and `src/components/Layout.tsx`:

```typescript
// App.tsx — add import
import { Tournament } from './pages/Tournament';

// App.tsx — add to Page type
export type Page = 'dashboard' | 'pipeline' | 'leaderboard' | 'agents' | 'research' | 'playSheets' | 'tournament';

// App.tsx — add to PageRenderer
case 'tournament':
  return <Tournament />;
```

```typescript
// Layout.tsx — add Trophy icon (already imported for Leaderboard) or Swords icon
import { Activity, GitMerge, Trophy, Bot, Settings, Search, Bell, Play, Square, Terminal as TerminalIcon, Plus, AlertCircle, ChevronDown, ChevronUp, Menu, X, BookOpen, ClipboardList, Swords } from 'lucide-react';

// Layout.tsx — add to navItems, after 'leaderboard'
{ id: 'tournament', label: 'Tournament', icon: Swords },
```

### 10.2 Page Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Tournament                                       [Start New]   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ Regime Indicator (when enabled) ────────────────────────┐  │
│  │  📊 MidVol-Bull  VIX: 18.5 (42nd %ile)  Mom: +6.3% 63d  │  │
│  │  Confidence: ████████░░ 72%                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Tournament Bracket ─────────────────────────────────────┐  │
│  │                                                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │ 🏆 SMB Capital│  │ Momentum Ign. │  │ VWAP Revers. │   │  │
│  │  │ Score: 82    │  │ Score: 71    │  │ Score: 55    │   │  │
│  │  │ PROMOTED ✓   │  │ PROMOTED ✓   │  │ RANKED       │   │  │
│  │  │ ██████████   │  │ ████████░░   │  │ █████░░░░░   │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │  │
│  │                                                           │  │
│  │  ┌──────────────┐  ┌──────────────┐                      │  │
│  │  │ Earn. Catal. │  │ Sector Rot.  │                      │  │
│  │  │ Score: 48    │  │ Score: 42    │                      │  │
│  │  │ RANKED       │  │ RANKED       │                      │  │
│  │  │ ████░░░░░░   │  │ ███░░░░░░░   │                      │  │
│  │  └──────────────┘  └──────────────┘                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Historical Tournaments ──────────────────────────────────┐  │
│  │  Date          │ Winner         │ Score │ Archetypes      │  │
│  │  2026-05-04    │ SMB Capital    │ 82    │ 5/5 completed   │  │
│  │  2026-05-03    │ Momentum Swing │ 76    │ 5/5 completed   │  │
│  │  2026-05-02    │ VWAP Reversion │ 68    │ 3/5 (2 failed) │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 Components

**`RegimeIndicator`** — Small banner showing current regime (when enabled)
- Shows regime label, VIX level, percentile, momentum
- Confidence bar
- Hidden when `regime_detection.enabled: false`

**`TournamentBracket`** — Grid of archetype cards
- Each card shows: style pack name, score bar, state badge (PROMOTED/RANKED/KILLED/FAILED/RUNNING)
- Winner card highlighted with gold border + trophy icon
- Live progress: during tournament, cards update via SSE
- Click card to navigate to experiment detail (`/api/experiments/:id/detail`)

**`StartTournamentDialog`** — Modal form
- Prompt input (text area)
- Mode selector (DayTrade / Swing)
- Asset class selector
- Style pack multi-select (or "Auto (regime-based)")
- Parallel toggle
- Promotion threshold slider

**`TournamentHistory`** — Table of past tournaments
- Links to individual tournament results
- Shows winner, score, archetype count, completion status

### 10.4 SSE Events

The tournament broadcasts progress via the existing SSE infrastructure (`broadcast()` in `server.ts`):

```typescript
// Tournament start
{ type: 'tournament_start', tournamentId: 'TNM-A3F8K2M1', archetypeCount: 5 }

// Regime detected (if enabled)
{ type: 'tournament_regime', tournamentId: 'TNM-A3F8K2M1', regime: { regime_label: 'MidVol-Bull', ... } }

// Single archetype completed
{ type: 'tournament_archetype_complete', tournamentId: 'TNM-A3F8K2M1', result: { experimentId: '...', stylePackId: '...', score: 82, state: 'PROMOTED' } }

// Tournament complete
{ type: 'tournament_complete', tournamentId: 'TNM-A3F8K2M1', winner: { stylePackId: 'smb_capital', score: 82 }, results: [...] }
```

---

## 11. Data Requirements

### 11.1 VIX Daily Data

- **Source:** yfinance (`^VIX` ticker)
- **Cost:** Free, no API key needed
- **Availability:** ~332 days of daily data confirmed working
- **Storage:** `data/vix_daily.parquet` (in QF project root)
- **Columns:** `date` (index), `vix_close`
- **Refresh:** Daily cron at 6am (`0 6 * * *`)

### 11.2 SPY Daily Data

- **Source:** Already exists in `/home/Zev/development/quant-projects/financial-data/`
- **Used for:** 63-day momentum, 50/200 SMA trend signals
- **No additional download needed**

### 11.3 No Polygon API Key Needed

Unlike the original PRD which required VIX data from Polygon (which uses `I:VIX` format), this design uses yfinance which is free and requires no API key. This removes a critical prerequisite.

### 11.4 VIX Download Script

See §6.4 — `scripts/download_vix.py` (~20 lines Python)

### 11.5 Data Flow Summary

```
yfinance (^VIX) ──► scripts/download_vix.py ──► data/vix_daily.parquet
                                                      │
yfinance (SPY) ──► (already exists in financial-data/) │
                                                      │
                                                      ▼
                                              server/regimeDetector.ts
                                                      │
                                                      ▼
                                              RegimeClassification
                                                      │
                                                      ▼
                                              server/regimeArchetypeMap.ts
                                                      │
                                                      ▼
                                              stylePackIds[] ──► tournamentRunner.ts
```

---

## 12. Testing Strategy

### 12.1 LLM Semaphore Unit Tests

**File:** `server/llm/llmSemaphore.test.ts`

```
1. test_semaphore_allows_up_to_max_concurrent
   - Create mock provider, set maxConcurrent=3
   - Fire 3 calls simultaneously, all should start immediately
   - Verify all 3 return successfully

2. test_semaphore_queues_excess_calls
   - Fire 5 calls simultaneously (maxConcurrent=3)
   - First 3 should start, 2 should queue
   - When first completes, 4th should start
   - When second completes, 5th should start

3. test_semaphore_timeout_prevents_deadlock
   - Create provider that never resolves (hangs forever)
   - Fire 4 calls (maxConcurrent=3), set timeout=100ms
   - 4th call should timeout after 100ms and throw
   - Verify no deadlock — other calls can still complete

4. test_semaphore_429_backoff
   - Mock provider throws 429 error
   - Verify the semaphore releases the slot even on error
   - Verify the existing SqliteQuotaScheduler handles retry

5. test_semaphore_noop_for_single_call
   - Fire 1 call with maxConcurrent=3
   - Should start immediately, no queuing

6. test_semaphore_release_on_error
   - Mock provider throws for one call
   - Verify slot is released (next queued call can proceed)
```

### 12.2 Tournament Runner Integration Tests

**File:** `server/tournamentRunner.test.ts`

```
7. test_tournament_sequential_mode
   - Mock runExperiment to return predetermined scores
   - Run tournament with parallel=false, 3 archetypes
   - Verify archetypes run one at a time (check call order)
   - Verify results sorted by score

8. test_tournament_parallel_mode
   - Mock runExperiment with delays
   - Run tournament with parallel=true, 5 archetypes
   - Verify all start within short time window (not sequential)
   - Verify results collected and ranked

9. test_tournament_regime_enrichment
   - Mock detectRegime to return MidVol-Bull
   - Verify archetype selection uses regime map
   - Verify prompt is enriched with regime context string

10. test_tournament_promotion_threshold
    - Run tournament with scores: 82, 71, 55, 48, 42
    - promotionThreshold=70
    - Verify top 2 are PROMOTED, rest are RANKED

11. test_tournament_cancellation
    - Start tournament with 5 archetypes
    - Cancel after 2 complete
    - Verify remaining 3 are not started
    - Verify tournament status is CANCELLED

12. test_tournament_explicit_stylepackids
    - Pass stylePackIds in config
    - Verify regime selection is skipped
    - Verify only specified packs are tested
```

### 12.3 Regime Detector Unit Tests

**File:** `server/regimeDetector.test.ts` (Tier 3)

```
13. test_detect_regime_placeholder
    - With regime_detection.enabled=false
    - Returns Unknown with confidence=0

14. test_regime_classification_crisis_bear
    - Synthetic VIX at 95th percentile, momentum -8%
    - Returns "Crisis-Bear" with high confidence

15. test_regime_classification_low_vol_bull
    - Synthetic VIX at 20th percentile, momentum +7%
    - Returns "LowVol-Bull" with high confidence

16. test_regime_insufficient_data
    - < 60 data points
    - Returns Unknown with warning

17. test_regime_archetype_map
    - getArchetypesForRegime("MidVol-Bull") returns correct packs
    - getArchetypesForRegime("Unknown") returns all packs
    - Filters out packs that don't exist
```

### 12.4 End-to-End Test

**File:** `tests/test_tournament_e2e.test.ts`

```
18. test_full_tournament_with_mock_pipeline
    - Use MockBacktestAdapter
    - Run POST /api/tournament with 3 archetypes
    - Poll GET /api/tournament/:id until COMPLETED
    - Verify 3 experiment rows in experiments table
    - Verify tournament has 3 results, sorted by score
    - Verify winner is PROMOTED

19. test_tournament_api_list
    - Create 3 tournaments
    - GET /api/tournament?limit=2
    - Returns latest 2
```

### 12.5 Rate Limit Test

```
20. test_429_exponential_backoff
    - Mock LLM provider to throw 429 on first call, succeed on retry
    - Verify tournament runner handles this gracefully
    - Verify exponential backoff is applied by SqliteQuotaScheduler
    - Verify final result is still collected
```

### 12.6 Parallel Safety Test

```
21. test_parallel_no_sqlite_collisions
    - Run 5 archetypes in parallel
    - Each runExperiment writes to experiments table
    - Verify no "database is locked" errors
    - Verify all 5 experiment rows exist after completion
    - Verify no mixed log streams (each experiment has its own expId in logs)

22. test_parallel_log_isolation
    - Run 3 archetypes in parallel
    - Capture all log output
    - Verify each log line contains the correct experimentId
    - Verify no interleaving of stage outputs between experiments
```

---

## 13. What This Does NOT Replace

### 13.1 Existing Single-Strategy `/api/run` Endpoint — Unchanged

The existing `POST /api/run` endpoint in `server.ts` (line 167) continues to work identically. It runs one experiment with one style pack. The tournament is a **new** endpoint (`/api/tournament`). No changes to the existing `/api/run` code path.

### 13.2 The Ralph Loop — Unchanged

The Ralph Loop (postmortem → mutation_plan → vision_keep) runs on the **winning archetype only**, after the tournament completes. This is the existing `autoOptimize: true` behavior in `runExperiment()`. The tournament runs each archetype for 1 iteration only (`maxIterations: 1, autoOptimize: false`). After the tournament, the user (or a future automation) can run the winner through the Ralph Loop via the existing `/api/run` endpoint.

### 13.3 CrabQuant Loops — Unchanged

The existing CrabQuant loops (diversity-explorer, sharpe-optimizer, meta-analyzer) are completely unaffected. This system lives entirely within Quant Factory (TypeScript). No Python code changes in CrabQuant.

### 13.4 Manual Override System — Unchanged

The existing manual override system (`/api/override/:stage/:id`) works during tournament experiments just as it does during single experiments. Each tournament archetype is a full experiment with its own stages, so overrides can be applied per-archetype.

---

## 14. Implementation Phases

### Tier 1: Core (~5 hrs)

| Task | File | Est. Lines |
|------|------|-----------|
| LLM semaphore | `server/llm/llmSemaphore.ts` | ~30 |
| Tournament runner | `server/tournamentRunner.ts` | ~80 |
| Tournament API endpoints | `server.ts` (add ~30 lines) | ~30 |
| Tournament DB schema | `server/db.ts` (add migration) | ~15 |
| Config parsing | `server.ts` (read .env vars) | ~10 |
| Sequential mode only | Default `parallel: false` | — |
| Semaphore unit tests | `server/llm/llmSemaphore.test.ts` | ~60 |
| Tournament unit tests | `server/tournamentRunner.test.ts` | ~80 |

**Deliverables:**
- `POST /api/tournament` works (sequential mode)
- `GET /api/tournament/:id` returns results
- `GET /api/tournament` lists history
- LLM semaphore gates concurrent calls
- All unit tests passing

### Tier 2: Parallel (~3 hrs)

| Task | File | Est. Lines |
|------|------|-----------|
| Parallel tournament mode | `server/tournamentRunner.ts` (modify) | ~20 |
| Cancel endpoint | `server.ts` (add ~15 lines) | ~15 |
| Parallel safety tests | `tests/test_tournament_parallel.test.ts` | ~60 |
| Rate limit backoff test | `tests/test_tournament_ratelimit.test.ts` | ~30 |
| Dashboard Tournament tab | `src/pages/Tournament.tsx` | ~200 |
| Navigation integration | `src/App.tsx`, `src/components/Layout.tsx` | ~5 |
| SSE event broadcasting | `server/tournamentRunner.ts` | ~20 |

**Deliverables:**
- Parallel mode works (5 archetypes simultaneously)
- Cancel endpoint works
- No SQLite collisions in parallel mode
- Dashboard tab shows tournament bracket with live scores
- All parallel safety tests passing

### Tier 3: Regime — Outline (~2 hrs)

| Task | File | Est. Lines |
|------|------|-----------|
| Regime detector interfaces + placeholder | `server/regimeDetector.ts` | ~50 |
| Regime archetype map | `server/regimeArchetypeMap.ts` | ~50 |
| VIX download script | `scripts/download_vix.py` | ~20 |
| Regime API endpoint | `server.ts` (add ~15 lines) | ~15 |
| Regime unit tests (synthetic data) | `server/regimeDetector.test.ts` | ~60 |
| Wire into tournament runner | `server/tournamentRunner.ts` (modify) | ~15 |
| Dashboard regime indicator | `src/pages/Tournament.tsx` (modify) | ~30 |

**Deliverables:**
- `GET /api/regime` returns placeholder when disabled
- Regime archetype map returns correct style pack IDs
- VIX download script works
- Unit tests with synthetic VIX data
- Dashboard shows regime indicator (placeholder state)
- Tournament uses regime map when enabled

### Tier 4: Integration (Later)

| Task | Description |
|------|-------------|
| Wire regime context into playbook prompt | Append regime summary to playbook researcher prompt in `runExperiment()` |
| Regime memory | Recall past tournament results by regime from `factory.db` |
| LLM-powered archetype selection | Use LLM to pick archetypes instead of heuristic map |
| Dashboard regime indicators | Full regime history chart, confidence trends |
| Ralph Loop on winner | Auto-trigger Ralph Loop on tournament winner |
| Cron integration | Scheduled daily tournaments via Gas City or system cron |

---

## 15. Risk Register & Mitigations

| # | Risk | Probability | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | LLM rate limiting (429 errors) during parallel tournament | Medium | Medium | LLM semaphore limits concurrency to 3. Existing `SqliteQuotaScheduler` handles 429 backoff with exponential delays. If all 3 concurrent calls hit 429, they retry with backoff. Worst case: tournament slows down but doesn't fail. |
| 2 | Parallel SQLite write collisions | Medium | High | SQLite in WAL mode allows concurrent reads. Writes are serialized by `better-sqlite3`'s synchronous API (Node.js single-threaded). Each `runExperiment()` uses a separate `expId` and writes to separate rows. The `saveIterationData` transaction in `orchestrator.ts` is already atomic. Risk is low but test explicitly (test #21). |
| 3 | Semaphore deadlocks | Low | Critical | Every queued call has a 60-second timeout. If timeout fires, the call is removed from the queue and throws. The `try/finally` in the semaphore ensures `release()` is always called even on error. Test explicitly (test #3). |
| 4 | VIX data unavailable from yfinance | Low | Low | yfinance is free and confirmed working with ~332 days of data. If unavailable, `detectRegime()` returns Unknown/neutral placeholder. Tournament falls back to all archetypes. No disruption. |
| 5 | Tournament takes too long (wall clock) | Medium | Low | Sequential mode: 5 archetypes × ~3 min each = ~15 min. Parallel mode: ~3-5 min total (LLM-bound). Configurable `maxArchetypes` to limit. Cancel endpoint available. |
| 6 | Disrupts existing single-strategy pipeline | Very Low | Critical | Tournament uses a separate `isTournamentRunning` flag. Existing `/api/run` is completely unchanged. `runExperiment()` is called with the same signature. No modifications to the orchestrator. |
| 7 | LLM semaphore adds latency to single experiments | Low | Low | When only 1 experiment runs, `active` is 0 or 1, never exceeds `maxConcurrent`. `acquire()` returns immediately. Zero overhead for single-experiment mode. |
| 8 | Style pack not found in regime map | Low | Low | `getArchetypesForRegime()` filters to existing packs only. If mapped pack doesn't exist, it's skipped. Falls back to `getAllStylePacks()` if no valid packs found. |
| 9 | Tournament results inconsistent due to LLM non-determinism | Medium | Low | Same issue as existing single-experiment runs. Each archetype runs 1 iteration. Scores are comparable (same prompt, different style pack). Non-determinism is a feature (tests robustness). |

---

## 16. Acceptance Criteria

### Must-Have (Tier 1 — Core)

- [ ] `server/llm/llmSemaphore.ts` — wraps LLM provider with max-concurrent limit
- [ ] `server/tournamentRunner.ts` — orchestrates N archetype runs through `runExperiment()`
- [ ] `POST /api/tournament` — starts tournament, returns tournamentId immediately
- [ ] `GET /api/tournament/:id` — returns tournament status and results
- [ ] `GET /api/tournament` — lists tournament history
- [ ] Sequential mode works (default `parallel: false`)
- [ ] Results ranked by score descending
- [ ] Top results marked PROMOTED when score >= `promotion_threshold`
- [ ] Tournament row written to `tournaments` table in `factory.db`
- [ ] LLM semaphore unit tests pass (6 tests)
- [ ] Tournament runner unit tests pass (6 tests)
- [ ] Existing `/api/run` endpoint works identically (no regressions)
- [ ] Config via `.env` variables (`TOURNAMENT_ENABLED`, `TOURNAMENT_MAX_CONCURRENT_LLM`, etc.)

### Must-Have (Tier 2 — Parallel)

- [ ] Parallel mode works (`tournament.parallel: true`)
- [ ] `POST /api/tournament/:id/cancel` — cancels running tournament
- [ ] No SQLite collisions in parallel mode (test #21 passing)
- [ ] No mixed log streams between parallel experiments (test #22 passing)
- [ ] Rate limit backoff test passing (test #20)
- [ ] Dashboard Tournament tab renders with bracket, scores, and history
- [ ] SSE events broadcast tournament progress
- [ ] Navigation shows "Tournament" tab in sidebar

### Must-Have (Tier 3 — Regime Outline)

- [ ] `server/regimeDetector.ts` — interfaces + placeholder implementation
- [ ] `server/regimeArchetypeMap.ts` — maps 12 regime labels to style pack IDs
- [ ] `GET /api/regime` — returns placeholder when `regime_detection.enabled: false`
- [ ] `scripts/download_vix.py` — downloads VIX data from yfinance
- [ ] Regime detector unit tests with synthetic data (5 tests)
- [ ] Tournament uses regime map for archetype selection when enabled
- [ ] Dashboard shows regime indicator (placeholder state when disabled)

### Nice-to-Have (Tier 4 — Integration)

- [ ] Regime context injected into playbook researcher prompt
- [ ] Regime memory (recall past tournament results by regime)
- [ ] LLM-powered archetype selection
- [ ] Dashboard regime history chart
- [ ] Auto-trigger Ralph Loop on tournament winner
- [ ] Scheduled daily tournaments via cron

---

## 17. Changelog

### 2026-05-04 — v2: Architectural Pivot to QF-Integrated Tournament

**Major changes from v1:**

- **Architecture:** Completely redesigned. No longer a separate CrabQuant Python loop. Now integrates directly into Quant Factory as a tournament mode endpoint.
- **Language:** All TypeScript in QF. No Python code in this PRD.
- **Removed:** `loops/regime-advisor/` directory, `program.md`, `regime_history.tsv`, `tournament_runner.py`, `params.yaml`, all Python modules
- **Removed:** Part C — Gas City / HermGC Integration section
- **Removed:** References to CrabQuant loop chain position, `loops/registry.yaml` registration
- **Removed:** HMM regime detection (explicitly deferred to later phase)
- **Added:** Tournament runner (`server/tournamentRunner.ts`) — orchestrates N `runExperiment()` calls
- **Added:** LLM semaphore (`server/llm/llmSemaphore.ts`) — max 3 concurrent LLM calls
- **Added:** Regime archetype map (`server/regimeArchetypeMap.ts`) — maps regimes to QF style packs
- **Added:** Tournament config via `.env` variables
- **Added:** 5 new API endpoints (`/api/tournament/*`, `/api/regime`)
- **Added:** Dashboard Tournament tab with bracket view
- **Added:** Parallel safety tests (SQLite collision, log isolation)
- **Added:** Implementation phases (Tier 1-4) with time estimates
- **Updated:** Risk register with LLM rate limiting, parallel SQLite safety, semaphore deadlocks
- **Updated:** Acceptance criteria for tournament mode
- **Updated:** Data requirements — yfinance instead of Polygon (free, no API key)
- **Updated:** Style pack references to match actual QF style packs (10+ packs)

### 2026-05-04 — v1: Initial Draft (Regime Advisor Loop)

- Created full PRD with CrabQuant loop architecture, Python modules, API endpoints, dashboard layout
- 1273 lines covering regime detection, proposal generation, tournament runner, cross-session memory

---

## Appendix A: Existing Style Packs Reference

Current QF style packs in `server/stylePacks.ts` (10 unique packs):

| style_id | Name | Asset Class |
|----------|------|-------------|
| `momentum_ignition` | Momentum Ignition | equities |
| `vwap_reversion` | VWAP Reversion | equities |
| `crypto_perp_breakout` | Crypto Perp Breakout | crypto |
| `crypto_funding_reversion` | Crypto Funding Reversion | crypto |
| `smb_capital` | SMB Capital Swing | equities |
| `momentum_swing` | Momentum Swing | equities |
| `momentum_breakout` | Momentum Breakout | equities |
| `earnings_catalyst` | Earnings Catalyst | equities |
| `mean_reversion_daily` | Mean Reversion Daily | equities |
| `sector_rotation` | Sector Rotation | equities |

Note: `mean_reversion` is an alias for `mean_reversion_daily`.

## Appendix B: Existing LLM Infrastructure

The QF codebase already has significant LLM infrastructure in `server/llm/`:

| File | Purpose |
|------|---------|
| `index.ts` | `LLMProvider` interface, `GeminiProvider`, `OpenAICompatibleProvider`, phase policies |
| `demo.ts` | `DemoProvider` for deterministic testing |
| `sqliteQuotaScheduler.ts` | Rate limiting with SQLite-backed quota tracking, cooldown, circuit breaker |
| `responseCache.ts` | Response caching to avoid duplicate LLM calls |

The semaphore (`llmSemaphore.ts`) composes with the existing `SqliteQuotaScheduler` — semaphore gates concurrency, scheduler gates rate.

## Appendix C: File Inventory

### New Files to Create

| File | Tier | Lines | Purpose |
|------|------|-------|---------|
| `server/llm/llmSemaphore.ts` | 1 | ~30 | LLM concurrency limiter |
| `server/llm/llmSemaphore.test.ts` | 1 | ~60 | Semaphore unit tests |
| `server/tournamentRunner.ts` | 1 | ~80 | Tournament coordinator |
| `server/tournamentRunner.test.ts` | 1 | ~80 | Tournament unit tests |
| `server/regimeDetector.ts` | 3 | ~50 | Regime detection interfaces + placeholder |
| `server/regimeDetector.test.ts` | 3 | ~60 | Regime unit tests |
| `server/regimeArchetypeMap.ts` | 3 | ~50 | Regime → style pack mapping |
| `scripts/download_vix.py` | 3 | ~20 | VIX data download |
| `src/pages/Tournament.tsx` | 2 | ~200 | Dashboard tournament tab |

### Existing Files to Modify

| File | Change | Tier |
|------|--------|------|
| `server.ts` | Add `/api/tournament/*` + `/api/regime` endpoints (~30 lines) | 1+3 |
| `server/db.ts` | Add `tournaments` table migration (~15 lines) | 1 |
| `src/App.tsx` | Add Tournament page import, Page type, PageRenderer case | 2 |
| `src/components/Layout.tsx` | Add Swords icon import, navItem | 2 |
| `.env.example` | Add tournament + regime config variables | 1 |
