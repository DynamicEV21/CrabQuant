# Diversity Explorer — Portfolio Quality-Diversity Optimization Loop

**Source:** Adapted from QuantEvolve (arxiv 2510.18569) quality-diversity optimization.
**Inspired by:** `alpha-lab/docs/reference-loops/strategy-evolver/program.md`

**Goal:** Ensure the CrabQuant strategy PORTFOLIO is diverse across regimes, archetypes,
tickers, and risk profiles — while maintaining minimum quality standards. This loop does NOT
discover new strategies from scratch. Instead, it identifies GAPS in the existing portfolio's
coverage and creates targeted mandates for the refinement pipeline to fill those gaps.

**Why this exists:** CrabQuant has 118+ strategies, but they may be clustered in certain
archetype-ticker combinations (e.g., many momentum strategies for SPY) while leaving other
completely uncovered (e.g., no volatility strategies for QQQ). A diverse portfolio is more
robust to regime changes and provides better risk-adjusted returns when combined.

**Primary metric:** Quality-Diversity Score (QD-Score) = `quality × novelty`, where:
- `quality` = Sharpe × √(trades/20) × (1 - |max_drawdown|) — matches CrabQuant's backtest engine
- `novelty` = how different the strategy is from existing portfolio, based on feature-space distance

**Time budget:** ~15 minutes per mandate (refinement pipeline). Target 5-10 mandates per session.

---

## Setup

Before starting the loop, complete these steps exactly once:

### 1. Read the codebase for full context

- `loops/sandbox.py` — all helpers (load_registry, compute_diversity_coverage, identify_gaps,
  create_mandate_for_gap, run_mandate, evaluate_result, diversity_report, git_commit, git_revert)
- `loops/diversity-explorer/feature_map.yaml` — behavioral feature dimensions for this loop
- `loops/diversity-explorer/gaps_log.tsv` — log of all gap exploration attempts
- `crabquant/refinement/config.py` — validation config (walk-forward thresholds, quality gates)
- `crabquant/refinement/archetypes.py` — strategy archetype definitions and skeletons
- `strategies/production/registry.json` — the production strategy registry (read-only)
- `scripts/crabquant_cron.py` — the cron entry point that runs mandates

### 2. Verify the project environment

```bash
cd /home/Zev/development/CrabQuant
source .venv/bin/activate
python -c "import loops.sandbox; print('sandbox OK')"
```

### 3. Run the initial diversity report

```python
import sys
sys.path.insert(0, "/home/Zev/development/CrabQuant")
from loops.sandbox import load_registry, diversity_report, compute_diversity_coverage, identify_gaps

registry = load_registry()
report = diversity_report(registry)
print(report)
```

Save this report. It establishes the baseline.

### 4. Record the baseline coverage

```python
coverage = compute_diversity_coverage(registry)
gaps = identify_gaps(coverage)

print(f"Baseline: {len(registry)} strategies, {len(gaps)} gaps identified")
print(f"Archetypes covered: {list(coverage['by_archetype'].keys())}")
print(f"Tickers covered: {len(coverage['by_ticker'])} / 33")
```

### 5. Confirm and go

Once baseline is recorded, start the loop immediately.

---

## Sandbox — Files You CAN Create/Modify

Only modify these files. Everything else is the judge and must not change:

| File | What you can change |
|------|-------------------|
| `loops/diversity-explorer/mandates/*.json` | Create new mandate files for the pipeline |
| `loops/diversity-explorer/gaps_log.tsv` | Append gap exploration records |
| `loops/diversity-explorer/config.json` | Loop configuration (max mandates, quality floor, etc.) |

## Judge — Files You CANNOT Modify

- `crabquant/` — all source code under this directory
- `scripts/` — entry points and cron scripts
- `strategies/` — strategy code and registry
- `results/` — backtest results, sweep results, winners
- `tests/` — test suite
- `loops/sandbox.py` — helper utilities
- `loops/diversity-explorer/program.md` — this file
- `loops/diversity-explorer/feature_map.yaml` — feature map definition

---

## Rules

1. **Diversity is a first-class objective.** A strategy that fills an empty cell AND meets
   quality thresholds is more valuable than a slightly higher-Sharpe strategy that duplicates
   an already-covered combination.
2. **Quality floor is sacred.** Every strategy must meet: Sharpe >= 0.5, rolling WF >= 3/6
   windows passed, walk_forward_robust=True. No exceptions.
3. **Only ADD strategies, never REPLACE.** If a gap already has a robust strategy, don't
   try to override it. Only fill truly empty cells.
4. **Don't write strategy code.** This loop creates mandates — JSON instructions for the
   existing refinement pipeline. The LLM-driven pipeline invents the actual strategy code.
5. **Use the existing infrastructure.** `crabquant_cron.py` runs mandates. The refinement
   pipeline handles invention, backtesting, and promotion. Don't reinvent any of this.
6. **Track everything in gaps_log.tsv.** Every gap exploration attempt must be logged,
   whether it succeeds or fails. This prevents re-attempting the same gaps.
7. **Simplicity counts.** A mandate that fills a gap is better than a complex mandate that
   doesn't. Keep mandates focused on one archetype-ticker combination.
8. **Anti-patterns:**
   - Don't create mandates for already-covered archetype-ticker combinations
   - Don't optimize for diversity alone (quality floor exists and is non-negotiable)
   - Don't create mandates for multi_signal_ensemble archetype (these are composed, not invented)
   - Don't spam the same gap — check gaps_log.tsv before creating a new mandate
   - Don't modify crabquant/ source code, backtest engine, or validation logic
   - Don't create more than 3 mandates for the same gap before reassessing

---

## Diversity Dimensions

The portfolio is analyzed across 6 dimensions. Each strategy occupies a position in this space:

### Dimension 1: Archetype (6 types)
- `momentum` — rides trends via acceleration signals (regime: trending)
- `mean_reversion` — buys oversold, sells overbought (regime: ranging)
- `breakout` — enters on range expansion (regime: volatile)
- `volatility` — trades vol expansion/contraction (regime: volatile)
- `statistical_arb` — z-score based entry/exit (regime: ranging)
- `multi_signal_ensemble` — multi-indicator consensus (regime: any)

### Dimension 2: Ticker (33 instruments)
Core ETFs: SPY, QQQ, IWM, GLD (highest priority)
MAG7: AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
Sectors: Financials (BAC, JPM, V), Healthcare (ABBV, LLY, UNH), etc.

### Dimension 3: Regime Affinity (3+1 types)
- `trending` — strategies that excel when markets trend
- `ranging` — strategies that excel in sideways markets
- `volatile` — strategies that excel during volatility spikes
- `any` — regime-agnostic approaches

### Dimension 4: Indicator Family (5 types)
- `momentum_indicators` — EMA, SMA, ROC, MACD, ADX, Supertrend
- `mean_reversion_indicators` — RSI, Bollinger Bands, CCI, Stochastic, Williams %R
- `volatility_indicators` — ATR, VIX, Keltner
- `volume_indicators` — OBV, Volume, VWAP
- `trend_indicators` — EMA, SMA, Donchian, Supertrend

### Dimension 5: Trade Frequency (3 bins)
- `low` — fewer than 20 trades/year (patient)
- `medium` — 20-60 trades/year (moderate)
- `high` — more than 60 trades/year (active)

### Dimension 6: Sharpe Tier (4 bins)
- `elite` — Sharpe > 2.0
- `strong` — Sharpe 1.0-2.0
- `viable` — Sharpe 0.5-1.0
- `marginal` — Sharpe 0-0.5

---

## The Experiment Loop

LOOP FOREVER (until manually stopped or max mandates reached):

### Step 1: Check current state

```python
import sys
sys.path.insert(0, "/home/Zev/development/CrabQuant")
from loops.sandbox import (
    load_registry,
    compute_diversity_coverage,
    identify_gaps,
    load_gaps_log,
    diversity_report,
    git_current_commit,
)

registry = load_registry()
coverage = compute_diversity_coverage(registry)
gaps = identify_gaps(coverage)
gaps_log = load_gaps_log()

print(f"State: {len(registry)} strategies, {len(gaps)} gaps, {len(gaps_log)} attempts logged")
```

- Read `loops/diversity-explorer/gaps_log.tsv` for previous attempts
- Filter out gaps that have been recently attempted (within last 3 attempts) and failed
- Count successful gap-fills vs failed attempts
- Check if any recent mandates produced strategies that passed validation

### Step 2: Select the highest-priority gap

Prioritize gaps using this scoring:

1. **Empty archetype-ticker cells** (priority 5) — most impactful for diversity
2. **Archetype deficits** — archetypes with < 2 strategies (priority varies)
3. **Ticker deficits** — tickers with no strategies at all (priority 15+)
4. **Regime deficits** — regimes with < 5 strategies (priority varies)

Filter logic:
- Skip gaps that were attempted 3+ times with no success (status=failed)
- Skip gaps for multi_signal_ensemble archetype (composed, not invented directly)
- Prefer gaps targeting high-priority tickers (SPY, QQQ, GLD, IWM) when tied
- If all gaps have been attempted, check if any succeeded and the registry hasn't been
  updated — if so, the pipeline may need a manual check

```python
# Filter out over-attempted gaps
recently_failed = set()
for log_entry in gaps_log:
    if log_entry.get("status") == "failed":
        gap_desc = log_entry.get("gap_description", "")
        recently_failed.add(gap_desc)

# Count failures per gap
from collections import Counter
failure_counts = Counter()
for log_entry in gaps_log:
    if log_entry.get("status") == "failed":
        failure_counts[log_entry.get("gap_description", "")] += 1

# Filter gaps
eligible_gaps = [
    g for g in gaps
    if failure_counts.get(g["description"], 0) < 3
    and g.get("archetype") != "multi_signal_ensemble"
]

if not eligible_gaps:
    print("All gaps exhausted or over-attempted. Stopping.")
    break

target_gap = eligible_gaps[0]  # Already sorted by priority
print(f"Target gap: {target_gap['description']}")
```

### Step 3: Create a mandate for the gap

```python
from loops.sandbox import create_mandate_for_gap, log_gap_attempt

mandate_path = create_mandate_for_gap(target_gap)
print(f"Created mandate: {mandate_path}")

# Log the attempt
log_gap_attempt(
    target_gap,
    mandate_path=mandate_path,
    status="pending",
    notes="Mandate created, about to run",
)
```

The mandate will be a JSON file in `loops/diversity-explorer/mandates/` with the format:
```json
{
  "name": "Diversity Mandate: No volatility strategy for QQQ (regime: volatile)",
  "description": "Fill portfolio gap: No volatility strategy for QQQ...",
  "tickers": ["QQQ"],
  "primary_ticker": "QQQ",
  "period": "2y",
  "strategy_archetype": "volatility",
  "target_regime": "volatile",
  "max_turns": 7,
  "sharpe_target": 1.0,
  "constraints": {"min_trades": 5, "max_drawdown_pct": 25.0},
  ...
}
```

### Step 4: Run the mandate via the refinement pipeline

Copy the mandate to the main mandates directory (so crabquant_cron.py can find it),
then run the cron script:

```python
from loops.sandbox import run_mandate
import shutil

# Copy mandate to the main mandates directory so cron picks it up
import os
main_mandate_path = f"/home/Zev/development/CrabQuant/refinement/mandates/{mandate_path.name}"
shutil.copy(str(mandate_path), main_mandate_path)

# Run the pipeline
result = run_mandate(main_mandate_path, timeout=900)  # 15 min timeout
print(f"Pipeline result: success={result['success']}, duration={result['duration']:.0f}s")

if not result["success"]:
    print(f"Error: {result['stderr']}")
```

**Alternative approach (if crabquant_cron.py doesn't support single-mandate targeting):**

You can also use the refinement pipeline directly:

```bash
cd /home/Zev/development/CrabQuant
source .venv/bin/activate
python scripts/refinement_loop.py --mandate loops/diversity-explorer/mandates/<mandate_name>.json
```

### Step 5: Evaluate the result

After the pipeline completes, check what happened:

```python
from loops.sandbox import evaluate_result, log_gap_attempt

# Find the run directory (naming convention varies)
import glob
run_dirs = sorted(glob.glob("refinement_runs/*"), key=os.path.getmtime, reverse=True)
if run_dirs:
    eval_result = evaluate_result(run_dirs[0])
    print(f"Run result: {eval_result}")

    # Update the gaps log
    if eval_result.get("converged"):
        log_gap_attempt(
            target_gap,
            mandate_path=mandate_path,
            result_sharpe=eval_result.get("best_sharpe"),
            status="success",
            notes=f"Converged in {eval_result.get('total_turns', 0)} turns"
        )
    else:
        log_gap_attempt(
            target_gap,
            mandate_path=mandate_path,
            result_sharpe=eval_result.get("best_sharpe"),
            status="failed",
            notes=f"Did not converge. Failure: {eval_result.get('failure_mode', 'unknown')}"
        )
```

### Step 6: Verify the strategy was added to the registry

If the pipeline reported success, verify:

```python
from loops.sandbox import load_registry

updated_registry = load_registry()
new_entries = [
    e for e in updated_registry
    if e.get("diversity_source") == "diversity-explorer"
    or e.get("strategy_name", "").startswith("diversity_")
]

if new_entries:
    print(f"NEW STRATEGIES ADDED: {len(new_entries)}")
    for entry in new_entries:
        print(f"  - {entry['strategy_name']} ({entry['ticker']}) "
              f"Sharpe={entry.get('sharpe', 0):.2f} "
              f"WF={'robust' if entry.get('walk_forward_robust') else 'NOT ROBUST'}")
else:
    print("No new strategies added to registry from this mandate.")
    print("The pipeline may have invented strategies but they didn't pass validation.")
```

### Step 7: Re-compute diversity and check improvement

```python
updated_coverage = compute_diversity_coverage(updated_registry)
updated_gaps = identify_gaps(updated_coverage)
improvement = len(gaps) - len(updated_gaps)

print(f"Coverage improvement: {improvement} gaps closed ({len(gaps)} → {len(updated_gaps)})")
print(f"Archetypes: {list(updated_coverage['by_archetype'].keys())}")
print(f"Tickers covered: {len(updated_coverage['by_ticker'])} / 33")
```

### Step 8: Print periodic coverage report

Every 5 mandates, print a full diversity report:

```python
from loops.sandbox import diversity_report
report = diversity_report(updated_registry)
print(report)
```

### Step 9: Repeat

Go back to Step 1. Do NOT stop. Loop until:
- All gaps are filled (unlikely in one session), OR
- Max mandates reached (check `loops/diversity-explorer/config.json`), OR
- 10 consecutive failures without any gap being filled

---

## Quality Gates for Keeping Strategies

A strategy from a diversity mandate must meet ALL of these to be considered successful:

1. **Sharpe >= 0.5** on the primary ticker (minimum viable quality)
2. **Walk-forward robust** — rolling WF with >= 3/6 windows passing
3. **walk_forward_test_sharpe > 0** — not a data artifact
4. **Not a duplicate** — different archetype OR different ticker from existing strategies
5. **Positive trades** — at least 5 trades in the backtest period

A strategy that meets these gates AND fills an empty cell should be KEPT even if its
Sharpe is lower than the portfolio average. Diversity contribution justifies inclusion.

---

## Gap Priority Scoring

Gaps are scored and sorted by priority:

| Gap Type | Base Priority | Multiplier |
|----------|--------------|------------|
| Ticker with NO strategies | 15 | × 1.0 |
| Archetype with < 2 strategies | varies | × (min - current) × 10 |
| Regime with < 5 strategies | varies | × (min - current) × 8 |
| Empty archetype-ticker cell | 5 | × 1.0 |

Additional multipliers:
- High-priority tickers (SPY, QQQ, GLD, IWM): × 1.5
- Previously attempted and failed: × 0.3 (de-prioritize)
- Recently succeeded for same archetype: × 0.5 (maybe pivot)

---

## Handling Stagnation

### After 5 consecutive failed mandates:

1. Review the gaps_log.tsv for patterns — are certain archetypes consistently failing?
2. Check if the refinement pipeline is having systemic issues (code gen failures, etc.)
3. Try lowering the Sharpe target for the next mandate (e.g., from 1.0 to 0.7)
4. Try a different gap type — switch from archetype-ticker to regime deficits

### After 10 consecutive failed mandates:

1. Print a full diversity report and assess whether further attempts are worthwhile
2. Consider that some gaps may be fundamentally hard to fill (e.g., volatility strategies
   for tickers with low volatility)
3. Mark persistent gaps as "low feasibility" in notes
4. Stop the loop and report findings

### Mandate Sharpe Target Adjustment:

Start with sharpe_target=1.0. If a gap fails 2+ times:
- Reduce to 0.7 for the 3rd attempt
- Reduce to 0.5 (the minimum) for the 4th attempt
- If it still fails at 0.5, the gap may be genuinely unfillable — mark and skip

---

## Output Files

### gaps_log.tsv
Tab-separated log of all gap exploration attempts:
```
timestamp    gap_description    archetype    ticker    regime    mandate_created    result_sharpe    result_wf_windows    status    notes
```

### mandates/*.json
Mandate files created for each gap. Naming: `diversity_{archetype}_{ticker}_{timestamp}.json`

### config.json (optional)
Loop configuration:
```json
{
  "max_mandates_per_session": 20,
  "min_sharpe": 0.5,
  "min_wf_windows": 3,
  "max_consecutive_failures": 10,
  "sharpe_target_start": 1.0,
  "sharpe_target_min": 0.5,
  "max_attempts_per_gap": 3
}
```

---

## Success Criteria

A successful session should:
1. **Close at least 1-2 gaps** — add strategies that fill previously empty cells
2. **Maintain quality floor** — all new strategies pass walk-forward validation
3. **Not regress existing coverage** — no existing robust strategies are displaced
4. **Log all attempts** — gaps_log.tsv has complete records for every mandate

Long-term target (after multiple sessions):
- At least 2 strategies per archetype
- At least 1 strategy per major ticker (SPY, QQQ, GLD, IWM, AAPL, NVDA, MSFT)
- At least 5 strategies per regime (trending, ranging, volatile)
- Coverage across all trade frequency bins (not all low-frequency or all high-frequency)

---

## Example Session Flow

```
Session Start
├── Load registry (118 strategies)
├── Compute coverage
│   ├── momentum: 45 strategies (SPY, AAPL, GOOGL, MSFT, NVDA, ...)
│   ├── mean_reversion: 30 strategies (SPY, AAPL, QQQ, ...)
│   ├── breakout: 25 strategies (SPY, GLD, CAT, ...)
│   ├── volatility: 10 strategies (SPY, GLD, ...)
│   └── No strategies for: TSLA, NFLX, DIS, INTC, SLV
├── Identify gaps (priority sorted)
│   ├── 1. No volatility strategies for QQQ (priority 5)
│   ├── 2. No mean_reversion for TSLA (priority 5)
│   ├── 3. No breakout for NFLX (priority 5)
│   └── ...
├── Loop iteration 1
│   ├── Create mandate: diversity_volatility_qqq_20260503_110000.json
│   ├── Run pipeline (12 min)
│   ├── Result: converged, Sharpe=1.2, WF=4/6
│   ├── Strategy added to registry
│   └── Gap closed! volatility|QQQ now covered
├── Loop iteration 2
│   ├── Create mandate: diversity_mean_reversion_tsla_20260503_111500.json
│   ├── Run pipeline (14 min)
│   ├── Result: did not converge, best Sharpe=0.3
│   └── Logged as failed, gap remains
├── Loop iteration 3
│   ├── Create mandate: diversity_breakout_nflx_20260503_113000.json
│   ├── Run pipeline (10 min)
│   ├── Result: converged, Sharpe=0.8, WF=3/6
│   └── Strategy added, gap closed
├── ... continue until max mandates or all gaps filled
└── Session End
    ├── Print final diversity report
    ├── Summary: 2 gaps filled, 1 failed, 118 → 120 strategies
    └── Commit: "diversity-explorer: filled volatility/QQQ and breakout/NFLX gaps"
```

---

## Integration with Existing CrabQuant Infrastructure

This loop integrates with CrabQuant's existing systems:

### Refinement Pipeline
- Mandates created here use the same JSON format as `refinement/mandates/`
- The pipeline (refinement_loop.py) processes them identically
- All quality gates, walk-forward validation, and promotion logic apply unchanged

### Registry
- New strategies are added to `strategies/production/registry.json` by the promotion pipeline
- This loop does NOT directly modify the registry
- Strategies are only added after passing all validation gates

### Walk-Forward Validation
- Uses the same rolling walk-forward (6 windows, min_avg_test_sharpe=0.4, min_windows=3)
- This loop does NOT change validation thresholds

### Governance
- Reports can be shared with the Director layer for KPI tracking
- Diversity coverage % can be added as a secondary KPI
- Gap-fill rate tracks the loop's effectiveness over time
