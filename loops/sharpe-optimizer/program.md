# Sharpe Optimizer — Single-Metric Strategy Refinement Loop

**Goal:** Maximize the Sharpe ratio of existing strategies that are close to the promotion
threshold but haven't passed walk-forward (WF) validation. This loop takes "near-miss"
strategies from the registry, re-runs them through the refinement pipeline with targeted
optimization parameters, and tracks improvements across runs.

**Why this exists:** CrabQuant has 118+ strategies in the registry, but many have decent
backtest Sharpe ratios (0.8-1.5) that failed walk-forward validation. These strategies
are "almost good enough" — they likely need parameter tuning, indicator swaps, or minor
structural changes to cross the promotion threshold. This loop systematically targets
these near-misses and applies escalating optimization treatments until they either
pass validation or are conclusively discarded.

**Primary metric:** Sharpe ratio improvement from original to optimized.
Secondary metrics: walk-forward windows passed, walk-forward avg test Sharpe, trade count.

**Key difference from diversity-explorer:** The diversity-explorer loop INVENTS new
strategies to fill coverage gaps. This loop IMPROVES existing strategies. It feeds
the original strategy code into the refinement pipeline as a starting point, rather
than asking the LLM to invent from scratch.

**Time budget:** ~15 minutes per mandate (refinement pipeline). Target 5-10 optimizations
per session. Total session budget: ~2 hours.

---

## Setup

Before starting the loop, complete these steps exactly once:

### 1. Read the codebase for full context

- `loops/sandbox.py` — shared helpers (load_registry, run_mandate, evaluate_result,
  git_commit, compute_quality_score, diversity_report, etc.)
- `loops/sharpe-optimizer/params.yaml` — optimization parameter space definition
- `loops/sharpe-optimizer/optimization_log.tsv` — log of all optimization attempts
- `crabquant/refinement/config.py` — RefinementConfig fields (sharpe_target, max_turns,
  cross_run_learning, parallel_invention, feature_importance, adaptive_sharpe_target, etc.)
- `strategies/production/registry.json` — the production strategy registry (read-only)
- `scripts/crabquant_cron.py` — the cron entry point that runs mandates
- `refinement/mandates/` — example mandate JSON files for format reference

### 2. Verify the project environment

```bash
cd /home/Zev/development/CrabQuant
source .venv/bin/activate
python -c "import loops.sandbox; print('sandbox OK')"
```

### 3. Scan the registry for optimization candidates

```python
import sys
sys.path.insert(0, "/home/Zev/development/CrabQuant")
from loops.sandbox import load_registry

registry = load_registry()
print(f"Total strategies in registry: {len(registry)}")
```

### 4. Identify near-miss strategies

A "near-miss" is a strategy that:
- Has a backtest Sharpe between 0.8 and 2.0 (decent but not elite)
- Either failed walk-forward validation (walk_forward_robust != True), OR
- Has a low walk-forward test Sharpe (< 0.8), OR
- Has fewer than 3 WF windows passed
- Is NOT already tagged as "ROBUST" in the verdict (those are already promoted)

```python
candidates = []
for entry in registry:
    name = entry.get("strategy_name", "")
    ticker = entry.get("ticker", "")
    sharpe = entry.get("sharpe", 0) or 0
    verdict = entry.get("verdict", "")
    wf_robust = entry.get("walk_forward_robust", False)
    wf_test_sharpe = entry.get("walk_forward_test_sharpe", 0) or 0
    trades = entry.get("trades", 0) or 0

    # Skip strategies already marked ROBUST
    if verdict == "ROBUST" and wf_robust:
        continue

    # Focus on strategies with meaningful but insufficient Sharpe
    if 0.5 <= sharpe < 2.0 and trades >= 3:
        # Score potential: higher original Sharpe = closer to promotion
        potential = sharpe  # Simple: closest to 1.5 target first
        candidates.append({
            "entry": entry,
            "strategy_name": name,
            "ticker": ticker,
            "original_sharpe": sharpe,
            "wf_robust": wf_robust,
            "wf_test_sharpe": wf_test_sharpe,
            "trades": trades,
            "potential": potential,
        })

# Sort by potential (highest Sharpe first — these are closest to promotion)
candidates.sort(key=lambda c: c["potential"], reverse=True)

print(f"Found {len(candidates)} optimization candidates")
for i, c in enumerate(candidates[:10]):
    print(f"  {i+1}. {c['strategy_name']} ({c['ticker']}) "
          f"Sharpe={c['original_sharpe']:.2f} "
          f"WF_robust={c['wf_robust']} WF_sharpe={c['wf_test_sharpe']:.2f}")
```

### 5. Load previous optimization log

```python
import csv
from pathlib import Path

log_path = Path("/home/Zev/development/CrabQuant/loops/sharpe-optimizer/optimization_log.tsv")
previous_attempts = {}

if log_path.exists() and log_path.stat().st_size > 0:
    with open(log_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["strategy_name"], row["ticker"])
            if key not in previous_attempts:
                previous_attempts[key] = []
            previous_attempts[key].append(row)

print(f"Previous optimization attempts: {len(previous_attempts)} strategies")
```

### 6. Confirm and go

Once candidates are identified and previous attempts are loaded, start the loop.

---

## Sandbox — Files You CAN Create/Modify

Only modify these files. Everything else is the judge and must not change:

| File | What you can change |
|------|-------------------|
| `loops/sharpe-optimizer/mandates/*.json` | Create new mandate files for the pipeline |
| `loops/sharpe-optimizer/optimization_log.tsv` | Append optimization records |
| `loops/sharpe-optimizer/config.json` | Loop configuration (max mandates, quality floor, etc.) |

## Judge — Files You CANNOT Modify

- `crabquant/` — all source code under this directory
- `scripts/` — entry points and cron scripts
- `strategies/` — strategy code and registry
- `results/` — backtest results, sweep results, winners
- `tests/` — test suite
- `loops/sandbox.py` — helper utilities
- `loops/sharpe-optimizer/program.md` — this file
- `loops/sharpe-optimizer/params.yaml` — parameter space definition

---

## Rules

1. **Sharpe improvement is the primary objective.** Every mandate should aim to push
   the strategy's Sharpe ratio higher than its current level, targeting the promotion
   threshold (Sharpe >= 1.0 + walk-forward robustness).
2. **Quality floor is sacred.** Every optimized strategy must meet: Sharpe >= original,
   rolling WF >= 3/6 windows passed, walk_forward_robust=True. If optimization can't
   beat the original while passing WF, it's a failed attempt.
3. **Don't write strategy code.** This loop creates mandates — JSON instructions for
   the existing refinement pipeline. The LLM-driven pipeline generates the actual
   strategy modifications.
4. **Use the existing infrastructure.** `crabquant_cron.py` runs mandates. The refinement
   pipeline handles code generation, backtesting, and promotion. Don't reinvent anything.
5. **Escalate treatments.** Start conservative (few turns, low temperature), then
   escalate if the strategy doesn't improve. Use params.yaml treatment sequences.
6. **Track everything in optimization_log.tsv.** Every optimization attempt must be
   logged, whether it succeeds or fails. This prevents re-attempting the same treatment
   on the same strategy.
7. **Don't spam the same strategy.** Maximum 4 attempts per strategy per session.
   If it fails 4 times, move on — the strategy may be fundamentally limited.
8. **Respect the API budget.** Each mandate costs LLM API calls. Don't waste calls on
   strategies with very low potential (Sharpe < 0.5).
9. **Anti-patterns:**
   - Don't create mandates for strategies already marked ROBUST
   - Don't optimize for Sharpe alone (walk-forward robustness is the real gate)
   - Don't skip the optimization log (every attempt must be recorded)
   - Don't modify crabquant/ source code, backtest engine, or validation logic
   - Don't create more than 4 mandates for the same strategy before moving on
   - Don't use sharpe_target below 1.0 for optimization (the pipeline's default floor)

---

## Candidate Scoring and Prioritization

### Candidate Score Formula

```
score = original_sharpe * (1 + 0.3 * has_source_code) * (1 + 0.2 * not_previously_attempted)
```

Where:
- `original_sharpe`: The strategy's backtest Sharpe (higher = closer to promotion)
- `has_source_code`: 1 if the strategy has accessible Python source, 0.5 otherwise
- `not_previously_attempted`: 1 if never optimized before, 0.5 if attempted once, 0 if attempted 3+ times

### Prioritization Order

1. **Strategies with Sharpe 1.0-1.5 that failed WF** — highest priority, closest to promotion
2. **Strategies with Sharpe 0.8-1.0 that failed WF** — good potential with moderate optimization
3. **Strategies with Sharpe 1.5-2.0 that failed WF** — already good but need WF robustness
4. **Strategies with Sharpe 0.5-0.8 that failed WF** — lower priority, bigger gap to close
5. **Strategies never backtested** — skip, not enough data to optimize

### Skip Conditions

Skip a candidate if ANY of these are true:
- Already ROBUST in the registry (verdict == "ROBUST" and walk_forward_robust == True)
- Original Sharpe < 0.5 (too far from promotion to be worth the API cost)
- Already attempted 4+ times this session
- Already attempted 3+ times total and never improved
- No source code available and no refinement run to build on

---

## The Optimization Loop

LOOP (until max mandates reached or no candidates remain):

### Step 1: Select the next candidate

```python
# Filter candidates
eligible = [
    c for c in candidates
    if previous_attempts.get((c["strategy_name"], c["ticker"]), [])
    and len(previous_attempts[(c["strategy_name"], c["ticker"])]) < 4
]

if not eligible:
    print("No more eligible candidates. Stopping.")
    break

# Pick the highest-potential candidate
target = eligible[0]
print(f"Target: {target['strategy_name']} ({target['ticker']}) "
      f"original Sharpe={target['original_sharpe']:.2f}")

# Count previous attempts for this strategy
attempt_num = len(previous_attempts.get(
    (target["strategy_name"], target["ticker"]), []
)) + 1
print(f"This is attempt #{attempt_num} for this strategy")
```

### Step 2: Design the optimization treatment

Based on the attempt number and the strategy's characteristics, select parameters
from `params.yaml`. Follow the treatment escalation sequence:

```python
def design_treatment(attempt_num, original_sharpe, strategy_name):
    """Select optimization parameters based on attempt number."""

    # Determine target Sharpe: always aim higher than original
    if attempt_num == 1:
        target_sharpe = max(original_sharpe + 0.5, 1.5)
        max_turns = 7
        temperature = 0.7
        archetype = None  # Keep original structure
        cross_run = True
        parallel = False
        feat_imp = True
        adaptive = False
        param_opt = True
        multi_ticker = False
    elif attempt_num == 2:
        target_sharpe = max(original_sharpe + 1.0, 2.0)
        max_turns = 10
        temperature = 0.7
        # Try a different archetype based on the original
        archetype = suggest_alternative_archetype(strategy_name)
        cross_run = True
        parallel = True
        feat_imp = True
        adaptive = True
        param_opt = True
        multi_ticker = False
    elif attempt_num == 3:
        target_sharpe = max(original_sharpe + 1.5, 2.5)
        max_turns = 15
        temperature = 0.9
        archetype = suggest_alternative_archetype(strategy_name, attempt=3)
        cross_run = True
        parallel = True
        feat_imp = True
        adaptive = True
        param_opt = True
        multi_ticker = True
    else:  # attempt 4+: nuclear
        target_sharpe = 3.0
        max_turns = 15
        temperature = 0.9
        archetype = suggest_alternative_archetype(strategy_name, attempt=4)
        cross_run = True
        parallel = True
        feat_imp = True
        adaptive = True
        param_opt = True
        multi_ticker = True

    return {
        "sharpe_target": target_sharpe,
        "max_turns": max_turns,
        "temperature": temperature,
        "strategy_archetype": archetype,
        "cross_run_learning": cross_run,
        "parallel_invention": parallel,
        "feature_importance": feat_imp,
        "adaptive_sharpe_target": adaptive,
        "param_optimization": param_opt,
        "multi_ticker_backtest": multi_ticker,
    }


def suggest_alternative_archetype(strategy_name, attempt=2):
    """Suggest a different archetype from the original strategy."""
    name_lower = strategy_name.lower()

    # Infer original archetype
    original = "momentum"  # default
    if any(kw in name_lower for kw in ["mean_rev", "reversion", "rsi", "bollinger"]):
        original = "mean_reversion"
    elif any(kw in name_lower for kw in ["breakout", "channel", "donchian"]):
        original = "breakout"
    elif any(kw in name_lower for kw in ["volatility", "vol_", "squeeze", "atr"]):
        original = "volatility"

    # Rotate through alternatives
    archetypes = ["momentum", "mean_reversion", "breakout", "volatility"]
    idx = archetypes.index(original) if original in archetypes else 0
    # Shift by attempt number to get different alternatives each time
    new_idx = (idx + attempt) % len(archetypes)
    return archetypes[new_idx]
```

### Step 3: Create the optimization mandate

Build a mandate JSON that instructs the refinement pipeline to improve the target strategy:

```python
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

def create_optimization_mandate(candidate, treatment, attempt_num):
    """Create a mandate JSON for optimizing an existing strategy."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    mandate_name = f"sharpe_opt_{candidate['strategy_name']}_{candidate['ticker'].lower()}_{timestamp}"

    mandate = {
        # ── Identity ──────────────────────────────────────────
        "name": f"Sharpe Optimization: {candidate['strategy_name']} ({candidate['ticker']}) "
                f"attempt #{attempt_num}",
        "description": (
            f"Optimize existing strategy '{candidate['strategy_name']}' on {candidate['ticker']}. "
            f"Original backtest Sharpe: {candidate['original_sharpe']:.2f}. "
            f"Target Sharpe: {treatment['sharpe_target']:.1f}. "
            f"Attempt #{attempt_num} with treatment: "
            f"turns={treatment['max_turns']}, temp={treatment['temperature']}, "
            f"archetype={'keep' if treatment['strategy_archetype'] is None else treatment['strategy_archetype']}. "
            f"Created by sharpe-optimizer loop."
        ),

        # ── Ticker and Period ─────────────────────────────────
        "tickers": [candidate["ticker"]],
        "primary_ticker": candidate["ticker"],
        "period": "2y",

        # ── Strategy Guidance ─────────────────────────────────
        "strategy_archetype": treatment["strategy_archetype"],
        "base_strategy": candidate["strategy_name"],  # Hint: start from this strategy
        "optimization_mode": True,  # Signal: this is an optimization, not invention
        "original_sharpe": candidate["original_sharpe"],

        # ── Refinement Parameters ─────────────────────────────
        "sharpe_target": treatment["sharpe_target"],
        "max_turns": treatment["max_turns"],

        # ── Constraints ───────────────────────────────────────
        "constraints": {
            "min_trades": 5,
            "max_drawdown_pct": 25.0,
        },

        # ── Accelerators (map to RefinementConfig fields) ─────
        "cross_run_learning": treatment["cross_run_learning"],
        "parallel_invention": treatment["parallel_invention"],
        "feature_importance": treatment["feature_importance"],
        "adaptive_sharpe_target": treatment["adaptive_sharpe_target"],
        "param_optimization": treatment["param_optimization"],
        "multi_ticker_backtest": treatment["multi_ticker_backtest"],

        # ── Backtest Config ───────────────────────────────────
        "backtest_config": {
            "start_date": "2023-01-01",
            "initial_capital": 100000,
            "commission": 0.001,
            "slippage": 0.0001,
        },

        # ── Loop Metadata ─────────────────────────────────────
        "sharpe_optimizer_source": "sharpe-optimizer",
        "sharpe_optimizer_attempt": attempt_num,
        "sharpe_optimizer_original_sharpe": candidate["original_sharpe"],
        "sharpe_optimizer_treatment": str(treatment),
    }

    # Write to the sharpe-optimizer mandates directory
    mandates_dir = Path("/home/Zev/development/CrabQuant/loops/sharpe-optimizer/mandates")
    mandates_dir.mkdir(parents=True, exist_ok=True)
    mandate_path = mandates_dir / f"{mandate_name}.json"
    mandate_path.write_text(json.dumps(mandate, indent=2))

    return mandate_path, mandate_name
```

### Step 4: Run the mandate via the refinement pipeline

Copy the mandate to the main mandates directory and run the cron script:

```python
from loops.sandbox import run_mandate
import shutil

# Copy to main mandates directory so crabquant_cron.py picks it up
main_mandates_dir = Path("/home/Zev/development/CrabQuant/refinement/mandates")
main_mandate_path = main_mandates_dir / mandate_path.name
shutil.copy(str(mandate_path), str(main_mandate_path))

# Run the pipeline (15 minute timeout)
result = run_mandate(str(main_mandate_path), timeout=900)
print(f"Pipeline result: success={result['success']}, duration={result['duration']:.0f}s")

if not result["success"]:
    print(f"Error: {result['stderr'][-500:]}")
```

### Step 5: Parse and evaluate the results

After the pipeline completes, find the run directory and parse state.json:

```python
from loops.sandbox import evaluate_result
import glob
import os

# Find the most recent run directory
run_dirs = sorted(
    glob.glob("/home/Zev/development/CrabQuant/refinement_runs/*"),
    key=os.path.getmtime,
    reverse=True,
)

eval_result = {
    "run_dir": "",
    "found": False,
    "status": "unknown",
    "best_sharpe": 0,
    "best_score": 0,
    "converged": False,
    "total_turns": 0,
    "strategy_name": "",
    "failure_mode": "",
    "wf_windows_passed": "",
    "wf_avg_sharpe": 0,
    "result_trades": 0,
}

if run_dirs:
    eval_result = evaluate_result(run_dirs[0])
    print(f"Run evaluation: {eval_result}")

    # Try to extract walk-forward details from the run directory
    run_dir = Path(run_dirs[0])

    # Check for walk-forward results in state.json
    state_path = run_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        eval_result["total_turns"] = state.get("current_turn", 0)

        # Extract WF details from history
        history = state.get("history", [])
        for entry in history:
            if entry.get("walk_forward"):
                wf = entry["walk_forward"]
                eval_result["wf_windows_passed"] = str(
                    wf.get("windows_passed", 0)
                )
                eval_result["wf_avg_sharpe"] = wf.get("avg_test_sharpe", 0)

    # Also check report files
    for report_file in run_dir.glob("*.md"):
        content = report_file.read_text()
        # Extract Sharpe from report
        import re
        sharpe_match = re.search(r"Sharpe[:\s]+([\d.]+)", content)
        if sharpe_match:
            eval_result["best_sharpe"] = float(sharpe_match.group(1))
        # Extract WF windows
        wf_match = re.search(r"(\d+)/(\d+)\s*windows?\s*passed", content)
        if wf_match:
            eval_result["wf_windows_passed"] = f"{wf_match.group(1)}/{wf_match.group(2)}"
        # Extract trades
        trades_match = re.search(r"(\d+)\s*trades?", content)
        if trades_match:
            eval_result["result_trades"] = int(trades_match.group(1))
```

### Step 6: Classify the result and log it

Determine the outcome based on improvement and walk-forward status:

```python
import csv
from datetime import datetime, timezone

def classify_result(eval_result, original_sharpe, target_sharpe):
    """Classify the optimization result into a status category."""

    result_sharpe = eval_result.get("best_sharpe", 0)
    improved = result_sharpe > original_sharpe
    wf_passed = eval_result.get("wf_windows_passed", "")
    wf_windows = int(wf_passed.split("/")[0]) if "/" in wf_passed else 0
    wf_total = int(wf_passed.split("/")[1]) if "/" in wf_passed else 0
    converged = eval_result.get("converged", False)

    # Decision tree
    if converged and wf_windows >= 3 and improved:
        return "promoted", (
            f"PROMOTED: Sharpe {original_sharpe:.2f} -> {result_sharpe:.2f}, "
            f"WF {wf_passed}"
        )
    elif improved and wf_windows >= 3:
        return "improved", (
            f"Improved Sharpe {original_sharpe:.2f} -> {result_sharpe:.2f}, "
            f"WF {wf_passed}, but convergence unclear"
        )
    elif improved:
        return "improved", (
            f"Improved Sharpe {original_sharpe:.2f} -> {result_sharpe:.2f}, "
            f"but WF {wf_passed} (< 3 windows)"
        )
    elif result_sharpe >= original_sharpe * 0.9:
        return "neutral", (
            f"No significant change: Sharpe {original_sharpe:.2f} -> {result_sharpe:.2f}"
        )
    else:
        return "discard", (
            f"Regressed: Sharpe {original_sharpe:.2f} -> {result_sharpe:.2f}"
        )


status, notes = classify_result(eval_result, target["original_sharpe"], treatment["sharpe_target"])

# Log to TSV
log_path = Path("/home/Zev/development/CrabQuant/loops/sharpe-optimizer/optimization_log.tsv")

timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
row = {
    "timestamp": timestamp,
    "strategy_name": target["strategy_name"],
    "original_sharpe": f"{target['original_sharpe']:.4f}",
    "target_sharpe": f"{treatment['sharpe_target']:.1f}",
    "max_turns": str(treatment["max_turns"]),
    "result_sharpe": f"{eval_result.get('best_sharpe', 0):.4f}",
    "result_trades": str(eval_result.get("result_trades", 0)),
    "wf_windows_passed": eval_result.get("wf_windows_passed", ""),
    "wf_avg_sharpe": f"{eval_result.get('wf_avg_sharpe', 0):.4f}",
    "status": status,
    "notes": notes,
}

# Write header if file doesn't exist or is empty
write_header = not log_path.exists() or log_path.stat().st_size == 0

with open(log_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys(), delimiter="\t")
    if write_header:
        writer.writeheader()
    writer.writerow(row)

print(f"Result: {status} — {notes}")
```

### Step 7: Update tracking and decide next action

```python
# Update previous attempts
key = (target["strategy_name"], target["ticker"])
if key not in previous_attempts:
    previous_attempts[key] = []
previous_attempts[key].append(row)

# Update the candidate's potential score
if status == "promoted":
    # Remove from candidates — it's been promoted
    candidates = [c for c in candidates if c["strategy_name"] != target["strategy_name"]
                  or c["ticker"] != target["ticker"]]
    print(f"Strategy PROMOTED! Removed from candidate list.")
elif status == "discard" and attempt_num >= 4:
    # Max attempts reached, remove
    candidates = [c for c in candidates if c["strategy_name"] != target["strategy_name"]
                  or c["ticker"] != target["ticker"]]
    print(f"Strategy discarded after {attempt_num} attempts.")
elif status == "improved":
    # Keep in candidates but update Sharpe
    for c in candidates:
        if c["strategy_name"] == target["strategy_name"] and c["ticker"] == target["ticker"]:
            c["original_sharpe"] = eval_result.get("best_sharpe", c["original_sharpe"])
            break
    print(f"Strategy improved, will retry with higher target.")
```

### Step 8: Clean up the mandate file

```python
# Remove the mandate from the main directory to prevent re-running
if main_mandate_path.exists():
    main_mandate_path.unlink()
```

### Step 9: Repeat

Go back to Step 1. Loop until:
- No more eligible candidates remain, OR
- Max mandates per session reached (default: 10), OR
- 5 consecutive discards without any improvement

---

## Walk-Forward Validation Requirements

For a strategy to be considered "promoted" by this loop, it must meet ALL of these:

1. **Backtest Sharpe >= sharpe_target** (the target set in the mandate)
2. **Rolling walk-forward robust** — >= 3 out of 6 windows passing
3. **Walk-forward avg test Sharpe > 0** — not a data artifact
4. **Walk-forward test Sharpe >= 0.4** — CrabQuant's minimum from VALIDATION_CONFIG
5. **At least 5 trades** in the backtest period
6. **Max drawdown < 25%** — risk management gate

The validation thresholds are defined in `crabquant/refinement/config.py` under
`VALIDATION_CONFIG` and are NOT modifiable by this loop.

---

## Treatment Escalation Strategy

### Attempt 1: Conservative Optimization (default)

Goal: Nudge the existing strategy structure slightly higher.
- Sharpe target: original + 0.5 (min 1.5)
- 7 turns, temperature 0.7
- Keep original archetype (no strategy_archetype override)
- Enable cross_run_learning and feature_importance
- Disable parallel_invention and adaptive_sharpe_target
- Enable param_optimization (let scipy DE tune parameters)

Rationale: Most near-miss strategies just need minor tweaks. The param optimizer
alone can often push Sharpe up by 0.2-0.5 without any LLM turns.

### Attempt 2: Structural Exploration

Goal: Try a fundamentally different approach to the same ticker.
- Sharpe target: original + 1.0 (min 2.0)
- 10 turns, temperature 0.7
- Switch to a DIFFERENT archetype (momentum → mean_reversion, etc.)
- Enable parallel_invention (explore 3 variants on turn 1)
- Enable adaptive_sharpe_target (start easy, ramp up)

Rationale: If the original structure can't reach the target, a different
architectural approach might. Parallel invention gives the LLM multiple shots.

### Attempt 3: Aggressive Exploration

Goal: Push hard with maximum refinement power.
- Sharpe target: original + 1.5 (min 2.5)
- 15 turns, temperature 0.9
- Try yet another archetype
- Enable ALL accelerators
- Enable multi_ticker_backtest (catch overfitting)

Rationale: At this point, the strategy needs significant rework. Higher
temperature encourages more creative solutions. Multi-ticker testing
ensures the strategy isn't just memorizing one ticker's patterns.

### Attempt 4: Nuclear Option

Goal: Last resort — maximum exploration budget.
- Sharpe target: 3.0
- 15 turns, temperature 0.9
- Most different archetype from original
- All accelerators enabled
- If this fails, discard the strategy

---

## Handling Stagnation

### After 3 consecutive discards (different strategies):

1. Review optimization_log.tsv for patterns — are certain archetypes consistently failing?
2. Check if the refinement pipeline is having systemic issues (look at stderr output)
3. Try lowering the Sharpe target incrementally (1.5 → 1.2 → 1.0)
4. Consider that the near-miss pool may be exhausted

### After 5 consecutive discards:

1. Print a summary of all attempts this session
2. Check if any strategies were improved but not promoted (status=improved)
3. For improved strategies, consider if they'd benefit from one more attempt
4. Stop the loop and report findings

### Pipeline-level stagnation:

CrabQuant's refinement pipeline has built-in stagnation detection (see
`crabquant/refinement/config.py`: stagnation_abandon_threshold, stagnation_nuclear_threshold).
If the pipeline triggers stagnation recovery, it will automatically:
- Pivot to a different approach after 70% of turns with no improvement
- Nuclear restart after 60% degradation from best
- Broaden the search space after 50% stagnation

This loop should NOT override these mechanisms. Trust the pipeline's internal
stagnation handling.

---

## Mandate JSON Field Reference

When creating mandates, use these fields (all optional — defaults from RefinementConfig apply):

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Human-readable mandate name |
| `description` | string | Detailed description for the LLM |
| `tickers` | list[str] | Tickers to trade (e.g., ["AAPL"]) |
| `primary_ticker` | string | Main ticker for backtesting |

### Optimization Fields (Sharpe Optimizer Specific)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sharpe_target` | float | 1.5 | Sharpe ratio to target |
| `max_turns` | int | 7 | Maximum LLM refinement iterations |
| `strategy_archetype` | string/null | null | Force a specific structural approach |
| `base_strategy` | string | null | Name of strategy to improve (hint for LLM) |
| `optimization_mode` | bool | false | Signal this is optimization, not invention |

### RefinementConfig Accelerator Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cross_run_learning` | bool | true | Feed winners into LLM context |
| `parallel_invention` | bool | false | Spawn N strategies in parallel on turn 1 |
| `parallel_invention_count` | int | 3 | How many parallel strategies to spawn |
| `feature_importance` | bool | true | Analyze indicator contributions |
| `param_optimization` | bool | true | scipy DE parameter sweep after each turn |
| `adaptive_sharpe_target` | bool | false | Ramp Sharpe target over early turns |
| `adaptive_start_factor` | float | 0.5 | Initial target multiplier |
| `adaptive_ramp_turns` | int | 3 | Turns to ramp to full target |
| `multi_ticker_backtest` | bool | false | Test on multiple tickers during refinement |
| `soft_promote` | bool | false | Promote "good enough" strategies |

### Constraint Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `constraints.min_trades` | int | 5 | Minimum trades for a valid backtest |
| `constraints.max_drawdown_pct` | float | 25.0 | Maximum allowed drawdown percentage |

### Backtest Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backtest_config.start_date` | string | "2023-01-01" | Backtest start date |
| `backtest_config.initial_capital` | int | 100000 | Starting capital |
| `backtest_config.commission` | float | 0.001 | Per-trade commission |
| `backtest_config.slippage` | float | 0.0001 | Per-trade slippage |

---

## Output Files

### optimization_log.tsv
Tab-separated log of all optimization attempts:
```
timestamp	strategy_name	original_sharpe	target_sharpe	max_turns	result_sharpe	result_trades	wf_windows_passed	wf_avg_sharpe	status	notes
```

### mandates/*.json
Mandate files created for each optimization attempt.
Naming: `sharpe_opt_{strategy_name}_{ticker}_{timestamp}.json`

### config.json (optional)
Loop configuration:
```json
{
  "max_mandates_per_session": 10,
  "max_attempts_per_strategy": 4,
  "min_sharpe_candidate": 0.5,
  "max_sharpe_candidate": 2.0,
  "max_consecutive_discards": 5,
  "treatment_sequence": ["conservative", "structural", "aggressive", "nuclear"]
}
```

---

## Success Criteria

A successful session should:
1. **Promote at least 1-2 strategies** — push near-miss strategies through WF validation
2. **Improve at least 3-5 strategies** — measurably higher Sharpe even if not promoted
3. **Not regress any existing strategies** — optimization only, never degradation
4. **Log all attempts** — optimization_log.tsv has complete records for every mandate
5. **Stay within budget** — no more than 10 mandates per session

Long-term targets (after multiple sessions):
- All strategies with Sharpe >= 1.0 should pass walk-forward validation
- The portfolio's average Sharpe should increase by 0.2-0.5 over baseline
- The "near-miss" pool (Sharpe 0.8-1.5, failed WF) should be depleted

---

## Example Session Flow

```
Session Start
├── Load registry (118 strategies)
├── Identify near-miss candidates (23 strategies with Sharpe 0.5-2.0, failed WF)
├── Load previous optimization log (12 past attempts, 2 promoted)
├── Filter: 15 eligible candidates (not yet attempted 4x)
│
├── Loop iteration 1
│   ├── Target: refined_momentum_spy (SPY) original Sharpe=1.32, attempt #1
│   ├── Treatment: conservative (target 1.5, 7 turns, temp 0.7)
│   ├── Create mandate, run pipeline (11 min)
│   ├── Result: Sharpe 1.32 → 1.67, WF 4/6, converged
│   ├── Status: PROMOTED
│   └── Log + remove from candidates
│
├── Loop iteration 2
│   ├── Target: rsi_bollinger_aapl (AAPL) original Sharpe=0.92, attempt #1
│   ├── Treatment: conservative (target 1.5, 7 turns, temp 0.7)
│   ├── Create mandate, run pipeline (13 min)
│   ├── Result: Sharpe 0.92 → 1.15, WF 2/6, not converged
│   ├── Status: improved (better but WF still failing)
│   └── Log + update candidate Sharpe to 1.15
│
├── Loop iteration 3
│   ├── Target: rsi_bollinger_aapl (AAPL) updated Sharpe=1.15, attempt #2
│   ├── Treatment: structural (target 2.0, 10 turns, temp 0.7, archetype=breakout)
│   ├── Create mandate, run pipeline (14 min)
│   ├── Result: Sharpe 1.15 → 1.43, WF 3/6, converged
│   ├── Status: improved (WF passing but convergence unclear)
│   └── Log + will retry
│
├── Loop iteration 4
│   ├── Target: vol_squeeze_gld (GLD) original Sharpe=0.78, attempt #1
│   ├── Treatment: conservative (target 1.5, 7 turns, temp 0.7)
│   ├── Create mandate, run pipeline (10 min)
│   ├── Result: Sharpe 0.78 → 0.82, WF 1/6, not converged
│   ├── Status: neutral (no significant change)
│   └── Log + keep in candidates
│
├── ... continue until max mandates or candidates exhausted
│
└── Session End
    ├── Summary: 1 promoted, 2 improved, 1 neutral, 0 discarded
    ├── Total mandates: 10
    ├── Total time: ~2 hours
    └── Commit: "sharpe-optimizer: promoted 1, improved 2 strategies"
```

---

## Integration with Existing CrabQuant Infrastructure

### Refinement Pipeline
- Mandates created here use the same JSON format as `refinement/mandates/`
- The pipeline (crabquant_cron.py) processes them identically
- All quality gates, walk-forward validation, and promotion logic apply unchanged
- The `optimization_mode: true` flag is a hint for future pipeline enhancements

### Parameter Optimizer
- The scipy differential evolution optimizer (`crabquant/refinement/param_optimizer.py`)
  runs automatically when `param_optimization: true` in the mandate
- It sweeps nearby parameter combinations and finds better settings without LLM turns
- This is extremely effective for strategies that are structurally sound but poorly tuned

### Stagnation Detection
- The pipeline's built-in stagnation system (`crabquant/refinement/stagnation.py`)
  handles intra-run stagnation (no improvement across turns)
- This loop handles inter-run stagnation (no improvement across separate mandates)
- Both systems complement each other

### Feature Importance
- When `feature_importance: true`, the pipeline analyzes which indicators contribute
  to returns and feeds this back to the LLM
- This helps the LLM remove harmful indicators and double down on useful ones
- Particularly effective for strategies with many indicators where some may be noise

### Walk-Forward Validation
- Uses the same rolling walk-forward (6 windows, min_avg_test_sharpe=0.4, min_windows=3)
- This loop does NOT change validation thresholds
- Strategies must pass WF to be considered "promoted"

### Cross-Run Learning
- When `cross_run_learning: true`, the pipeline feeds proven winners from
  `results/winners/winners.json` into the LLM context
- This helps the LLM learn patterns from successful strategies across runs
- Very effective for later optimization attempts on the same strategy

---

## Comparison with Other Loops

| Aspect | diversity-explorer | sharpe-optimizer |
|--------|-------------------|-----------------|
| Goal | Fill coverage gaps | Improve Sharpe ratio |
| Strategy source | Invent from scratch | Improve existing |
| Primary metric | QD-Score (quality × novelty) | Sharpe improvement |
| Candidate pool | Empty archetype-ticker cells | Near-miss strategies |
| Treatment | One mandate per gap | Escalating treatments |
| Success criterion | Fill an empty cell | Pass WF validation |
| Re-attempt policy | 3 attempts per gap | 4 attempts per strategy |
| Typical outcome | New strategy in portfolio | Better version of existing |
