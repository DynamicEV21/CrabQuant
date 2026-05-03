# Agent Loops

Autonomous experiment loops for CrabQuant, inspired by [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch).

Each loop is a self-contained experiment directory with everything an AI agent needs to iterate autonomously: instructions, constraints, evaluation criteria, and logging.

## How It Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  program.md  │────▶│  Agent runs  │────▶│  Metric check│
│  (the boss)  │     │  experiment  │     │  (the judge) │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
                            │                     │
                     ┌──────▼───────┐     ┌──────▼───────┐
                     │  Modified?   │     │  Improved?   │
                     │  strategy    │     │  vs baseline │
                     └──────────────┘     └──────┬───────┘
                                                 │
                                          ┌──────▼───────┐
                                          │ KEEP: commit │
                                          │ DISCARD:     │
                                          │ git revert   │
                                          └──────────────┘
```

## Architecture

```
loops/
  README.md                    ← You are here
  registry.yaml                ← Loop catalog with metadata
  sandbox.py                   ← Shared helpers (metrics, git, TSV logging, diversity)
  run.py                       ← CLI entry point for discovering and launching loops
  diversity-explorer/          ← Loop: quality-diversity optimization
    program.md                 ← Agent instructions (the "boss")
    feature_map.yaml           ← Feature-to-dimension mapping
    gaps_log.tsv               ← Experiment log (auto-created on first run)
  sharpe-optimizer/            ← Loop: Sharpe ratio optimization
    program.md                 ← Agent instructions
    params.yaml                ← Search parameters and bounds
    optimization_log.tsv       ← Experiment log
  meta-analyzer/               ← Loop: cross-strategy pattern mining
    program.md                 ← Agent instructions
    analysis_config.yaml       ← Analysis configuration
    insights_knowledge.tsv     ← Knowledge base
    analysis_report.md         ← Latest analysis report
```

### Three Roles (per loop)

- **Boss** — `program.md`: Human writes, agent reads. Defines goal, constraints, strategy hints.
- **Sandbox** — Files in the loop directory (configs, params, feature maps): Agent modifies these.
- **Judge** — Everything else (crabquant source code, backtest engine, validation logic): Read-only to agent.

### sandbox.py Helpers

`loops/sandbox.py` provides reusable functions any loop can import:

**Diversity explorer helpers:**
- `load_registry()` — Load the strategy registry JSON
- `load_sweep_results()` — Load sweep results from disk
- `load_winners()` — Load only winning (robust) strategies
- `compute_quality_score()` — Score a strategy by Sharpe, WF pass rate, etc.
- `extract_strategy_features()` — Extract archetype/ticker/regime features
- `compute_diversity_coverage()` — Compute coverage across feature dimensions
- `identify_gaps()` — Find under-explored regions of the feature space
- `create_mandate_for_gap()` — Generate a refinement mandate for a gap
- `run_mandate()` — Execute a mandate through the refinement pipeline
- `evaluate_result()` — Score a completed refinement run
- `diversity_report()` — Generate a human-readable coverage report
- `build_population_features()` — Build feature vectors for the population
- `compute_novelty_score()` — Novelty of a strategy vs the population
- `compute_qd_score()` — Composite quality-diversity score
- `log_gap_attempt()` — Append a row to the gaps log
- `load_gaps_log()` — Read all entries from the gaps log

**Git helpers:**
- `git_commit()` — Stage and commit modified files
- `git_current_commit()` — Short commit hash
- `git_revert()` — Checkout files to HEAD state
- `git_has_uncommitted_changes()` — Check working tree state

**Shared infrastructure helpers:**
- `list_loops()` — List all registered loops from registry.yaml
- `load_loop_registry()` — Load the loop registry.yaml
- `validate_loop()` — Validate a loop has required files
- `log_experiment()` — Append a row to any TSV experiment log
- `generate_report()` — Parse a TSV log and generate a summary report
- `parse_mandate()` — Read and validate a mandate JSON file
- `parse_run_state()` — Read state.json from a refinement run
- `safe_json_load()` — Load JSON with error handling
- `safe_json_save()` — Save JSON with error handling

## Active Loops

- **diversity-explorer** — Quality-diversity optimization across regimes, archetypes, and tickers. Builds a population of robust strategies and systematically fills coverage gaps in the feature space (archetype × ticker × regime × trade frequency × Sharpe tier). Uses QD-score (quality × diversity) to guide exploration toward under-explored regions.

- **sharpe-optimizer** — Iterative Sharpe optimization of near-miss strategies. Targets strategies that almost passed walk-forward validation (high Sharpe but marginal WF consistency) and iteratively refines their parameters to push them over the robustness threshold.

- **meta-analyzer** — Cross-strategy pattern mining and knowledge generation. Analyzes the full strategy registry to extract patterns, correlations, and actionable insights about what makes strategies succeed or fail across different market conditions.

## Running a Loop

### Via run.py (recommended)

```bash
cd ~/development/CrabQuant

# List all available loops
python loops/run.py list

# Prepare context for an agent to run a loop
python loops/run.py diversity-explorer

# Validate loop config without running
python loops/run.py diversity-explorer --dry-run

# View experiment stats
python loops/run.py diversity-explorer --report
```

This is **not** an autonomous runner — it prepares context (prints program.md) for an AI agent to read and execute the loop.

### Via Hermes Agent

```
delegate_task(
  goal="Read loops/diversity-explorer/program.md and run the experiment loop.",
  context="You are an autonomous quant researcher. Follow program.md exactly.",
  toolsets=["terminal", "file"],
)
```

### Via Cron (overnight)

```bash
# Run iterations overnight, deliver results to Telegram
hermes cron create '0 23 * * *' \
  --prompt "Read ~/development/CrabQuant/loops/diversity-explorer/program.md. Run iterations of the experiment loop. Report results." \
  --deliver telegram
```

## Adding a New Loop

1. Create `loops/<loop-name>/` directory
2. Create `loops/<loop-name>/program.md` with full agent instructions (goal, setup, sandbox, experiment loop, output format)
3. Create any config files the loop needs (e.g., `params.yaml`, `feature_map.yaml`)
4. Add an entry to `loops/registry.yaml` with metadata (description, metric, category, tsv_log)
5. Add any shared helper functions to `loops/sandbox.py` if needed
6. Verify with `python loops/run.py <loop-name> --dry-run`

### program.md Template

```markdown
# <Loop Name>

## Goal
<Single sentence: what metric to optimize, direction, constraints>

## Setup
1. Read these files for context: ...
2. Verify data exists: ...
3. Run baseline first: ...

## Sandbox (files you CAN modify)
- params.yaml
- config.yaml

## Judge (files you CANNOT modify)
- crabquant/*.py
- backtest engine

## Experiment Loop
1. Read current best metric
2. Propose a change
3. Run experiment: <command>
4. Extract metric: <how>
5. If improved → commit, log "keep"
6. If not → revert, log "discard"
7. NEVER STOP — loop forever until manually stopped

## Output Format
<What the experiment outputs, how to parse the metric>
```

### Design Patterns for Writing program.md

1. **One metric rule** — Always specify a single primary metric. Multiple objectives should be composed into one score.
2. **Explicit file ownership** — Clearly separate Sandbox (modifiable) from Judge (read-only).
3. **Concrete knobs** — List specific parameters with value ranges.
4. **Escape hatches** — Include "when stuck" strategies for after repeated failures.
5. **Time budgets** — Specify expected time per iteration and timeout thresholds.
6. **Anti-patterns** — Explicitly list things NOT to do.
7. **Reproducible setup** — Include exact commands with flags.
8. **Output format spec** — Show example output so agents know how to parse results.

## Loop Compositions

Loops can be chained for more complex workflows:

- **Discovery → Optimization**: diversity-explorer (find gaps) → sharpe-optimizer (refine near-misses)
- **Analysis → Discovery**: meta-analyzer (find patterns) → diversity-explorer (fill new gaps)
- **Full Pipeline**: meta-analyzer → diversity-explorer → sharpe-optimizer
