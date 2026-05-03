# Meta-Analyzer Loop — Program

> **Loop Identity**
> - **Name:** meta-analyzer
> - **Category:** analysis (READ-ONLY — no backtesting, no mandate creation, no strategy code modification)
> - **Metric:** insights_count (number of new actionable insights discovered)
> - **Direction:** maximize
> - **Estimated Time:** ~2 minutes per full analysis cycle
> - **Dependencies:** None (reads existing data only)
> - **Output:** `insights_knowledge.tsv` + `analysis_report.md`

---

## Goal

Analyze ALL historical strategy data across CrabQuant to discover patterns, identify what makes strategies succeed or fail, and generate actionable insights that feed into the other loops (diversity-explorer, sharpe-optimizer) and the human operator.

This loop is the **brain** of the loop system. It never runs backtests, never creates mandates, never modifies strategy code. It reads, analyzes, and reports.

---

## Setup

### Step 1: Load the Sandbox

```python
import sys
sys.path.insert(0, "/home/Zev/development/CrabQuant")
from loops.sandbox import (
    load_registry, load_sweep_results, load_winners,
    compute_quality_score, extract_strategy_features,
    safe_json_load, list_loops
)
```

### Step 2: Load All Data Sources

```python
registry = load_registry()          # list[dict] — 118 strategies from production registry
sweep_results = load_sweep_results()  # dict — 113 strategies with rolling WF window data
winners = load_winners()            # list[dict] — 181 winner entries across cycles

print(f"Registry: {len(registry)} strategies")
print(f"Sweep results: {len(sweep_results)} entries")
print(f"Winners: {len(winners)} entries")
```

### Step 3: Understand the Data Schemas

**Registry entry** (from `strategies/production/registry.json`):
- `name` (str): Strategy identifier
- `ticker` (str): Target ticker (e.g., "AAPL", "SPY")
- `archetype` (str): One of: momentum, mean_reversion, breakout, volatility, statistical_arb, multi_signal_ensemble, volatility_breakout
- `sharpe` (float): Discovery/backtest Sharpe ratio
- `walk_forward_robust` (bool): Whether strategy passed rolling WF validation
- `verdict` (str): Human or automated verdict
- `params` (dict): Strategy parameters (indicators, thresholds, etc.)
- `indicators_used` (list[str], optional): Names of technical indicators used
- `total_return` (float, optional): Total return percentage
- `max_drawdown` (float, optional): Maximum drawdown
- `num_trades` (int, optional): Number of trades
- `score` (float, optional): Composite quality score
- `discovered_cycle` (int, optional): Which discovery cycle found this

**Sweep results entry** (from `results/sweep_results_cycle13.json`):
- `strategy_name` (str): Strategy identifier
- `windows` (list[dict]): Rolling WF windows, each with:
  - `sharpe` (float): Test-window Sharpe
  - `total_return` (float): Test-window return
  - `max_drawdown` (float): Test-window drawdown
  - `num_trades` (int): Test-window trades
  - `score` (float): Quality score
  - `passed` (bool): Whether this window passed validation

**Winner entry** (from `results/winners/winners.json`):
- `name` (str): Strategy identifier
- `ticker` (str): Target ticker
- `archetype` (str): Strategy archetype
- `sharpe` (float): Discovery Sharpe
- `score` (float): Composite score
- `cycle` (int): Discovery cycle number

### Step 4: Validate Data Loaded Correctly

```python
assert len(registry) > 0, "Registry is empty!"
assert len(sweep_results) > 0, "Sweep results are empty!"
assert len(winners) > 0, "Winners are empty!"

# Spot-check fields
sample = registry[0]
assert "name" in sample and "ticker" in sample and "sharpe" in sample, \
    f"Registry entry missing fields: {list(sample.keys())}"
print("All data sources loaded and validated ✓")
```

---

## Analysis Suite

Run these analyses in order. Each produces a findings dict. At the end, all findings are compiled into the report and knowledge base.

### Analysis 1: Archetype Walk-Forward Pass Rates (HIGH PRIORITY)

**Question:** Which strategy archetypes have the highest WF pass rate? Are some archetypes fundamentally more robust than others?

```python
from collections import defaultdict

def analyze_archetype_wf_rates(registry):
    """Compute per-archetype WF pass rates and statistics."""
    archetype_data = defaultdict(lambda: {
        "total": 0, "wf_robust": 0, "sharpes": [], "trades": [],
        "tickers": defaultdict(int)
    })

    for entry in registry:
        arch = entry.get("archetype", "unknown")
        d = archetype_data[arch]
        d["total"] += 1
        if entry.get("walk_forward_robust", False):
            d["wf_robust"] += 1
        d["sharpes"].append(entry.get("sharpe", 0))
        d["trades"].append(entry.get("num_trades", 0))
        d["tickers"][entry.get("ticker", "unknown")] += 1

    results = {}
    for arch, d in archetype_data.items():
        results[arch] = {
            "total": d["total"],
            "wf_robust": d["wf_robust"],
            "wf_pass_rate": d["wf_robust"] / d["total"] if d["total"] > 0 else 0,
            "avg_sharpe": sum(d["sharpes"]) / len(d["sharpes"]) if d["sharpes"] else 0,
            "median_sharpe": sorted(d["sharpes"])[len(d["sharpes"])//2] if d["sharpes"] else 0,
            "avg_trades": sum(d["trades"]) / len(d["trades"]) if d["trades"] else 0,
            "top_ticker": max(d["tickers"], key=d["tickers"].get) if d["tickers"] else "N/A"
        }
    return results

archetype_rates = analyze_archetype_wf_rates(registry)
for arch, stats in sorted(archetype_rates.items(), key=lambda x: -x[1]["wf_pass_rate"]):
    print(f"  {arch:25s}  {stats['wf_robust']:3d}/{stats['total']:3d} "
          f"({stats['wf_pass_rate']*100:5.1f}% pass)  avg_sharpe={stats['avg_sharpe']:.3f}")
```

**What to look for:**
- Archetypes with <10% WF pass rate are fundamentally broken — insights for other loops
- Archetypes with >50% WF pass rate are "safe bets" — recommend for diversity-explorer
- Big gap between avg Sharpe and WF pass rate = overfitting-prone archetype

### Analysis 2: Ticker Sharpe Distribution (HIGH PRIORITY)

**Question:** Do some tickers offer more alpha opportunity than others?

```python
def analyze_ticker_performance(registry):
    """Per-ticker Sharpe distributions and success rates."""
    ticker_data = defaultdict(lambda: {
        "strategies": [], "wf_robust": 0, "archetypes": defaultdict(int)
    })

    for entry in registry:
        ticker = entry.get("ticker", "unknown")
        d = ticker_data[ticker]
        d["strategies"].append({
            "name": entry.get("name", ""),
            "sharpe": entry.get("sharpe", 0),
            "wf_robust": entry.get("walk_forward_robust", False),
            "archetype": entry.get("archetype", "unknown"),
            "score": entry.get("score", 0),
        })
        if entry.get("walk_forward_robust", False):
            d["wf_robust"] += 1
        d["archetypes"][entry.get("archetype", "unknown")] += 1

    results = {}
    for ticker, d in ticker_data.items():
        sharpes = sorted([s["sharpe"] for s in d["strategies"]])
        n = len(sharpes)
        results[ticker] = {
            "total": n,
            "wf_robust": d["wf_robust"],
            "wf_pass_rate": d["wf_robust"] / n if n > 0 else 0,
            "mean_sharpe": sum(sharpes) / n if n > 0 else 0,
            "median_sharpe": sharpes[n//2] if n > 0 else 0,
            "best_sharpe": sharpes[-1] if n > 0 else 0,
            "worst_sharpe": sharpes[0] if n > 0 else 0,
            "sharpe_spread": (sharpes[-1] - sharpes[0]) if n > 0 else 0,
            "top_archetype": max(d["archetypes"], key=d["archetypes"].get) if d["archetypes"] else "N/A"
        }
    return results

ticker_perf = analyze_ticker_performance(registry)
for ticker, stats in sorted(ticker_perf.items(), key=lambda x: -x[1]["wf_pass_rate"]):
    print(f"  {ticker:6s}  {stats['total']:3d} strats  {stats['wf_pass_rate']*100:5.1f}% pass  "
          f"median={stats['median_sharpe']:.2f}  best={stats['best_sharpe']:.2f}")
```

**What to look for:**
- Tickers with 0 strategies = coverage gap for diversity-explorer
- Tickers with high Sharpe spread = high variance, potentially exploitable
- Tickers with 0% WF pass rate despite decent Sharpe = overfitting problem

### Analysis 3: Indicator Family Success Rates (HIGH PRIORITY)

**Question:** Which technical indicator families produce the most robust strategies?

```python
def classify_indicator(indicator_name):
    """Classify an indicator into its family."""
    name = indicator_name.lower()
    families = {
        "momentum": ["ema", "sma", "wma", "macd", "roc", "tsi", "dpo", "moving_avg", "ma_"],
        "mean_reversion": ["rsi", "bbands", "bollinger", "stoch", "cci", "willr", "keltner"],
        "volatility": ["atr", "adx", "supertrend", "kelner", "true_range", "volatility"],
        "volume": ["obv", "vwap", "ad", "cmf", "mfi", "volume"],
        "trend": ["ichimoku", "dema", "tema", "psar", "parabolic"]
    }
    for family, keywords in families.items():
        for kw in keywords:
            if kw in name:
                return family
    return "other"

def analyze_indicator_effectiveness(registry):
    """Which indicator families lead to robust WF-passing strategies."""
    family_data = defaultdict(lambda: {
        "strategies": set(), "wf_robust": 0, "sharpes": []
    })

    for entry in registry:
        # Get indicators from params or indicators_used
        indicators = entry.get("indicators_used", [])
        if not indicators:
            # Try to extract from params
            params = entry.get("params", {})
            for key, val in params.items():
                if any(kw in key.lower() for kw in
                       ["ema", "sma", "rsi", "macd", "atr", "bbands", "obv", "vwap", "adx"]):
                    indicators.append(key)

        families_seen = set()
        for ind in indicators:
            fam = classify_indicator(ind)
            if fam != "other":
                families_seen.add(fam)

        if not families_seen:
            families_seen.add("other")

        name = entry.get("name", "")
        # Also classify by name pattern
        for fam_keywords in [
            (["momentum", "trend_follow", "ma_cross"], "momentum"),
            (["mean_rev", "rsi_", "bollinger"], "mean_reversion"),
            (["breakout", "volatility", "atr_"], "volatility"),
        ]:
            keywords, fam = fam_keywords
            if any(kw in name.lower() for kw in keywords):
                families_seen.add(fam)

        for fam in families_seen:
            family_data[fam]["strategies"].add(name)
            family_data[fam]["sharpes"].append(entry.get("sharpe", 0))
            if entry.get("walk_forward_robust", False):
                family_data[fam]["wf_robust"] += 1

    results = {}
    for fam, d in family_data.items():
        n = len(d["strategies"])
        sharpes = d["sharpes"]
        results[fam] = {
            "n_strategies": n,
            "wf_robust": d["wf_robust"],
            "wf_pass_rate": d["wf_robust"] / n if n > 0 else 0,
            "avg_sharpe": sum(sharpes) / len(sharpes) if sharpes else 0
        }
    return results

indicator_rates = analyze_indicator_effectiveness(registry)
for fam, stats in sorted(indicator_rates.items(), key=lambda x: -x[1]["wf_pass_rate"]):
    print(f"  {fam:20s}  {stats['n_strategies']:3d} strats  {stats['wf_pass_rate']*100:5.1f}% pass")
```

**What to look for:**
- Families with >40% WF pass rate = reliable, recommend for mandates
- Families with <10% WF pass rate = unreliable, avoid in mandates
- If "other" (unclassifiable) has high rate = missing family definitions

### Analysis 4: Overfitting Detection (HIGH PRIORITY)

**Question:** How much do strategies degrade from discovery to walk-forward? Is this a systemic problem?

```python
def analyze_overfitting(registry, sweep_results):
    """Discovery Sharpe vs WF Sharpe correlation — overfitting signal."""
    # Build lookup from sweep results
    sweep_by_name = {}
    if isinstance(sweep_results, dict):
        for name, data in sweep_results.items():
            windows = data.get("windows", [])
            if windows:
                avg_wf_sharpe = sum(w.get("sharpe", 0) for w in windows) / len(windows)
                windows_passed = sum(1 for w in windows if w.get("passed", False))
                sweep_by_name[name] = {
                    "avg_wf_sharpe": avg_wf_sharpe,
                    "windows_passed": windows_passed,
                    "total_windows": len(windows)
                }

    pairs = []  # (discovery_sharpe, wf_sharpe, degradation)
    for entry in registry:
        name = entry.get("name", "")
        if name in sweep_by_name:
            disc_sharpe = entry.get("sharpe", 0)
            wf_sharpe = sweep_by_name[name]["avg_wf_sharpe"]
            if disc_sharpe > 0:
                degradation = (disc_sharpe - wf_sharpe) / disc_sharpe
                pairs.append({
                    "name": name,
                    "discovery_sharpe": disc_sharpe,
                    "wf_sharpe": wf_sharpe,
                    "degradation": degradation,
                    "windows_passed": sweep_by_name[name]["windows_passed"],
                    "total_windows": sweep_by_name[name]["total_windows"]
                })

    if not pairs:
        print("  No matched pairs found between registry and sweep results")
        return {"pairs": [], "correlation": 0, "avg_degradation": 0}

    # Compute correlation (simple Pearson)
    n = len(pairs)
    x = [p["discovery_sharpe"] for p in pairs]
    y = [p["wf_sharpe"] for p in pairs]
    mean_x, mean_y = sum(x)/n, sum(y)/n
    cov = sum((xi-mean_x)*(yi-mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi-mean_x)**2 for xi in x) / n) ** 0.5
    std_y = (sum((yi-mean_y)**2 for yi in y) / n) ** 0.5
    correlation = cov / (std_x * std_y) if std_x > 0 and std_y > 0 else 0

    avg_degradation = sum(p["degradation"] for p in pairs) / n
    high_deg = [p for p in pairs if p["degradation"] > 0.5]

    return {
        "n_pairs": n,
        "correlation": round(correlation, 4),
        "avg_discovery_sharpe": round(sum(x)/n, 3),
        "avg_wf_sharpe": round(sum(y)/n, 3),
        "avg_degradation_pct": round(avg_degradation * 100, 1),
        "high_degradation_count": len(high_deg),
        "high_degradation_pct": round(len(high_deg)/n*100, 1),
        "worst_5": sorted(pairs, key=lambda p: -p["degradation"])[:5]
    }

overfit = analyze_overfitting(registry, sweep_results)
print(f"  Correlation (disc vs WF): {overfit['correlation']:.4f}")
print(f"  Avg degradation: {overfit['avg_degradation_pct']:.1f}%")
print(f"  High degradation (>50%): {overfit['high_degradation_count']}/{overfit['n_pairs']}")
```

**What to look for:**
- Correlation < 0.3 = discovery Sharpe is NOT predictive of WF performance (big problem)
- Correlation > 0.7 = good, discovery backtests are reliable
- Avg degradation > 40% = systemic overfitting
- Specific archetypes with worst degradation = target for other loops

### Analysis 5: Trade Frequency vs Robustness (MEDIUM PRIORITY)

**Question:** Do higher-frequency or lower-frequency strategies survive walk-forward better?

```python
def analyze_trade_frequency(registry, sweep_results):
    """Trade count buckets vs WF pass rate."""
    buckets = {
        "low (<20/yr)": {"range": (0, 20), "strategies": [], "wf_robust": 0},
        "medium (20-60/yr)": {"range": (20, 60), "strategies": [], "wf_robust": 0},
        "high (>60/yr)": {"range": (60, 99999), "strategies": [], "wf_robust": 0},
    }

    for entry in registry:
        trades = entry.get("num_trades", 0)
        for bucket_name, bucket in buckets.items():
            lo, hi = bucket["range"]
            if lo <= trades < hi:
                bucket["strategies"].append(entry)
                if entry.get("walk_forward_robust", False):
                    bucket["wf_robust"] += 1
                break

    results = {}
    for name, bucket in buckets.items():
        n = len(bucket["strategies"])
        sharpes = [e.get("sharpe", 0) for e in bucket["strategies"]]
        results[name] = {
            "n": n,
            "wf_robust": bucket["wf_robust"],
            "wf_pass_rate": bucket["wf_robust"] / n if n > 0 else 0,
            "avg_sharpe": sum(sharpes) / n if n > 0 else 0
        }
    return results

freq_data = analyze_trade_frequency(registry, sweep_results)
for bucket, stats in freq_data.items():
    print(f"  {bucket:25s}  {stats['n']:3d} strats  {stats['wf_pass_rate']*100:5.1f}% pass")
```

**What to look for:**
- One bucket dramatically outperforming others = frequency sweet spot
- Very low-frequency (<10 trades/yr) with high Sharpe but low WF = curve fitting
- Very high-frequency with low Sharpe = too much noise

### Analysis 6: Stagnation Analysis (MEDIUM PRIORITY)

**Question:** What are the most common failure modes? Which stagnation traps are hardest to escape?

```python
def analyze_stagnation(sweep_results):
    """Analyze strategies that FAILED walk-forward to find common patterns."""
    failed = []
    if isinstance(sweep_results, dict):
        for name, data in sweep_results.items():
            windows = data.get("windows", [])
            if windows:
                passed = sum(1 for w in windows if w.get("passed", False))
                total = len(windows)
                if passed < 3:  # Failed WF (need 3/6)
                    avg_sharpe = sum(w.get("sharpe", 0) for w in windows) / total
                    avg_trades = sum(w.get("num_trades", 0) for w in windows) / total
                    avg_dd = sum(abs(w.get("max_drawdown", 0)) for w in windows) / total
                    failed.append({
                        "name": name,
                        "windows_passed": passed,
                        "total_windows": total,
                        "avg_wf_sharpe": avg_sharpe,
                        "avg_trades": avg_trades,
                        "avg_drawdown": avg_dd,
                        "sharpe_variance": (sum((w.get("sharpe",0)-avg_sharpe)**2 for w in windows) / total) ** 0.5
                    })

    # Categorize failure modes
    failure_modes = {
        "zero_sharpe": [f for f in failed if f["avg_wf_sharpe"] < 0.1],
        "low_sharpe": [f for f in failed if 0.1 <= f["avg_wf_sharpe"] < 0.4],
        "inconsistent": [f for f in failed if f["sharpe_variance"] > 1.0],
        "few_trades": [f for f in failed if f["avg_trades"] < 5],
        "high_drawdown": [f for f in failed if f["avg_drawdown"] > 0.3],
    }

    results = {
        "total_failed": len(failed),
        "total_analyzed": len(sweep_results) if isinstance(sweep_results, dict) else 0,
        "failure_rate": len(failed) / len(sweep_results) if isinstance(sweep_results, dict) and len(sweep_results) > 0 else 0,
        "failure_modes": {mode: len(strats) for mode, strats in failure_modes.items()},
        "worst_offenders": sorted(failed, key=lambda f: f["avg_wf_sharpe"])[:10]
    }
    return results

stagnation = analyze_stagnation(sweep_results)
print(f"  Failed WF: {stagnation['total_failed']}/{stagnation['total_analyzed']} "
      f"({stagnation['failure_rate']*100:.1f}%)")
for mode, count in stagnation["failure_modes"].items():
    print(f"    {mode:20s}: {count}")
```

**What to look for:**
- Dominant failure mode = systemic issue (e.g., most fail due to inconsistency → regime sensitivity)
- Specific names in worst offenders = candidates for sharpe-optimizer to retry with different config

### Analysis 7: Complexity vs Robustness (MEDIUM PRIORITY)

**Question:** Do strategies with more indicators/parameters have lower WF pass rates?

```python
def analyze_complexity(registry):
    """Number of indicators/params vs WF pass rate."""
    complexity_buckets = {
        "simple (1-2 params)": {"range": (0, 3), "strategies": [], "wf_robust": 0},
        "moderate (3-5 params)": {"range": (3, 6), "strategies": [], "wf_robust": 0},
        "complex (6-8 params)": {"range": (6, 9), "strategies": [], "wf_robust": 0},
        "very complex (9+ params)": {"range": (9, 999), "strategies": [], "wf_robust": 0},
    }

    for entry in registry:
        params = entry.get("params", {})
        indicators = entry.get("indicators_used", [])
        # Count complexity as number of params + number of indicators
        complexity = len(params) + len(indicators)
        for bucket_name, bucket in complexity_buckets.items():
            lo, hi = bucket["range"]
            if lo <= complexity < hi:
                bucket["strategies"].append(entry)
                if entry.get("walk_forward_robust", False):
                    bucket["wf_robust"] += 1
                break

    results = {}
    for name, bucket in complexity_buckets.items():
        n = len(bucket["strategies"])
        sharpes = [e.get("sharpe", 0) for e in bucket["strategies"]]
        results[name] = {
            "n": n,
            "wf_robust": bucket["wf_robust"],
            "wf_pass_rate": bucket["wf_robust"] / n if n > 0 else 0,
            "avg_sharpe": sum(sharpes) / n if n > 0 else 0
        }
    return results

complexity = analyze_complexity(registry)
for bucket, stats in complexity.items():
    print(f"  {bucket:30s}  {stats['n']:3d} strats  {stats['wf_pass_rate']*100:5.1f}% pass")
```

---

## Knowledge Generation

After running all analyses, generate insights and write them to the knowledge base.

### Insight Format

Each insight in `insights_knowledge.tsv` has these columns:
- `timestamp` — ISO 8601
- `category` — archetype, ticker, indicator, overfitting, frequency, stagnation, complexity, cross_pattern
- `insight` — Human-readable finding (one sentence)
- `confidence` — high (n>30), medium (n>10), low (n<=10)
- `evidence` — Key statistic supporting the insight
- `n_strategies` — Number of strategies this is based on
- `actionable` — yes/no — can another loop act on this?
- `source_files` — Which data sources were used

### Writing Insights

```python
from datetime import datetime, timezone

def write_insight(tsv_path, category, insight, confidence, evidence, n_strategies, actionable, source_files):
    """Append an insight to the knowledge base TSV."""
    import csv
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = [timestamp, category, insight, confidence, evidence, str(n_strategies), actionable, source_files]
    with open(tsv_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row)

# Example:
write_insight(
    "loops/meta-analyzer/insights_knowledge.tsv",
    "archetype",
    "Momentum strategies have 2.3x higher WF pass rate than mean_reversion strategies",
    "high",
    "momentum: 45% pass (50/111), mean_reversion: 20% pass (3/15)",
    126,
    "yes",
    "registry.json, sweep_results_cycle13.json"
)
```

### Confidence Rules

- **high**: Based on 30+ strategies, effect size >1.5x, consistent across subgroups
- **medium**: Based on 10-30 strategies, effect size 1.3-1.5x
- **low**: Based on <10 strategies, or effect size <1.3x, or conflicting evidence

### Actionability Rules

An insight is "actionable" if another loop can use it:
- ✅ "Momentum on AAPL has highest WF pass rate" → diversity-explorer should prioritize AAPL momentum mandates
- ✅ "Strategies with >6 params have 60% lower WF pass rate" → sharpe-optimizer should limit param count
- ❌ "Average Sharpe across all strategies is 1.2" → not actionable, just a statistic
- ❌ "SPY has the most strategies" → obvious, not useful

---

## Incremental Learning

### Compare Against Previous Insights

Before writing new insights, check if they already exist:

```python
def load_existing_insights(tsv_path):
    """Load existing insights from knowledge base."""
    import csv
    insights = []
    try:
        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                insights.append(row)
    except FileNotFoundError:
        pass
    return insights

def is_duplicate(new_insight, existing_insights, similarity_threshold=0.7):
    """Check if an insight is substantially similar to an existing one."""
    # Simple word overlap check
    new_words = set(new_insight.lower().split())
    for existing in existing_insights:
        old_words = set(existing.get("insight", "").lower().split())
        overlap = len(new_words & old_words) / max(len(new_words), 1)
        if overlap > similarity_threshold:
            return True
    return False

existing = load_existing_insights("loops/meta-analyzer/insights_knowledge.tsv")
print(f"Existing insights: {len(existing)}")
```

### Update Confidence

If an existing insight is reinforced by new data, update its confidence:

```python
# When you find an insight that matches an existing one but with more data:
# 1. Remove the old row from the TSV
# 2. Write a new row with updated confidence and evidence
# This is done by rewriting the entire TSV (it's small)
```

---

## Report Generation

After all analyses are complete, fill in the report template at `analysis_report.md`.

### Template Variables

The report template uses `{{VARIABLE}}` placeholders. Replace them:

```python
def generate_report(template_path, output_path, all_findings):
    """Fill in the analysis report template."""
    with open(template_path, "r") as f:
        template = f.read()

    from datetime import datetime, timezone
    replacements = {
        "{{TIMESTAMP}}": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "{{N_REGISTRY}}": str(len(registry)),
        "{{N_SWEEP}}": str(len(sweep_results)),
        "{{N_WINNERS}}": str(len(winners)),
        "{{CYCLE_NUMBER}}": str(len(existing_insights) + 1),
        # ... fill in all template variables from all_findings
    }

    for key, value in replacements.items():
        template = template.replace(key, value)

    with open(output_path, "w") as f:
        f.write(template)
```

### Report Sections

1. **Executive Summary** — 3-5 bullet points with the most important findings
2. **Archetype Success Rates** — table + key findings
3. **Ticker Performance Matrix** — table + key findings
4. **Indicator Family Effectiveness** — table + key findings
5. **Overfitting Detection** — correlation stats + signals
6. **Trade Frequency vs Robustness** — table + findings
7. **Stagnation Analysis** — failure mode breakdown
8. **Refinement Efficiency** — turn-to-success stats
9. **Complexity vs Robustness** — table + findings
10. **Top 10 Actionable Insights** — ranked list
11. **Knowledge Base Updates** — what's new this cycle
12. **Recommendations for Other Loops** — specific, actionable guidance

---

## Anti-Patterns

Things this loop must NOT do:

1. **Do NOT modify crabquant/ source code** — this loop is read-only
2. **Do NOT create mandates** — that's diversity-explorer and sharpe-optimizer's job
3. **Do NOT run backtests** — use existing results only
4. **Do NOT report obvious findings** — "strategies with higher Sharpe have higher Sharpe" is useless
5. **Do NOT overfit the analysis** — don't cherry-pick subsets that tell a story; report the full picture
6. **Do NOT recommend specific parameter values** — that's the job of individual strategy optimization
7. **Do NOT duplicate existing insights** — always check the knowledge base first

---

## Example Session Flow

Here's a complete analysis session:

```
1. Load sandbox functions
2. Load all data sources (registry, sweep_results, winners)
3. Validate data (assert counts > 0)
4. Run Analysis 1: Archetype WF Pass Rates
   → Finding: momentum dominates (57%), mean_reversion rare, volatility absent
   → Insight: "Mean_reversion archetype needs more exploration — only 3 strategies"
5. Run Analysis 2: Ticker Performance
   → Finding: AAPL has best WF pass rate, JNJ has worst
   → Insight: "JNJ strategies have 0% WF pass rate despite avg Sharpe 0.8 — overfitting"
6. Run Analysis 3: Indicator Effectiveness
   → Finding: momentum indicators (EMA, MACD) most reliable
   → Insight: "Volume-based indicators have 2x lower WF pass rate than momentum indicators"
7. Run Analysis 4: Overfitting Detection
   → Finding: correlation 0.35, avg degradation 45%
   → Insight: "Discovery Sharpe explains only 12% of WF variance (r²=0.12) — overfitting is systemic"
8. Run Analysis 5: Trade Frequency
   → Finding: all strategies are low-frequency
   → Insight: "100% of strategies trade <20 times/year — frequency diversity is zero"
9. Run Analysis 6: Stagnation
   → Finding: 60% fail WF, most common mode is inconsistency
   → Insight: "Inconsistent Sharpe across windows is the #1 failure mode (40% of failures)"
10. Run Analysis 7: Complexity
    → Finding: complex strategies fail more
    → Insight: "Strategies with 6+ params have 50% lower WF pass rate than simple ones"
11. Load existing insights, check for duplicates
12. Write 8-12 new insights to knowledge base
13. Fill in report template
14. Save report
15. Print summary: "8 new insights added, 2 existing confirmed, report saved"
```

---

## Time Budget

- Data loading + validation: ~10 seconds
- Each analysis: ~10-30 seconds
- Knowledge generation + dedup: ~30 seconds
- Report generation: ~30 seconds
- **Total: ~2-3 minutes**

This is the fastest loop in the system because it never runs backtests or LLM calls.

---

## Integration with Other Loops

The meta-analyzer produces knowledge that other loops consume:

| This Loop Produces | Consumed By | How |
|---|---|---|
| Archetype WF pass rates | diversity-explorer | Prioritize gap-filling for high-success archetypes |
| Ticker alpha factors | diversity-explorer | Focus mandate creation on high-opportunity tickers |
| Indicator effectiveness | sharpe-optimizer | Constrain indicator choices in mandates |
| Overfitting metrics | sharpe-optimizer | Set realistic Sharpe targets (account for degradation) |
| Trade frequency findings | diversity-explorer | Identify frequency gaps to fill |
| Stagnation patterns | sharpe-optimizer | Pre-emptively avoid known failure modes |
| Complexity thresholds | sharpe-optimizer | Enforce max param count in mandates |

**To read meta-analyzer insights from another loop:**
```python
import csv
insights = []
with open("loops/meta-analyzer/insights_knowledge.tsv", "r") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        insights.append(row)
# Filter by category
archetype_insights = [i for i in insights if i["category"] == "archetype"]
actionable = [i for i in insights if i["actionable"] == "yes"]
```
