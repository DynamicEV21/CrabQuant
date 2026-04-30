# CrabQuant — Vision

**Last Updated:** 2026-04-29
**Current Phase:** Phase 6 — Production Validation (prep)

---

## What CrabQuant Is

CrabQuant is an **autonomous, always-on quantitative strategy research system** that continuously invents, tests, validates, refines, and promotes trading strategies without human intervention.

It's not a backtesting tool. It's not a strategy library. It's a **research engine** — a system that does what a quant researcher does: come up with ideas, test them, figure out why they fail, iterate, and keep the winners.

The end state: you give it compute and an API budget. It runs. It finds edges. It invents new approaches. It validates them. It promotes the survivors. You check in, review results, and decide what to deploy.

---

## Core Philosophy

### Always-On, Not Periodic

The system should run **continuously** — not as a set of cron jobs that fire every few hours, but as an always-on engine that's constantly churning through strategies.

### LLMs as Consultants, Not Conductors

Python owns the loop. JSON on disk is state. LLMs are called when intelligence is needed (inventing strategies, diagnosing failures, proposing fixes) but they don't orchestrate anything.

### Realistic Realism

No shortcuts that sacrifice accuracy. Real OHLCV data from Yahoo Finance. Realistic fill simulation. Walk-forward validation. Cross-ticker validation. Composite scoring that penalizes overfit.

If a strategy looks too good, it probably is. The system's job is to catch that.

### Self-Improving

Every cycle makes the system smarter. The meta-analyzer studies what indicator families work on which tickers, which failure modes are most common, which actions lead to breakthroughs. That knowledge feeds back into invention prompts.

### Fast Iteration

Minutes between strategy discoveries, not hours. The refinement pipeline should produce a complete strategy lifecycle (invent → backtest → diagnose → refine → validate) in under 15 minutes per mandate.

---

## What's Built

### Research Engine (Refinement Pipeline)
The core. An LLM-driven iterative refinement loop that takes a strategy mandate and loops up to 7 turns, using LLM intelligence to improve strategies until they hit a Sharpe target.

31 components: orchestrator, LLM API, validation gates, backtest engine, circuit breaker, stagnation detection, action analytics, auto-promotion, wave manager, regime tagger, rolling walk-forward, cross-run learning, feature importance, stagnation recovery, semantic action validation, and more. All tested (4430+ tests passing).

### Backtest Engine
VectorBT-based with composite scoring, parallel execution across tickers, real OHLCV from Yahoo Finance with pickle caching (20hr TTL).

### Strategy Library
25+ strategies across archetypes (momentum, mean reversion, breakout, trend, volume). Mix of hand-crafted and LLM-invented.

### Validation
Walk-forward validation (configurable train/test split), cross-ticker validation across 15+ tickers, overfitting detection, guardrails.

### Daemon
Persistent process with PID management, state persistence, graceful shutdown, health check endpoint. Has run 11 waves and 23 mandates.

---

## Current Reality (April 2026)

The pipeline runs end-to-end. Strategies get invented and backtested. Some hit Sharpe >2.0 in 2 turns. The promotion pipeline gap has been fixed — 118 strategies now in the production registry.

| Metric | Value | Problem? |
|--------|-------|----------|
| Total mandates run | 30 unique (147 turns) | — |
| Backtest successes (real mandates) | 6.8% per-turn | 🔴 Low |
| Mandate convergence (≥1 success) | 33% (7/21 real) | 🟡 Needs work |
| Code gen failure rate (real mandates) | **0%** | ✅ Fixed |
| Strategies in production registry | **99** (ROBUST, audited) | ✅ Exceeded target |
| Registry entries demoted (integrity audit) | 19 (Cycle 19) | ✅ Cleaned |
| winners.json entries | 178 (119 promoted) | ✅ Strong pipeline |
| Validation pass rate | 84% (97/116 re-validated) | ✅ Exceeded target |

**The funnel is open.** After fixing threshold bugs (Cycle 10-12), the promotion pipeline gap (Cycle 13), and the registry integrity audit (Cycle 19), 99 genuinely validated strategies remain in the production registry. The other 19 were demoted after failing re-validation with proper rolling walk-forward (6 windows, min_avg_test_sharpe=0.5, min_windows_passed=3).

**Key insight (Cycle 15):** The "54% code gen failure rate" was a phantom metric inflated by smoke_test and test_mandate entries. Real production mandates have **0% code gen failures**. The real bottleneck is **strategy quality** — 83% of real mandate turns fail on performance metrics:
- low_sharpe: 35% — strategies run but underperform
- regime_fragility: 25% — strategies work in some years but not others
- too_few_trades: 24% — strategies too selective (< 5 trades)
- excessive_drawdown: 10% — strategies lose too much

**Fixes applied (Cycle 15):**
- Sharpe Root Cause Analyzer: 12 diagnosis patterns for low_sharpe with specific actionable fixes
- Regime Diagnosis System: 9 regime patterns with per-year breakdown and targeted fixes

**Phase 5 fixes the funnel. Phase 5.5 adds regime awareness. Phase 5.6 accelerates invention.** See ROADMAP.md.

---

## Success Metrics

| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| Validation pass rate | >50% | 67% (119/178) | ✅ Met |
| Strategies in registry (from invention) | 10+ | 118 promoted | ✅ Exceeded |
| Mandate convergence rate | >50% | 33% (7/21) | 🟡 17% |
| Per-turn success rate | >20% | 6.8% (10/147) | 🔴 13% |
| Code gen failure rate | <30% | 0% (real) | ✅ Fixed |
| Test coverage | 100% of new code | 4430+ tests | ✅ Surpassed |

---

## Continuous Improvement Engine

This section tells the orchestrator what to do when all planned tasks are complete. It does NOT need human direction — it reads this section, identifies priorities, and works autonomously.

### How the Priority System Works

1. **Read the Success Metrics table above** — the Gap column shows relative urgency
2. **The largest gap = current priority** — work on whatever moves the biggest gap metric
3. **Blocking chains matter** — some metrics block others (see below)
4. **Check diminishing returns** — if a metric has improved >50% toward target, deprioritize it
5. **One priority at a time** — focus workers on the current P0, don't spread thin

### Priority Queue

The orchestrator should recalculate priorities every cycle by looking at actual metric values (not just what's written here — check real files like `results/winners/`, `STRATEGY_REGISTRY`, test counts, etc.):

**🔴 P0 — Improve Per-Turn Success Rate (6.8% → >20%)**
- The biggest gap. 83% of real mandate turns fail on performance, not code quality.
- **low_sharpe (35%)** — ✅ Sharpe Root Cause Analyzer added (Cycle 15). Verified wired correctly (Cycle 17 wiring audit).
- **regime_fragility (25%)** — ✅ Regime Diagnosis System added (Cycle 15). Verified wired correctly.
- **too_few_trades (24%)** — ✅ Guidance template added (Cycle 15 orchestrator fix). Verified wired correctly.
- **excessive_drawdown (10%)** — Minor. Existing guidance is adequate.
- **Wiring bugs fixed (Cycle 17)**: `detect_regime` imported from wrong module, `analyze_failure_patterns` called with wrong arg type, 2 silent `except-pass` blocks now log warnings. All 3 bugs would have caused dead-code paths in the Turn 1 and Turn 2+ prompt builders.
- **Full wiring audit (Cycle 17)**: Zero new bugs found across all 17 modules. All function signatures match their call sites.
- **Positive Feedback Analyzer (a3dc168)**: Added to prevent LLM regression — tracks what works and feeds successful patterns back into prompts. Agent should verify this is wired and not regressed.
- Action: Run live mandates to verify all diagnosis systems work end-to-end (now P1).

**✅ P0.5 — Registry Data Integrity Audit — COMPLETE (Cycle 19)**
- ~~116/118 strategies had `walk_forward_test_sharpe=0.0` but `walk_forward_robust=True`.~~ FIXED.
- Re-ran rolling walk-forward on all 116 flagged entries (6 windows, min_avg_test_sharpe=0.5, min_windows_passed=3).
- 97 PASSED (avg_test_sharpe 0.526-2.188, median 0.990).
- 19 DEMOTED (avg_test_sharpe below 0.5 threshold). Registry updated.
- All 118 strategy .py files import and execute cleanly (0 errors).

**🟡 P1 — Run Live Mandates to Verify Improvements**
- The diagnosis systems (sharpe, regime) were built based on historical data analysis
- Need to run actual mandates to verify they improve the per-turn success rate
- Run explorer mode mandates on diverse tickers (not just SPY)
- Verify the feedback loop actually changes LLM behavior
- **param_optimizer** (wired in commits c8ee32b, 1900a12): Automated parameter optimization tool that rescued a mandate to Sharpe 1.54 (saved 3 turns). The agent should use `param_optimizer` more aggressively — it's a fast, deterministic alternative to LLM-driven refinement when a strategy's logic is sound but parameters need tuning. Try it before spending LLM turns on parameter tweaks.

**🟢 P2 — Polish (only after P0/P1 are significantly improved)**
- Mandate convergence rate (33% → >50%)
- ROADMAP Phase 6 prep items — daemon config, budget tracking, Telegram briefs

**⚪ P3 — Diminished Returns (avoid unless directly tied to P0/P1 work)**
- Test expansion — we have 4430+ tests. Only write tests for code you're actively modifying for a higher-priority goal.

### Dynamic Priority Rules

These rules override the static queue above when conditions change:

| Condition | Action |
|-----------|--------|
| A metric has improved >50% toward target | Deprioritize — shift workers to next biggest gap |
| P0 is partially resolved (e.g., validation rate hits 20%) | Keep P0 but split workers: 2 on P0, 1 on P1 |
| Same task has failed 3+ times across cycles | Stop retrying — escalate to orchestrator self-rescue or skip |
| A worker discovers a bug blocking P0 | Immediately redirect 1 worker to fix it |
| All P0 tasks are in-progress or blocked | Move to P1 — don't idle |
| Test count for a module exceeds 50 | STOP adding tests to that module — diminishing returns |

### Work Categories

| Category | Examples | When to Use |
|----------|----------|-------------|
| **Fix blockers** | Debug validation failures, fix code gen errors, resolve import bugs | Always — highest value |
| **Build features** | Implement ROADMAP items, add modules from current phase plan | When planned tasks exist |
| **Investigate & diagnose** | Run diagnostic mandates, analyze failure patterns, profile bottlenecks | When root cause is unclear |
| **Improve quality** | Better error messages, type hints, logging, docstrings | Only alongside active feature/fix work |
| **Expand tests** | Add tests for new/modified modules | Only for code being changed for P0/P1 goals |

### Discovery Budget

Each cycle, the orchestrator MAY allocate **at most 1 worker slot** to "discovered" work — something the agent finds on its own that wasn't in the task queue.

**Allowed discovery:**
- Finding and fixing bugs that block success metrics
- Improving prompts or config that would move P0/P1 metrics
- Adding small utility functions that reduce code duplication
- Implementing ROADMAP items that weren't in the task queue but align with current phase

**NOT allowed discovery:**
- Expanding test files for well-tested modules
- Refactoring working code that isn't blocking anything
- Adding features not in VISION.md or ROADMAP.md
- Changing business logic or algorithm behavior without investigation

Discovery workers should commit with prefix `discovery:` so their work is distinguishable from planned work.

### Anti-Patterns (Do NOT Do These)

- ❌ Expanding test files for modules with >50 tests (diminishing returns)
- ❌ Adding tests that don't correspond to any feature/fix/bug work
- ❌ Working on P3 priorities when P0/P1 gaps exist
- ❌ Refactoring code that's already working and not blocking anything
- ❌ Repeating a task that failed 3+ times without changing approach
- ❌ Writing code without understanding why previous attempts failed

### Data Integrity Checks

The agent MUST verify data integrity in production files. The batch promotion system (`sweep_promote_remaining.py`) has been observed hardcoding `walk_forward_robust=True` without re-running validation — 116/118 registry entries have `walk_forward_test_sharpe=0.000` despite being marked robust. This is a data integrity bug, not a strategy quality bug (the strategies passed rolling WF with median Sharpe 0.947 during the sweep).

**Minimum sanity checks (apply to ALL registry operations):**
- `walk_forward_test_sharpe > 0` — entries with sharpe=0.0 and robust=True are suspect
- `walk_forward_robust=True` must correspond to actual rolling walk-forward results
- `validation_status` in winners.json must match registry.json status

**Periodic audit task:** Load `strategies/production/registry.json`. Flag entries where `walk_forward_robust=True` but `walk_forward_test_sharpe=0.0`. Re-run `rolling_walk_forward()` on flagged entries (6 windows, min_avg_test_sharpe=0.5, min_windows_passed=3). Remove entries that fail. Update winners.json accordingly.

### Quality Score

Track outcome-based quality per cycle to measure whether the cron agent's contributions actually improve over time:

```
quality_score = (new strategies passing validation) / (total strategies attempted)
```

Store in `results/quality_score.json`:
```json
{
  "cycles": [
    {"cycle": 18, "attempted": 10, "passed": 3, "score": 0.3, "timestamp": "2026-04-30T01:00:00Z"}
  ]
}
```

The agent should read this file at the start of each cycle. If score is declining over 3+ cycles, stop and investigate why (prompt drift, threshold changes, data staleness).

---

## Pipeline Enhancements (North Star Additions)

These are concrete improvements to the existing pipeline, identified through deep analysis of the refinement source code and comparison with external projects (dietmarwo/autoresearch-trading, CarloNicolini/autoresearch-skfolio, SYNR-AI/StrategyArena). These are NOT a separate system — they're targeted upgrades to specific files in the existing codebase.

### Enhancement 1: Replace Grid Optimizer with scipy DE

**File:** `crabquant/refinement/param_optimizer.py`
**Problem:** Current optimizer is a 3-point grid search (3^N combos, max 20, ±50% around defaults). That's ~20 evaluations per turn — barely scratching the surface of parameter space. Many strategies with good structure fail because default params are wrong.
**Solution:** Replace with `scipy.optimize.differential_evolution` (already installed — scipy 1.17.1). Population-based global search, converges reliably, zero extra dependencies.

```python
from scipy.optimize import differential_evolution

result = differential_evolution(
    objective_func,  # negative sharpe on test window
    bounds=[(lo, hi), ...],  # from strategy's DEFAULT_PARAMS
    maxiter=1000,
    tol=1e-8,
    polish=True,  # local refinement after global search
)
```

**Why scipy DE over fcmaes/BiteOpt:**
- Already installed, zero extra dependencies
- Best convergence reliability in benchmarks (always finds global optimum on Rosenbrock)
- Well-maintained, huge community, excellent docs
- fcmaes has open bugs (Windows C++ crashes, poor worker scaling, single maintainer)
- fcmaes's "10K evals/sec" claim is misleading — actual benchmarks show ~1,400 evals/sec on cheap functions, and parallel overhead makes it slower for functions with <10ms eval time

**Why NOT vectorbt for optimization:** Already a dependency but its matrix approach can't express CrabQuant's custom `get_strategy()/simulate()` state machine pattern. Vectorbt is good for entry/exit signal sweeps, not for the numba strategy architecture CrabQuant uses.

### Enhancement 2: Deflated Sharpe Ratio

**File:** New `crabquant/refinement/deflated_sharpe.py` (~150 lines), wired into `promotion.py` and `validation_gates.py`
**Source:** Port from CarloNicolini/autoresearch-skfolio (`dsr.py`, MIT license)
**Problem:** CrabQuant has run 4,614+ experiments but has ZERO correction for multiple testing bias. After that many trials, some strategies look good by pure luck. There's no mechanism to ask "would this Sharpe still look impressive if I ran 4,614 random strategies?"
**Solution:** Implement Deflated Sharpe Ratio (Bailey & López de Prado 2014):

```python
def deflated_sharpe_ratio(observed_sharpe, sharpe_std, n_trials, sr0=0.0):
    """
    Adjusts observed Sharpe for the number of independent trials.
    Returns probability that observed Sharpe is NOT due to luck.
    DSR < 0.05 → reject (likely overfit to multiple testing)
    """
    expected_max_sharpe = compute_expected_max_sr(n_trials, sr0, sharpe_std)
    psr = probabilistic_sharpe_ratio(observed_sharpe, sr0, sharpe_std)
    # DSR = PSR evaluated at expected_max_SR instead of benchmark SR
    return probabilistic_sharpe_ratio(observed_sharpe, expected_max_sharpe, sharpe_std)
```

**Key inputs:** Track `n_trials` (total experiments run), compute `sharpe_std` from walk-forward fold variation. Wire into promotion: strategy must have DSR > 0.05 to be accepted as a winner.

### Enhancement 3: Complexity Scoring

**File:** New `crabquant/refinement/complexity.py` (~100 lines), wired into `param_optimizer.py` scoring and `context_builder.py`
**Problem:** No penalty for parameter count. A 12-param strategy and 3-param strategy with the same Sharpe score equally. Adding more indicators always "looks better" in-sample. No way to detect if the system is overfitting through complexity.
**Solution:** Parse strategy code AST → compute complexity dimensions:

```python
@dataclass
class ComplexityScore:
    n_params: int          # from DEFAULT_PARAMS length
    n_indicators: int      # count of indicator function calls in simulate()
    n_branches: int        # if/elif/else count in _execute()
    n_features: int        # distinct data arrays used (close, high, low, volume)
    total: float           # weighted combination
```

**Uses:**
1. **Scoring penalty:** `adjusted_threshold = base_threshold * (1 + 0.05 * (n_params - 4))` — more params → harder to get promoted
2. **Feedback to LLM:** "Your last strategy had complexity 8/10 and failed. Try complexity < 5."
3. **Overfitting detection:** Plot complexity vs score over time. If complexity rises but scores stay flat → the system is fitting noise, not finding edge.
4. **Context building:** Show LLM "strategies with complexity 3-5 have 2x higher acceptance rate than 7+"

### Enhancement 4: Explainer Agent

**File:** New `crabquant/refinement/explainer.py` (~80 lines), wired into `scripts/refinement_loop.py` after each turn
**Problem:** When a strategy fails, the classifier says `low_sharpe` but doesn't explain WHY in natural language. The LLM gets numerical feedback but no structured reasoning about what specifically went wrong. Future turns can't learn from past failures beyond "Sharpe was 0.3, target was 1.5."
**Solution:** After each experiment, call GLM-5 with the backtest report and a focused prompt:

```python
EXPLAINER_PROMPT = """You are a concise trading strategy analyst. Given this backtest result, explain in 2-3 sentences WHY this strategy failed. Be specific about the mechanism — not "Sharpe was low" but "the RSI entry triggers too late in fast moves, causing the strategy to enter after the move is over and then get stopped out."

Strategy: {name}
Failure mode: {failure_mode}
Sharpe: {sharpe} (target: {target})
Per-fold breakdown: {fold_breakdown}
Top trades: {top_trades_summary}
Indicators used: {indicators}

2-3 sentence explanation:"""
```

**Store in** `results/explanations.json` keyed by experiment_id. Feed back into context_builder.py: "Similar strategies failed because..." — gives the LLM concrete, actionable failure reasoning instead of raw numbers.

### Enhancement 5: AST Safety Sanitizer

**File:** New `crabquant/refinement/ast_sanitizer.py` (~60 lines), wired into `validation_gates.py` Gate 1
**Source:** Inspired by SYNR-AI/StrategyArena's sanitizer pattern
**Problem:** LLM-generated code can contain look-ahead bias (`shift(-N)`), forbidden imports (`os`, `sys`, `requests`), or dangerous patterns (`eval`, `exec`, `__import__`). Current Gate 1 checks syntax/AST imports but doesn't catch look-ahead or forbidden builtins.
**Solution:** Add to Gate 1:

```python
BLOCKED_IMPORTS = {"os", "sys", "subprocess", "requests", "urllib", "socket"}
BLOCKED_BUILTINS = {"eval", "exec", "__import__", "compile", "open"}
LOOKAHEAD_PATTERNS = ["shift(-", "[:-N]", ".shift(-"]  # look-ahead bias in pandas

def sanitize_strategy(code: str) -> list[str]:
    violations = []
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: {alias.name}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_BUILTINS:
                violations.append(f"Blocked builtin: {node.func.id}")
    for pattern in LOOKAHEAD_PATTERNS:
        if pattern in code:
            violations.append(f"Possible look-ahead bias: {pattern}")
    return violations
```

### Enhancement 6: dietmarwo Upstream Features to Port

**Source:** dietmarwo/autoresearch-trading (MIT, actively maintained, CrabQuant's original inspiration)

Specific features from latest commits worth porting:
1. **Shadow replay** — Re-test promoted winners with conservative fills (wider slippage, higher fees). If a winner fails shadow replay, demote it. Prevents strategies that only survive under optimistic assumptions.
2. **Strategy family labeling** — Tag each strategy with a family (breakout, mean_reversion, trend, volatility, etc.). Track per-family success rates. Use in context building: "breakout strategies have 15% success rate on SPY but 30% on BTC."
3. **Crypto indicator helpers** — `rolling_vwap_np`, `vwap_deviation_np`, `choppiness_index_np`, `realized_volatility_np`, `distance_from_high_np`, `distance_from_low_np`. Add to `strategy_helpers.py` if not already present.

### Enhancement 7: Time-Reversed Validation (Overfitting Detector)

**File:** `crabquant/refinement/walk_forward.py` — add reversed check (~20 lines)
**Source:** CarloNicolini/autoresearch-skfolio's time-reversed validation pattern
**Problem:** A strategy that looks good on normal data but also "works" on reversed data is exploiting a statistical artifact, not a real edge. After 4,614 trials, some winners survive by luck. CrabQuant has no check for this.
**Solution:** After walk-forward validation passes, run the same strategy on time-reversed data:

```python
def time_reversed_check(strategy_fn, data, threshold=0.3):
    """If strategy performs similarly on reversed data, it's overfit."""
    reversed_data = data.iloc[::-1].copy()
    normal_score = run_backtest(strategy_fn, data)
    reversed_score = run_backtest(strategy_fn, reversed_data)
    # If reversed score is >30% of normal score, strategy is suspicious
    if reversed_score / max(normal_score, 1e-8) > threshold:
        return False, f"Overfit: reversed Sharpe {reversed_score:.2f} is {reversed_score/normal_score:.0%} of normal"
    return True, "Passed time-reversed check"
```

**Why this matters:** Trivially implementable (reverse a DataFrame), catches the most insidious form of overfitting — the kind where the strategy accidentally memorizes temporal structure.

### Enhancement 8: Mandate-Aware Forced Exploration on Plateau

**File:** `crabquant/refinement/stagnation.py` — enhance existing stagnation with forced pivot (~80 lines)
**Source:** dietmarwo/autoresearch-trading's explore/exploit alternation (adapted for mandate system)
**Problem:** `stagnation.py` already detects indicator ruts (`track_indicator_diversity`) and stagnation traps (`detect_stagnation_trap`), but only *suggests* alternatives — it never forces a structural pivot. The LLM keeps generating variants of the same family for all 7 turns, wasting them. Crucially, this must **respect the mandate's archetype** — a manual `momentum_aapl.json` mandate should not get hijacked into mean reversion.
**Solution:** Two-tier pivot system. On 3+ consecutive failures in same family without a KEEP:

1. **Within-archetype pivot** (always, respects mandate): Force different indicators, logic structure, or timeframe within the same strategy family
2. **Cross-archetype pivot** (only when mandate allows): Jump to a completely different family

```python
def check_family_plateau(turn_history, mandate, max_same_family=3):
    """Check if LLM is stuck in a strategy rut. Returns (should_pivot, pivot_type, message)."""
    families = [classify_family(t) for t in turn_history]
    recent = families[-max_same_family:]
    recent_statuses = [t.status for t in turn_history[-max_same_family:]]

    if not (len(set(recent)) == 1 and all(s != "keep" for s in recent_statuses)):
        return False, None, None

    stuck_family = recent[0]
    mandate_arch = mandate.get("strategy_archetype", "")
    allow_cross = mandate.get("force_diversify", False)  # Only auto-generated mandates default True

    if stuck_family == mandate_arch and not allow_cross:
        # Within-archetype: different indicators, logic, timeframe
        return True, "within", (
            f"STUCK: All {max_same_family} turns used {stuck_family} indicators with no progress.\n"
            f"You MUST use a completely different set of {stuck_family} indicators and logic structure.\n"
            f"Try a different approach within {stuck_family}: "
            f"if using MACD try ROC/TSI, if using single timeframe try multi-timeframe, "
            f"if using entry signals try a filter-based approach.\n"
            f"Do NOT repeat your previous indicator combination."
        )
    else:
        # Cross-archetype: completely different family
        alternatives = [f for f in _INDICATOR_FAMILIES if f != stuck_family]
        alt_hint = ", ".join(alternatives[:3])
        return True, "cross", (
            f"STUCK: All {max_same_family} turns used {stuck_family} strategies with no progress.\n"
            f"You must switch to a DIFFERENT strategy family entirely: {alt_hint}.\n"
            f"Do NOT use any {stuck_family} indicators on the next turn."
        )
```

**Mandate field:** Add `"force_diversify": true` to auto-generated mandates. Manual mandates default to `false`, preserving user intent when deliberately exploring a specific archetype.

**Impact:** 25% of failures are regime_fragility — forcing within-archetype structural pivots recovers wasted turns without breaking deliberate deep-dives.

### Enhancement 9: Error-Type-Specific Repair Prompts

**File:** `crabquant/refinement/llm_api.py` — enhance repair logic (~40 lines)
**Source:** dietmarwo/autoresearch-trading's crash recovery with error-type-specific repair
**Problem:** When strategy code fails to execute, CrabQuant retries with a generic "fix this" prompt. But a `SyntaxError` (indentation) needs different guidance than an `ImportError` (wrong module) or a `TypeError` (wrong function signature). Generic repair wastes LLM turns.
**Solution:** Parse the error type and inject targeted guidance:

```python
REPAIR_GUIDANCE = {
    "SyntaxError": "Check indentation — Python uses 4-space indent, not tabs. Check for missing colons after if/for/def/with.",
    "IndentationError": "Fix indentation. All code inside if/for/def/with/class must be indented 4 spaces more than the parent.",
    "ImportError": "The module '{module}' is not available. Use only: numpy (np), numba, math, and functions from strategy_helpers.py.",
    "TypeError": "Function '{func}' was called with wrong argument types. Check the function signature — it expects {expected_sig}.",
    "NameError": "'{name}' is not defined. Check for typos. Available functions are listed in strategy_helpers.py.",
    "IndexError": "Array index out of bounds. Your data has {n_bars} bars — check that your lookback window doesn't exceed this.",
    "ZeroDivisionError": "Division by zero. Add a small epsilon (1e-8) to denominators, or check for zero before dividing.",
}
```

### Enhancement 10: Degenerate Strategy Rejection

**File:** `crabquant/refinement/validation_gates.py` — add to Gate 2 (~10 lines)
**Source:** dietmarwo/autoresearch-trading's flat/no-trade detection
**Problem:** 24% of failures are "too_few_trades." Some strategies achieve this by literally never trading — zero trades = zero drawdown = might pass some metrics. CrabQuant should reject these immediately without wasting a validation slot.
**Solution:**

```python
def reject_degenerate(results):
    if results.get("num_trades", 0) == 0:
        return False, "DEGENERATE: strategy executed zero trades"
    if results.get("total_return", 0) == 0 and results.get("max_drawdown", 0) == 0:
        return False, "DEGENERATE: strategy never entered any position"
    return True, None
```

### Enhancement 11: HODL Baseline Comparison

**File:** `crabquant/refinement/scoring.py` — add to composite score (~15 lines)
**Source:** dietmarwo/autoresearch-trading's logarithmic diagrams with HODL baseline
**Problem:** A strategy with Sharpe 1.2 that underperforms buy-and-hold is being promoted alongside strategies that actually beat the market. There's no baseline comparison.
**Solution:** Compute HODL return over the same period. If strategy underperforms HODL, add a penalty or flag:

```python
def hodl_penalty(strategy_return, benchmark_return, threshold=0.8):
    """If strategy return < 80% of HODL return, it's not adding value."""
    if benchmark_return > 0 and strategy_return < benchmark_return * threshold:
        return -0.3  # Significant penalty
    return 0.0
```

### Enhancement 12: Tighten Walk-Forward Pass Criteria

**File:** `crabquant/refinement/config.py` — raise thresholds (config change only)
**Problem:** Current walk-forward pass criteria are dangerously permissive. With 6 rolling windows (18mo train / 6mo test / 6mo step over 5y data), only `min_avg_test_sharpe >= 0.3` and `min_windows_passed >= 1` are required. A strategy that works in just 1 out of 6 time windows and has a mediocre average Sharpe gets promoted. This is a major source of the 6.8% success rate — flukes pass validation and then fail in production.

**Probability argument:** If first 2 windows average Sharpe ~0.1, the remaining 4 must each average ≥0.4 to hit the 0.3 floor — meaning each remaining window must exceed the threshold itself. The chance of a strategy that fails in 2 distinct time periods suddenly thriving in 4 others is under 5%.

**Solution:** Tighten thresholds in `VALIDATION_CONFIG`:

```python
VALIDATION_CONFIG: dict = {
    # rolling_walk_forward() — tightened thresholds
    "min_avg_test_sharpe": 0.4,        # was 0.3 — raise the floor
    "min_windows_passed": 3,            # was 1 — require majority (3/6 windows)
    "min_window_test_sharpe": 0.1,      # was 0.0 (disabled) — each window must be at least slightly positive
    "max_window_degradation": 0.8,      # was 1.0 (disabled) — cap train→test drop at 80%
    ...
}
```

**Why 3/6 windows:** Majority rule ensures the strategy works in *most* time periods, not just one lucky one. With 6 windows covering different market regimes (bull, bear, sideways, volatile, etc.), passing 3 means the strategy has genuine edge across at least half the observed market conditions.

**Impact:** Directly filters out fluke strategies before they reach the winners pool. Will initially reduce the number of promoted strategies but dramatically increase the quality of those that pass — directly attacking the 6.8% success rate from the validation gate rather than trying to fix bad strategies downstream.

**Optional follow-up:** Add fast-fail optimization (~15 lines in `validation/__init__.py`) to abort after first 2 windows if median Sharpe < 0.15, saving ~40 seconds per garbage strategy. This is a performance optimization, not a quality improvement — tighten the bar first.

### Enhancement 13: Missing Crypto Indicators

**File:** `crabquant/strategies/strategy_helpers.py` — add ~150 lines
**Source:** dietmarwo/autoresearch-trading's `strategy_helpers.py` (1522 lines, 99 indicators)
**Problem:** CrabQuant's indicator catalog is smaller. Missing indicators the LLM might want to use but can't because they don't exist.
**Add these (all @njit, MIT licensed from dietmarwo):**
- `rolling_vwap_np(ohlc, period)` — Volume-weighted average price
- `vwap_deviation_np(ohlc, period)` — Deviation from VWAP (mean reversion signal)
- `choppiness_index_np(high, low, close, period)` — Trend vs ranging detection
- `realized_volatility_np(close, period)` — Annualized realized vol (useful for position sizing)
- `distance_from_high_np(high, period)` — % from period high (drawdown proxy)
- `distance_from_low_np(low, period)` — % from period low (bounce proxy)
- `frama_np(close, period)` — Fractal adaptive moving average
- `kama_np(close, period)` — Kaufman adaptive moving average
- `vortex_np(high, low, close, period)` — Vortex indicator (trend direction + strength)

### Enhancement 14: Per-Ticker Alpha Decomposition in Prompts

**File:** `crabquant/refinement/context_builder.py` — enhance failure feedback (~30 lines)
**Source:** dietmarwo/autoresearch-trading's per-ticker alpha breakdown
**Problem:** 25% of failures are regime_fragility — strategies work on some tickers but not others. The LLM gets "failed on cross-ticker validation" but doesn't know WHICH tickers worked and WHY.
**Solution:** After cross-ticker validation, inject into the repair prompt:

```python
def build_ticker_alpha_feedback(ticker_results):
    """Show LLM which tickers worked and which failed."""
    winners = [t for t in ticker_results if t.sharpe > 1.0]
    losers = [t for t in ticker_results if t.sharpe < 0.5]
    feedback = f"Cross-ticker results: {len(winners)} winners, {len(losers)} failures.\n"
    if winners:
        feedback += f"  Worked on: {', '.join(w.ticker for w in winners)} (Sharpe: {', '.join(f'{w.sharpe:.1f}' for w in winners)})\n"
    if losers:
        feedback += f"  Failed on: {', '.join(l.ticker for l in losers)} (Sharpe: {', '.join(f'{l.sharpe:.1f}' for l in losers)})\n"
        feedback += "Consider: Does this strategy need trending markets? High volatility? Specific sector behavior?"
    return feedback
```

### Implementation Priority

| # | Enhancement | Impact on 6.8% success rate | Effort | Why |
|---|---|---|---|---|
| **1** | **scipy DE optimizer** | 🔴 HIGH | 2-3h | Many strategies fail on params, not logic. 20→1000+ evals |
| **12** | **Tighten walk-forward criteria** | 🔴 HIGH | 15min | Config-only: 1/6→3/6 windows, enables per-window floor. Directly attacks 6.8% |
| **8** | **Mandate-aware forced exploration** | 🔴 HIGH | 1-2h | Breaks family plateaus (25% failures). Within-archetype pivot respects mandate |
| **9** | **Error-type-specific repair** | 🔴 HIGH | 1h | Saves repair turns. Different error = different fix |
| **7** | **Time-reversed validation** | 🔴 HIGH | 30min | Catches overfitting false winners. 5 lines of code |
| **10** | **Degenerate strategy rejection** | 🟡 HIGH | 30min | Instant-reject zero-trade strategies (24% of failures) |
| **1** | **scipy DE optimizer** | 🟡 HIGH | 2-3h | Many strategies fail on params, not logic. 20→1000+ evals |
| **4** | **Explainer Agent** | 🟡 MEDIUM | 1-2h | Better feedback loop for LLM |
| **14** | **Per-ticker alpha in prompts** | 🟡 MEDIUM | 1h | Helps fix regime_fragility (25% of failures) |
| **3** | **Complexity scoring** | 🟡 MEDIUM | 2-3h | Penalizes overfit, improves promotion quality |
| **11** | **HODL baseline comparison** | 🟡 MEDIUM | 1h | Prevents promoting strategies that underperform buy-and-hold |
| **2** | **Deflated Sharpe Ratio** | 🟡 MEDIUM | 2-3h | Prevents accepting statistical noise after 4,614 trials |
| **13** | **Missing crypto indicators** | 🟢 LOW | 2h | More tools for LLM. Won't move success rate directly |
| **5** | **AST sanitizer (enhanced)** | 🟢 LOW | 2h | Security hardening. Add dunder blocking + AST look-ahead check |
| **6** | **dietmarwo ports (shadow replay, family labeling)** | 🟢 LOW | 3h | Polish, not core pipeline improvement |

**Recommended implementation order:**
1. **Config fix** (15min): Tighten walk-forward criteria (#12) — biggest bang for zero code
2. **Quick wins** (30min each): Time-reversed validation (#7) → Degenerate rejection (#10)
3. **High-impact** (1-2h each): Mandate-aware exploration (#8) → Error-type repair (#9) → Per-ticker alpha (#14)
4. **Core upgrades** (2-3h each): scipy DE (#1) → Explainer Agent (#4) → Complexity (#3) → Deflated Sharpe (#2)
5. **Polish** (1-3h each): HODL baseline (#11) → Indicators (#13) → Enhanced sanitizer (#5) → Shadow replay/families (#6)

**Estimated total impact:** Implementing the 3 quick wins + 3 high-impact items could move 6.8% → **15-20%**. The core upgrades push toward **25%+**.

---

## The Dream

> You give it a vision. It runs. It finds edges. It invents new approaches. It validates them. It improves itself. You check in, review the results, and decide what to deploy.

This is the north star. Every architectural decision should move us closer to this.
