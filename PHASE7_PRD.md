# Phase 7 PRD — Deployment Readiness

**Goal:** Validate that refined strategies would survive real market conditions — not just backtests with perfect fills. Prepare for paper trading and eventual live deployment.

**Scope:** Slippage + commission integration, paper trading engine (architecture), performance dashboard (Telegram), multi-timeframe support, walk-forward in refinement loop, regime-aware validation.
**Out of scope:** Broker integration, risk management layer, live trading (Phase 8).

**Dependencies:** Phase 6 complete (portfolio of promoted strategies, intelligence layer providing decay detection and correlation).

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   Deployment Readiness                        │
│                                                               │
│  ┌──────────────────┐   ┌──────────────────┐                │
│  │  Slippage +      │   │  Walk-Forward    │                │
│  │  Commission      │──►│  in Refinement   │                │
│  │  Integration     │   │  Loop            │                │
│  └──────────────────┘   └──────────────────┘                │
│          │                                                     │
│          ▼                                                     │
│  ┌──────────────────┐   ┌──────────────────┐                │
│  │  Paper Trading   │   │  Multi-Timeframe │                │
│  │  Engine          │   │  Support         │                │
│  └────────┬─────────┘   └──────────────────┘                │
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────┐                                         │
│  │  Telegram        │                                         │
│  │  Dashboard       │                                         │
│  └──────────────────┘                                         │
│                                                               │
│  ┌──────────────────┐                                         │
│  │  Regime-Aware    │                                         │
│  │  Validation      │                                         │
│  └──────────────────┘                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Component 1: Slippage + Commission Integration

**Files:** 
- `~/development/CrabQuant/crabquant/confirm/` (existing — enhance)
- `~/development/CrabQuant/crabquant/refinement/diagnostics.py` (enhance)

### 2.1 Current State

The `confirm/` module exists with:
- `runner.py` — bar-by-bar backtest with slippage simulation
- `batch.py` — batch confirmation across periods and slippage levels
- `strategy_converter.py` — converts VBT strategies to backtesting.py format

However, `confirm/` is **not wired into the refinement loop**. Strategies are promoted based on raw VectorBT backtests without slippage validation.

### 2.2 Requirements

1. **Wire slippage check into refinement diagnostics**: After a strategy passes the Sharpe target in the refinement loop, run a quick slippage simulation before declaring it a winner
2. **Default slippage model**: 5 basis points (0.05%) commission per trade + 1-tick slippage on market orders
3. **Promotion gate**: Strategies must retain ≥70% of their raw Sharpe after slippage to be promoted. If Sharpe drops below the target after slippage, the strategy is NOT promoted
4. **Lightweight check**: Don't run full batch confirmation (9 backtests) during refinement — just a single quick check on the same period
5. **Report degradation**: Log the Sharpe degradation from slippage in the run state

### 2.3 Interface

```python
@dataclass
class SlippageCheckResult:
    raw_sharpe: float
    slippage_sharpe: float
    degradation_pct: float       # (raw - slip) / raw
    commission_per_trade_bps: float
    slippage_ticks: float
    passes_threshold: bool       # True if slippage_sharpe >= 0.7 * raw_sharpe
    passes_sharpe_target: bool   # True if slippage_sharpe >= sharpe_target

def quick_slippage_check(
    strategy_code: str,
    ticker: str = "SPY",
    period: str = "2y",
    commission_bps: float = 5.0,
    slippage_ticks: float = 1.0,
    min_retention_pct: float = 0.70,
    sharpe_target: float | None = None,
) -> SlippageCheckResult:
    """Run a quick slippage simulation on a strategy.
    
    Uses the VectorBT engine with simulated slippage (not full backtesting.py confirmation).
    Applies commission and slippage to the equity curve post-hoc.
    
    This is intentionally lightweight — full confirmation runs in the production pipeline.
    """
```

### 2.4 Integration Points

- `crabquant/refinement/diagnostics.py` — add slippage check as an optional diagnostic after successful backtest
- `crabquant/refinement/promotion.py` — require slippage check to pass before promotion
- `crabquant/refinement/orchestrator.py` — include slippage result in Turn 2+ context ("Your strategy lost 40% Sharpe with slippage — consider reducing trade frequency")

### 2.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_slippage_integration.py` (8+ tests)

- `test_slippage_reduces_sharpe`
- `test_high_frequency_strategy_fails_slippage`
- `test_low_frequency_strategy_passes_slippage`
- `test_retention_threshold_enforced`
- `test_sharpe_target_check`
- `test_zero_slippage_no_change`
- `test_custom_commission_rate`
- `test_slippage_result_serialization`

### 2.6 Effort Estimate
**S-M (1 session)** — existing confirm module, need lightweight wrapper + integration.

---

## 3. Component 2: Paper Trading Engine (Architecture)

**Directory:** `~/development/CrabQuant/crabquant/paper_trading/` (new)

### 3.1 Requirements

Build a minimal paper trading engine that:
1. Subscribes to delayed price data (15-min delayed Yahoo Finance or free Alpaca data)
2. Generates signals from promoted strategies on a daily basis
3. Tracks a virtual portfolio with cash management
4. Logs all trades for later analysis

**This is architecture only** — the engine should be functional but not production-grade. It's a foundation for Phase 8.

### 3.2 Interface

```python
@dataclass
class PaperPosition:
    ticker: str
    shares: int
    entry_price: float
    entry_date: str
    strategy_name: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

@dataclass 
class PaperPortfolio:
    cash: float
    positions: list[PaperPosition]
    total_value: float
    daily_returns: list[float]
    inception_date: str

class PaperTradingEngine:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        data_source: str = "yahoo",  # "yahoo" or "alpaca"
        max_position_pct: float = 0.10,  # Max 10% per position
    ): ...
    
    def generate_signals(self, strategies: list[str], tickers: list[str]) -> dict:
        """Generate today's signals from promoted strategies.
        
        Returns: {ticker: {"action": "BUY"/"SELL"/"HOLD", "strategy": "name", "confidence": 0.8}}
        """
    
    def execute_signals(self, signals: dict) -> list[dict]:
        """Execute signals against virtual portfolio.
        
        Returns list of trade records.
        """
    
    def update_prices(self) -> None:
        """Fetch latest prices and update position values."""
    
    def get_portfolio_summary(self) -> dict:
        """Return current portfolio state: cash, positions, total value, daily P&L."""
    
    def save_state(self, path: str) -> None:
        """Persist portfolio state to JSON."""
    
    def load_state(self, path: str) -> None:
        """Load portfolio state from JSON."""
```

### 3.3 Data Source Strategy

- **Default**: Yahoo Finance delayed data (free, no API key)
- **Upgrade path**: Alpaca paper API (free tier, real-time paper trading)
- **Frequency**: Check signals daily at market close (4:00 PM ET)
- **Fallback**: If data fetch fails, hold all positions (don't sell)

### 3.4 Integration Points

- Supervisor cron — run paper trading engine daily after market close
- Status reporter — include paper portfolio P&L in daily report
- Dashboard — show paper trading performance

### 3.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_paper_trading.py` (8+ tests)

- `test_portfolio_initialization`
- `test_buy_signal_execution`
- `test_sell_signal_execution`
- `test_max_position_limit`
- `test_cash_management`
- `test_state_persistence`
- `test_hold_signal_no_action`
- `test_portfolio_summary_fields`

### 3.6 Effort Estimate
**M (1-2 sessions)** — new module, but intentionally simple.

---

## 4. Component 3: Telegram Performance Dashboard

**File:** `~/development/CrabQuant/crabquant/dashboard/telegram_dashboard.py` (new)

### 4.1 Requirements

A Telegram-based dashboard that shows:
1. **Portfolio performance**: Daily P&L curve (as text chart), cumulative return, current drawdown
2. **Strategy signals**: Today's signals from all promoted strategies
3. **Live positions**: Current paper trading positions with unrealized P&L
4. **Sharpe tracking**: Rolling 30-day Sharpe of paper portfolio
5. **Alerts**: Drawdown warnings, strategy decay notifications, API budget alerts

### 4.2 Interface

```python
class TelegramDashboard:
    def __init__(self, chat_id: str | None = None): ...
    
    def send_daily_performance(self, portfolio: dict, signals: dict) -> bool:
        """Send daily performance summary."""
    
    def send_signal_alert(self, signal: dict) -> bool:
        """Send immediate alert on new signal."""
    
    def send_drawdown_alert(self, current_dd: float, threshold: float = 0.10) -> bool:
        """Alert when drawdown exceeds threshold."""
    
    def send_decay_alert(self, strategy: str, decline_pct: float) -> bool:
        """Alert when strategy decay detected."""
    
    def format_text_chart(self, values: list[float], width: int = 40, height: int = 10) -> str:
        """Generate a text-based sparkline chart for Telegram."""
    
    def format_positions_table(self, positions: list[dict]) -> str:
        """Format positions as a readable table."""
```

### 4.3 Dashboard Format (Telegram)

```
📊 Paper Portfolio — 2026-04-27
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 Value: $102,347 (+2.35%)
📉 Cash: $45,200 | Invested: $57,147
📈 Today: +$847 (+0.83%)
📉 Max DD: -3.2%

📈 30d Performance:
▁▂▃▅▆▇█▇▆▅▃▂▃▅▆▇▇██▇▆▅
Rolling Sharpe: 1.42

📋 Today's Signals:
  BUY SPY (momentum_macd, conf 0.85)
  SELL TSLA (rsi_crossover, conf 0.72)
  HOLD AAPL (bollinger_squeeze)

📌 Positions:
  SPY: 50 shares @ $521.30 → $523.10 (+$90, +0.35%)
  NVDA: 20 shares @ $892.40 → $878.50 (-$278, -1.56%)
  AAPL: 30 shares @ $178.20 → $180.10 (+$57, +1.07%)
```

### 4.4 Integration Points

- Paper trading engine — read portfolio state and signals
- Supervisor cron — send daily report after market close
- Status reporter — reuse formatting utilities

### 4.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_telegram_dashboard.py` (8+ tests)

- `test_daily_performance_format`
- `test_text_chart_generation`
- `test_positions_table_format`
- `test_drawdown_alert_trigger`
- `test_decay_alert_format`
- `test_signal_alert_format`
- `test_empty_portfolio_graceful`
- `test_char_limit_respected`

### 4.6 Effort Estimate
**S (1 session)** — formatting + Telegram send.

---

## 5. Component 4: Multi-Timeframe Support

**Files:**
- `~/development/CrabQuant/crabquant/data/__init__.py` (enhance)
- `~/development/CrabQuant/crabquant/refinement/prompts.py` (enhance)

### 5.1 Requirements

Support strategies that combine signals across multiple timeframes:
1. **Data loading**: Fetch daily, weekly, and (optionally) hourly data for each ticker
2. **Strategy interface**: Extend `generate_signals()` to accept a dict of DataFrames keyed by timeframe: `{"daily": df_daily, "weekly": df_weekly}`
3. **Mandate support**: Allow mandates to specify multi-timeframe strategies
4. **Examples**: Add 2-3 multi-timeframe strategy examples to LLM context (e.g., weekly trend filter + daily entry)

### 5.2 Interface

```python
# Enhanced data loading
def load_multi_timeframe_data(
    ticker: str,
    timeframes: list[str] = ["daily", "weekly"],
    period: str = "2y",
) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for multiple timeframes.
    
    Returns: {"daily": df_1d, "weekly": df_1wk}
    Weekly data is resampled from daily using last-price convention.
    """

# Enhanced strategy interface (backward compatible)
def generate_signals(
    df: pd.DataFrame | dict[str, pd.DataFrame],
    params: dict,
) -> tuple[pd.Series, pd.Series]:
    """Generate entry/exit signals.
    
    If df is a dict (multi-timeframe), extract individual timeframes.
    If df is a single DataFrame (legacy), use as daily data.
    Backward compatible with all existing strategies.
    """
```

### 5.3 Multi-Timeframe Strategy Example (for LLM context)

```python
def generate_signals(df, params):
    """Weekly trend filter + daily RSI entry.
    
    Uses weekly EMA crossover for trend direction,
    daily RSI for entry timing.
    """
    # Extract timeframes
    if isinstance(df, dict):
        daily = df["daily"]
        weekly = df["weekly"]
    else:
        daily = df
        weekly = df.resample("W").last()
    
    # Weekly trend
    weekly_ema_fast = weekly["close"].ewm(span=params["weekly_fast"]).mean()
    weekly_ema_slow = weekly["close"].ewm(span=params["weekly_slow"]).mean()
    weekly_trend = (weekly_ema_fast > weekly_ema_slow).reindex(daily.index, method="ffill")
    
    # Daily entry
    rsi = ta.rsi(daily["close"], length=params["rsi_length"])
    
    entries = weekly_trend & (rsi < params["rsi_oversold"])
    exits = ~weekly_trend | (rsi > params["rsi_overbought"])
    
    return entries, exits
```

### 5.4 Integration Points

- `crabquant/data/__init__.py` — add `load_multi_timeframe_data()`
- `crabquant/refinement/prompts.py` — include multi-timeframe examples in LLM context
- `crabquant/refinement/module_loader.py` — handle multi-timeframe data passing
- `crabquant/refinement/diagnostics.py` — pass multi-timeframe data when running backtests

### 5.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_multi_timeframe.py` (8+ tests)

- `test_load_daily_and_weekly`
- `test_weekly_resampling`
- `test_backward_compat_single_df`
- `test_multi_timeframe_strategy_signals`
- `test_mandate_with_multi_timeframe`
- `test_data_cache_per_timeframe`
- `test_empty_data_graceful`
- `test_hourly_timeframe_optional`

### 5.6 Effort Estimate
**M (1-2 sessions)** — data loading + backward compatibility + examples.

---

## 5. Component 5: Walk-Forward in Refinement Loop

**File:** `~/development/CrabQuant/crabquant/refinement/diagnostics.py` (enhance)

### 5.1 Current State

Walk-forward validation (`walk_forward_test()`) only runs at promotion time. During the refinement loop, strategies are evaluated on a single backtest period.

### 5.2 Requirements

Add optional walk-forward diagnostic at Turn 4+ to catch in-sample overfit early:

1. **Trigger**: At Turn 4 or later, if the strategy has achieved ≥80% of the Sharpe target, run a quick walk-forward check
2. **Lightweight**: Use a single train/test split (not the full rolling window) to keep it fast
3. **Context injection**: Include walk-forward results in Turn 5+ context: "Your strategy achieves Sharpe 1.8 in-sample but only 0.9 out-of-sample. Consider simplifying."
4. **Gate**: If out-of-sample Sharpe drops below 50% of in-sample Sharpe, add a warning to the LLM context

### 5.3 Interface

```python
@dataclass
class QuickWalkForwardResult:
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    retention_pct: float          # oos / is
    is_overfit: bool              # True if retention < 0.50
    train_period: str             # e.g., "2024-04-27 to 2025-10-27"
    test_period: str              # e.g., "2025-10-27 to 2026-04-27"

def quick_walk_forward_check(
    strategy_code: str,
    ticker: str = "SPY",
    train_pct: float = 0.70,
    overfit_threshold: float = 0.50,
) -> QuickWalkForwardResult:
    """Run a quick walk-forward check on a strategy.
    
    Splits data 70/30 (train/test). Runs backtest on both periods.
    Returns retention percentage and overfit flag.
    """
```

### 5.4 Integration Points

- `crabquant/refinement/orchestrator.py` — at Turn 4+, if Sharpe approaching target, call `quick_walk_forward_check()` and inject results into context.
- `crabquant/refinement/context_builder.py` — format walk-forward results for LLM.

### 5.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_quick_walkforward.py` (6+ tests)

- `test_walkforward_runs`
- `test_overfit_detected`
- `test_good_strategy_passes`
- `test_result_serialization`
- `test_custom_train_pct`
- `test_bad_strategy_code_handled`

### 5.6 Effort Estimate
**S (1 session)** — reuse existing walk_forward_test(), add quick wrapper.

---

## 6. Component 6: Regime-Aware Validation

**File:** `~/development/CrabQuant/crabquant/validation/__init__.py` (enhance)

### 6.1 Current State

`walk_forward_test()` reports overall Sharpe but doesn't break it down by market regime.

### 6.2 Requirements

1. **Per-regime Sharpe**: Report Sharpe, return, and drawdown broken down by market regime within the test period
2. **Regime balance check**: Reject strategies that only work in one regime (e.g., bull-market-only strategies)
3. **Minimum regime coverage**: Require strategy to have positive Sharpe in at least 2 of 5 regimes
4. **Weighted Sharpe**: Compute a regime-weighted Sharpe that penalizes strategies concentrated in rare regimes

### 6.3 Interface

```python
@dataclass
class RegimeValidationResult:
    per_regime_sharpe: dict[str, float]
    per_regime_trades: dict[str, int]
    regimes_positive: int
    total_regimes: int
    passes_coverage: bool          # True if ≥2 regimes positive
    weighted_sharpe: float         # Sharpe weighted by regime frequency
    is_regime_balanced: bool

def regime_aware_validation(
    strategy_code: str,
    ticker: str = "SPY",
    min_regimes_positive: int = 2,
) -> RegimeValidationResult:
    """Run backtest and analyze performance by market regime.
    
    Splits the backtest period into regime segments using crabquant.regime.
    Computes Sharpe within each regime segment.
    """
```

### 6.4 Integration Points

- `crabquant/refinement/promotion.py` — require regime-aware validation to pass before promotion
- `crabquant/refinement/context_builder.py` — include regime breakdown in Turn 2+ context
- Status reporter — include regime performance of promoted strategies

### 6.5 Test Requirements

**File:** `~/development/CrabQuant/tests/refinement/test_regime_validation.py` (6+ tests)

- `test_per_regime_sharpe_computed`
- `test_coverage_check_passes`
- `test_coverage_check_fails_single_regime`
- `test_weighted_sharpe_computation`
- `test_empty_regime_segment`
- `test_result_serialization`

### 6.6 Effort Estimate
**S (1 session)** — reuse regime.py, add per-segment analysis.

---

## 7. Dependencies Between Components

```
Slippage Integration ──────────► Promotion Gate (require slippage pass)
Walk-Forward in Loop ──────────► Context Builder (inject OOS results)
Regime-Aware Validation ───────► Promotion Gate (require regime balance)
Multi-Timeframe ───────────────► Data Loading + Strategy Interface
Paper Trading Engine ──────────► Dashboard (display portfolio)
Dashboard ─────────────────────► Status Reporter (reuse formatting)
```

**Build order:**
1. **Slippage Integration** (no deps — high impact, do first)
2. **Walk-Forward in Loop** (no deps — can parallel with #1)
3. **Regime-Aware Validation** (no deps — can parallel with #1-2)
4. **Multi-Timeframe Support** (no deps — independent feature)
5. **Paper Trading Engine** (depends on slippage model being integrated)
6. **Telegram Dashboard** (depends on paper trading + status reporter)

---

## 8. Success Criteria

- [ ] Slippage integrated: at least 1 strategy passes raw Sharpe target but fails slippage check (proving the filter works)
- [ ] Walk-forward check runs at Turn 4+ and injects results into LLM context
- [ ] Regime-aware validation rejects strategies that only work in one regime
- [ ] Multi-timeframe mandate completes successfully through the full refinement loop
- [ ] Paper trading engine runs daily with ≥3 strategies generating signals
- [ ] Dashboard sends daily performance report to Telegram with all sections
- [ ] All unit tests pass (target: 44+ new tests)
- [ ] PHASE_CHECKLIST.md completed

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Slippage kills all strategies | No survivors | Expected — this is the point. If 0% survive, reduce slippage model or focus on longer holding periods |
| Paper trading API rate limits | Can't get real-time data | Use delayed Yahoo Finance data; 15-min delay is fine for daily strategies |
| Multi-timeframe increases data requirements | Slower pipeline | Only use multi-timeframe for final validation, not every refinement turn |
| Dashboard becomes maintenance burden | Distraction from research | Keep it minimal — text formatting only, no frameworks, no charts |
| Walk-forward at Turn 4 slows pipeline | Longer refinement loops | Only trigger when Sharpe ≥80% of target; most mandates won't hit this |
| Regime detection too coarse | Misleading validation | Use existing 5-regime model; it's proven sufficient for strategy classification |

---

## 10. Effort Summary

| Component | Effort | Tests | Dependencies |
|-----------|--------|-------|-------------|
| Slippage Integration | S-M (1 session) | 8+ | None |
| Walk-Forward in Loop | S (1 session) | 6+ | None |
| Regime-Aware Validation | S (1 session) | 6+ | None |
| Multi-Timeframe Support | M (1-2 sessions) | 8+ | None |
| Paper Trading Engine | M (1-2 sessions) | 8+ | Slippage |
| Telegram Dashboard | S (1 session) | 8+ | Paper Trading |
| Integration + Wiring | M (1 session) | — | All components |
| **Total** | **L (5-7 sessions)** | **44+** | — |

---

## 11. File Structure

```
~/development/CrabQuant/
├── crabquant/
│   ├── data/
│   │   └── __init__.py             # ENHANCED — load_multi_timeframe_data()
│   ├── validation/
│   │   └── __init__.py             # ENHANCED — regime_aware_validation()
│   ├── refinement/
│   │   ├── diagnostics.py          # ENHANCED — slippage check, quick walk-forward
│   │   ├── promotion.py            # ENHANCED — slippage + regime gates
│   │   ├── orchestrator.py         # ENHANCED — walk-forward trigger at Turn 4+
│   │   ├── context_builder.py      # ENHANCED — walk-forward + regime context
│   │   ├── prompts.py              # ENHANCED — multi-timeframe examples
│   │   └── module_loader.py        # ENHANCED — multi-timeframe data passing
│   ├── paper_trading/              # NEW directory
│   │   ├── __init__.py
│   │   ├── engine.py               # PaperTradingEngine
│   │   └── portfolio.py            # PaperPortfolio, PaperPosition
│   └── dashboard/                  # NEW directory
│       ├── __init__.py
│       └── telegram_dashboard.py   # TelegramDashboard
└── tests/refinement/
    ├── test_slippage_integration.py     # NEW
    ├── test_quick_walkforward.py        # NEW
    ├── test_regime_validation.py        # NEW
    ├── test_multi_timeframe.py          # NEW
    ├── test_paper_trading.py            # NEW
    └── test_telegram_dashboard.py       # NEW
```
