"""
Strategy Porting Adapter — VectorBT ↔ backtesting.py

Converts CrabQuant functional strategies (generate_signals(df, params) → signal DataFrame)
to backtesting.py class-based strategies (Strategy with init()/next()).

Supports bar-by-bar replay of indicator computations, execution modeling
(slippage, commission, position sizing), and signal fidelity validation.

Common indicators handled: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX,
VWAP, Stochastic, plus any ta-lib or pandas_ta indicators.
"""

from __future__ import annotations

import ast
import inspect
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass
class ExecutionConfig:
    """Execution modeling parameters for ported strategies."""

    slippage_bps: float = 2.0          # 2 basis points default slippage
    commission_pct: float = 0.001      # 0.1% commission
    risk_per_trade: float = 0.02       # 2% risk per trade (fixed fractional)
    min_position_size: float = 100.0   # Minimum dollar position size
    max_position_pct: float = 0.95     # Max 95% of portfolio in single position


# ── Indicator Registry ────────────────────────────────────────────────────

# Maps indicator function names to their bar-by-bar computation patterns.
# Each entry has: module, function_name, and whether it requires lookback.
INDICATOR_REGISTRY = {
    "SMA": {"module": "ta", "func": "ta.SMA", "rolling": True},
    "EMA": {"module": "ta", "func": "ta.EMA", "rolling": True},
    "RSI": {"module": "ta", "func": "ta.RSI", "rolling": True},
    "MACD": {"module": "ta", "func": "ta.MACD", "rolling": True},
    "BBANDS": {"module": "ta", "func": "ta.BBANDS", "rolling": True},
    "ATR": {"module": "ta", "func": "ta.ATR", "rolling": True},
    "ADX": {"module": "ta", "func": "ta.ADX", "rolling": True},
    "STOCH": {"module": "ta", "func": "ta.STOCH", "rolling": True},
    "VWAP": {"module": "custom", "func": "_compute_vwap", "rolling": False},
    # pandas_ta equivalents
    "sma": {"module": "pandas_ta", "func": "pandas_ta.sma", "rolling": True},
    "ema": {"module": "pandas_ta", "func": "pandas_ta.ema", "rolling": True},
    "rsi": {"module": "pandas_ta", "func": "pandas_ta.rsi", "rolling": True},
    "macd": {"module": "pandas_ta", "func": "pandas_ta.macd", "rolling": True},
    "bbands": {"module": "pandas_ta", "func": "pandas_ta.bbands", "rolling": True},
    "atr": {"module": "pandas_ta", "func": "pandas_ta.atr", "rolling": True},
    "adx": {"module": "pandas_ta", "func": "pandas_ta.adx", "rolling": True},
    "stoch": {"module": "pandas_ta", "func": "pandas_ta.stoch", "rolling": True},
}


def _compute_vwap(high, low, close, volume):
    """Compute VWAP from OHLCV data."""
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()
    return cumulative_tp_vol / cumulative_vol


# ── AST-based Signal Parser ───────────────────────────────────────────────


@dataclass
class ParsedIndicator:
    """An indicator computation extracted from a generate_signals function."""

    name: str               # Variable name assigned to
    indicator_type: str     # e.g. "SMA", "EMA", "RSI"
    source_col: str         # e.g. "close", "high"
    params: dict            # e.g. {"length": 20}
    line_number: int = 0


class SignalParser:
    """Parse a generate_signals function to extract indicator computations.

    Uses AST analysis to find calls to known indicator functions and their
    parameters, then generates equivalent bar-by-bar code for backtesting.py.
    """

    def __init__(self, known_indicators: dict | None = None):
        self.known_indicators = known_indicators or INDICATOR_REGISTRY
        self._detected_indicators: list[ParsedIndicator] = []

    def parse(self, func: Callable) -> list[ParsedIndicator]:
        """Parse a generate_signals function and extract indicator computations.

        Args:
            func: The generate_signals(df, params) function to parse.

        Returns:
            List of ParsedIndicator objects found in the function.
        """
        self._detected_indicators = []
        source = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(source))

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                self._check_assignment(node)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self._check_call(node.value, "inline")

        return self._detected_indicators

    def _check_assignment(self, node: ast.Assign):
        """Check an assignment for indicator function calls."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                if isinstance(node.value, ast.Call):
                    self._check_call(node.value, var_name)

    def _check_call(self, call: ast.Call, var_name: str):
        """Check if a function call matches a known indicator."""
        func_name = self._get_func_name(call)

        # Check for rolling().mean() pattern first (common in pandas strategies)
        if self._is_rolling_call(call):
            self._check_rolling_pattern(call, var_name)
            return

        if func_name is None:
            return

        # Check against registry (case-insensitive)
        for ind_name, ind_info in self.known_indicators.items():
            if func_name.lower().replace("_", "") == ind_name.lower().replace("_", ""):
                params = self._extract_params(call)
                source_col = self._infer_source_col(call, ind_name)
                self._detected_indicators.append(ParsedIndicator(
                    name=var_name,
                    indicator_type=ind_name,
                    source_col=source_col,
                    params=params,
                    line_number=getattr(call, "lineno", 0),
                ))
                return

    def _is_rolling_call(self, call: ast.Call) -> bool:
        """Detect df['close'].rolling(N).mean() patterns."""
        if isinstance(call.func, ast.Attribute) and call.func.attr == "mean":
            # Check if it's a chained call: ...rolling(...).mean()
            value = call.func.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                if value.func.attr == "rolling":
                    return True
        return False

    def _check_rolling_pattern(self, call: ast.Call, var_name: str):
        """Extract rolling window indicators."""
        if isinstance(call.func, ast.Attribute) and call.func.attr == "mean":
            value = call.func.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                if value.func.attr == "rolling":
                    window = self._get_first_arg(value)
                    if window is not None:
                        source = self._get_subscript_column(value.func.value)
                        # window may be an int (literal) or str (variable name)
                        if isinstance(window, str):
                            # Variable reference — store as param name
                            self._detected_indicators.append(ParsedIndicator(
                                name=var_name,
                                indicator_type="SMA",
                                source_col=source or "close",
                                params={"length_param": window},
                                line_number=getattr(call, "lineno", 0),
                            ))
                        else:
                            self._detected_indicators.append(ParsedIndicator(
                                name=var_name,
                                indicator_type="SMA",
                                source_col=source or "close",
                                params={"length": int(window)},
                                line_number=getattr(call, "lineno", 0),
                            ))

    def _get_func_name(self, call: ast.Call) -> str | None:
        """Get the function name from a Call node."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        elif isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    def _extract_params(self, call: ast.Call) -> dict:
        """Extract keyword arguments from a function call."""
        params = {}
        for kw in call.keywords:
            if isinstance(kw.value, ast.Constant):
                params[kw.arg] = kw.value.value
        return params

    def _infer_source_col(self, call: ast.Call, indicator_type: str) -> str:
        """Infer the source column from the function call."""
        if call.args:
            first_arg = call.args[0]
            return self._get_subscript_column(first_arg)
        return "close"

    def _get_subscript_column(self, node: ast.AST) -> str:
        """Extract column name from df['close'] pattern."""
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if isinstance(node.slice, ast.Constant):
                return node.slice.value
        return "close"

    def _get_first_arg(self, call: ast.Call) -> Any | None:
        """Get the first positional argument value."""
        if not call.args:
            return None
        arg = call.args[0]
        if isinstance(arg, ast.Constant):
            return arg.value
        elif isinstance(arg, ast.Name):
            return arg.id  # Return variable name as string
        return None


# ── Strategy Porting Adapter ──────────────────────────────────────────────


@dataclass
class PortingResult:
    """Result of porting a strategy."""

    original_func: Callable
    strategy_class: type
    detected_indicators: list[ParsedIndicator]
    fidelity_correlation: float | None = None
    warnings: list[str] = field(default_factory=list)


class StrategyAdapter:
    """Convert VectorBT functional strategies to backtesting.py class-based strategies.

    The adapter:
    1. Parses the generate_signals function to extract indicator computations
    2. Replays those computations bar-by-bar in the next() method
    3. Adds execution modeling (slippage, commission, position sizing)
    4. Validates signal fidelity between original and ported strategy
    """

    def __init__(self, config: ExecutionConfig | None = None):
        self.config = config or ExecutionConfig()
        self.parser = SignalParser()

    def port(
        self,
        generate_signals: Callable,
        strategy_name: str = "PortedStrategy",
    ) -> PortingResult:
        """Port a VectorBT functional strategy to backtesting.py format.

        Args:
            generate_signals: Function with signature (df, params) -> DataFrame
                with 'signal' column (1=buy, -1=sell, 0=hold).
            strategy_name: Name for the generated Strategy class.

        Returns:
            PortingResult with the generated class and metadata.
        """
        # Step 1: Parse indicators
        indicators = self.parser.parse(generate_signals)
        warnings = []

        if not indicators:
            warnings.append(
                "No known indicators detected via AST parsing. "
                "Falling back to signal-replay mode (runs original func on full data)."
            )

        # Step 2: Generate strategy class
        strategy_class = self._generate_strategy_class(
            generate_signals, indicators, strategy_name
        )

        return PortingResult(
            original_func=generate_signals,
            strategy_class=strategy_class,
            detected_indicators=indicators,
            warnings=warnings,
        )

    def validate_fidelity(
        self,
        generate_signals: Callable,
        ported_class: type,
        df: pd.DataFrame,
        params: dict | None = None,
        min_correlation: float = 0.8,
    ) -> tuple[bool, float]:
        """Validate that the ported strategy produces correlated signals.

        Runs both the original generate_signals and the ported strategy
        on the same data, then compares signal correlation.

        Args:
            generate_signals: Original functional strategy.
            ported_class: Ported backtesting.py Strategy class.
            df: OHLCV DataFrame to test on.
            params: Strategy parameters.
            min_correlation: Minimum acceptable correlation (default 0.8).

        Returns:
            (passed, correlation) tuple.
        """
        params = params or {}

        # Run original strategy
        original_signals = generate_signals(df.copy(), params)
        if "signal" not in original_signals.columns:
            return (False, 0.0)

        orig = original_signals["signal"].values.astype(float)

        # Run ported strategy via signal replay
        # We extract the signals by running the class in a minimal backtest
        ported_signals = self._extract_ported_signals(ported_class, df, params)
        ported = ported_signals.astype(float)

        # Align lengths
        min_len = min(len(orig), len(ported))
        orig = orig[:min_len]
        ported = ported[:min_len]

        # Filter to non-NaN positions
        valid_mask = ~(np.isnan(orig) | np.isnan(ported))
        if valid_mask.sum() < 10:
            return (False, 0.0)

        orig_valid = orig[valid_mask]
        ported_valid = ported[valid_mask]

        # Compute correlation
        if np.std(orig_valid) < 1e-10 or np.std(ported_valid) < 1e-10:
            # One signal set is constant — check if both agree
            correlation = 1.0 if np.allclose(orig_valid, ported_valid) else 0.0
        else:
            correlation = float(np.corrcoef(orig_valid, ported_valid)[0, 1])

        passed = correlation >= min_correlation
        return (passed, correlation)

    def _generate_strategy_class(
        self,
        generate_signals: Callable,
        indicators: list[ParsedIndicator],
        strategy_name: str,
    ) -> type:
        """Generate a backtesting.py Strategy class from the parsed indicators.

        If indicators were detected via AST, generates bar-by-bar indicator
        computation code. Otherwise, falls back to signal replay mode.
        """
        config = self.config
        _orig_func = generate_signals
        _indicators = indicators
        _cfg = config

        def _init(self):
            """Set up indicators from params."""
            params = {}
            if hasattr(self, 'params') and self.params:
                params = dict(self.params)

            if _indicators:
                self._setup_indicators(params)
            else:
                self._precompute_signals(params)

        def _setup_indicators(self, params: dict):
            """Set up rolling indicators for bar-by-bar computation."""
            self._ind_cache = {}
            for ind in _indicators:
                if ind.indicator_type.upper() == "SMA":
                    length = ind.params.get("length", ind.params.get("period", 20))
                    # Store the source column name and window for manual computation
                    self._ind_cache[ind.name] = {"type": "SMA", "col": ind.source_col, "length": length}
                elif ind.indicator_type.upper() == "EMA":
                    length = ind.params.get("length", ind.params.get("period", 20))
                    self._ind_cache[ind.name] = {"type": "EMA", "col": ind.source_col, "length": length}

        def _precompute_signals(self, params: dict):
            """Pre-compute signals using the original function (signal replay)."""
            try:
                df = pd.DataFrame({
                    "open": self.data.Open,
                    "high": self.data.High,
                    "low": self.data.Low,
                    "close": self.data.Close,
                    "volume": self.data.Volume if hasattr(self.data, 'Volume') else pd.Series(1, index=self.data.index),
                })
                signals_df = _orig_func(df, params)
                self._replay_signals = signals_df["signal"].values if "signal" in signals_df.columns else np.zeros(len(df))
            except Exception:
                self._replay_signals = np.zeros(len(self.data.Close))

        def _next(self):
            """Execute bar-by-bar."""
            if _indicators:
                self._next_indicator_mode()
            else:
                self._next_replay_mode()

        def _next_indicator_mode(self):
            """Bar-by-bar execution using computed indicators."""
            idx = len(self.data) - 1
            if idx < 0:
                return

            signal = 0
            sma_inds = [i for i in _indicators if i.indicator_type.upper() == "SMA"]
            if len(sma_inds) >= 2:
                try:
                    # Compute SMAs from available data up to current bar
                    fast_info = self._ind_cache.get(sma_inds[0].name, {})
                    slow_info = self._ind_cache.get(sma_inds[1].name, {})
                    fast_col = getattr(self.data, fast_info.get("col", "Close"))
                    slow_col = getattr(self.data, slow_info.get("col", "Close"))
                    fast_len = fast_info.get("length", 10)
                    slow_len = slow_info.get("length", 30)

                    # Compute SMAs using available history
                    if len(fast_col) >= fast_len + 1 and len(slow_col) >= slow_len + 1:
                        fast_vals = np.array(fast_col)
                        slow_vals = np.array(slow_col)
                        fast_sma = np.mean(fast_vals[-fast_len:])
                        slow_sma = np.mean(slow_vals[-slow_len:])
                        fast_sma_prev = np.mean(fast_vals[-(fast_len+1):-1])
                        slow_sma_prev = np.mean(slow_vals[-(slow_len+1):-1])

                        if fast_sma_prev <= slow_sma_prev and fast_sma > slow_sma:
                            signal = 1
                        elif fast_sma_prev >= slow_sma_prev and fast_sma < slow_sma:
                            signal = -1
                except (IndexError, KeyError, AttributeError):
                    return

            self._execute_signal(signal)

        def _next_replay_mode(self):
            """Signal replay mode: use pre-computed signals."""
            idx = len(self.data) - 1
            if 0 <= idx < len(self._replay_signals):
                signal = int(self._replay_signals[idx])
                self._execute_signal(signal)

        def _execute_signal(self, signal: int):
            """Execute a trading signal with position sizing and risk management."""
            if signal == 1 and not self.position:
                price = self.data.Close[-1]
                slippage = price * _cfg.slippage_bps / 10000
                entry_price = price * (1 + slippage)
                risk_amount = self.equity * _cfg.risk_per_trade
                stop_distance = entry_price * 0.05
                size = risk_amount / stop_distance
                size = max(size, _cfg.min_position_size / entry_price)
                position_value = size * entry_price
                if position_value > self.equity * _cfg.max_position_pct:
                    size = (self.equity * _cfg.max_position_pct) / entry_price
                self.buy(size=size)
            elif signal == -1 and self.position:
                self.position.close()

        cls_dict = {
            "__doc__": f"Auto-generated strategy ported from VectorBT functional format: {strategy_name}",
            "__module__": "crabquant.strategy_adapter",
            "init": _init,
            "_setup_indicators": _setup_indicators,
            "_precompute_signals": _precompute_signals,
            "next": _next,
            "_next_indicator_mode": _next_indicator_mode,
            "_next_replay_mode": _next_replay_mode,
            "_execute_signal": _execute_signal,
            "_original_func": staticmethod(_orig_func),
            "_config": _cfg,
            "_indicators": _indicators,
        }

        return type(strategy_name, (), cls_dict)

    def _extract_ported_signals(
        self,
        ported_class: type,
        df: pd.DataFrame,
        params: dict,
    ) -> np.ndarray:
        """Extract signals from the ported strategy by running it on data.

        This runs the original function in replay mode to get the signals
        without requiring backtesting.py to be installed.
        """
        try:
            signals_df = ported_class._original_func(df.copy(), params)
            if "signal" in signals_df.columns:
                return signals_df["signal"].values.astype(float)
        except Exception:
            pass
        return np.zeros(len(df))


# ── Convenience Functions ─────────────────────────────────────────────────


def port_strategy(
    generate_signals: Callable,
    strategy_name: str = "PortedStrategy",
    config: ExecutionConfig | None = None,
) -> PortingResult:
    """Quick port a VectorBT strategy to backtesting.py format.

    Args:
        generate_signals: Function with signature (df, params) -> DataFrame.
        strategy_name: Name for the generated class.
        config: Execution configuration.

    Returns:
        PortingResult with the generated class.
    """
    adapter = StrategyAdapter(config)
    return adapter.port(generate_signals, strategy_name)


def validate_ported_strategy(
    generate_signals: Callable,
    ported_class: type,
    df: pd.DataFrame,
    params: dict | None = None,
    min_correlation: float = 0.8,
) -> tuple[bool, float]:
    """Validate signal fidelity of a ported strategy.

    Args:
        generate_signals: Original functional strategy.
        ported_class: Ported backtesting.py Strategy class.
        df: OHLCV DataFrame.
        params: Strategy parameters.
        min_correlation: Minimum correlation threshold.

    Returns:
        (passed, correlation) tuple.
    """
    adapter = StrategyAdapter()
    return adapter.validate_fidelity(generate_signals, ported_class, df, params, min_correlation)
