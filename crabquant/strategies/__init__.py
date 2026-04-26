"""
CrabQuant Strategy Library

Each strategy is a self-contained module with:
- generate_signals(df, params) -> (entries, exits)
- DEFAULT_PARAMS: dict of default parameter values
- PARAM_GRID: dict of parameter options for optimization
- DESCRIPTION: what the strategy does and when it works
"""

from crabquant.strategies.rsi_crossover import (
    generate_signals as rsi_crossover_signals,
    DEFAULT_PARAMS as rsi_crossover_defaults,
    PARAM_GRID as rsi_crossover_grid,
    DESCRIPTION as rsi_crossover_desc,
)
from crabquant.strategies.macd_momentum import (
    generate_signals as macd_momentum_signals,
    DEFAULT_PARAMS as macd_momentum_defaults,
    PARAM_GRID as macd_momentum_grid,
    DESCRIPTION as macd_momentum_desc,
)
from crabquant.strategies.adx_pullback import (
    generate_signals as adx_pullback_signals,
    DEFAULT_PARAMS as adx_pullback_defaults,
    PARAM_GRID as adx_pullback_grid,
    DESCRIPTION as adx_pullback_desc,
)
from crabquant.strategies.atr_channel_breakout import (
    generate_signals as atr_channel_signals,
    DEFAULT_PARAMS as atr_channel_defaults,
    PARAM_GRID as atr_channel_grid,
    DESCRIPTION as atr_channel_desc,
)
from crabquant.strategies.volume_breakout import (
    generate_signals as volume_breakout_signals,
    DEFAULT_PARAMS as volume_breakout_defaults,
    PARAM_GRID as volume_breakout_grid,
    DESCRIPTION as volume_breakout_desc,
)
from crabquant.strategies.multi_rsi_confluence import (
    generate_signals as multi_rsi_signals,
    DEFAULT_PARAMS as multi_rsi_defaults,
    PARAM_GRID as multi_rsi_grid,
    DESCRIPTION as multi_rsi_desc,
)
from crabquant.strategies.ema_ribbon_reversal import (
    generate_signals as ema_ribbon_signals,
    DEFAULT_PARAMS as ema_ribbon_defaults,
    PARAM_GRID as ema_ribbon_grid,
    DESCRIPTION as ema_ribbon_desc,
)
from crabquant.strategies.bollinger_squeeze import (
    generate_signals as bollinger_squeeze_signals,
    DEFAULT_PARAMS as bollinger_squeeze_defaults,
    PARAM_GRID as bollinger_squeeze_grid,
    DESCRIPTION as bollinger_squeeze_desc,
)
from crabquant.strategies.invented_momentum_rsi_atr import (
    generate_signals as invented_momentum_rsi_atr_signals,
    DEFAULT_PARAMS as invented_momentum_rsi_atr_defaults,
    PARAM_GRID as invented_momentum_rsi_atr_grid,
    DESCRIPTION as invented_momentum_rsi_atr_desc,
)
from crabquant.strategies.ichimoku_trend import (
    generate_signals as ichimoku_signals,
    DEFAULT_PARAMS as ichimoku_defaults,
    PARAM_GRID as ichimoku_grid,
    DESCRIPTION as ichimoku_desc,
)

# Registry: name -> (fn, defaults, grid, description)
STRATEGY_REGISTRY = {
    "rsi_crossover": (rsi_crossover_signals, rsi_crossover_defaults, rsi_crossover_grid, rsi_crossover_desc),
    "macd_momentum": (macd_momentum_signals, macd_momentum_defaults, macd_momentum_grid, macd_momentum_desc),
    "adx_pullback": (adx_pullback_signals, adx_pullback_defaults, adx_pullback_grid, adx_pullback_desc),
    "atr_channel_breakout": (atr_channel_signals, atr_channel_defaults, atr_channel_grid, atr_channel_desc),
    "volume_breakout": (volume_breakout_signals, volume_breakout_defaults, volume_breakout_grid, volume_breakout_desc),
    "multi_rsi_confluence": (multi_rsi_signals, multi_rsi_defaults, multi_rsi_grid, multi_rsi_desc),
    "ema_ribbon_reversal": (ema_ribbon_signals, ema_ribbon_defaults, ema_ribbon_grid, ema_ribbon_desc),
    "bollinger_squeeze": (bollinger_squeeze_signals, bollinger_squeeze_defaults, bollinger_squeeze_grid, bollinger_squeeze_desc),
    "ichimoku_trend": (ichimoku_signals, ichimoku_defaults, ichimoku_grid, ichimoku_desc),
    "invented_momentum_rsi_atr": (invented_momentum_rsi_atr_signals, invented_momentum_rsi_atr_defaults, invented_momentum_rsi_atr_grid, invented_momentum_rsi_atr_desc),
}

__all__ = ["STRATEGY_REGISTRY"]
