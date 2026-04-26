"""
CrabQuant Strategy Library

Each strategy is a self-contained module with:
- generate_signals(df, params) -> (entries, exits)
- generate_signals_matrix(df, param_grid) -> (entries_df, exits_df, param_list)
- DEFAULT_PARAMS: dict of default parameter values
- PARAM_GRID: dict of parameter options for optimization
- DESCRIPTION: what the strategy does and when it works

Invented strategies (by the self-improvement agent) are prefixed with "invented_".
"""

from crabquant.strategies.rsi_crossover import (
    generate_signals as rsi_crossover_signals,
    generate_signals_matrix as rsi_crossover_matrix,
    DEFAULT_PARAMS as rsi_crossover_defaults,
    PARAM_GRID as rsi_crossover_grid,
    DESCRIPTION as rsi_crossover_desc,
)
from crabquant.strategies.macd_momentum import (
    generate_signals as macd_momentum_signals,
    generate_signals_matrix as macd_momentum_matrix,
    DEFAULT_PARAMS as macd_momentum_defaults,
    PARAM_GRID as macd_momentum_grid,
    DESCRIPTION as macd_momentum_desc,
)
from crabquant.strategies.adx_pullback import (
    generate_signals as adx_pullback_signals,
    generate_signals_matrix as adx_pullback_matrix,
    DEFAULT_PARAMS as adx_pullback_defaults,
    PARAM_GRID as adx_pullback_grid,
    DESCRIPTION as adx_pullback_desc,
)
from crabquant.strategies.atr_channel_breakout import (
    generate_signals as atr_channel_signals,
    generate_signals_matrix as atr_channel_matrix,
    DEFAULT_PARAMS as atr_channel_defaults,
    PARAM_GRID as atr_channel_grid,
    DESCRIPTION as atr_channel_desc,
)
from crabquant.strategies.volume_breakout import (
    generate_signals as volume_breakout_signals,
    generate_signals_matrix as volume_breakout_matrix,
    DEFAULT_PARAMS as volume_breakout_defaults,
    PARAM_GRID as volume_breakout_grid,
    DESCRIPTION as volume_breakout_desc,
)
from crabquant.strategies.multi_rsi_confluence import (
    generate_signals as multi_rsi_signals,
    generate_signals_matrix as multi_rsi_matrix,
    DEFAULT_PARAMS as multi_rsi_defaults,
    PARAM_GRID as multi_rsi_grid,
    DESCRIPTION as multi_rsi_desc,
)
from crabquant.strategies.ema_ribbon_reversal import (
    generate_signals as ema_ribbon_signals,
    generate_signals_matrix as ema_ribbon_matrix,
    DEFAULT_PARAMS as ema_ribbon_defaults,
    PARAM_GRID as ema_ribbon_grid,
    DESCRIPTION as ema_ribbon_desc,
)
from crabquant.strategies.bollinger_squeeze import (
    generate_signals as bollinger_squeeze_signals,
    generate_signals_matrix as bollinger_squeeze_matrix,
    DEFAULT_PARAMS as bollinger_squeeze_defaults,
    PARAM_GRID as bollinger_squeeze_grid,
    DESCRIPTION as bollinger_squeeze_desc,
)
from crabquant.strategies.ichimoku_trend import (
    generate_signals as ichimoku_signals,
    generate_signals_matrix as ichimoku_matrix,
    DEFAULT_PARAMS as ichimoku_defaults,
    PARAM_GRID as ichimoku_grid,
    DESCRIPTION as ichimoku_desc,
)
from crabquant.strategies.invented_momentum_rsi_atr import (
    generate_signals as invented_mrsa_signals,
    generate_signals_matrix as invented_mrsa_matrix,
    DEFAULT_PARAMS as invented_mrsa_defaults,
    PARAM_GRID as invented_mrsa_grid,
    DESCRIPTION as invented_mrsa_desc,
)
from crabquant.strategies.invented_momentum_rsi_stoch import (
    generate_signals as invented_mrss_signals,
    generate_signals_matrix as invented_mrss_matrix,
    DEFAULT_PARAMS as invented_mrss_defaults,
    PARAM_GRID as invented_mrss_grid,
    DESCRIPTION as invented_mrss_desc,
)
from crabquant.strategies.vpt_crossover import (
    generate_signals as vpt_crossover_signals,
    generate_signals_matrix as vpt_crossover_matrix,
    DEFAULT_PARAMS as vpt_crossover_defaults,
    PARAM_GRID as vpt_crossover_grid,
    DESCRIPTION as vpt_crossover_desc,
)
from crabquant.strategies.roc_ema_volume import (
    generate_signals as roc_ema_volume_signals,
    generate_signals_matrix as roc_ema_volume_matrix,
    DEFAULT_PARAMS as roc_ema_volume_defaults,
    PARAM_GRID as roc_ema_volume_grid,
    DESCRIPTION as roc_ema_volume_desc,
)
from crabquant.strategies.bb_stoch_macd import (
    generate_signals as bb_stoch_macd_signals,
    generate_signals_matrix as bb_stoch_macd_matrix,
    DEFAULT_PARAMS as bb_stoch_macd_defaults,
    PARAM_GRID as bb_stoch_macd_grid,
    DESCRIPTION as bb_stoch_macd_desc,
)
from crabquant.strategies.rsi_regime_dip import (
    generate_signals as rsi_regime_dip_signals,
    generate_signals_matrix as rsi_regime_dip_matrix,
    DEFAULT_PARAMS as rsi_regime_dip_defaults,
    PARAM_GRID as rsi_regime_dip_grid,
    DESCRIPTION as rsi_regime_dip_desc,
)
from crabquant.strategies.ema_crossover import (
    generate_signals as ema_crossover_signals,
    generate_signals_matrix as ema_crossover_matrix,
    DEFAULT_PARAMS as ema_crossover_defaults,
    PARAM_GRID as ema_crossover_grid,
    DESCRIPTION as ema_crossover_desc,
)
from crabquant.strategies.injected_momentum_atr_volume import (
    generate_signals as injected_momentum_atr_volume_signals,
    generate_signals_matrix as injected_momentum_atr_volume_matrix,
    DEFAULT_PARAMS as injected_momentum_atr_volume_defaults,
    PARAM_GRID as injected_momentum_atr_volume_grid,
    DESCRIPTION as injected_momentum_atr_volume_desc,
)
from crabquant.strategies.informed_simple_adaptive import (
    generate_signals as informed_simple_adaptive_signals,
    generate_signals_matrix as informed_simple_adaptive_matrix,
    DEFAULT_PARAMS as informed_simple_adaptive_defaults,
    PARAM_GRID as informed_simple_adaptive_grid,
    DESCRIPTION as informed_simple_adaptive_desc,
)

# Registry: name -> (fn, defaults, grid, description, matrix_fn)
STRATEGY_REGISTRY = {
    "rsi_crossover": (rsi_crossover_signals, rsi_crossover_defaults, rsi_crossover_grid, rsi_crossover_desc, rsi_crossover_matrix),
    "macd_momentum": (macd_momentum_signals, macd_momentum_defaults, macd_momentum_grid, macd_momentum_desc, macd_momentum_matrix),
    "adx_pullback": (adx_pullback_signals, adx_pullback_defaults, adx_pullback_grid, adx_pullback_desc, adx_pullback_matrix),
    "atr_channel_breakout": (atr_channel_signals, atr_channel_defaults, atr_channel_grid, atr_channel_desc, atr_channel_matrix),
    "volume_breakout": (volume_breakout_signals, volume_breakout_defaults, volume_breakout_grid, volume_breakout_desc, volume_breakout_matrix),
    "multi_rsi_confluence": (multi_rsi_signals, multi_rsi_defaults, multi_rsi_grid, multi_rsi_desc, multi_rsi_matrix),
    "ema_ribbon_reversal": (ema_ribbon_signals, ema_ribbon_defaults, ema_ribbon_grid, ema_ribbon_desc, ema_ribbon_matrix),
    "bollinger_squeeze": (bollinger_squeeze_signals, bollinger_squeeze_defaults, bollinger_squeeze_grid, bollinger_squeeze_desc, bollinger_squeeze_matrix),
    "ichimoku_trend": (ichimoku_signals, ichimoku_defaults, ichimoku_grid, ichimoku_desc, ichimoku_matrix),
    # Invented strategies
    "invented_momentum_rsi_atr": (invented_mrsa_signals, invented_mrsa_defaults, invented_mrsa_grid, invented_mrsa_desc, invented_mrsa_matrix),
    "invented_momentum_rsi_stoch": (invented_mrss_signals, invented_mrss_defaults, invented_mrss_grid, invented_mrss_desc, invented_mrss_matrix),
    # QF proven patterns
    "vpt_crossover": (vpt_crossover_signals, vpt_crossover_defaults, vpt_crossover_grid, vpt_crossover_desc, vpt_crossover_matrix),
    "roc_ema_volume": (roc_ema_volume_signals, roc_ema_volume_defaults, roc_ema_volume_grid, roc_ema_volume_desc, roc_ema_volume_matrix),
    "bb_stoch_macd": (bb_stoch_macd_signals, bb_stoch_macd_defaults, bb_stoch_macd_grid, bb_stoch_macd_desc, bb_stoch_macd_matrix),
    "rsi_regime_dip": (rsi_regime_dip_signals, rsi_regime_dip_defaults, rsi_regime_dip_grid, rsi_regime_dip_desc, rsi_regime_dip_matrix),
    "ema_crossover": (ema_crossover_signals, ema_crossover_defaults, ema_crossover_grid, ema_crossover_desc, ema_crossover_matrix),
    # New injected strategy
    "injected_momentum_atr_volume": (injected_momentum_atr_volume_signals, injected_momentum_atr_volume_defaults, injected_momentum_atr_volume_grid, injected_momentum_atr_volume_desc, injected_momentum_atr_volume_matrix),
    # New informed strategy
    "informed_simple_adaptive": (informed_simple_adaptive_signals, informed_simple_adaptive_defaults, informed_simple_adaptive_grid, informed_simple_adaptive_desc, informed_simple_adaptive_matrix),
}

# Diverse ticker list for cross-ticker validation (excludes low-liquidity / meme stocks)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",      # mega-cap tech
    "JPM", "GS", "V", "MA",                         # financials
    "JNJ", "UNH", "PFE", "ABBV",                     # healthcare
    "XOM", "CVX", "COP",                              # energy
    "CAT", "DE", "HON",                               # industrials
    "WMT", "TGT", "COST",                             # retail
    "PLD", "AMT",                                       # real estate
]

__all__ = ["STRATEGY_REGISTRY", "DEFAULT_TICKERS"]