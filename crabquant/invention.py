"""
Strategy invention and improvement system.
"""

import json
import tempfile
from pathlib import Path
import importlib.util
import sys

from crabquant.strategies import STRATEGY_REGISTRY

RESULTS_DIR = Path(__file__).parent.parent / "results"


def analyze_market_data(tickers: list = None) -> dict:
    """Analyze market data to identify regimes and patterns."""
    
    # Mock analysis for now - in reality this would analyze actual market data
    return {
        "momentum_tickers": ["CAT", "SPY", "JPM", "XOM", "GLD"],
        "mean_reversion_tickers": ["NVDA", "TSLA", "NFLX"], 
        "high_volatility_tickers": ["NVDA", "TSLA", "NFLX"],
        "trending_tickers": ["AAPL", "NVDA", "CAT", "SPY", "JPM"],
        "range_bound_tickers": ["JNJ", "ORCL", "PLTR"]
    }


def get_strategy_catalog() -> str:
    """Get descriptions of all existing strategies for the inventor to learn from."""
    catalog = []
    for name, (_, _, _, _, desc) in STRATEGY_REGISTRY.items():
        catalog.append(f"  - {name}: {desc}")
    return "\n".join(catalog)


def get_top_winners_summary() -> str:
    """Get summary of top winning strategies for pattern analysis."""
    winners_file = RESULTS_DIR / "winners" / "winners.json"
    if not winners_file.exists():
        return "  No winners yet."

    with open(winners_file) as f:
        winners = json.load(f)

    lines = []
    for w in winners[:10]:
        lines.append(
            f"  - {w['ticker']}/{w['strategy']}: Sharpe={w['sharpe']:.2f}, "
            f"Return={w['return']:.1%}, Trades={w['trades']}, "
            f"Params={json.dumps(w['params'])}"
        )
    return "\n".join(lines)


def get_market_regime_summary() -> str:
    """Get summary of market regimes for the current dataset."""
    regime_data = analyze_market_data()
    
    return (
        "  Current market analysis shows:\n"
        f"    - Momentum: {', '.join(regime_data['momentum_tickers'])}\n"
        f"    - Mean-reversion: {', '.join(regime_data['mean_reversion_tickers'])}\n"
        f"    - High volatility: {', '.join(regime_data['high_volatility_tickers'])}\n"
        f"    - Trending: {', '.join(regime_data['trending_tickers'])}\n"
        f"    - Range-bound: {', '.join(regime_data['range_bound_tickers'])}"
    )


def test_strategy_code(strategy_code: str, strategy_name: str) -> bool:
    """Test if strategy code can be imported and has required functions."""
    try:
        # Create temporary file for the strategy
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(strategy_code)
            temp_file = f.name
        
        # Try to import the module
        spec = importlib.util.spec_from_file_location(strategy_name, temp_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check for required functions
        required_functions = ['generate_signals', 'generate_signals_matrix']
        has_required = all(hasattr(module, func) for func in required_functions)
        
        # Check for required attributes
        required_attrs = ['DEFAULT_PARAMS', 'PARAM_GRID', 'DESCRIPTION']
        has_attrs = all(hasattr(module, attr) for attr in required_attrs)
        
        # Cleanup
        Path(temp_file).unlink()
        
        return has_required and has_attrs
        
    except Exception as e:
        print(f"Error testing strategy {strategy_name}: {e}")
        return False


def save_invented_strategy(strategy_name: str, strategy_code: str) -> bool:
    """Save invented strategy to the strategies directory."""
    try:
        strategy_file = Path(__file__).parent / "strategies" / f"{strategy_name}.py"
        
        with open(strategy_file, 'w') as f:
            f.write(strategy_code)
        
        # Reload strategies module to include new strategy
        import crabquant.strategies
        importlib.reload(crabquant.strategies)
        
        return True
        
    except Exception as e:
        print(f"Error saving strategy {strategy_name}: {e}")
        return False


def generate_invention_prompt(insights: dict, market_data: dict) -> str:
    """Generate a prompt for the strategy inventor based on analysis results."""
    
    total_results = insights.get("total_results", 0)
    total_winners = insights.get("total_winners", 0)
    win_rate = insights.get("win_rate", 0)
    
    strategy_stats = insights.get("strategy_stats", {})
    ticker_stats = insights.get("ticker_stats", {})
    
    top_strategies = sorted(
        strategy_stats.items(),
        key=lambda x: x[1].get("win_rate", 0),
        reverse=True
    )[:5]
    
    top_tickers = sorted(
        ticker_stats.items(),
        key=lambda x: x[1].get("win_rate", 0),
        reverse=True
    )[:5]
    
    prompt = f"""
You are the CrabQuant Strategy Inventor. Your task is to create a new trading strategy.

Current Performance Overview:
- Total Results Analyzed: {total_results:,}
- Winning Strategies: {total_winners:,}
- Overall Win Rate: {win_rate:.1%}

Top Performing Strategies:
{chr(10).join(f"- {name}: {stat['win_rate']:.1%} win rate ({stat.get('won', 0)}/{stat.get('tested', 0)})" 
               for name, stat in top_strategies)}

Best Performing Tickers:
{chr(10).join(f"- {name}: {stat['win_rate']:.1%} win rate ({stat.get('won', 0)}/{stat.get('tested', 0)})" 
               for name, stat in top_tickers)}

Existing Strategies to Learn From:
{get_strategy_catalog()}

Market Regimes:
{get_market_regime_summary()}

Task:
1. Analyze what's working and what's not
2. Create a NEW strategy file at crabquant/strategies/invented_XXXX.py
3. The strategy must work on at least 2 of AAPL, NVDA, CAT, SPY
4. Use pandas_ta indicators properly (ADX_14, BBU_20_2.0_2.0)
5. Include proper DEFAULT_PARAMS, PARAM_GRID, and DESCRIPTION

Strategy Requirements:
- generate_signals(df, params) -> (entries: pd.Series[bool], exits: pd.Series[bool])
- DEFAULT_PARAMS dict with reasonable defaults
- PARAM_GRID dict with 3+ values per parameter
- DESCRIPTION explaining when the strategy works
- Must handle df columns: open, high, low, close, volume (all lowercase)

Output your strategy code directly. Focus on what works based on the data.
"""

    return prompt.strip()