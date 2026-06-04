"""
CrabQuant Parallel Backtesting

Run strategy backtests across multiple tickers in parallel using ProcessPoolExecutor.
Each worker loads its own data (cached via pickle) and runs run_vectorized() for one ticker.

Resource-aware: Uses ResourceMonitor to dynamically throttle worker count based on
available CPU and RAM. Prevents OOM kills and CPU thrashing during heavy backtesting.
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Optional

from crabquant.engine.resource_monitor import ResourceMonitor, compute_optimal_workers

logger = logging.getLogger(__name__)


def _worker_backtest(args: tuple) -> list[dict]:
    """
    Worker function executed in a subprocess.
    Loads data, generates signals, runs vectorized backtest, returns serialized results.
    
    Args:
        args: (strategy_name, ticker, param_grid, period)
    
    Returns:
        List of BacktestResult dicts (serializable for IPC)
    """
    strategy_name, ticker, param_grid, period = args

    try:
        from crabquant.data import load_data
        from crabquant.engine.backtest import BacktestEngine
        from crabquant.strategies import STRATEGY_REGISTRY

        if strategy_name not in STRATEGY_REGISTRY:
            logger.error(f"Unknown strategy: {strategy_name}")
            return []

        _, defaults, _, _, matrix_fn = STRATEGY_REGISTRY[strategy_name]

        if not param_grid:
            logger.info(f"[{ticker}] No param grid for {strategy_name}, skipping")
            return []

        df = load_data(ticker, period=period)
        engine = BacktestEngine()

        entries_df, exits_df, param_list = matrix_fn(df, param_grid)
        if entries_df.empty or len(param_list) == 0:
            logger.info(f"[{ticker}] No param combos generated")
            return []

        results = engine.run_vectorized(df, entries_df, exits_df, param_list, strategy_name, ticker)
        return [asdict(r) for r in results]

    except Exception as e:
        logger.error(f"[{ticker}] Worker error: {e}")
        return [{"_error": True, "ticker": ticker, "strategy": strategy_name, "error": str(e)}]


def parallel_backtest(
    strategy_name: str,
    tickers: list[str],
    param_grid: dict,
    max_workers: Optional[int] = None,
    period: str = "2y",
    *,
    resource_monitor: Optional[ResourceMonitor] = None,
) -> list:
    """
    Run a strategy across multiple tickers in parallel.

    Uses ResourceMonitor to dynamically throttle worker count based on
    available CPU and RAM, preventing OOM kills and CPU thrashing.

    Args:
        strategy_name: Strategy name from STRATEGY_REGISTRY
        tickers: List of ticker symbols to backtest
        param_grid: Dict of param_name -> list of values
        max_workers: Max parallel workers (default: min(cpu_count, len(tickers)))
        period: Data period (default: '2y')
        resource_monitor: Optional ResourceMonitor for adaptive throttling.
            If None, creates one for the duration of this call.

    Returns:
        Flat list of BacktestResult objects from all tickers
    """
    from crabquant.engine.backtest import BacktestResult

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(tickers))

    # Apply resource-aware throttling
    actual_workers = compute_optimal_workers(max_workers)
    actual_workers = min(actual_workers, len(tickers))

    args_list = [(strategy_name, t, param_grid, period) for t in tickers]

    all_results = []
    completed = 0
    errors = []

    logger.info(
        f"Starting parallel backtest: {strategy_name} across {len(tickers)} tickers "
        f"(requested {max_workers} workers, using {actual_workers} after resource check)"
    )

    t0 = time.time()

    # Use resource monitor for adaptive throttling if provided
    own_monitor = resource_monitor is None
    if own_monitor:
        resource_monitor = ResourceMonitor()

    with resource_monitor:
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(_worker_backtest, args): args[1] for args in args_list}

            for future in as_completed(futures):
                ticker = futures[future]
                completed += 1
                try:
                    result_dicts = future.result()
                    for rd in result_dicts:
                        if rd.get("_error"):
                            errors.append(f"{ticker}: {rd['error']}")
                            continue
                        all_results.append(BacktestResult(**rd))

                    status = f"[{completed}/{len(tickers)}]"
                    if result_dicts and not result_dicts[0].get("_error"):
                        n_results = len(result_dicts)
                        logger.info(f"{status} {ticker}: {n_results} results")
                    else:
                        logger.info(f"{status} {ticker}: FAILED")

                except Exception as e:
                    errors.append(f"{ticker}: {e}")
                    logger.error(f"[{completed}/{len(tickers)}] {ticker}: {e}")

    elapsed = time.time() - t0
    traded = [r for r in all_results if r.num_trades > 0]
    passed = [r for r in all_results if r.passed]

    logger.info(f"Parallel backtest complete in {elapsed:.1f}s: "
                f"{len(all_results)} results, {len(traded)} with trades, "
                f"{len(passed)} passed, {len(errors)} errors")

    if errors:
        for err in errors:
            logger.warning(f"Error: {err}")

    return all_results
