"""
Walk-forward optimization functionality for the backtesting system.

This module provides functionality for validating trading strategies
using walk-forward optimization techniques.
"""

from typing import Dict, Any, Optional, List, Tuple, Callable
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import itertools

from backtester.core import Backtester
from backtester.data import DataSource
from backtester.strategy import Strategy
from backtester.portfolio import PortfolioManager

logger = logging.getLogger(__name__)

def create_synthetic_data(days: int = 252, trend: str = 'random', volatility: float = 0.01) -> pd.DataFrame:
    """
    Create synthetic price data for testing.
    
    Args:
        days: Number of days of data
        trend: Type of trend ('up', 'down', 'random', 'cycle')
        volatility: Daily volatility
        
    Returns:
        DataFrame with OHLCV data
    """
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(days)]
    
    # Generate price series based on trend
    if trend == 'up':
        drift = 0.0005  # Positive drift
        prices = 100 * np.cumprod(1 + np.random.normal(drift, volatility, days))
    elif trend == 'down':
        drift = -0.0005  # Negative drift
        prices = 100 * np.cumprod(1 + np.random.normal(drift, volatility, days))
    elif trend == 'cycle':
        # Create a cyclical pattern
        t = np.linspace(0, 4*np.pi, days)
        cycle = np.sin(t) * 20
        prices = 100 + cycle + np.cumsum(np.random.normal(0, volatility, days))
    else:  # random
        prices = 100 * np.cumprod(1 + np.random.normal(0, volatility, days))
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.uniform(100, 1000, days)
    })
    
    data = data.set_index('timestamp')
    return data

def run_walk_forward(
    data_source: Optional[DataSource] = None,
    strategy: Optional[Strategy] = None,
    portfolio_manager: Optional[PortfolioManager] = None,
    window_size: int = 60,
    step_size: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol: str = "BTC-USD",
    transaction_costs: float = 0.001,
    slippage: float = 0.0005,
    parameter_grid: Optional[Dict[str, List[Any]]] = None,
    strategy_factory: Optional[Callable[..., Strategy]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run walk-forward optimization on a trading strategy.
    
    Args:
        data_source: Data source for market data
        strategy: Strategy for generating signals (used if strategy_factory is None)
        portfolio_manager: Portfolio manager for executing trades
        window_size: Size of each window in days
        step_size: Size of each step in days
        start_date: Optional start date for the backtest
        end_date: Optional end date for the backtest
        symbol: Trading symbol
        transaction_costs: Transaction costs as a fraction of trade value
        slippage: Slippage as a fraction of price
        parameter_grid: Dictionary of parameter names and values to optimize
        strategy_factory: Function to create a strategy with given parameters
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary with walk-forward optimization results
    """
    logger.info(f"Running walk-forward optimization with window size {window_size} and step size {step_size}")
    
    # If no data source is provided, create synthetic data
    if data_source is None:
        logger.warning("No data source provided. Using synthetic data.")
        data = create_synthetic_data(days=252, trend='random')
        
        # Create a simple data source from the DataFrame
        from backtester.data import DataSource
        class DataFrameSource(DataSource):
            def __init__(self, data, event_bus=None):
                super().__init__(event_bus)
                self._data = data
                
            def load_data(self, **kwargs) -> pd.DataFrame:
                """Load data from the DataFrame."""
                return self._data
            
            def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
                """No preprocessing needed as data is already in the correct format."""
                return data
        
        data_source = DataFrameSource(data)
    else:
        # Load data from the provided data source
        data = data_source.load_data()
        data = data_source.preprocess_data(data)
    
    # If no strategy factory is provided but parameter grid is, use a default factory
    if parameter_grid is not None and strategy_factory is None:
        if strategy is None:
            logger.warning("No strategy provided. Using MovingAverageCrossover.")
            from backtester.strategy import MovingAverageCrossover
            
            def default_factory(**kwargs):
                return MovingAverageCrossover(**kwargs)
            
            strategy_factory = default_factory
            
            # Default parameter grid for MovingAverageCrossover if none provided
            if parameter_grid is None:
                parameter_grid = {
                    'short_window': [5, 10, 20],
                    'long_window': [30, 50, 100]
                }
        else:
            # Use the provided strategy class but create new instances with different parameters
            strategy_class = strategy.__class__
            
            def default_factory(**kwargs):
                return strategy_class(**kwargs)
            
            strategy_factory = default_factory
    
    # If no strategy is provided and no factory, use a simple moving average crossover
    if strategy is None and strategy_factory is None:
        logger.warning("No strategy provided. Using MovingAverageCrossover.")
        from backtester.strategy import MovingAverageCrossover
        strategy = MovingAverageCrossover(short_window=10, long_window=30)
    
    # If no portfolio manager is provided, use a basic one
    if portfolio_manager is None:
        logger.warning("No portfolio manager provided. Using BasicPortfolioManager.")
        from backtester.portfolio import BasicPortfolioManager
        portfolio_manager = BasicPortfolioManager(initial_capital=10000)
    
    # Filter data by date if provided
    if start_date:
        data = data[data.index >= pd.Timestamp(start_date)]
    if end_date:
        data = data[data.index <= pd.Timestamp(end_date)]
    
    # Calculate number of windows
    total_days = len(data)
    num_windows = max(1, (total_days - window_size) // step_size + 1)
    
    window_results = []
    
    for i in range(num_windows):
        # Calculate window indices
        train_start_idx = i * step_size
        train_end_idx = train_start_idx + window_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + step_size, total_days)
        
        # Split data into training and test sets
        train_data = data.iloc[train_start_idx:train_end_idx]
        test_data = data.iloc[test_start_idx:test_end_idx]
        
        # Skip if test data is empty
        if len(test_data) == 0:
            continue
        
        # If parameter optimization is enabled
        if parameter_grid is not None and strategy_factory is not None:
            # Generate all parameter combinations
            param_names = list(parameter_grid.keys())
            param_values = list(parameter_grid.values())
            param_combinations = list(itertools.product(*param_values))
            
            best_return = -float('inf')
            best_params = None
            best_strategy = None
            
            # Test each parameter combination on the training data
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                
                # Create strategy with these parameters
                current_strategy = strategy_factory(**param_dict)
                
                # Create a data source for the training data
                train_data_source = DataFrameSource(train_data)
                
                # Create and run backtester
                backtester = Backtester(
                    data_source=train_data_source,
                    strategy=current_strategy,
                    portfolio_manager=portfolio_manager,
                    transaction_costs=transaction_costs,
                    slippage=slippage
                )
                
                # Run backtest
                results = backtester.run(symbol=symbol)
                
                # Check if this is the best strategy so far
                if results.total_return > best_return:
                    best_return = results.total_return
                    best_params = param_dict
                    best_strategy = current_strategy
            
            # Use the best strategy for the test data
            strategy_to_use = best_strategy
        else:
            # Use the provided strategy
            strategy_to_use = strategy
        
        # Create a data source for the test data
        test_data_source = DataFrameSource(test_data)
        
        # Create and run backtester on test data
        backtester = Backtester(
            data_source=test_data_source,
            strategy=strategy_to_use,
            portfolio_manager=portfolio_manager,
            transaction_costs=transaction_costs,
            slippage=slippage
        )
        
        # Run backtest
        results = backtester.run(symbol=symbol)
        
        # Store results
        window_result = {
            "window": i + 1,
            "train_start_date": train_data.index[0].strftime("%Y-%m-%d"),
            "train_end_date": train_data.index[-1].strftime("%Y-%m-%d"),
            "test_start_date": test_data.index[0].strftime("%Y-%m-%d"),
            "test_end_date": test_data.index[-1].strftime("%Y-%m-%d"),
            "total_return": results.total_return,
            "sharpe_ratio": results.metrics.get("sharpe_ratio", 0),
            "max_drawdown": results.metrics.get("max_drawdown", 0),
            "num_trades": len(results.trades)
        }
        
        # Add best parameters if parameter optimization was used
        if parameter_grid is not None and strategy_factory is not None and best_params is not None:
            window_result["best_params"] = best_params
        
        window_results.append(window_result)
    
    # Calculate aggregate statistics
    returns = [r["total_return"] for r in window_results]
    
    walk_forward_results = {
        "average_return": np.mean(returns) if returns else 0.0,
        "best_window_return": max(returns) if returns else 0.0,
        "worst_window_return": min(returns) if returns else 0.0,
        "window_results": window_results
    }
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        with open(os.path.join(output_dir, "walk_forward_summary.txt"), "w") as f:
            f.write(f"Walk-Forward Optimization Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Window size: {window_size} days\n")
            f.write(f"Step size: {step_size} days\n")
            f.write(f"Number of windows: {len(window_results)}\n")
            f.write(f"Average return: {walk_forward_results['average_return']:.4f}\n")
            f.write(f"Best window return: {walk_forward_results['best_window_return']:.4f}\n")
            f.write(f"Worst window return: {walk_forward_results['worst_window_return']:.4f}\n\n")
            
            f.write(f"Window results:\n")
            for result in window_results:
                f.write(f"  Window {result['window']}:\n")
                f.write(f"    Training period: {result['train_start_date']} to {result['train_end_date']}\n")
                f.write(f"    Testing period: {result['test_start_date']} to {result['test_end_date']}\n")
                f.write(f"    Return: {result['total_return']:.4f}\n")
                f.write(f"    Sharpe ratio: {result['sharpe_ratio']:.4f}\n")
                f.write(f"    Max drawdown: {result['max_drawdown']:.4f}\n")
                f.write(f"    Number of trades: {result['num_trades']}\n")
                if "best_params" in result:
                    f.write(f"    Best parameters: {result['best_params']}\n")
                f.write("\n")
    
    return walk_forward_results 