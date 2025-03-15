"""
Cross-validation functionality for the backtesting system.

This module provides functionality for validating trading strategies
using cross-validation techniques.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

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

def run_cross_validation(
    data_source: Optional[DataSource] = None,
    strategy: Optional[Strategy] = None,
    portfolio_manager: Optional[PortfolioManager] = None,
    folds: int = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol: str = "BTC-USD",
    transaction_costs: float = 0.001,
    slippage: float = 0.0005,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run cross-validation on a trading strategy.
    
    Args:
        data_source: Data source for market data
        strategy: Strategy for generating signals
        portfolio_manager: Portfolio manager for executing trades
        folds: Number of folds for cross-validation
        start_date: Optional start date for the backtest
        end_date: Optional end date for the backtest
        symbol: Trading symbol
        transaction_costs: Transaction costs as a fraction of trade value
        slippage: Slippage as a fraction of price
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary with cross-validation results
    """
    logger.info(f"Running cross-validation with {folds} folds")
    
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
    
    # If no strategy is provided, use a simple moving average crossover
    if strategy is None:
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
    
    # Calculate fold size
    fold_size = len(data) // folds
    
    # Run backtest for each fold
    fold_results = []
    
    for i in range(folds):
        # Calculate fold indices
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < folds - 1 else len(data)
        
        # Split data into training and test sets
        test_data = data.iloc[start_idx:end_idx]
        
        # Create a data source for this fold
        fold_data_source = DataFrameSource(test_data)
        
        # Create and run backtester
        backtester = Backtester(
            data_source=fold_data_source,
            strategy=strategy,
            portfolio_manager=portfolio_manager,
            transaction_costs=transaction_costs,
            slippage=slippage
        )
        
        # Run backtest
        results = backtester.run(symbol=symbol)
        
        # Store results
        fold_results.append({
            "fold": i + 1,
            "start_date": test_data.index[0].strftime("%Y-%m-%d"),
            "end_date": test_data.index[-1].strftime("%Y-%m-%d"),
            "total_return": results.total_return,
            "sharpe_ratio": results.metrics.get("sharpe_ratio", 0),
            "max_drawdown": results.metrics.get("max_drawdown", 0),
            "num_trades": len(results.trades)
        })
    
    # Calculate aggregate statistics
    returns = [r["total_return"] for r in fold_results]
    sharpe_ratios = [r["sharpe_ratio"] for r in fold_results]
    
    cross_val_results = {
        "average_return": np.mean(returns),
        "std_return": np.std(returns),
        "max_return": max(returns),
        "min_return": min(returns),
        "sharpe_ratios": sharpe_ratios,
        "fold_results": fold_results
    }
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        with open(os.path.join(output_dir, "cross_validation_summary.txt"), "w") as f:
            f.write(f"Cross-Validation Summary\n")
            f.write(f"=======================\n\n")
            f.write(f"Number of folds: {folds}\n")
            f.write(f"Average return: {cross_val_results['average_return']:.4f}\n")
            f.write(f"Standard deviation: {cross_val_results['std_return']:.4f}\n")
            f.write(f"Max return: {cross_val_results['max_return']:.4f}\n")
            f.write(f"Min return: {cross_val_results['min_return']:.4f}\n\n")
            
            f.write(f"Fold results:\n")
            for result in fold_results:
                f.write(f"  Fold {result['fold']}:\n")
                f.write(f"    Period: {result['start_date']} to {result['end_date']}\n")
                f.write(f"    Return: {result['total_return']:.4f}\n")
                f.write(f"    Sharpe ratio: {result['sharpe_ratio']:.4f}\n")
                f.write(f"    Max drawdown: {result['max_drawdown']:.4f}\n")
                f.write(f"    Number of trades: {result['num_trades']}\n\n")
    
    return cross_val_results 