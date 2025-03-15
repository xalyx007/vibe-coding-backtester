"""
Monte Carlo simulation functionality for the backtesting system.

This module provides functionality for validating trading strategies
using Monte Carlo simulation techniques.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import random

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

def perturb_data(data: pd.DataFrame, volatility_factor: float = 0.5) -> pd.DataFrame:
    """
    Create a perturbed version of the input data for Monte Carlo simulation.
    
    Args:
        data: Original price data
        volatility_factor: Factor to scale the random perturbations
        
    Returns:
        Perturbed DataFrame
    """
    # Create a copy of the data
    perturbed = data.copy()
    
    # Calculate daily returns
    returns = data['close'].pct_change().dropna().values
    
    # Calculate volatility
    volatility = np.std(returns)
    
    # Generate random perturbations
    random_perturbations = np.random.normal(0, volatility * volatility_factor, len(data))
    
    # Apply perturbations to close prices
    perturbed_returns = returns + random_perturbations[1:len(returns)+1]
    perturbed_prices = data['close'].iloc[0] * np.cumprod(1 + np.append([0], perturbed_returns))
    
    # Update OHLC based on perturbed close prices
    perturbed['close'] = perturbed_prices
    perturbed['open'] = perturbed['close'] * (data['open'] / data['close'])
    perturbed['high'] = perturbed['close'] * (data['high'] / data['close'])
    perturbed['low'] = perturbed['close'] * (data['low'] / data['close'])
    
    return perturbed

def run_monte_carlo(
    data_source: Optional[DataSource] = None,
    strategy: Optional[Strategy] = None,
    portfolio_manager: Optional[PortfolioManager] = None,
    simulations: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol: str = "BTC-USD",
    transaction_costs: float = 0.001,
    slippage: float = 0.0005,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation on a trading strategy.
    
    Args:
        data_source: Data source for market data
        strategy: Strategy for generating signals
        portfolio_manager: Portfolio manager for executing trades
        simulations: Number of Monte Carlo simulations to run
        start_date: Optional start date for the backtest
        end_date: Optional end date for the backtest
        symbol: Trading symbol
        transaction_costs: Transaction costs as a fraction of trade value
        slippage: Slippage as a fraction of price
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary with Monte Carlo simulation results
    """
    logger.info(f"Running Monte Carlo simulation with {simulations} simulations")
    
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
    
    # Save original data if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        data.to_csv(os.path.join(output_dir, "base_data.csv"))
    
    # Run simulations
    simulation_results = []
    
    for i in range(simulations):
        # Perturb data for this simulation
        perturbed_data = perturb_data(data)
        
        # Create a data source for this simulation
        sim_data_source = DataFrameSource(perturbed_data)
        
        # Create and run backtester
        backtester = Backtester(
            data_source=sim_data_source,
            strategy=strategy,
            portfolio_manager=portfolio_manager,
            transaction_costs=transaction_costs,
            slippage=slippage
        )
        
        # Run backtest
        results = backtester.run(symbol=symbol)
        
        # Store results
        simulation_results.append({
            "simulation": i + 1,
            "total_return": results.total_return,
            "sharpe_ratio": results.metrics.get("sharpe_ratio", 0),
            "max_drawdown": results.metrics.get("max_drawdown", 0),
            "num_trades": len(results.trades)
        })
    
    # Calculate aggregate statistics
    returns = [r["total_return"] for r in simulation_results]
    returns_array = np.array(returns)
    
    # Calculate Value at Risk (VaR)
    var_95 = np.percentile(returns_array, 5)
    var_99 = np.percentile(returns_array, 1)
    
    monte_carlo_results = {
        "average_return": np.mean(returns),
        "var_95": var_95,
        "var_99": var_99,
        "max_return": max(returns),
        "min_return": min(returns),
        "simulation_results": simulation_results
    }
    
    # Save results if output directory is provided
    if output_dir:
        # Save summary
        with open(os.path.join(output_dir, "monte_carlo_summary.txt"), "w") as f:
            f.write(f"Monte Carlo Simulation Summary\n")
            f.write(f"==============================\n\n")
            f.write(f"Number of simulations: {simulations}\n")
            f.write(f"Average return: {monte_carlo_results['average_return']:.4f}\n")
            f.write(f"Value at Risk (95%): {monte_carlo_results['var_95']:.4f}\n")
            f.write(f"Value at Risk (99%): {monte_carlo_results['var_99']:.4f}\n")
            f.write(f"Max return: {monte_carlo_results['max_return']:.4f}\n")
            f.write(f"Min return: {monte_carlo_results['min_return']:.4f}\n\n")
            
            f.write(f"Distribution of returns:\n")
            bins = np.linspace(min(returns), max(returns), 10)
            hist, _ = np.histogram(returns, bins=bins)
            for i in range(len(hist)):
                f.write(f"  {bins[i]:.4f} to {bins[i+1]:.4f}: {hist[i]}\n")
    
    return monte_carlo_results 