"""
Monte Carlo Validation Script for Backtester

This script performs Monte Carlo simulations to validate the robustness
of the backtesting system by introducing random variations in the data
and analyzing the distribution of results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data import CSVDataSource, DataSource
from backtester.strategy import MovingAverageCrossover, BollingerBandsStrategy, RSIStrategy
from backtester.portfolio import BasicPortfolioManager
from backtester.core import Backtester
from backtester.utils.constants import SignalType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../output/logs/monte_carlo_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("monte_carlo_validation")

# Create results directory if it doesn't exist
os.makedirs("../../output/results/monte_carlo", exist_ok=True)


# Create a proper DataFrameSource class that implements the required abstract methods
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


def add_noise_to_data(data: pd.DataFrame, noise_level: float = 0.001) -> pd.DataFrame:
    """
    Add random noise to price data.
    
    Args:
        data: Original price data
        noise_level: Standard deviation of noise to add
        
    Returns:
        DataFrame with noise added
    """
    noisy_data = data.copy()
    
    # Add noise to price columns
    for col in ['open', 'high', 'low', 'close']:
        noise = np.random.normal(0, noise_level, len(data))
        noisy_data[col] = data[col] * (1 + noise)
    
    # Ensure high >= open, close, low and low <= open, close
    noisy_data['high'] = noisy_data[['high', 'open', 'close']].max(axis=1)
    noisy_data['low'] = noisy_data[['low', 'open', 'close']].min(axis=1)
    
    return noisy_data


def run_monte_carlo_simulation(
    base_data: pd.DataFrame, 
    strategy_class, 
    strategy_params: Dict[str, Any],
    num_simulations: int = 100,
    noise_level: float = 0.001
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulations by adding noise to the data.
    
    Args:
        base_data: Base price data
        strategy_class: Strategy class to use
        strategy_params: Parameters for the strategy
        num_simulations: Number of simulations to run
        noise_level: Level of noise to add to the data
        
    Returns:
        Dictionary with simulation results
    """
    logger.info(f"Running {num_simulations} Monte Carlo simulations with {strategy_class.__name__}")
    
    results = []
    
    for i in tqdm(range(num_simulations), desc="Running simulations"):
        # Add noise to data
        noisy_data = add_noise_to_data(base_data, noise_level)
        
        # Create strategy and portfolio manager
        strategy = strategy_class(**strategy_params)
        portfolio_manager = BasicPortfolioManager(initial_capital=10000)
        
        # Create a data source from the DataFrame
        data_source = DataFrameSource(noisy_data)
        
        # Create and run backtester
        backtester = Backtester(
            data_source=data_source,
            strategy=strategy,
            portfolio_manager=portfolio_manager,
            transaction_costs=0.001,
            slippage=0.0005
        )
        
        # Run backtest
        backtest_results = backtester.run()
        
        # Extract key metrics
        results.append({
            "final_value": backtest_results.portfolio_values['portfolio_value'].iloc[-1],
            "total_return": backtest_results.total_return,
            "max_drawdown": backtest_results.metrics.get('max_drawdown', 0),
            "sharpe_ratio": backtest_results.metrics.get('sharpe_ratio', 0),
            "num_trades": len(backtest_results.trades)
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    stats = {
        "mean_return": results_df['total_return'].mean(),
        "std_return": results_df['total_return'].std(),
        "min_return": results_df['total_return'].min(),
        "max_return": results_df['total_return'].max(),
        "mean_final_value": results_df['final_value'].mean(),
        "std_final_value": results_df['final_value'].std(),
        "mean_max_drawdown": results_df['max_drawdown'].mean(),
        "mean_sharpe_ratio": results_df['sharpe_ratio'].mean(),
        "mean_num_trades": results_df['num_trades'].mean(),
        "std_num_trades": results_df['num_trades'].std()
    }
    
    # Calculate confidence intervals (95%)
    ci_lower = np.percentile(results_df['total_return'], 2.5)
    ci_upper = np.percentile(results_df['total_return'], 97.5)
    stats["return_95ci_lower"] = ci_lower
    stats["return_95ci_upper"] = ci_upper
    
    # Plot histogram of returns
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['total_return'], bins=20, alpha=0.7)
    plt.axvline(stats["mean_return"], color='r', linestyle='dashed', linewidth=2, label=f"Mean: {stats['mean_return']:.4f}")
    plt.axvline(ci_lower, color='g', linestyle='dashed', linewidth=2, label=f"2.5%: {ci_lower:.4f}")
    plt.axvline(ci_upper, color='g', linestyle='dashed', linewidth=2, label=f"97.5%: {ci_upper:.4f}")
    plt.title(f'Distribution of Returns - {strategy_class.__name__}')
    plt.xlabel('Total Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'../../output/results/monte_carlo/{strategy_class.__name__}_returns_histogram.png')
    
    # Plot histogram of final values
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['final_value'], bins=20, alpha=0.7)
    plt.axvline(stats["mean_final_value"], color='r', linestyle='dashed', linewidth=2, label=f"Mean: {stats['mean_final_value']:.2f}")
    plt.title(f'Distribution of Final Portfolio Values - {strategy_class.__name__}')
    plt.xlabel('Final Portfolio Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'../../output/results/monte_carlo/{strategy_class.__name__}_final_values_histogram.png')
    
    # Plot scatter of returns vs drawdowns
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['total_return'], results_df['max_drawdown'], alpha=0.5)
    plt.title(f'Returns vs Max Drawdown - {strategy_class.__name__}')
    plt.xlabel('Total Return')
    plt.ylabel('Max Drawdown')
    plt.savefig(f'../../output/results/monte_carlo/{strategy_class.__name__}_return_vs_drawdown.png')
    
    # Save results to CSV
    results_df.to_csv(f'../../output/results/monte_carlo/{strategy_class.__name__}_simulation_results.csv', index=False)
    
    return {
        "results": results_df,
        "stats": stats
    }


def run_parameter_sensitivity_analysis(
    base_data: pd.DataFrame,
    strategy_class,
    base_params: Dict[str, Any],
    param_ranges: Dict[str, List[Any]],
    num_simulations: int = 10
) -> Dict[str, Any]:
    """
    Run parameter sensitivity analysis.
    
    Args:
        base_data: Base price data
        strategy_class: Strategy class to use
        base_params: Base parameters for the strategy
        param_ranges: Dictionary of parameter names and ranges to test
        num_simulations: Number of simulations to run for each parameter set
        
    Returns:
        Dictionary with sensitivity analysis results
    """
    logger.info(f"Running parameter sensitivity analysis for {strategy_class.__name__}")
    
    results = []
    
    # Generate all parameter combinations
    param_combinations = []
    param_names = list(param_ranges.keys())
    
    def generate_combinations(index, current_params):
        if index == len(param_names):
            param_combinations.append(current_params.copy())
            return
        
        param_name = param_names[index]
        for param_value in param_ranges[param_name]:
            current_params[param_name] = param_value
            generate_combinations(index + 1, current_params)
    
    generate_combinations(0, base_params.copy())
    
    # Run simulations for each parameter combination
    for params in tqdm(param_combinations, desc="Testing parameter combinations"):
        # Run Monte Carlo simulations for this parameter set
        simulation_results = run_monte_carlo_simulation(
            base_data=base_data,
            strategy_class=strategy_class,
            strategy_params=params,
            num_simulations=num_simulations,
            noise_level=0.001
        )
        
        # Extract key statistics
        results.append({
            **params,
            "mean_return": simulation_results["stats"]["mean_return"],
            "std_return": simulation_results["stats"]["std_return"],
            "mean_max_drawdown": simulation_results["stats"]["mean_max_drawdown"],
            "mean_sharpe_ratio": simulation_results["stats"]["mean_sharpe_ratio"],
            "mean_num_trades": simulation_results["stats"]["mean_num_trades"]
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(f'../../output/results/monte_carlo/{strategy_class.__name__}_parameter_sensitivity.csv', index=False)
    
    # Create visualizations for each parameter
    for param_name in param_names:
        plt.figure(figsize=(10, 6))
        
        # Group by parameter value and calculate mean return
        param_values = sorted(param_ranges[param_name])
        mean_returns = [results_df[results_df[param_name] == val]['mean_return'].mean() for val in param_values]
        
        plt.plot(param_values, mean_returns, marker='o')
        plt.title(f'Parameter Sensitivity: {param_name} vs Mean Return')
        plt.xlabel(param_name)
        plt.ylabel('Mean Return')
        plt.grid(True)
        plt.savefig(f'../../output/results/monte_carlo/{strategy_class.__name__}_{param_name}_sensitivity.png')
    
    return {
        "results": results_df
    }


def run_monte_carlo_validation():
    """Run Monte Carlo validation for different strategies."""
    logger.info("Starting Monte Carlo validation")
    
    # Create synthetic data for testing
    base_data = create_synthetic_data(days=252, trend='random')
    
    # Save the base data for reference
    base_data.to_csv('../../output/results/monte_carlo/base_data.csv')
    
    # Run Monte Carlo simulations for different strategies
    
    # 1. Moving Average Crossover
    ma_results = run_monte_carlo_simulation(
        base_data=base_data,
        strategy_class=MovingAverageCrossover,
        strategy_params={"short_window": 10, "long_window": 30},
        num_simulations=100,
        noise_level=0.001
    )
    
    # 2. RSI Strategy
    rsi_results = run_monte_carlo_simulation(
        base_data=base_data,
        strategy_class=RSIStrategy,
        strategy_params={"window": 14, "overbought": 70, "oversold": 30},
        num_simulations=100,
        noise_level=0.001
    )
    
    # 3. Bollinger Bands
    bb_results = run_monte_carlo_simulation(
        base_data=base_data,
        strategy_class=BollingerBandsStrategy,
        strategy_params={"window": 20, "num_std": 2},
        num_simulations=100,
        noise_level=0.001
    )
    
    # Run parameter sensitivity analysis for Moving Average Crossover
    ma_sensitivity = run_parameter_sensitivity_analysis(
        base_data=base_data,
        strategy_class=MovingAverageCrossover,
        base_params={"short_window": 10, "long_window": 30},
        param_ranges={
            "short_window": [5, 10, 15, 20],
            "long_window": [20, 30, 40, 50]
        },
        num_simulations=10
    )
    
    # Create summary report
    with open("../../output/results/monte_carlo/summary.txt", "w") as f:
        f.write(f"Monte Carlo Validation Summary\n")
        f.write(f"============================\n\n")
        
        f.write(f"Moving Average Crossover Strategy:\n")
        f.write(f"  Mean Return: {ma_results['stats']['mean_return']:.4f}\n")
        f.write(f"  Std Dev Return: {ma_results['stats']['std_return']:.4f}\n")
        f.write(f"  95% CI: [{ma_results['stats']['return_95ci_lower']:.4f}, {ma_results['stats']['return_95ci_upper']:.4f}]\n")
        f.write(f"  Mean Max Drawdown: {ma_results['stats']['mean_max_drawdown']:.4f}\n")
        f.write(f"  Mean Sharpe Ratio: {ma_results['stats']['mean_sharpe_ratio']:.4f}\n")
        f.write(f"  Mean Number of Trades: {ma_results['stats']['mean_num_trades']:.1f}\n\n")
        
        f.write(f"RSI Strategy:\n")
        f.write(f"  Mean Return: {rsi_results['stats']['mean_return']:.4f}\n")
        f.write(f"  Std Dev Return: {rsi_results['stats']['std_return']:.4f}\n")
        f.write(f"  95% CI: [{rsi_results['stats']['return_95ci_lower']:.4f}, {rsi_results['stats']['return_95ci_upper']:.4f}]\n")
        f.write(f"  Mean Max Drawdown: {rsi_results['stats']['mean_max_drawdown']:.4f}\n")
        f.write(f"  Mean Sharpe Ratio: {rsi_results['stats']['mean_sharpe_ratio']:.4f}\n")
        f.write(f"  Mean Number of Trades: {rsi_results['stats']['mean_num_trades']:.1f}\n\n")
        
        f.write(f"Bollinger Bands Strategy:\n")
        f.write(f"  Mean Return: {bb_results['stats']['mean_return']:.4f}\n")
        f.write(f"  Std Dev Return: {bb_results['stats']['std_return']:.4f}\n")
        f.write(f"  95% CI: [{bb_results['stats']['return_95ci_lower']:.4f}, {bb_results['stats']['return_95ci_upper']:.4f}]\n")
        f.write(f"  Mean Max Drawdown: {bb_results['stats']['mean_max_drawdown']:.4f}\n")
        f.write(f"  Mean Sharpe Ratio: {bb_results['stats']['mean_sharpe_ratio']:.4f}\n")
        f.write(f"  Mean Number of Trades: {bb_results['stats']['mean_num_trades']:.1f}\n\n")
        
        f.write(f"Parameter Sensitivity Analysis:\n")
        f.write(f"  See ../../output/results/monte_carlo/MovingAverageCrossover_parameter_sensitivity.csv for details\n")
    
    logger.info("Monte Carlo validation complete")
    logger.info("Results saved to ../../output/results/monte_carlo/")
    
    return True


if __name__ == "__main__":
    success = run_monte_carlo_validation()
    sys.exit(0 if success else 1) 