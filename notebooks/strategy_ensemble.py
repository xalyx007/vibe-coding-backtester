#!/usr/bin/env python
"""
Strategy Ensemble Example

This script demonstrates how to create and backtest a strategy ensemble using the Modular Backtesting System.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Import backtester modules
from backtester.data import CSVDataSource
from backtester.strategy import MovingAverageCrossover, StrategyEnsemble
from backtester.portfolio import SimplePortfolioManager
from backtester.backtester import Backtester
from backtester.utils.metrics import calculate_metrics
from backtester.utils.visualization import plot_equity_curve, plot_drawdown

def main():
    """Run a strategy ensemble example."""
    print("Running strategy ensemble example...")
    
    # Create a data source
    print("Loading data...")
    data_source = CSVDataSource(
        file_path="data/BTC-USD.csv",
        date_format="%Y-%m-%d",
        timestamp_column="Date",
        open_column="Open",
        high_column="High",
        low_column="Low",
        close_column="Close",
        volume_column="Volume"
    )
    
    # Load the data
    data = data_source.load_data()
    print(f"Loaded {len(data)} data points.")
    
    # Create individual strategies
    print("Creating strategies...")
    ma_fast = MovingAverageCrossover(
        short_window=10,
        long_window=30
    )
    
    ma_medium = MovingAverageCrossover(
        short_window=20,
        long_window=50
    )
    
    ma_slow = MovingAverageCrossover(
        short_window=50,
        long_window=200
    )
    
    # Create a strategy ensemble
    print("Creating strategy ensemble...")
    ensemble = StrategyEnsemble(
        strategies=[ma_fast, ma_medium, ma_slow],
        weights=[0.4, 0.3, 0.3]  # Weights should sum to 1
    )
    
    # Create a portfolio manager
    print("Creating portfolio manager...")
    portfolio_manager = SimplePortfolioManager(
        initial_capital=10000,
        position_size=0.1
    )
    
    # Run individual backtests for comparison
    print("Running individual backtests for comparison...")
    results = {}
    
    # Run backtest for each individual strategy
    for name, strategy in [("MA Fast", ma_fast), ("MA Medium", ma_medium), ("MA Slow", ma_slow)]:
        print(f"Running backtest for {name}...")
        backtester = Backtester(
            data=data,
            strategy=strategy,
            portfolio_manager=portfolio_manager
        )
        results[name] = backtester.run()
    
    # Run backtest for the ensemble
    print("Running backtest for the ensemble...")
    backtester = Backtester(
        data=data,
        strategy=ensemble,
        portfolio_manager=portfolio_manager
    )
    results["Ensemble"] = backtester.run()
    
    # Calculate metrics for all strategies
    print("Calculating performance metrics...")
    metrics = {}
    for name, result in results.items():
        metrics[name] = calculate_metrics(result)
    
    # Display metrics
    print("\nPerformance Metrics:")
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df[['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']])
    
    # Save metrics to CSV
    metrics_df.to_csv("notebooks/assets/ensemble_metrics.csv")
    
    # Plot equity curves for all strategies
    print("\nPlotting equity curves...")
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        plt.plot(result.index, result['equity_curve'], label=name)
    
    plt.title('Equity Curves')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig("notebooks/assets/ensemble_equity_curves.png")
    
    # Plot drawdowns for all strategies
    print("Plotting drawdowns...")
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        plt.plot(result.index, result['drawdown'], label=name)
    
    plt.title('Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("notebooks/assets/ensemble_drawdowns.png")
    
    # Plot bar chart of key metrics
    print("Plotting key metrics comparison...")
    plt.figure(figsize=(12, 8))
    
    # Select key metrics for comparison
    key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    
    # Create a new DataFrame with just the key metrics
    key_metrics_df = metrics_df[key_metrics]
    
    # Plot each metric as a separate subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(key_metrics):
        key_metrics_df[metric].plot(kind='bar', ax=axes[i])
        axes[i].set_title(metric)
        axes[i].set_ylabel(metric)
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig("notebooks/assets/ensemble_metrics_comparison.png")
    
    print("\nStrategy ensemble example completed. Results saved to notebooks/assets/")

if __name__ == "__main__":
    main() 