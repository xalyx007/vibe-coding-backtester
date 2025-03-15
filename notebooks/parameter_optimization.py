#!/usr/bin/env python
"""
Parameter Optimization Example

This script demonstrates how to optimize strategy parameters using the Modular Backtesting System.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time

# Set up plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Import backtester modules
from backtester.data import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import SimplePortfolioManager
from backtester.backtester import Backtester
from backtester.utils.metrics import calculate_metrics

def run_backtest(data, short_window, long_window, initial_capital=10000, position_size=0.1):
    """Run a backtest with the given parameters."""
    # Create a strategy
    strategy = MovingAverageCrossover(
        short_window=short_window,
        long_window=long_window
    )
    
    # Create a portfolio manager
    portfolio_manager = SimplePortfolioManager(
        initial_capital=initial_capital,
        position_size=position_size
    )
    
    # Create a backtester
    backtester = Backtester(
        data=data,
        strategy=strategy,
        portfolio_manager=portfolio_manager
    )
    
    # Run the backtest
    results = backtester.run()
    
    # Calculate performance metrics
    metrics = calculate_metrics(results)
    
    return metrics

def main():
    """Run a parameter optimization example."""
    print("Running parameter optimization example...")
    
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
    
    # Define parameter ranges
    short_windows = range(5, 51, 5)  # 5, 10, 15, ..., 50
    long_windows = range(20, 201, 20)  # 20, 40, 60, ..., 200
    
    # Create a results dataframe
    results = []
    
    # Run backtests for all parameter combinations
    print("Running backtests for all parameter combinations...")
    start_time = time.time()
    
    for short_window, long_window in product(short_windows, long_windows):
        # Skip invalid combinations (short window must be less than long window)
        if short_window >= long_window:
            continue
        
        print(f"Testing short_window={short_window}, long_window={long_window}")
        
        # Run the backtest
        metrics = run_backtest(data, short_window, long_window)
        
        # Add parameters to metrics
        metrics['short_window'] = short_window
        metrics['long_window'] = long_window
        
        # Add to results
        results.append(metrics)
    
    end_time = time.time()
    print(f"Completed {len(results)} backtests in {end_time - start_time:.2f} seconds.")
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio (descending)
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    # Display top 10 results
    print("\nTop 10 parameter combinations by Sharpe ratio:")
    print(results_df.head(10)[['short_window', 'long_window', 'sharpe_ratio', 'total_return', 'max_drawdown']])
    
    # Save results to CSV
    results_df.to_csv("notebooks/assets/parameter_optimization_results.csv", index=False)
    print("Results saved to notebooks/assets/parameter_optimization_results.csv")
    
    # Plot heatmap of Sharpe ratios
    print("Plotting heatmap of Sharpe ratios...")
    pivot_table = results_df.pivot_table(
        index='short_window',
        columns='long_window',
        values='sharpe_ratio'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=False, cmap='viridis')
    plt.title('Sharpe Ratio by Moving Average Window Lengths')
    plt.xlabel('Long Window')
    plt.ylabel('Short Window')
    plt.savefig("notebooks/assets/parameter_optimization_heatmap.png")
    
    # Plot scatter plot of Sharpe ratio vs. total return
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=results_df,
        x='total_return',
        y='sharpe_ratio',
        hue='short_window',
        size='long_window',
        sizes=(20, 200),
        alpha=0.7
    )
    plt.title('Sharpe Ratio vs. Total Return')
    plt.xlabel('Total Return (%)')
    plt.ylabel('Sharpe Ratio')
    plt.savefig("notebooks/assets/parameter_optimization_scatter.png")
    
    print("\nParameter optimization example completed. Results saved to notebooks/assets/")

if __name__ == "__main__":
    main() 