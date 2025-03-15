#!/usr/bin/env python
"""
Basic Backtest Example

This script demonstrates how to use the Modular Backtesting System to run a simple backtest
using a moving average crossover strategy.
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
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import SimplePortfolioManager
from backtester.backtester import Backtester
from backtester.utils.metrics import calculate_metrics
from backtester.utils.visualization import plot_equity_curve, plot_drawdown

def main():
    """Run a basic backtest example."""
    print("Running basic backtest example...")
    
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
    
    # Create a strategy
    print("Creating strategy...")
    strategy = MovingAverageCrossover(
        short_window=20,  # 20-day short-term moving average
        long_window=50    # 50-day long-term moving average
    )
    
    # Generate signals
    signals = strategy.generate_signals(data)
    print(f"Generated {len(signals)} signals.")
    
    # Create a portfolio manager
    print("Creating portfolio manager...")
    portfolio_manager = SimplePortfolioManager(
        initial_capital=10000,  # Starting with $10,000
        position_size=0.1       # Invest 10% of capital in each position
    )
    
    # Create a backtester
    print("Running backtest...")
    backtester = Backtester(
        data=data,
        strategy=strategy,
        portfolio_manager=portfolio_manager
    )
    
    # Run the backtest
    results = backtester.run()
    print(f"Backtest completed with {len(results)} results.")
    
    # Calculate performance metrics
    print("Calculating performance metrics...")
    metrics = calculate_metrics(results)
    
    # Display the metrics
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Plot the equity curve
    print("\nPlotting equity curve...")
    plot_equity_curve(results)
    plt.savefig("notebooks/assets/equity_curve.png")
    
    # Plot the drawdown
    print("Plotting drawdown...")
    plot_drawdown(results)
    plt.savefig("notebooks/assets/drawdown.png")
    
    print("\nBacktest example completed. Results saved to notebooks/assets/")

if __name__ == "__main__":
    main() 