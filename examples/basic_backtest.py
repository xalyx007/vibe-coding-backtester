#!/usr/bin/env python
"""
Basic example of running a backtest with the Modular Backtesting System.

This example demonstrates how to use the system to backtest a simple
moving average crossover strategy on Bitcoin price data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from backtester.inputs import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import BasicPortfolioManager, PercentageSizer
from backtester.backtest import Backtester
from backtester.events import EventBus


def main():
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Check if the data file exists, if not, create a sample file
    data_file = "data/btc_daily.csv"
    if not os.path.exists(data_file):
        print(f"Sample data file not found: {data_file}")
        print("Creating a sample data file with random data...")
        create_sample_data(data_file)
    
    # Create event bus for logging events
    event_bus = EventBus()
    
    # Subscribe to events
    event_bus.subscribe("backtest_started", lambda event: print(f"Backtest started: {event['data']}"))
    event_bus.subscribe("backtest_completed", lambda event: print(f"Backtest completed: {event['data']}"))
    
    # Create components
    data_source = CSVDataSource(
        file_path=data_file,
        date_column="timestamp",
        event_bus=event_bus
    )
    
    strategy = MovingAverageCrossover(
        short_window=20,
        long_window=50,
        event_bus=event_bus
    )
    
    position_sizer = PercentageSizer(percentage=0.1)
    
    portfolio_manager = BasicPortfolioManager(
        initial_capital=10000,
        position_sizer=position_sizer,
        event_bus=event_bus
    )
    
    # Create backtester
    backtester = Backtester(
        data_source=data_source,
        strategy=strategy,
        portfolio_manager=portfolio_manager,
        event_bus=event_bus,
        transaction_costs=0.001,  # 0.1% transaction costs
        slippage=0.0005  # 0.05% slippage
    )
    
    # Run backtest
    print("Running backtest...")
    results = backtester.run(
        start_date="2020-01-01",
        end_date="2021-01-01",
        symbol="BTC-USD"
    )
    
    # Print results summary
    print("\nBacktest Results Summary:")
    summary = results.summary()
    print(f"Initial Capital: ${summary['initial_capital']:.2f}")
    print(f"Final Capital: ${summary['final_capital']:.2f}")
    print(f"Total Return: {summary['total_return']:.2%}")
    print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {summary.get('max_drawdown', 0):.2%}")
    print(f"Number of Trades: {summary['total_trades']}")
    
    # Plot results
    print("\nGenerating plots...")
    fig1 = results.plot_equity_curve()
    fig2 = results.plot_drawdown()
    fig3 = results.plot_trades()
    
    # Save plots
    os.makedirs("results", exist_ok=True)
    fig1.savefig("results/equity_curve.png")
    fig2.savefig("results/drawdown.png")
    fig3.savefig("results/trades.png")
    
    print("Plots saved to 'results' directory.")
    
    # Save results to JSON
    results.to_json("results/backtest_results.json")
    print("Results saved to 'results/backtest_results.json'.")
    
    # Show plots
    plt.show()


def create_sample_data(file_path):
    """Create a sample data file with random price data."""
    import numpy as np
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2021, 1, 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random price data
    np.random.seed(42)  # For reproducibility
    
    # Start with a base price
    base_price = 10000
    
    # Generate random returns with a slight upward bias
    returns = np.random.normal(0.001, 0.02, len(dates))
    
    # Calculate prices
    prices = base_price * np.cumprod(1 + returns)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * (1 + np.random.uniform(0.01, 0.03, len(dates))),
        'low': prices * (1 - np.random.uniform(0.01, 0.03, len(dates))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, len(dates)) * prices
    })
    
    # Ensure high is the highest and low is the lowest
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    # Save to CSV
    data.to_csv(file_path, index=False)
    print(f"Sample data saved to {file_path}")


if __name__ == "__main__":
    main() 