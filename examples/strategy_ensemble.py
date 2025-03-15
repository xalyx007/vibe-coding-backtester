#!/usr/bin/env python
"""
Example of combining multiple strategies with the Modular Backtesting System.

This example demonstrates how to use the StrategyEnsemble class to combine
multiple trading strategies and backtest the ensemble.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from backtester.inputs import CSVDataSource
from backtester.strategy import RSIStrategy, BollingerBandsStrategy, StrategyEnsemble
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
    
    # Create data source
    data_source = CSVDataSource(
        file_path=data_file,
        date_column="timestamp",
        event_bus=event_bus
    )
    
    # Create individual strategies
    rsi_strategy = RSIStrategy(
        period=14,
        overbought=70,
        oversold=30,
        event_bus=event_bus
    )
    
    bbands_strategy = BollingerBandsStrategy(
        window=20,
        num_std=2,
        event_bus=event_bus
    )
    
    # Create strategy ensemble
    ensemble = StrategyEnsemble(
        strategies=[rsi_strategy, bbands_strategy],
        weights=[0.5, 0.5],  # Equal weights
        combination_method="weighted",
        event_bus=event_bus
    )
    
    # Create portfolio manager
    position_sizer = PercentageSizer(percentage=0.1)
    
    portfolio_manager = BasicPortfolioManager(
        initial_capital=10000,
        position_sizer=position_sizer,
        event_bus=event_bus
    )
    
    # Create backtester
    backtester = Backtester(
        data_source=data_source,
        strategy=ensemble,
        portfolio_manager=portfolio_manager,
        event_bus=event_bus,
        transaction_costs=0.001,
        slippage=0.0005
    )
    
    # Run backtest
    print("Running backtest with strategy ensemble...")
    ensemble_results = backtester.run(
        start_date="2020-01-01",
        end_date="2021-01-01",
        symbol="BTC-USD"
    )
    
    # Run individual strategy backtests for comparison
    print("\nRunning backtest with RSI strategy...")
    rsi_backtester = Backtester(
        data_source=data_source,
        strategy=rsi_strategy,
        portfolio_manager=BasicPortfolioManager(
            initial_capital=10000,
            position_sizer=position_sizer
        ),
        transaction_costs=0.001,
        slippage=0.0005
    )
    
    rsi_results = rsi_backtester.run(
        start_date="2020-01-01",
        end_date="2021-01-01",
        symbol="BTC-USD"
    )
    
    print("\nRunning backtest with Bollinger Bands strategy...")
    bbands_backtester = Backtester(
        data_source=data_source,
        strategy=bbands_strategy,
        portfolio_manager=BasicPortfolioManager(
            initial_capital=10000,
            position_sizer=position_sizer
        ),
        transaction_costs=0.001,
        slippage=0.0005
    )
    
    bbands_results = bbands_backtester.run(
        start_date="2020-01-01",
        end_date="2021-01-01",
        symbol="BTC-USD"
    )
    
    # Print results comparison
    print("\nResults Comparison:")
    print(f"{'Strategy':<20} {'Total Return':<15} {'Sharpe Ratio':<15} {'Max Drawdown':<15} {'Trades':<10}")
    print("-" * 75)
    
    ensemble_summary = ensemble_results.summary()
    rsi_summary = rsi_results.summary()
    bbands_summary = bbands_results.summary()
    
    print(f"{'Ensemble':<20} {ensemble_summary['total_return']:.2%:<15} "
          f"{ensemble_summary.get('sharpe_ratio', 0):.2f:<15} "
          f"{ensemble_summary.get('max_drawdown', 0):.2%:<15} "
          f"{ensemble_summary['total_trades']:<10}")
    
    print(f"{'RSI':<20} {rsi_summary['total_return']:.2%:<15} "
          f"{rsi_summary.get('sharpe_ratio', 0):.2f:<15} "
          f"{rsi_summary.get('max_drawdown', 0):.2%:<15} "
          f"{rsi_summary['total_trades']:<10}")
    
    print(f"{'Bollinger Bands':<20} {bbands_summary['total_return']:.2%:<15} "
          f"{bbands_summary.get('sharpe_ratio', 0):.2f:<15} "
          f"{bbands_summary.get('max_drawdown', 0):.2%:<15} "
          f"{bbands_summary['total_trades']:<10}")
    
    # Plot equity curves for comparison
    plt.figure(figsize=(12, 6))
    
    # Get portfolio values
    ensemble_values = ensemble_results.portfolio_values
    rsi_values = rsi_results.portfolio_values
    bbands_values = bbands_results.portfolio_values
    
    # Ensure timestamps are datetime
    ensemble_values['timestamp'] = pd.to_datetime(ensemble_values['timestamp'])
    rsi_values['timestamp'] = pd.to_datetime(rsi_values['timestamp'])
    bbands_values['timestamp'] = pd.to_datetime(bbands_values['timestamp'])
    
    # Plot equity curves
    plt.plot(ensemble_values['timestamp'], ensemble_values['portfolio_value'], 
             label='Ensemble', linewidth=2)
    plt.plot(rsi_values['timestamp'], rsi_values['portfolio_value'], 
             label='RSI', linewidth=1, alpha=0.7)
    plt.plot(bbands_values['timestamp'], bbands_values['portfolio_value'], 
             label='Bollinger Bands', linewidth=1, alpha=0.7)
    
    plt.title('Equity Curve Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/ensemble_comparison.png")
    
    print("\nComparison plot saved to 'results/ensemble_comparison.png'.")
    
    # Save results to JSON
    ensemble_results.to_json("results/ensemble_results.json")
    rsi_results.to_json("results/rsi_results.json")
    bbands_results.to_json("results/bbands_results.json")
    
    print("Results saved to 'results' directory.")
    
    # Show plot
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