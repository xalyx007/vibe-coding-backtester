#!/usr/bin/env python
"""
Bitcoin Backtest Example

This example demonstrates how to run a backtest using Bitcoin data from Yahoo Finance.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from backtester.inputs import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import BasicPortfolioManager
from backtester.backtest import Backtester

# Check if Bitcoin data exists, if not, download it
BTC_DATA_PATH = "data/yahoo/1d/BTC_USD.csv"
if not os.path.exists(BTC_DATA_PATH):
    print(f"Bitcoin data not found at {BTC_DATA_PATH}")
    print("Downloading Bitcoin data...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(BTC_DATA_PATH), exist_ok=True)
    
    # Import and run the download script
    import sys
    sys.path.append("scripts/data_downloaders")
    from download_yahoo_crypto import download_btc_data
    
    # Download Bitcoin data
    download_btc_data()
    
    print(f"Bitcoin data downloaded to {BTC_DATA_PATH}")

# Load data
print("Loading Bitcoin data...")
data_source = CSVDataSource(BTC_DATA_PATH)

# Create strategy
print("Creating strategy...")
strategy = MovingAverageCrossover(short_window=20, long_window=50)

# Create portfolio manager
print("Creating portfolio manager...")
portfolio_manager = BasicPortfolioManager(initial_capital=10000, position_size=0.01)

# Create and run backtester
print("Running backtest...")
backtester = Backtester(data_source, strategy, portfolio_manager)
results = backtester.run()

# Print performance metrics
print("\nPerformance Metrics:")
print(f"Total Return: {results.total_return:.2%}")

# Access metrics from the metrics dictionary
metrics = results.metrics
print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Save trades to CSV
if len(results.trades) > 0:
    results.trades.to_csv("results/trades.csv", index=False)
    print(f"Saved {len(results.trades)} trades to results/trades.csv")
else:
    print("No trades were executed during the backtest.")

# Plot results
print("\nGenerating plots...")
results.plot_equity_curve()
plt.title("Bitcoin Strategy - Equity Curve")
plt.savefig("results/btc_equity_curve.png")
plt.close()

results.plot_drawdown()
plt.title("Bitcoin Strategy - Drawdown")
plt.savefig("results/btc_drawdown.png")
plt.close()

results.plot_trades()
plt.title("Bitcoin Strategy - Trades")
plt.savefig("results/btc_trades.png")
plt.close()

print("\nBacktest completed successfully!")
print("Results saved to the 'results' directory.")

# Optional: Show the plots
# plt.show() 