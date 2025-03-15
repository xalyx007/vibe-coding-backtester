# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bitcoin Data Analysis and Backtesting
#
# This notebook demonstrates how to download Bitcoin data from Yahoo Finance, analyze it, and run a backtest using the Modular Backtesting System.

# %% [markdown]
# ## 1. Setup and Data Download

# %%
# Import required libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Add project root to path
sys.path.append('..')

# %% [markdown]
# ### Download Bitcoin Data
#
# We'll use the `download_yahoo_crypto.py` script to download Bitcoin data from Yahoo Finance.

# %%
# Check if Bitcoin data exists, if not, download it
BTC_DATA_PATH = "../data/yahoo/1d/BTC_USD.csv"

if not os.path.exists(BTC_DATA_PATH):
    print(f"Bitcoin data not found at {BTC_DATA_PATH}")
    print("Downloading Bitcoin data...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(BTC_DATA_PATH), exist_ok=True)
    
    # Import and run the download script
    sys.path.append("../scripts/data_downloaders")
    from download_yahoo_crypto import download_btc_data
    
    # Download Bitcoin data
    download_btc_data()
    
    print(f"Bitcoin data downloaded to {BTC_DATA_PATH}")
else:
    print(f"Bitcoin data found at {BTC_DATA_PATH}")

# %% [markdown]
# ## 2. Data Exploration and Analysis

# %%
# Load the data
btc_data = pd.read_csv(BTC_DATA_PATH)

# Display basic information
print("Bitcoin data shape:", btc_data.shape)
print("\nFirst few rows:")
btc_data.head()

# %%
# Convert date to datetime
btc_data['date'] = pd.to_datetime(btc_data['date'])

# Set date as index
btc_data = btc_data.set_index('date')

# Display summary statistics
btc_data.describe()

# %% [markdown]
# ### Price Chart

# %%
# Plot Bitcoin price
plt.figure(figsize=(14, 7))
plt.plot(btc_data.index, btc_data['close'], label='Close Price', color='#ff9900')
plt.title('Bitcoin Price (USD)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Volume Chart

# %%
# Plot Bitcoin volume
plt.figure(figsize=(14, 7))
plt.bar(btc_data.index, btc_data['volume'], color='#3d85c6', alpha=0.7)
plt.title('Bitcoin Trading Volume', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volume', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Returns Analysis

# %%
# Calculate daily returns
btc_data['daily_return'] = btc_data['close'].pct_change() * 100

# Plot daily returns
plt.figure(figsize=(14, 7))
plt.plot(btc_data.index, btc_data['daily_return'], color='#6aa84f')
plt.title('Bitcoin Daily Returns (%)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Return (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Plot return distribution
plt.figure(figsize=(14, 7))
sns.histplot(btc_data['daily_return'].dropna(), kde=True, bins=100, color='#6aa84f')
plt.title('Bitcoin Daily Returns Distribution', fontsize=16)
plt.xlabel('Daily Return (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Volatility Analysis

# %%
# Calculate rolling volatility (30-day standard deviation of returns)
btc_data['volatility_30d'] = btc_data['daily_return'].rolling(window=30).std()

# Plot volatility
plt.figure(figsize=(14, 7))
plt.plot(btc_data.index, btc_data['volatility_30d'], color='#cc0000')
plt.title('Bitcoin 30-Day Rolling Volatility', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility (Std Dev of Returns)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Technical Indicators

# %%
# Calculate Simple Moving Averages
btc_data['SMA_20'] = btc_data['close'].rolling(window=20).mean()
btc_data['SMA_50'] = btc_data['close'].rolling(window=50).mean()
btc_data['SMA_200'] = btc_data['close'].rolling(window=200).mean()

# Plot price with moving averages
plt.figure(figsize=(14, 7))
plt.plot(btc_data.index, btc_data['close'], label='Close Price', color='#ff9900')
plt.plot(btc_data.index, btc_data['SMA_20'], label='20-Day SMA', color='#3d85c6')
plt.plot(btc_data.index, btc_data['SMA_50'], label='50-Day SMA', color='#6aa84f')
plt.plot(btc_data.index, btc_data['SMA_200'], label='200-Day SMA', color='#cc0000')
plt.title('Bitcoin Price with Moving Averages', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Calculate Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

btc_data['RSI_14'] = calculate_rsi(btc_data['close'], window=14)

# Plot RSI
plt.figure(figsize=(14, 7))
plt.plot(btc_data.index, btc_data['RSI_14'], color='#674ea7')
plt.axhline(y=70, color='#cc0000', linestyle='--', alpha=0.5)
plt.axhline(y=30, color='#6aa84f', linestyle='--', alpha=0.5)
plt.title('Bitcoin 14-Day RSI', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('RSI', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Backtesting with the Modular Backtesting System

# %%
# Import backtesting components
from backtester.data import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import SimplePortfolioManager
from backtester.backtester import Backtester

# %% [markdown]
# ### Moving Average Crossover Strategy

# %%
# Create data source
data_source = CSVDataSource(BTC_DATA_PATH)

# Create strategy
strategy = MovingAverageCrossover(short_window=20, long_window=50)

# Create portfolio manager
portfolio_manager = SimplePortfolioManager(initial_capital=10000, position_size=0.1)

# Create and run backtester
backtester = Backtester(data_source, strategy, portfolio_manager)
results = backtester.run()

# Print performance metrics
print("Performance Metrics:")
print(f"Total Return: {results.total_return:.2%}")
print(f"Annualized Return: {results.annualized_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
print(f"Profit Factor: {results.profit_factor:.2f}")

# %% [markdown]
# ### Visualize Backtest Results

# %%
# Plot equity curve
results.plot_equity_curve()
plt.title("Bitcoin Strategy - Equity Curve", fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Plot drawdown
results.plot_drawdown()
plt.title("Bitcoin Strategy - Drawdown", fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Plot trades
results.plot_trades()
plt.title("Bitcoin Strategy - Trades", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Parameter Optimization

# %%
# Define parameter ranges
short_windows = range(5, 51, 5)  # 5, 10, 15, ..., 50
long_windows = range(20, 201, 20)  # 20, 40, 60, ..., 200

# Initialize results storage
optimization_results = []

# Run backtests for each parameter combination
for short_window in short_windows:
    for long_window in long_windows:
        # Skip invalid combinations (short window must be less than long window)
        if short_window >= long_window:
            continue
            
        # Create strategy with current parameters
        strategy = MovingAverageCrossover(short_window=short_window, long_window=long_window)
        
        # Create and run backtester
        backtester = Backtester(data_source, strategy, portfolio_manager)
        results = backtester.run()
        
        # Store results
        optimization_results.append({
            'short_window': short_window,
            'long_window': long_window,
            'total_return': results.total_return,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown
        })

# Convert to DataFrame
optimization_df = pd.DataFrame(optimization_results)

# Display top 10 strategies by Sharpe ratio
print("Top 10 Strategies by Sharpe Ratio:")
optimization_df.sort_values('sharpe_ratio', ascending=False).head(10)

# %% [markdown]
# ### Visualize Parameter Optimization Results

# %%
# Create pivot table for heatmap
heatmap_data = optimization_df.pivot_table(
    index='short_window', 
    columns='long_window', 
    values='sharpe_ratio'
)

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f')
plt.title('Sharpe Ratio by Moving Average Parameters', fontsize=16)
plt.xlabel('Long Window', fontsize=12)
plt.ylabel('Short Window', fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Conclusion
#
# In this notebook, we've:
#
# 1. Downloaded Bitcoin data from Yahoo Finance
# 2. Explored and analyzed the data
# 3. Calculated technical indicators
# 4. Backtested a Moving Average Crossover strategy
# 5. Optimized strategy parameters
#
# The Modular Backtesting System makes it easy to test different strategies and parameters on cryptocurrency data. 