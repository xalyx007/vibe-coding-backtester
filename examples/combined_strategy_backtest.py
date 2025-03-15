#!/usr/bin/env python
"""
Combined Strategy Backtest Example

This example demonstrates how to run a backtest using a combined strategy that incorporates
Moving Average Crossover, RSI, and MACD indicators on Bitcoin data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from backtester.inputs import CSVDataSource
from backtester.strategy import CombinedStrategy
from backtester.portfolio import BasicPortfolioManager
from backtester.backtest import Backtester
from backtester.events import EventBus


def main():
    # Create data directory if it doesn't exist
    os.makedirs("data/yahoo/1d", exist_ok=True)
    
    # Check if Bitcoin data exists, if not, download it
    BTC_DATA_PATH = "data/yahoo/1d/BTC_USD.csv"
    if not os.path.exists(BTC_DATA_PATH):
        print(f"Bitcoin data not found at {BTC_DATA_PATH}")
        print("Downloading Bitcoin data...")
        
        # Import and run the download script
        import sys
        sys.path.append("scripts/data_downloaders")
        from download_yahoo_crypto import download_btc_data
        
        # Download Bitcoin data
        download_btc_data()
        
        print(f"Bitcoin data downloaded to {BTC_DATA_PATH}")
    
    # Create event bus for logging events
    event_bus = EventBus()
    
    # Subscribe to events
    event_bus.subscribe("backtest_started", lambda event: print(f"Backtest started: {event['data']}"))
    event_bus.subscribe("backtest_completed", lambda event: print(f"Backtest completed: {event['data']}"))
    
    # Create components
    data_source = CSVDataSource(
        filepath=BTC_DATA_PATH,
        date_column="date",
        event_bus=event_bus
    )
    
    # Create the combined strategy with the specified parameters
    strategy = CombinedStrategy(
        short_ma_window=9,        # 9-day MA
        long_ma_window=50,        # 50-day MA
        rsi_window=14,            # 14-day RSI
        rsi_overbought=70,        # RSI overbought threshold
        rsi_oversold=30,          # RSI oversold threshold
        macd_fast=12,             # MACD fast EMA
        macd_slow=26,             # MACD slow EMA
        macd_signal=9,            # MACD signal line
        event_bus=event_bus
    )
    
    # Create portfolio manager with 10% position size
    portfolio_manager = BasicPortfolioManager(
        initial_capital=10000,
        position_size=0.1,
        event_bus=event_bus
    )
    
    # Create backtester
    backtester = Backtester(
        data_source=data_source,
        strategy=strategy,
        portfolio_manager=portfolio_manager,
        event_bus=event_bus,
        transaction_costs=0.001,  # 0.1% transaction costs
        slippage=0.0005           # 0.05% slippage
    )
    
    # Run backtest
    print("Running backtest with Combined Strategy (MA Crossover + RSI + MACD)...")
    results = backtester.run(
        start_date="2020-01-01",
        end_date="2023-01-01",
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
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save trades to CSV
    if len(results.trades) > 0:
        results.trades.to_csv("results/combined_strategy_trades.csv", index=False)
        print(f"Saved {len(results.trades)} trades to results/combined_strategy_trades.csv")
    else:
        print("No trades were executed during the backtest.")
    
    # Plot results
    print("\nGenerating plots...")
    
    # Plot equity curve
    fig1 = results.plot_equity_curve()
    plt.title("Combined Strategy - Equity Curve")
    plt.savefig("results/combined_strategy_equity_curve.png")
    plt.close()
    
    # Plot drawdown
    fig2 = results.plot_drawdown()
    plt.title("Combined Strategy - Drawdown")
    plt.savefig("results/combined_strategy_drawdown.png")
    plt.close()
    
    # Plot trades
    fig3 = results.plot_trades()
    plt.title("Combined Strategy - Trades")
    plt.savefig("results/combined_strategy_trades.png")
    plt.close()
    
    # Plot strategy indicators
    print("Plotting strategy indicators...")
    plot_strategy_indicators(results.signals, strategy)
    
    print("\nBacktest completed successfully!")
    print("Results saved to the 'results' directory.")


def plot_strategy_indicators(signals, strategy):
    """
    Plot the strategy indicators.
    
    Args:
        signals (pd.DataFrame): The signals DataFrame with indicators.
        strategy (CombinedStrategy): The strategy instance.
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot price and moving averages
    axs[0].plot(signals.index, signals['close'], label='Price', alpha=0.5)
    axs[0].plot(signals.index, signals[f'ma_{strategy.short_ma_window}'], label=f'{strategy.short_ma_window}-day MA')
    axs[0].plot(signals.index, signals[f'ma_{strategy.long_ma_window}'], label=f'{strategy.long_ma_window}-day MA')
    axs[0].set_title('Price and Moving Averages')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot RSI
    axs[1].plot(signals.index, signals['rsi'], label='RSI')
    axs[1].axhline(y=strategy.rsi_overbought, color='r', linestyle='--', label='Overbought')
    axs[1].axhline(y=strategy.rsi_oversold, color='g', linestyle='--', label='Oversold')
    axs[1].set_title('Relative Strength Index (RSI)')
    axs[1].set_ylabel('RSI')
    axs[1].set_ylim(0, 100)
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot MACD
    axs[2].plot(signals.index, signals['macd_line'], label='MACD Line')
    axs[2].plot(signals.index, signals['signal_line'], label='Signal Line')
    axs[2].bar(signals.index, signals['macd_hist'], label='Histogram', alpha=0.3)
    axs[2].set_title('MACD')
    axs[2].set_ylabel('MACD')
    axs[2].set_xlabel('Date')
    axs[2].legend()
    axs[2].grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("results/combined_strategy_indicators.png")
    plt.close()


if __name__ == "__main__":
    main() 