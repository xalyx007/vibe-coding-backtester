"""
Functions for visualizing backtest results.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from backtester.analysis.metrics import calculate_drawdowns


def plot_equity_curve(portfolio_values: pd.DataFrame, 
                     figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Plot the equity curve from portfolio values.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure timestamp is datetime
    if isinstance(portfolio_values["timestamp"].iloc[0], str):
        portfolio_values = portfolio_values.copy()
        portfolio_values["timestamp"] = pd.to_datetime(portfolio_values["timestamp"])
    
    # Plot equity curve
    ax.plot(portfolio_values["timestamp"], portfolio_values["portfolio_value"], 
           label="Portfolio Value", color="blue")
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.set_title("Equity Curve")
    ax.grid(True)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    return fig


def plot_drawdown(portfolio_values: pd.DataFrame, 
                 figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Plot the drawdown from portfolio values.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate drawdowns
    drawdowns = calculate_drawdowns(portfolio_values)
    
    # Ensure timestamp is datetime
    if isinstance(drawdowns["timestamp"].iloc[0], str):
        drawdowns = drawdowns.copy()
        drawdowns["timestamp"] = pd.to_datetime(drawdowns["timestamp"])
    
    # Plot drawdown
    ax.fill_between(drawdowns["timestamp"], 0, drawdowns["drawdown"] * 100, 
                   color="red", alpha=0.3, label="Drawdown")
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Portfolio Drawdown")
    ax.grid(True)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    return fig


def plot_trades(trades: pd.DataFrame, 
               signals: pd.DataFrame, 
               symbol: str,
               figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Plot trades on top of price data.
    
    Args:
        trades: DataFrame with trade details
        signals: DataFrame with strategy signals and price data
        symbol: Trading symbol
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure timestamp is datetime
    if isinstance(signals.index[0], str):
        signals = signals.copy()
        signals.index = pd.to_datetime(signals.index)
    
    # Plot price data
    ax.plot(signals.index, signals["close"], label=f"{symbol} Price", color="blue", alpha=0.5)
    
    # Plot buy and sell trades
    if "timestamp" in trades.columns and "type" in trades.columns and "price" in trades.columns:
        # Ensure timestamp is datetime
        if isinstance(trades["timestamp"].iloc[0], str):
            trades = trades.copy()
            trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        
        # Plot buy trades
        buy_trades = trades[trades["type"] == "BUY"]
        ax.scatter(buy_trades["timestamp"], buy_trades["price"], 
                  marker="^", color="green", s=100, label="Buy")
        
        # Plot sell trades
        sell_trades = trades[trades["type"] == "SELL"]
        ax.scatter(sell_trades["timestamp"], sell_trades["price"], 
                  marker="v", color="red", s=100, label="Sell")
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{symbol} Price and Trades")
    ax.grid(True)
    ax.legend()
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    return fig


def plot_monthly_returns(portfolio_values: pd.DataFrame, 
                        figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Plot monthly returns as a heatmap.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from backtester.analysis.metrics import calculate_monthly_returns
    
    # Calculate monthly returns
    monthly_returns = calculate_monthly_returns(portfolio_values)
    
    if len(monthly_returns) <= 1:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient data for monthly returns heatmap", 
               ha="center", va="center")
        return fig
    
    # Create pivot table
    pivot = monthly_returns.pivot(index="year", columns="month", values="return")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    cmap = plt.cm.RdYlGn  # Red for negative, green for positive
    im = ax.imshow(pivot, cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Monthly Return", rotation=-90, va="bottom")
    
    # Add labels
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(np.arange(len(month_names)))
    ax.set_xticklabels(month_names)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    # Add title
    ax.set_title("Monthly Returns Heatmap")
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(month_names)):
            if j < pivot.shape[1] and not np.isnan(pivot.iloc[i, j]):
                text = ax.text(j, i, f"{pivot.iloc[i, j]:.1%}",
                              ha="center", va="center", 
                              color="black" if abs(pivot.iloc[i, j]) < 0.1 else "white")
    
    return fig 