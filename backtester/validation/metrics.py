"""
Validation metrics for the backtesting system.

This module provides functionality for calculating validation metrics
for trading strategies.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate returns from a price series.
    
    Args:
        prices: Series of prices
        
    Returns:
        Series of returns
    """
    return prices.pct_change().dropna()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Annualization factor (252 for daily, 12 for monthly, etc.)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    return excess_returns.mean() / returns.std() * np.sqrt(annualization_factor)

def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate the maximum drawdown.
    
    Args:
        prices: Series of prices
        
    Returns:
        Maximum drawdown as a positive percentage
    """
    if len(prices) == 0:
        return 0.0
    
    # Calculate the running maximum
    running_max = prices.cummax()
    
    # Calculate the drawdown
    drawdown = (prices - running_max) / running_max
    
    # Return the maximum drawdown as a positive percentage
    return abs(drawdown.min())  # Using abs to ensure positive value

def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate the win rate from a list of trades.
    
    Args:
        trades: List of trade dictionaries with 'profit' key
        
    Returns:
        Win rate as a percentage
    """
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
    return winning_trades / len(trades)

def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate the profit factor from a list of trades.
    
    Args:
        trades: List of trade dictionaries with 'profit' key
        
    Returns:
        Profit factor (gross profit / gross loss)
    """
    if not trades:
        return 0.0
    
    gross_profit = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
    gross_loss = sum(abs(trade.get('profit', 0)) for trade in trades if trade.get('profit', 0) < 0)
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss

def calculate_average_trade(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate the average trade profit from a list of trades.
    
    Args:
        trades: List of trade dictionaries with 'profit' key
        
    Returns:
        Average trade profit
    """
    if not trades:
        return 0.0
    
    return sum(trade.get('profit', 0) for trade in trades) / len(trades)

def calculate_validation_metrics(
    prices: pd.Series,
    trades: List[Dict[str, Any]],
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> Dict[str, float]:
    """
    Calculate validation metrics for a trading strategy.
    
    Args:
        prices: Series of prices
        trades: List of trade dictionaries
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Annualization factor (252 for daily, 12 for monthly, etc.)
        
    Returns:
        Dictionary with validation metrics
    """
    # Calculate returns
    returns = calculate_returns(prices)
    
    # Calculate metrics
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1 if len(prices) > 0 else 0.0
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate, annualization_factor)
    max_drawdown = calculate_max_drawdown(prices)
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    average_trade = calculate_average_trade(trades)
    
    # Return metrics
    return {
        "total_return": total_return,
        "annualized_return": (1 + total_return) ** (annualization_factor / len(returns)) - 1 if len(returns) > 0 else 0.0,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_trade": average_trade,
        "num_trades": len(trades),
        "volatility": returns.std() * np.sqrt(annualization_factor) if len(returns) > 0 else 0.0,
        "calmar_ratio": (total_return / max_drawdown) if max_drawdown > 0 else 0.0
    }

def calculate_metrics(
    returns: List[float],
    positions: List[int],
    trades: List[tuple],
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> Dict[str, float]:
    """
    Calculate validation metrics for backtesting results.
    
    Args:
        returns: A list of returns.
        positions: A list of positions (-1, 0, or 1).
        trades: A list of trades (tuples of (direction, size, date)).
        risk_free_rate: The annualized risk-free rate.
        annualization_factor: The annualization factor (252 for daily data).
    
    Returns:
        A dictionary containing the calculated metrics.
    """
    # Convert to numpy arrays
    returns_array = np.array(returns)
    positions_array = np.array(positions)
    
    # Calculate total return
    total_return = np.prod(1 + returns_array) - 1
    
    # Calculate annualized return
    annualized_return = (1 + total_return) ** (annualization_factor / len(returns_array)) - 1
    
    # Calculate standard deviation
    std_dev = np.std(returns_array) * np.sqrt(annualization_factor)
    
    # Calculate Sharpe ratio
    if std_dev > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / std_dev
    else:
        sharpe_ratio = 0.0
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + returns_array)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak
    max_drawdown = np.max(drawdown)
    
    # Calculate win rate
    if len(trades) > 0:
        winning_trades = sum(1 for trade in trades if trade[0] * trade[1] > 0)
        win_rate = winning_trades / len(trades)
    else:
        win_rate = 0.0
    
    # Calculate profit factor
    if len(trades) > 0:
        gross_profit = sum(trade[0] * trade[1] for trade in trades if trade[0] * trade[1] > 0)
        gross_loss = abs(sum(trade[0] * trade[1] for trade in trades if trade[0] * trade[1] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        profit_factor = 0.0
    
    # Return metrics
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'std_dev': std_dev,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    } 