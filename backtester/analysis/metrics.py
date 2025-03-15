"""
Functions for calculating performance metrics from backtest results.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime


def calculate_metrics(portfolio_values: pd.DataFrame, 
                     trades: pd.DataFrame, 
                     initial_capital: float) -> Dict[str, Any]:
    """
    Calculate performance metrics from backtest results.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        trades: DataFrame with trade details
        initial_capital: Initial capital for the backtest
        
    Returns:
        Dictionary with performance metrics
    """
    metrics = {}
    
    # Return metrics
    if len(portfolio_values) > 0:
        # Extract portfolio values
        values = portfolio_values["portfolio_value"]
        
        # Total return
        initial_value = initial_capital
        final_value = values.iloc[-1]
        total_return = (final_value / initial_value) - 1
        metrics["total_return"] = total_return
        
        # Annualized return
        if len(values) > 1:
            start_date = portfolio_values["timestamp"].iloc[0]
            end_date = portfolio_values["timestamp"].iloc[-1]
            
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
                
            years = (end_date - start_date).days / 365.25
            if years > 0:
                annualized_return = (1 + total_return) ** (1 / years) - 1
                metrics["annualized_return"] = annualized_return
            else:
                metrics["annualized_return"] = 0
    
    # Risk metrics
    if len(portfolio_values) > 1:
        # Calculate daily returns
        values = portfolio_values["portfolio_value"]
        returns = values.pct_change().dropna()
        
        # Volatility (annualized)
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days
        metrics["volatility"] = annualized_volatility
        
        # Sharpe ratio (annualized, assuming risk-free rate of 0)
        if annualized_volatility > 0:
            sharpe_ratio = metrics.get("annualized_return", 0) / annualized_volatility
            metrics["sharpe_ratio"] = sharpe_ratio
        else:
            metrics["sharpe_ratio"] = 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        metrics["max_drawdown"] = max_drawdown
    
    # Trade metrics
    if len(trades) > 0:
        # Number of trades
        metrics["num_trades"] = len(trades)
        
        # Win rate
        if "profit" in trades.columns:
            winning_trades = trades[trades["profit"] > 0]
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            metrics["win_rate"] = win_rate
            
            # Profit factor
            gross_profit = winning_trades["profit"].sum() if len(winning_trades) > 0 else 0
            losing_trades = trades[trades["profit"] < 0]
            gross_loss = abs(losing_trades["profit"].sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            metrics["profit_factor"] = profit_factor
            
            # Average profit per trade
            avg_profit = trades["profit"].mean() if len(trades) > 0 else 0
            metrics["avg_profit_per_trade"] = avg_profit
            
            # Average win and loss
            avg_win = winning_trades["profit"].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades["profit"].mean() if len(losing_trades) > 0 else 0
            metrics["avg_win"] = avg_win
            metrics["avg_loss"] = avg_loss
    
    return metrics


def calculate_drawdowns(portfolio_values: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate drawdowns from portfolio values.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        
    Returns:
        DataFrame with drawdown information
    """
    if len(portfolio_values) <= 1:
        return pd.DataFrame(columns=["timestamp", "drawdown"])
    
    # Calculate returns
    values = portfolio_values["portfolio_value"]
    returns = values.pct_change().dropna()
    
    # Calculate drawdowns
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    
    # Create drawdown DataFrame
    drawdown_df = pd.DataFrame({
        "timestamp": portfolio_values["timestamp"].iloc[1:].reset_index(drop=True),
        "drawdown": drawdown.values
    })
    
    return drawdown_df


def calculate_monthly_returns(portfolio_values: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly returns from portfolio values.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        
    Returns:
        DataFrame with monthly returns
    """
    if len(portfolio_values) <= 1:
        return pd.DataFrame(columns=["year", "month", "return"])
    
    # Ensure timestamp is datetime
    if isinstance(portfolio_values["timestamp"].iloc[0], str):
        portfolio_values = portfolio_values.copy()
        portfolio_values["timestamp"] = pd.to_datetime(portfolio_values["timestamp"])
    
    # Set timestamp as index
    df = portfolio_values.set_index("timestamp")
    
    # Resample to monthly and calculate returns
    monthly_values = df["portfolio_value"].resample("M").last()
    monthly_returns = monthly_values.pct_change().dropna()
    
    # Create monthly returns DataFrame
    result = pd.DataFrame({
        "year": monthly_returns.index.year,
        "month": monthly_returns.index.month,
        "return": monthly_returns.values
    })
    
    return result 