"""
Class for storing and analyzing backtest results.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

from backtester.analysis.metrics import calculate_metrics
from backtester.analysis.visualization import plot_equity_curve, plot_drawdown, plot_trades


class BacktestResults:
    """
    Class for storing and analyzing backtest results.
    
    This class stores the results of a backtest and provides methods for
    calculating performance metrics and generating visualizations.
    """
    
    def __init__(self, 
                portfolio_values: pd.DataFrame, 
                trades: pd.DataFrame,
                signals: pd.DataFrame,
                strategy_parameters: Dict[str, Any],
                initial_capital: float,
                symbol: str):
        """
        Initialize the backtest results.
        
        Args:
            portfolio_values: DataFrame with portfolio values over time
            trades: DataFrame with trade details
            signals: DataFrame with strategy signals
            strategy_parameters: Dictionary of strategy parameters
            initial_capital: Initial capital for the backtest
            symbol: Trading symbol
        """
        self.portfolio_values = portfolio_values
        self.trades = trades
        self.signals = signals
        self.strategy_parameters = strategy_parameters
        self.initial_capital = initial_capital
        self.symbol = symbol
        
        # Calculate basic metrics
        self._calculate_basic_metrics()
    
    def _calculate_basic_metrics(self):
        """Calculate basic performance metrics."""
        if len(self.portfolio_values) > 0:
            initial_value = self.initial_capital
            final_value = self.portfolio_values["portfolio_value"].iloc[-1]
            
            self.total_return = (final_value / initial_value) - 1
            self.total_trades = len(self.trades)
            
            # Calculate more advanced metrics
            self.metrics = calculate_metrics(
                self.portfolio_values, 
                self.trades, 
                self.initial_capital
            )
        else:
            self.total_return = 0
            self.total_trades = 0
            self.metrics = {}
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the backtest results.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            "symbol": self.symbol,
            "initial_capital": self.initial_capital,
            "final_capital": self.portfolio_values["portfolio_value"].iloc[-1] if len(self.portfolio_values) > 0 else self.initial_capital,
            "total_return": self.total_return,
            "total_trades": self.total_trades,
            "strategy_parameters": self.strategy_parameters,
            **self.metrics
        }
    
    def plot_equity_curve(self, figsize=(12, 6)):
        """
        Plot the equity curve.
        
        Args:
            figsize: Figure size
        """
        return plot_equity_curve(self.portfolio_values, figsize=figsize)
    
    def plot_drawdown(self, figsize=(12, 6)):
        """
        Plot the drawdown.
        
        Args:
            figsize: Figure size
        """
        return plot_drawdown(self.portfolio_values, figsize=figsize)
    
    def plot_trades(self, figsize=(12, 6)):
        """
        Plot the trades.
        
        Args:
            figsize: Figure size
        """
        return plot_trades(self.trades, self.signals, self.symbol, figsize=figsize)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the results to a dictionary.
        
        Returns:
            Dictionary representation of the results
        """
        return {
            "summary": self.summary(),
            "portfolio_values": self.portfolio_values.to_dict(orient="records"),
            "trades": self.trades.to_dict(orient="records")
        }
    
    def to_json(self, path: str):
        """
        Save the results to a JSON file.
        
        Args:
            path: Path to save the JSON file
        """
        import json
        
        # Convert DataFrames to records
        result_dict = self.to_dict()
        
        # Save to JSON
        with open(path, 'w') as f:
            json.dump(result_dict, f, indent=4, default=str) 