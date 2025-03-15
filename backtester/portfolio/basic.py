"""
Basic Portfolio Manager for the Backtesting System.

This module implements a simple portfolio manager for backtesting.
"""

import pandas as pd
import numpy as np
from backtester.portfolio.base import PortfolioManager
from backtester.utils.constants import SignalType

class BasicPortfolioManager(PortfolioManager):
    """
    A basic portfolio manager that handles position sizing and trade execution.
    
    Attributes:
        initial_capital (float): The initial capital for the portfolio.
        position_size (float): The position size as a percentage of capital.
        current_capital (float): The current capital in the portfolio.
        positions (dict): The current positions in the portfolio.
        trades (list): The list of executed trades.
    """
    
    def __init__(self, initial_capital=10000.0, position_size=0.1, **kwargs):
        """
        Initialize the basic portfolio manager.
        
        Args:
            initial_capital (float): The initial capital for the portfolio.
            position_size (float): The position size as a percentage of capital.
            **kwargs: Additional arguments to pass to the base class.
        """
        super().__init__(initial_capital=initial_capital, **kwargs)
        self.position_size = position_size
        self.positions = {}
        self.equity_curve = []
        
    def calculate_position_size(self, symbol, price, signal_type):
        """
        Calculate the position size for a trade.
        
        Args:
            symbol (str): The symbol of the asset.
            price (float): The current price of the asset.
            signal_type (SignalType): The type of signal.
            
        Returns:
            float: The number of units to trade.
        """
        # For BUY signals, use a percentage of the current capital
        if signal_type == SignalType.BUY:
            trade_value = self.current_capital * self.position_size
            return trade_value / price
        # For SELL signals, sell all holdings of the symbol
        elif signal_type == SignalType.SELL:
            return self.positions.get(symbol, 0)
        else:
            return 0
    
    def update_position(self, timestamp, symbol, signal_type, price, metadata=None):
        """
        Update portfolio positions based on a signal.
        
        Args:
            timestamp: Timestamp of the signal
            symbol: Trading symbol (e.g., 'BTC-USD')
            signal_type: Type of signal (BUY, SELL, HOLD)
            price: Current price of the asset
            metadata: Additional information about the signal
            
        Returns:
            Dictionary with details of the executed trade
        """
        # Default to no trade
        trade_type = None
        
        if signal_type == SignalType.HOLD:
            return None
        elif signal_type == SignalType.BUY:
            trade_type = 'buy'
        elif signal_type == SignalType.SELL:
            trade_type = 'sell'
        else:
            return None  # Unknown signal type
            
        quantity = self.calculate_position_size(symbol, price, signal_type)
        
        return self.execute_trade(symbol, price, quantity, timestamp, trade_type)
    
    def execute_trade(self, symbol, price, quantity, timestamp, trade_type):
        """
        Execute a trade and update the portfolio.
        
        Args:
            symbol (str): The symbol of the asset.
            price (float): The price of the asset.
            quantity (float): The quantity to trade.
            timestamp (pd.Timestamp): The timestamp of the trade.
            trade_type (str): The type of trade ('buy' or 'sell').
            
        Returns:
            dict: The executed trade.
        """
        trade_value = price * quantity
        commission = trade_value * 0.001  # 0.1% commission
        
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'trade_type': trade_type,
            'trade_value': trade_value,
            'commission': commission
        }
        
        # Update positions and capital
        if trade_type == 'buy':
            self.current_capital -= (trade_value + commission)
            if symbol in self.positions:
                self.positions[symbol] += quantity
            else:
                self.positions[symbol] = quantity
        elif trade_type == 'sell':
            self.current_capital += (trade_value - commission)
            if symbol in self.positions:
                self.positions[symbol] -= quantity
                if self.positions[symbol] <= 0:
                    del self.positions[symbol]
        
        # Record the trade
        self.trades.append(trade)
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'capital': self.current_capital,
            'positions_value': self.calculate_positions_value(price),
            'total_equity': self.current_capital + self.calculate_positions_value(price)
        })
        
        # Emit trade event if event bus is available
        self._emit_trade_event(trade)
        
        return trade
    
    def calculate_positions_value(self, current_price):
        """
        Calculate the current value of all positions.
        
        Args:
            current_price (float): The current price of the asset.
            
        Returns:
            float: The total value of all positions.
        """
        total_value = 0.0
        for symbol, quantity in self.positions.items():
            total_value += quantity * current_price
        return total_value
    
    def get_equity_curve(self):
        """
        Get the equity curve of the portfolio.
        
        Returns:
            pd.DataFrame: The equity curve.
        """
        return pd.DataFrame(self.equity_curve).set_index('timestamp')
    
    def get_trades(self):
        """
        Get the list of executed trades.
        
        Returns:
            pd.DataFrame: The list of trades.
        """
        return pd.DataFrame(self.trades) 