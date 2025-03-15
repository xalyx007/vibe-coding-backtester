"""
RSI Strategy for the Backtesting System.

This module implements a strategy based on the Relative Strength Index (RSI).
"""

import pandas as pd
import numpy as np
from backtester.strategy.base import Strategy

class RSIStrategy(Strategy):
    """
    A strategy that generates signals based on RSI values.
    
    Attributes:
        window (int): The window size for RSI calculation.
        overbought (float): The overbought threshold.
        oversold (float): The oversold threshold.
        price_column (str): The column name for price data.
    """
    
    def __init__(self, window=14, overbought=70, oversold=30, price_column='close'):
        """
        Initialize the RSI strategy.
        
        Args:
            window (int): The window size for RSI calculation.
            overbought (float): The overbought threshold.
            oversold (float): The oversold threshold.
            price_column (str): The column name for price data.
        """
        super().__init__()
        self.window = window
        self.overbought = overbought
        self.oversold = oversold
        self.price_column = price_column
        
    def calculate_rsi(self, data):
        """
        Calculate the RSI for the given data.
        
        Args:
            data (pd.Series): The price data.
            
        Returns:
            pd.Series: The RSI values.
        """
        # Calculate price changes
        delta = data.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI values.
        
        Args:
            data (pd.DataFrame): The market data.
            
        Returns:
            pd.DataFrame: The input data with additional signal columns.
        """
        # Make a copy of the data to avoid modifying the original
        signals = data.copy()
        
        # Calculate RSI
        signals['rsi'] = self.calculate_rsi(signals[self.price_column])
        
        # Create signal column (1 for buy, -1 for sell, 0 for hold)
        signals['signal'] = 0.0
        
        # Generate signals based on RSI thresholds
        signals['signal'] = np.where(signals['rsi'] < self.oversold, 1.0, 0.0)
        signals['signal'] = np.where(signals['rsi'] > self.overbought, -1.0, signals['signal'])
        
        # Generate positions (1 for long, -1 for short, 0 for no position)
        signals['position'] = signals['signal'].diff()
        
        # Replace NaN values with 0
        signals.fillna(0, inplace=True)
        
        return signals 