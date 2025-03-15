"""
Bollinger Bands strategy for backtesting.

This module provides a strategy that generates trading signals based on Bollinger Bands.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from backtester.strategy.base import Strategy


class BollingerBandsStrategy(Strategy):
    """
    A strategy that generates trading signals based on Bollinger Bands.
    
    This strategy generates buy signals when the price crosses below the lower band
    and sell signals when the price crosses above the upper band.
    
    Attributes:
        window (int): The window size for the moving average.
        num_std (float): The number of standard deviations for the bands.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        """
        Initialize the Bollinger Bands strategy.
        
        Args:
            window: The window size for the moving average.
            num_std: The number of standard deviations for the bands.
        """
        super().__init__()
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            data: A DataFrame containing price data.
        
        Returns:
            A DataFrame containing the trading signals.
        """
        # Make a copy of the data
        df = data.copy()
        
        # Calculate the moving average
        df['ma'] = df['close'].rolling(window=self.window).mean()
        
        # Calculate the standard deviation
        df['std'] = df['close'].rolling(window=self.window).std()
        
        # Calculate the upper and lower bands
        df['upper_band'] = df['ma'] + (self.num_std * df['std'])
        df['lower_band'] = df['ma'] - (self.num_std * df['std'])
        
        # Initialize the signal column
        df['signal'] = 0
        
        # Generate buy signals when price crosses below the lower band
        df.loc[df['close'] < df['lower_band'], 'signal'] = 1
        
        # Generate sell signals when price crosses above the upper band
        df.loc[df['close'] > df['upper_band'], 'signal'] = -1
        
        # Calculate the position column (cumulative sum of signals)
        df['position'] = df['signal'].cumsum()
        
        # Ensure the position is either -1, 0, or 1
        df['position'] = df['position'].clip(-1, 1)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters of the strategy.
        
        Returns:
            A dictionary containing the parameters of the strategy.
        """
        return {
            'window': self.window,
            'num_std': self.num_std
        } 