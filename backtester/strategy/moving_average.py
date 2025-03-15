"""
Moving Average Crossover Strategy for the Backtesting System.

This module implements a simple moving average crossover strategy.
"""

import pandas as pd
import numpy as np
from backtester.strategy.base import Strategy
from backtester.utils.constants import SignalType

class MovingAverageCrossover(Strategy):
    """
    A strategy that generates signals based on moving average crossovers.
    
    Attributes:
        short_window (int): The window size for the short moving average.
        long_window (int): The window size for the long moving average.
        price_column (str): The column name for price data.
    """
    
    def __init__(self, short_window=20, long_window=50, price_column='close', **kwargs):
        """
        Initialize the moving average crossover strategy.
        
        Args:
            short_window (int): The window size for the short moving average.
            long_window (int): The window size for the long moving average.
            price_column (str): The column name for price data.
            **kwargs: Additional arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.price_column = price_column
        
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data (pd.DataFrame): The market data.
            
        Returns:
            pd.DataFrame: The input data with additional signal columns.
        """
        # Make a copy of the data to avoid modifying the original
        signals = data.copy()
        
        # Calculate moving averages
        signals[f'sma_{self.short_window}'] = signals[self.price_column].rolling(window=self.short_window).mean()
        signals[f'sma_{self.long_window}'] = signals[self.price_column].rolling(window=self.long_window).mean()
        
        # Create signal column (BUY, SELL, HOLD)
        signals['signal'] = SignalType.HOLD
        
        # Generate signals based on crossover
        signals.loc[signals[f'sma_{self.short_window}'] > signals[f'sma_{self.long_window}'], 'signal'] = SignalType.BUY
        signals.loc[signals[f'sma_{self.short_window}'] <= signals[f'sma_{self.long_window}'], 'signal'] = SignalType.SELL
        
        # Generate positions (1 for long, -1 for short, 0 for no position)
        # This is kept for backward compatibility but not used by the backtester
        signals['position'] = 0
        signals.loc[signals['signal'] == SignalType.BUY, 'position'] = 1
        signals.loc[signals['signal'] == SignalType.SELL, 'position'] = -1
        
        # Replace NaN values with HOLD
        signals['signal'] = signals['signal'].fillna(SignalType.HOLD)
        signals.fillna(0, inplace=True)
        
        # Emit signal events if event bus is available
        if self.event_bus:
            for idx, row in signals.iterrows():
                if row['signal'] != SignalType.HOLD:
                    self._emit_signal_event(
                        timestamp=idx,
                        signal_type=row['signal'],
                        metadata={
                            'price': row[self.price_column],
                            'short_ma': row[f'sma_{self.short_window}'],
                            'long_ma': row[f'sma_{self.long_window}']
                        }
                    )
        
        return signals 