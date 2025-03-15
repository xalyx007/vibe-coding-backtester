"""
Combined Strategy for the Backtesting System.

This module implements a strategy that combines Moving Average Crossover, RSI, and MACD indicators.
"""

import pandas as pd
import numpy as np
from backtester.strategy.base import Strategy
from backtester.utils.constants import SignalType


class CombinedStrategy(Strategy):
    """
    A strategy that combines Moving Average Crossover, RSI, and MACD indicators.
    
    Attributes:
        short_ma_window (int): The window size for the short moving average.
        long_ma_window (int): The window size for the long moving average.
        rsi_window (int): The window size for RSI calculation.
        rsi_overbought (float): The overbought threshold for RSI.
        rsi_oversold (float): The oversold threshold for RSI.
        macd_fast (int): The window size for the fast EMA in MACD.
        macd_slow (int): The window size for the slow EMA in MACD.
        macd_signal (int): The window size for the signal line in MACD.
        price_column (str): The column name for price data.
    """
    
    def __init__(self, 
                 short_ma_window=9, 
                 long_ma_window=50, 
                 rsi_window=14, 
                 rsi_overbought=70, 
                 rsi_oversold=30, 
                 macd_fast=12, 
                 macd_slow=26, 
                 macd_signal=9, 
                 price_column='close', 
                 **kwargs):
        """
        Initialize the combined strategy.
        
        Args:
            short_ma_window (int): The window size for the short moving average.
            long_ma_window (int): The window size for the long moving average.
            rsi_window (int): The window size for RSI calculation.
            rsi_overbought (float): The overbought threshold for RSI.
            rsi_oversold (float): The oversold threshold for RSI.
            macd_fast (int): The window size for the fast EMA in MACD.
            macd_slow (int): The window size for the slow EMA in MACD.
            macd_signal (int): The window size for the signal line in MACD.
            price_column (str): The column name for price data.
            **kwargs: Additional arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self.short_ma_window = short_ma_window
        self.long_ma_window = long_ma_window
        self.rsi_window = rsi_window
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
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
        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data):
        """
        Calculate the MACD for the given data.
        
        Args:
            data (pd.Series): The price data.
            
        Returns:
            tuple: (MACD line, Signal line, Histogram)
        """
        # Calculate EMAs
        ema_fast = data.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = data.ewm(span=self.macd_slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate Signal line
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        
        # Calculate Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the combined strategy.
        
        Args:
            data (pd.DataFrame): The market data.
            
        Returns:
            pd.DataFrame: The input data with additional signal columns.
        """
        # Make a copy of the data to avoid modifying the original
        signals = data.copy()
        
        # Calculate Moving Averages
        signals[f'ma_{self.short_ma_window}'] = signals[self.price_column].rolling(window=self.short_ma_window).mean()
        signals[f'ma_{self.long_ma_window}'] = signals[self.price_column].rolling(window=self.long_ma_window).mean()
        
        # Calculate RSI
        signals['rsi'] = self.calculate_rsi(signals[self.price_column])
        
        # Calculate MACD
        signals['macd_line'], signals['signal_line'], signals['macd_hist'] = self.calculate_macd(signals[self.price_column])
        
        # Create signal column (BUY, SELL, HOLD)
        signals['signal'] = SignalType.HOLD
        
        # Generate BUY signals based on combined conditions
        buy_condition = (
            (signals[f'ma_{self.short_ma_window}'] > signals[f'ma_{self.long_ma_window}']) &  # Short MA > Long MA
            (signals['macd_line'] > signals['signal_line']) &  # MACD line > Signal line
            (signals['rsi'] < self.rsi_overbought)  # RSI < 70 (not overbought)
        )
        
        # Generate SELL signals based on combined conditions
        sell_condition = (
            (signals[f'ma_{self.short_ma_window}'] < signals[f'ma_{self.long_ma_window}']) &  # Short MA < Long MA
            (signals['macd_line'] < signals['signal_line']) &  # MACD line < Signal line
            (signals['rsi'] > self.rsi_oversold)  # RSI > 30 (not oversold)
        )
        
        # Apply signals
        signals.loc[buy_condition, 'signal'] = SignalType.BUY
        signals.loc[sell_condition, 'signal'] = SignalType.SELL
        
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
                            'short_ma': row[f'ma_{self.short_ma_window}'],
                            'long_ma': row[f'ma_{self.long_ma_window}'],
                            'rsi': row['rsi'],
                            'macd_line': row['macd_line'],
                            'signal_line': row['signal_line']
                        }
                    )
        
        return signals 