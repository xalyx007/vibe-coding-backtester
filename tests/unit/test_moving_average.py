"""
Unit tests for the MovingAverageCrossover strategy.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtester.strategy import MovingAverageCrossover
from backtester.utils.constants import SignalType


class TestMovingAverageCrossover(unittest.TestCase):
    """Test cases for the MovingAverageCrossover strategy."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        
        # Create price data with a known pattern
        # First 50 days: downtrend
        # Last 50 days: uptrend
        prices = np.zeros(100)
        prices[:50] = np.linspace(100, 50, 50)  # Downtrend
        prices[50:] = np.linspace(50, 150, 50)  # Uptrend
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Set timestamp as index
        self.data.set_index('timestamp', inplace=True)
        
        # Create strategy
        self.strategy = MovingAverageCrossover(
            short_window=10,
            long_window=30
        )
    
    def test_generate_signals(self):
        """Test signal generation."""
        # Generate signals
        signals = self.strategy.generate_signals(self.data)
        
        # Check that signals DataFrame has the correct shape
        self.assertEqual(signals.shape[0], self.data.shape[0])
        self.assertIn('signal', signals.columns)
        
        # Check that signals are of the correct type
        for signal in signals['signal']:
            self.assertIsInstance(signal, SignalType)
        
        # Check that there are some BUY and SELL signals
        signal_types = signals['signal'].unique()
        self.assertIn(SignalType.BUY, signal_types)
        self.assertIn(SignalType.SELL, signal_types)
        
        # Check that signals occur at the expected times
        # In the downtrend, we expect SELL signals
        # In the uptrend, we expect BUY signals
        downtrend_signals = signals.iloc[40:50]['signal']  # End of downtrend
        uptrend_signals = signals.iloc[80:90]['signal']    # Middle of uptrend
        
        # Check that there's at least one SELL signal in the downtrend
        self.assertTrue(any(s == SignalType.SELL for s in downtrend_signals))
        
        # Check that there's at least one BUY signal in the uptrend
        self.assertTrue(any(s == SignalType.BUY for s in uptrend_signals))
    
    def test_get_parameters(self):
        """Test parameter retrieval."""
        params = self.strategy.get_parameters()
        
        # Check that parameters are correct
        self.assertEqual(params['short_window'], 10)
        self.assertEqual(params['long_window'], 30)
        self.assertEqual(params['price_column'], 'close')


if __name__ == '__main__':
    unittest.main() 