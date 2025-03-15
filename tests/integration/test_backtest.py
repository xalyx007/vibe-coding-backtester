"""
Integration tests for the backtesting system.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile

from backtester.data import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import BasicPortfolioManager, PercentageSizer
from backtester.core import Backtester


class TestBacktestIntegration(unittest.TestCase):
    """Integration tests for the backtesting system."""
    
    def setUp(self):
        """Set up test data and components."""
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
        
        # Create temporary CSV file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, 'test_data.csv')
        self.data.to_csv(self.csv_path, index=False)
        
        # Create components
        self.data_source = CSVDataSource(
            filepath=self.csv_path,
            date_column='timestamp'
        )
        
        self.strategy = MovingAverageCrossover(
            short_window=10,
            long_window=30
        )
        
        self.portfolio_manager = BasicPortfolioManager(
            initial_capital=10000,
            position_size=0.1
        )
        
        self.backtester = Backtester(
            data_source=self.data_source,
            strategy=self.strategy,
            portfolio_manager=self.portfolio_manager,
            transaction_costs=0.001,
            slippage=0.0005
        )
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_backtest_run(self):
        """Test running a backtest end-to-end."""
        # Run backtest
        results = self.backtester.run(symbol='TEST')
        
        # Check that results are not empty
        self.assertGreater(len(results.portfolio_values), 0)
        self.assertGreater(len(results.trades), 0)
        
        # Check that portfolio values are calculated correctly
        self.assertEqual(results.portfolio_values.iloc[0]['portfolio_value'], 10000)
        
        # Check that metrics are calculated
        summary = results.summary()
        self.assertIn('total_return', summary)
        self.assertIn('sharpe_ratio', summary)
        self.assertIn('max_drawdown', summary)
        
        # Check that the number of trades is reasonable
        # In this test data, we expect at least a few trades
        self.assertGreater(summary['total_trades'], 0)
    
    def test_parameter_sweep(self):
        """Test running a parameter sweep."""
        # Define parameter sweep
        short_windows = [5, 10, 15]
        long_windows = [20, 30, 40]
        
        results = []
        
        # Perform parameter sweep
        for short_window in short_windows:
            for long_window in long_windows:
                # Skip invalid combinations
                if short_window >= long_window:
                    continue
                
                # Create strategy with these parameters
                strategy = MovingAverageCrossover(
                    short_window=short_window,
                    long_window=long_window
                )
                
                # Create backtester
                backtester = Backtester(
                    data_source=self.data_source,
                    strategy=strategy,
                    portfolio_manager=self.portfolio_manager,
                    transaction_costs=0.001,
                    slippage=0.0005
                )
                
                # Run backtest
                result = backtester.run(symbol='TEST')
                
                # Store result
                results.append({
                    'short_window': short_window,
                    'long_window': long_window,
                    'total_return': result.total_return
                })
        
        # Check that we have the expected number of results
        expected_count = sum(1 for sw in short_windows for lw in long_windows if sw < lw)
        self.assertEqual(len(results), expected_count)
        
        # Check that the results vary with different parameters
        returns = [r['total_return'] for r in results]
        self.assertGreater(max(returns) - min(returns), 0.001)  # At least 0.1% difference


if __name__ == '__main__':
    unittest.main() 