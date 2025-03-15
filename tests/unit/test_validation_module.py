#!/usr/bin/env python
"""
Unit tests for the validation module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import shutil

from backtester.validation import (
    run_cross_validation,
    run_monte_carlo,
    run_walk_forward,
    calculate_metrics
)
from backtester.data.csv_source import CSVDataSource
from backtester.strategy.moving_average import MovingAverageCrossover
from backtester.portfolio.basic import BasicPortfolioManager


class TestValidationModule(unittest.TestCase):
    """Test cases for the validation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        prices = 100 * (1 + np.random.normal(0, 0.01, 100).cumsum() * 0.1)
        self.price_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, 'test_prices.csv')
        self.price_data.to_csv(self.csv_path, index=False)
        
        # Create sample portfolio values
        self.portfolio_values = pd.Series(prices, index=dates)
        
        # Create sample trades as dictionaries for calculate_win_rate
        self.trades_dict = [
            {'profit': 100, 'profit_pct': 0.01},
            {'profit': -50, 'profit_pct': -0.005},
            {'profit': 200, 'profit_pct': 0.02},
            {'profit': -30, 'profit_pct': -0.003}
        ]
        
        # Create sample trades as tuples for calculate_metrics
        self.trades_tuple = [
            (1.0, 0.01),  # (quantity, profit_pct) - winning trade
            (2.0, -0.005),  # losing trade
            (1.5, 0.02),  # winning trade
            (0.5, -0.003)  # losing trade
        ]
        
        # Create sample positions
        self.positions = pd.DataFrame({
            'symbol': ['AAPL'] * len(dates),
            'quantity': [10] * len(dates),
            'entry_price': [150.0] * len(dates)
        }, index=dates)
        
        # Create components
        self.data_source = CSVDataSource(self.csv_path)
        self.strategy = MovingAverageCrossover(short_window=10, long_window=50)
        self.portfolio_manager = BasicPortfolioManager(initial_capital=10000)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_calculate_metrics(self):
        """Test the metrics calculation function."""
        # Calculate returns from portfolio values
        returns = pd.Series(np.random.normal(0.001, 0.01, 100), 
                           index=pd.date_range(start='2020-01-01', periods=100, freq='D'))
        
        # Create positions list (-1, 0, or 1)
        positions = [1, 1, 0, -1, -1, 0, 1, 1, 0, 0] * 10
        
        metrics = calculate_metrics(returns, positions, self.trades_tuple)
        
        # Check that the metrics dictionary contains expected keys
        expected_keys = ['total_return', 'annualized_return', 'sharpe_ratio', 
                         'max_drawdown', 'win_rate', 'profit_factor']
        
        for key in expected_keys:
            self.assertIn(key, metrics)
            
        # Verify specific metrics
        self.assertEqual(metrics['win_rate'], 0.5)  # 2 winning trades out of 4
    
    @unittest.skip("Requires actual implementation of data source")
    def test_cross_validation(self):
        """Test the cross_validation function."""
        results = run_cross_validation(
            data_source=self.data_source,
            strategy=self.strategy,
            portfolio_manager=self.portfolio_manager,
            folds=5
        )
        
        # Verify that the results contain the expected keys
        self.assertIn('metrics', results)
        self.assertIn('fold_metrics', results)
        self.assertEqual(len(results['fold_metrics']), 5)
    
    @unittest.skip("Requires actual implementation of data source")
    def test_monte_carlo(self):
        """Test the monte_carlo function."""
        results = run_monte_carlo(
            data_source=self.data_source,
            strategy=self.strategy,
            portfolio_manager=self.portfolio_manager,
            simulations=10
        )
        
        # Verify that the results contain the expected keys
        self.assertIn('metrics', results)
        self.assertIn('simulation_metrics', results)
        self.assertEqual(len(results['simulation_metrics']), 10)
    
    @unittest.skip("Requires actual implementation of data source")
    def test_walk_forward(self):
        """Test the walk_forward function."""
        results = run_walk_forward(
            data_source=self.data_source,
            strategy=self.strategy,
            portfolio_manager=self.portfolio_manager,
            window_size=20,
            step_size=10
        )
        
        # Verify that the results contain the expected keys
        self.assertIn('metrics', results)
        self.assertIn('window_metrics', results)
        self.assertGreater(len(results['window_metrics']), 0)


if __name__ == '__main__':
    unittest.main() 