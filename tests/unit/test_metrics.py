#!/usr/bin/env python
"""
Unit tests for the metrics module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtester.validation.metrics import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_metrics
)


class TestMetrics(unittest.TestCase):
    """Test cases for the metrics module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample portfolio values
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        self.portfolio_values = 10000 * np.cumprod(1 + np.random.normal(0.001, 0.005, 100))
        self.portfolio_series = pd.Series(self.portfolio_values, index=dates)
        
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
        
        # Create positions
        self.positions = pd.Series([0, 1, 1, 1, 0, 0, 1, 1, 1, 0] * 10, index=dates)
    
    def test_calculate_returns(self):
        """Test the calculate_returns function."""
        returns = calculate_returns(self.portfolio_series)
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), len(self.portfolio_series) - 1)
    
    def test_calculate_max_drawdown(self):
        """Test the calculate_max_drawdown function."""
        # Create a specific series with a known drawdown
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        prices = pd.Series([100, 90, 95, 70, 80], index=dates)
        
        # Calculate the expected drawdown
        # The maximum drawdown should be (100 - 70) / 100 = 0.3 or 30%
        expected_drawdown = 0.3
        
        max_drawdown = calculate_max_drawdown(prices)
        self.assertIsInstance(max_drawdown, float)
        self.assertAlmostEqual(max_drawdown, expected_drawdown, places=6)
        
        # Also test with the portfolio series
        max_drawdown = calculate_max_drawdown(self.portfolio_series)
        self.assertIsInstance(max_drawdown, float)
        self.assertGreaterEqual(max_drawdown, 0)  # Max drawdown should be positive or zero
        self.assertLessEqual(max_drawdown, 1)  # Max drawdown should be less than or equal to 100%
    
    def test_calculate_sharpe_ratio(self):
        """Test the calculate_sharpe_ratio function."""
        returns = calculate_returns(self.portfolio_series)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe_ratio, float)
    
    def test_calculate_win_rate(self):
        """Test the win rate calculation."""
        win_rate = calculate_win_rate(self.trades_dict)
        self.assertIsInstance(win_rate, float)
        self.assertEqual(win_rate, 0.5)  # 2 winning trades out of 4
    
    def test_calculate_metrics(self):
        """Test the metrics calculation function."""
        # Calculate returns from portfolio values
        returns = calculate_returns(self.portfolio_series)
        
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


if __name__ == '__main__':
    unittest.main() 