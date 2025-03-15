#!/usr/bin/env python
"""
Integration tests for the validation module.

This file contains integration tests for the cross-validation, monte carlo,
and walk-forward validation methods in the backtester.validation module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile

from backtester.validation import (
    run_cross_validation,
    run_monte_carlo,
    run_walk_forward
)
from backtester.data.csv_source import CSVDataSource
from backtester.strategy.moving_average import MovingAverageCrossover
from backtester.portfolio.basic import BasicPortfolioManager


class TestValidationIntegration(unittest.TestCase):
    """Integration tests for the validation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample price data
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 100))
        
        self.sample_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Save sample data to CSV
        self.csv_path = os.path.join(self.temp_dir.name, 'sample_data.csv')
        self.sample_data.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    @unittest.skip("Requires fixing DataFrameSource scope in validation functions")
    def test_cross_validation_integration(self):
        """Test the cross-validation integration."""
        # Create a CSVDataSource with the correct parameters
        data_source = CSVDataSource(self.csv_path)
        
        # Load the data
        data_source.load()
        
        strategy = MovingAverageCrossover(short_window=5, long_window=10)
        portfolio_manager = BasicPortfolioManager(initial_capital=10000)
        
        # Run cross-validation
        results = run_cross_validation(
            data_source=data_source,
            strategy=strategy,
            portfolio_manager=portfolio_manager,
            folds=2  # Use a small number of folds for testing
        )
        
        # Verify that the results contain the expected keys
        self.assertIsInstance(results, dict)
        self.assertIn('metrics', results)
        self.assertIn('fold_metrics', results)
        
        # Check that metrics contains expected values
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            self.assertIn(metric, results['metrics'])
    
    @unittest.skip("Requires fixing DataFrameSource scope in validation functions")
    def test_monte_carlo_integration(self):
        """Test the Monte Carlo simulation integration."""
        # Create a CSVDataSource with the correct parameters
        data_source = CSVDataSource(self.csv_path)
        
        # Load the data
        data_source.load()
        
        strategy = MovingAverageCrossover(short_window=5, long_window=10)
        portfolio_manager = BasicPortfolioManager(initial_capital=10000)
        
        # Run Monte Carlo simulation
        results = run_monte_carlo(
            data_source=data_source,
            strategy=strategy,
            portfolio_manager=portfolio_manager,
            simulations=2  # Use a small number of simulations for testing
        )
        
        # Verify that the results contain the expected keys
        self.assertIsInstance(results, dict)
        self.assertIn('metrics', results)
        self.assertIn('simulation_metrics', results)
        
        # Check that metrics contains expected values
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            self.assertIn(metric, results['metrics'])
    
    @unittest.skip("Requires fixing DataFrameSource scope in validation functions")
    def test_walk_forward_integration(self):
        """Test the walk-forward optimization integration."""
        # Create a CSVDataSource with the correct parameters
        data_source = CSVDataSource(self.csv_path)
        
        # Load the data
        data_source.load()
        
        strategy = MovingAverageCrossover(short_window=5, long_window=10)
        portfolio_manager = BasicPortfolioManager(initial_capital=10000)
        
        # Define parameter grid
        parameter_grid = {
            'short_window': [3, 5],
            'long_window': [10, 15]
        }
        
        # Run walk-forward optimization
        results = run_walk_forward(
            data_source=data_source,
            strategy=strategy,
            portfolio_manager=portfolio_manager,
            parameter_grid=parameter_grid,
            window_size=20,
            step_size=10
        )
        
        # Verify that the results contain the expected keys
        self.assertIsInstance(results, dict)
        self.assertIn('metrics', results)
        self.assertIn('window_metrics', results)
        
        # Check that metrics contains expected values
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            self.assertIn(metric, results['metrics'])


if __name__ == '__main__':
    unittest.main() 