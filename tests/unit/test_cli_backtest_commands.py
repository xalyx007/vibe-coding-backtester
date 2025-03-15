#!/usr/bin/env python
"""
Unit tests for the CLI backtest commands module.
"""

import unittest
import os
import tempfile
import json
import yaml
import logging
from unittest.mock import patch, MagicMock

from backtester.cli.backtest_commands import run_backtest
from backtester.core import Backtester, BacktesterConfig, BacktestResults


class TestCliBacktestCommands(unittest.TestCase):
    """Test cases for the CLI backtest commands module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_config = {
            'data': {
                'source': 'csv',
                'path': 'data/btc_usd.csv',
                'symbol': 'BTC-USD'
            },
            'strategy': {
                'name': 'moving_average_crossover',
                'parameters': {
                    'short_window': 10,
                    'long_window': 50
                }
            },
            'portfolio': {
                'initial_capital': 10000,
                'position_size': 0.1
            },
            'backtest': {
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
                'transaction_costs': 0.001,
                'slippage': 0.0005
            }
        }
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a temporary config file
        self.config_path = os.path.join(self.temp_dir.name, 'config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.sample_config, f)
        
        # Create mock arguments
        self.args = MagicMock()
        self.args.config = self.config_path
        self.args.output_dir = os.path.join(self.temp_dir.name, 'output')
        self.args.start_date = '2020-01-01'
        self.args.end_date = '2020-12-31'
        self.args.symbol = 'BTC-USD'
        
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('test_logger')
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    @patch('backtester.cli.backtest_commands.create_data_source')
    @patch('backtester.cli.backtest_commands.create_strategy')
    @patch('backtester.cli.backtest_commands.create_portfolio_manager')
    @patch('backtester.cli.backtest_commands.Backtester')
    def test_run_backtest_with_config(self, mock_backtester_class, mock_create_portfolio_manager, 
                                    mock_create_strategy, mock_create_data_source):
        """Test running a backtest with a configuration object."""
        # Configure mocks
        mock_data_source = MagicMock()
        mock_strategy = MagicMock()
        mock_portfolio_manager = MagicMock()
        mock_backtester = MagicMock()
        mock_results = MagicMock(spec=BacktestResults)
        
        mock_create_data_source.return_value = mock_data_source
        mock_create_strategy.return_value = mock_strategy
        mock_create_portfolio_manager.return_value = mock_portfolio_manager
        mock_backtester_class.return_value = mock_backtester
        mock_backtester.run.return_value = mock_results
        
        # Create config
        config = BacktesterConfig()
        config.config = {
            'data_source': {'type': 'csv', 'path': 'data/btc_usd.csv'},
            'strategy': {'type': 'moving_average_crossover', 'parameters': {'short_window': 10, 'long_window': 50}},
            'portfolio_manager': {'type': 'basic', 'initial_capital': 10000},
            'backtest': {'start_date': '2020-01-01', 'end_date': '2020-12-31'}
        }
        
        # Run the backtest
        run_backtest(self.args, config, self.logger)
        
        # Verify that the correct functions were called
        mock_create_data_source.assert_called_once()
        mock_create_strategy.assert_called_once()
        mock_create_portfolio_manager.assert_called_once()
        mock_backtester_class.assert_called_once()
        mock_backtester.run.assert_called_once()
        mock_results.to_json.assert_called_once()
    
    @patch('backtester.cli.backtest_commands.create_data_source_from_args')
    @patch('backtester.cli.backtest_commands.create_strategy_from_args')
    @patch('backtester.cli.backtest_commands.create_portfolio_manager_from_args')
    @patch('backtester.cli.backtest_commands.Backtester')
    def test_run_backtest_without_config(self, mock_backtester_class, mock_create_portfolio_manager_from_args, 
                                       mock_create_strategy_from_args, mock_create_data_source_from_args):
        """Test running a backtest without a configuration object."""
        # Configure mocks
        mock_data_source = MagicMock()
        mock_strategy = MagicMock()
        mock_portfolio_manager = MagicMock()
        mock_backtester = MagicMock()
        mock_results = MagicMock(spec=BacktestResults)
        
        mock_create_data_source_from_args.return_value = mock_data_source
        mock_create_strategy_from_args.return_value = mock_strategy
        mock_create_portfolio_manager_from_args.return_value = mock_portfolio_manager
        mock_backtester_class.return_value = mock_backtester
        mock_backtester.run.return_value = mock_results
        
        # Run the backtest
        run_backtest(self.args, None, self.logger)
        
        # Verify that the correct functions were called
        mock_create_data_source_from_args.assert_called_once_with(self.args)
        mock_create_strategy_from_args.assert_called_once_with(self.args)
        mock_create_portfolio_manager_from_args.assert_called_once_with(self.args)
        mock_backtester_class.assert_called_once()
        mock_backtester.run.assert_called_once()
        mock_results.to_json.assert_called_once()


if __name__ == '__main__':
    unittest.main() 