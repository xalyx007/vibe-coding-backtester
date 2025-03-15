#!/usr/bin/env python
"""
Unit tests for the CLI main module.
"""

import unittest
import os
import tempfile
import json
import yaml
import logging
import sys
from unittest.mock import patch, MagicMock

from backtester.cli.main import parse_args, main
from backtester.core import BacktesterConfig


class TestCliMain(unittest.TestCase):
    """Test cases for the CLI main module."""
    
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
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_backtest(self, mock_parse_args):
        """Test parsing arguments for the backtest command."""
        # Configure the mock to return a namespace with backtest command arguments
        mock_args = MagicMock()
        mock_args.command = 'backtest'
        mock_args.verbose = True
        mock_args.config = self.config_path
        mock_args.output_dir = 'output'
        mock_args.start_date = '2020-01-01'
        mock_args.end_date = '2020-12-31'
        mock_args.symbol = 'BTC-USD'
        mock_parse_args.return_value = mock_args
        
        # Call parse_args
        args = parse_args()
        
        # Verify that the arguments were parsed correctly
        self.assertEqual(args.command, 'backtest')
        self.assertEqual(args.verbose, True)
        self.assertEqual(args.config, self.config_path)
        self.assertEqual(args.output_dir, 'output')
        self.assertEqual(args.start_date, '2020-01-01')
        self.assertEqual(args.end_date, '2020-12-31')
        self.assertEqual(args.symbol, 'BTC-USD')
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_validation(self, mock_parse_args):
        """Test parsing arguments for the validation command."""
        # Configure the mock to return a namespace with validation command arguments
        mock_args = MagicMock()
        mock_args.command = 'validation'
        mock_args.verbose = True
        mock_args.config = self.config_path
        mock_args.output_dir = 'output'
        mock_args.method = 'cross_validation'
        mock_args.folds = 5
        mock_parse_args.return_value = mock_args
        
        # Call parse_args
        args = parse_args()
        
        # Verify that the arguments were parsed correctly
        self.assertEqual(args.command, 'validation')
        self.assertEqual(args.verbose, True)
        self.assertEqual(args.config, self.config_path)
        self.assertEqual(args.output_dir, 'output')
        self.assertEqual(args.method, 'cross_validation')
        self.assertEqual(args.folds, 5)
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_analysis(self, mock_parse_args):
        """Test parsing arguments for the analysis command."""
        # Configure the mock to return a namespace with analysis command arguments
        mock_args = MagicMock()
        mock_args.command = 'analysis'
        mock_args.verbose = True
        mock_args.config = self.config_path
        mock_args.output_dir = 'output'
        mock_args.results_file = 'results.json'
        mock_args.analysis_type = 'metrics'
        mock_parse_args.return_value = mock_args
        
        # Call parse_args
        args = parse_args()
        
        # Verify that the arguments were parsed correctly
        self.assertEqual(args.command, 'analysis')
        self.assertEqual(args.verbose, True)
        self.assertEqual(args.config, self.config_path)
        self.assertEqual(args.output_dir, 'output')
        self.assertEqual(args.results_file, 'results.json')
        self.assertEqual(args.analysis_type, 'metrics')
    
    @patch('backtester.cli.main.parse_args')
    @patch('backtester.cli.main.setup_logging')
    @patch('backtester.cli.main.BacktesterConfig')
    @patch('backtester.cli.main.run_backtest')
    def test_main_backtest(self, mock_run_backtest, mock_backtester_config, mock_setup_logging, mock_parse_args):
        """Test the main function with the backtest command."""
        # Configure the mock to return a namespace with backtest command arguments
        mock_args = MagicMock()
        mock_args.command = 'backtest'
        mock_args.verbose = True
        mock_args.config = self.config_path
        mock_args.output_dir = 'output'
        mock_parse_args.return_value = mock_args
        
        # Configure the mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Configure the mock config
        mock_config = MagicMock()
        mock_backtester_config.return_value = mock_config
        
        # Call main
        main()
        
        # Verify that the correct functions were called
        mock_setup_logging.assert_called_once_with(verbose=True)
        mock_backtester_config.assert_called_once_with(self.config_path)
        mock_run_backtest.assert_called_once_with(mock_args, mock_config, mock_logger)
    
    @patch('backtester.cli.main.parse_args')
    @patch('backtester.cli.main.setup_logging')
    @patch('backtester.cli.main.BacktesterConfig')
    @patch('backtester.cli.main.run_validation')
    def test_main_validation(self, mock_run_validation, mock_backtester_config, mock_setup_logging, mock_parse_args):
        """Test the main function with the validation command."""
        # Configure the mock to return a namespace with validation command arguments
        mock_args = MagicMock()
        mock_args.command = 'validation'
        mock_args.verbose = True
        mock_args.config = self.config_path
        mock_args.output_dir = 'output'
        mock_parse_args.return_value = mock_args
        
        # Configure the mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Configure the mock config
        mock_config = MagicMock()
        mock_backtester_config.return_value = mock_config
        
        # Call main
        main()
        
        # Verify that the correct functions were called
        mock_setup_logging.assert_called_once_with(verbose=True)
        mock_backtester_config.assert_called_once_with(self.config_path)
        mock_run_validation.assert_called_once_with(mock_args, mock_config, mock_logger)
    
    @patch('backtester.cli.main.parse_args')
    @patch('backtester.cli.main.setup_logging')
    @patch('backtester.cli.main.BacktesterConfig')
    @patch('backtester.cli.main.run_analysis')
    def test_main_analysis(self, mock_run_analysis, mock_backtester_config, mock_setup_logging, mock_parse_args):
        """Test the main function with the analysis command."""
        # Configure the mock to return a namespace with analysis command arguments
        mock_args = MagicMock()
        mock_args.command = 'analysis'
        mock_args.verbose = True
        mock_args.config = self.config_path
        mock_args.output_dir = 'output'
        mock_parse_args.return_value = mock_args
        
        # Configure the mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Configure the mock config
        mock_config = MagicMock()
        mock_backtester_config.return_value = mock_config
        
        # Call main
        main()
        
        # Verify that the correct functions were called
        mock_setup_logging.assert_called_once_with(verbose=True)
        mock_backtester_config.assert_called_once_with(self.config_path)
        mock_run_analysis.assert_called_once_with(mock_args, mock_config, mock_logger)
    
    @patch('backtester.cli.main.parse_args')
    @patch('backtester.cli.main.setup_logging')
    @patch('backtester.cli.main.sys.exit')
    def test_main_no_command(self, mock_exit, mock_setup_logging, mock_parse_args):
        """Test the main function with no command."""
        # Configure the mock to return a namespace with no command
        mock_args = MagicMock()
        mock_args.command = None
        mock_args.verbose = True
        mock_args.config = None
        mock_args.output_dir = 'output'
        mock_parse_args.return_value = mock_args
        
        # Configure the mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Call main
        main()
        
        # Verify that the program exited with an error
        mock_exit.assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main() 