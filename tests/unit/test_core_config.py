#!/usr/bin/env python
"""
Unit tests for the core config module.
"""

import unittest
import os
import tempfile
import json
import yaml
from unittest.mock import patch, mock_open

from backtester.core.config import BacktesterConfig


class TestBacktesterConfig(unittest.TestCase):
    """Test cases for the BacktesterConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_config = {
            'data_source': {
                'type': 'csv',
                'path': 'data/btc_usd.csv',
                'symbol': 'BTC-USD'
            },
            'strategy': {
                'type': 'moving_average_crossover',
                'parameters': {
                    'short_window': 10,
                    'long_window': 50
                }
            },
            'portfolio_manager': {
                'type': 'basic',
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
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization_without_config(self):
        """Test initialization without a config file."""
        config = BacktesterConfig()
        self.assertEqual(config.config, {})
    
    def test_load_yaml_config(self):
        """Test loading a YAML config file."""
        # Create a temporary YAML config file
        config_path = os.path.join(self.temp_dir.name, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.sample_config, f)
        
        # Load the config
        config = BacktesterConfig(config_path)
        
        # Verify that the config was loaded correctly
        self.assertEqual(config.config, self.sample_config)
    
    def test_load_json_config(self):
        """Test loading a JSON config file."""
        # Create a temporary JSON config file
        config_path = os.path.join(self.temp_dir.name, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.sample_config, f)
        
        # Load the config
        config = BacktesterConfig(config_path)
        
        # Verify that the config was loaded correctly
        self.assertEqual(config.config, self.sample_config)
    
    def test_get_config_value(self):
        """Test getting a value from the config."""
        config = BacktesterConfig()
        config.config = self.sample_config
        
        # Test getting a top-level value
        self.assertEqual(config.get('data_source'), self.sample_config['data_source'])
        
        # Test getting a nested value
        self.assertEqual(config.get('strategy.type'), 'moving_average_crossover')
        self.assertEqual(config.get('strategy.parameters.short_window'), 10)
        
        # Test getting a value with a default
        self.assertEqual(config.get('nonexistent', 'default'), 'default')
        
        # Test getting a nested value with a default
        self.assertEqual(config.get('strategy.nonexistent', 'default'), 'default')
    
    def test_set_config_value(self):
        """Test setting a value in the config."""
        config = BacktesterConfig()
        
        # Test setting a top-level value
        config.set('data_source', {'type': 'csv'})
        self.assertEqual(config.get('data_source'), {'type': 'csv'})
        
        # Test setting a nested value
        config.set('strategy.type', 'rsi')
        self.assertEqual(config.get('strategy.type'), 'rsi')
        
        # Test setting a deeply nested value
        config.set('strategy.parameters.rsi_period', 14)
        self.assertEqual(config.get('strategy.parameters.rsi_period'), 14)
    
    def test_to_dict(self):
        """Test converting the config to a dictionary."""
        config = BacktesterConfig()
        config.config = self.sample_config
        
        # Verify that to_dict returns a copy of the config
        config_dict = config.to_dict()
        self.assertEqual(config_dict, self.sample_config)
        self.assertIsNot(config_dict, config.config)  # Should be a copy, not the same object
    
    def test_save_config(self):
        """Test saving the config to a file."""
        config = BacktesterConfig()
        config.config = self.sample_config
        
        # Test saving to a YAML file
        yaml_path = os.path.join(self.temp_dir.name, 'saved_config.yaml')
        config.save(yaml_path)
        
        # Verify that the file was created and contains the correct data
        with open(yaml_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        self.assertEqual(saved_config, self.sample_config)
        
        # Test saving to a JSON file
        json_path = os.path.join(self.temp_dir.name, 'saved_config.json')
        config.save(json_path)
        
        # Verify that the file was created and contains the correct data
        with open(json_path, 'r') as f:
            saved_config = json.load(f)
        self.assertEqual(saved_config, self.sample_config)


if __name__ == '__main__':
    unittest.main() 