"""
Configuration utilities for the backtesting system.
"""

import os
import yaml
import json
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file (YAML or JSON)
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file type based on extension
    _, ext = os.path.splitext(config_path)
    
    # Load configuration
    with open(config_path, 'r') as f:
        if ext.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif ext.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
    
    # Validate configuration
    validate_config(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Check required sections
    required_sections = ['data_source', 'strategy', 'portfolio_manager', 'backtest']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Check data source configuration
    data_source = config['data_source']
    if 'type' not in data_source:
        raise ValueError("Missing 'type' in data_source configuration")
    
    # Check strategy configuration
    strategy = config['strategy']
    if 'type' not in strategy:
        raise ValueError("Missing 'type' in strategy configuration")
    
    # Check portfolio manager configuration
    portfolio_manager = config['portfolio_manager']
    if 'type' not in portfolio_manager:
        raise ValueError("Missing 'type' in portfolio_manager configuration")
    
    # Check backtest configuration
    backtest = config['backtest']
    if 'symbol' not in backtest:
        raise ValueError("Missing 'symbol' in backtest configuration") 