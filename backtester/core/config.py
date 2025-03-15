"""
Configuration management for the backtesting system.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional


class BacktesterConfig:
    """
    Configuration manager for the backtesting system.
    
    This class handles loading, validating, and accessing configuration settings
    for the backtesting system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a configuration file
        """
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
            
        Returns:
            Dictionary with configuration settings
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file type from extension
        _, ext = os.path.splitext(config_path)
        
        # Load configuration
        if ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif ext.lower() == '.json':
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
        
        # Validate configuration
        self._validate_config()
        
        return self.config
    
    def _validate_config(self):
        """Validate the configuration."""
        # Check for required sections
        required_sections = ['data_source', 'strategy', 'portfolio_manager', 'backtest']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data source configuration
        if 'type' not in self.config['data_source']:
            raise ValueError("Data source configuration must include 'type'")
        
        # Validate strategy configuration
        if 'type' not in self.config['strategy']:
            raise ValueError("Strategy configuration must include 'type'")
        
        # Validate portfolio manager configuration
        if 'type' not in self.config['portfolio_manager']:
            raise ValueError("Portfolio manager configuration must include 'type'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default
        """
        # Handle nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            value = self.config
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            
            return value
        
        # Handle top-level keys
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            value: Value to set
        """
        # Handle nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            config = self.config
            
            # Navigate to the nested dictionary
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            
            # Set the value
            config[parts[-1]] = value
        else:
            # Set top-level key
            self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return self.config.copy()
    
    def save(self, path: str):
        """
        Save the configuration to a file.
        
        Args:
            path: Path to save the configuration file
        """
        # Determine file type from extension
        _, ext = os.path.splitext(path)
        
        # Save configuration
        if ext.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif ext.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=4)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}") 