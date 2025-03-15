"""
Logging utilities for the backtesting system.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str = 'backtester', 
                level: int = logging.INFO, 
                log_file: Optional[str] = None,
                log_dir: str = 'logs') -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Log file name (if None, a default name will be used)
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_file is not None or log_dir is not None:
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create log file name if not provided
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{name}_{timestamp}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    return logger


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up logging for the backtesting system.
    
    Args:
        verbose: Whether to enable verbose logging
        
    Returns:
        Configured logger
    """
    level = logging.DEBUG if verbose else logging.INFO
    log_dir = 'output/logs'
    return setup_logger(level=level, log_dir=log_dir) 