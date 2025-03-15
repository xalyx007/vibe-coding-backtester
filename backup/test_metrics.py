#!/usr/bin/env python
"""
Test script for metrics calculation functionality.

This script tests the metrics calculation functionality in the backtester.validation module.
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime

# Add the parent directory to the path so we can import the backtester module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_metrics")

# Import backtester modules
from backtester.validation import calculate_metrics

def test_metrics():
    """Test the metrics calculation functionality."""
    logger.info("Testing metrics calculation functionality...")
    
    try:
        # Create sample data
        returns = np.random.normal(0.001, 0.01, 100).tolist()
        positions = [1 if r > 0 else -1 if r < 0 else 0 for r in returns]
        trades = [(1, 100, "2020-01-01"), (-1, 100, "2020-01-05"), (1, 200, "2020-01-10")]
        
        # Calculate metrics
        metrics = calculate_metrics(
            returns=returns,
            positions=positions,
            trades=trades,
            risk_free_rate=0.0,
            annualization_factor=252
        )
        
        # Print metrics
        logger.info("Metrics calculation results:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
        logger.info("Metrics calculation test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during metrics calculation test: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_metrics()
    sys.exit(0 if success else 1) 