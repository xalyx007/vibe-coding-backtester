#!/usr/bin/env python
"""
Test script for all validation functionality.

This script runs all the validation tests to ensure that the validation module is working correctly.
"""

import os
import sys
import logging
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
logger = logging.getLogger("test_all_validation")

# Import test functions
from test_cross_validation import test_cross_validation
from test_monte_carlo import test_monte_carlo
from test_walk_forward import test_walk_forward
from test_metrics import test_metrics

def run_all_tests():
    """Run all validation tests."""
    logger.info("Running all validation tests...")
    
    # Run cross-validation test
    logger.info("=== Cross-Validation Test ===")
    cross_val_success = test_cross_validation()
    
    # Run Monte Carlo test
    logger.info("\n=== Monte Carlo Test ===")
    monte_carlo_success = test_monte_carlo()
    
    # Run walk-forward test
    logger.info("\n=== Walk-Forward Test ===")
    walk_forward_success = test_walk_forward()
    
    # Run metrics test
    logger.info("\n=== Metrics Test ===")
    metrics_success = test_metrics()
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Cross-Validation Test: {'PASSED' if cross_val_success else 'FAILED'}")
    logger.info(f"Monte Carlo Test: {'PASSED' if monte_carlo_success else 'FAILED'}")
    logger.info(f"Walk-Forward Test: {'PASSED' if walk_forward_success else 'FAILED'}")
    logger.info(f"Metrics Test: {'PASSED' if metrics_success else 'FAILED'}")
    
    # Overall success
    all_passed = all([cross_val_success, monte_carlo_success, walk_forward_success, metrics_success])
    logger.info(f"\nOverall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 