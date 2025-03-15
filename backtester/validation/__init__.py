"""
Validation module for the backtester.

This module provides functionality for validating trading strategies using various techniques.
"""

from backtester.validation.cross_validation import run_cross_validation
from backtester.validation.monte_carlo import run_monte_carlo
from backtester.validation.walk_forward import run_walk_forward
from backtester.validation.metrics import calculate_metrics

__all__ = [
    'run_cross_validation',
    'run_monte_carlo',
    'run_walk_forward',
    'calculate_metrics'
] 