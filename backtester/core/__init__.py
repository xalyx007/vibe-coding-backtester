"""
Core module for the backtesting system.

This module contains the central components of the backtesting system,
including the main engine and configuration management.
"""

from backtester.core.engine import Backtester
from backtester.core.results import BacktestResults
from backtester.core.config import BacktesterConfig

__all__ = ['Backtester', 'BacktestResults', 'BacktesterConfig'] 