"""
Portfolio Manager Module for the Backtesting System.

This module is responsible for managing positions and trades,
decoupled from strategy and backtesting logic.
"""

from backtester.portfolio.base import PortfolioManager
from backtester.portfolio.basic import BasicPortfolioManager
from backtester.portfolio.position_sizing import FixedAmountSizer, PercentageSizer

__all__ = [
    'PortfolioManager',
    'BasicPortfolioManager',
    'FixedAmountSizer',
    'PercentageSizer',
] 