"""
Modular Backtesting System for Trading Strategies.

A flexible, modular backend system for backtesting trading strategies
with a focus on cryptocurrencies, extensible to other asset classes.
"""

from backtester.core import Backtester, BacktestResults, BacktesterConfig
from backtester.cli import main

__version__ = "0.1.0"

__all__ = [
    'Backtester',
    'BacktestResults',
    'BacktesterConfig',
    'main'
] 