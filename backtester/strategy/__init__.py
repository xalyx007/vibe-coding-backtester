"""
Strategy Module for the Backtesting System.

This module is responsible for generating buy/sell signals based on
input data, fully decoupled from backtesting logic.
"""

from backtester.strategy.base import Strategy
from backtester.strategy.moving_average import MovingAverageCrossover
from backtester.strategy.rsi import RSIStrategy
from backtester.strategy.bollinger_bands import BollingerBandsStrategy
from backtester.strategy.ml_strategy import MLStrategy
from backtester.strategy.ensemble import StrategyEnsemble
from backtester.strategy.combined_strategy import CombinedStrategy

__all__ = [
    'Strategy',
    'MovingAverageCrossover',
    'RSIStrategy',
    'BollingerBandsStrategy',
    'MLStrategy',
    'StrategyEnsemble',
    'CombinedStrategy',
] 