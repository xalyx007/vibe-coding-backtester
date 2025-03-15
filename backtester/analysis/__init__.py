"""
Analysis Module for the Backtesting System.

This module is responsible for analyzing backtesting results and
generating performance metrics and visualizations.
"""

from backtester.analysis.metrics import calculate_metrics
from backtester.analysis.visualization import plot_equity_curve, plot_drawdown, plot_trades

__all__ = [
    'calculate_metrics',
    'plot_equity_curve',
    'plot_drawdown',
    'plot_trades',
] 