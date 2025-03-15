"""
Reporting module for the backtesting system.

This module provides functionality for generating reports from backtest results
in various formats such as HTML, PDF, and JSON.
"""

from backtester.analysis.reporting.html_report import generate_html_report
from backtester.analysis.reporting.pdf_report import generate_pdf_report

__all__ = [
    'generate_html_report',
    'generate_pdf_report'
] 