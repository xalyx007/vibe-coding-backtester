"""
Data module for the backtester.

This module provides data sources for the backtester.
"""

from backtester.data.base import DataSource
from backtester.data.csv_source import CSVDataSource
from backtester.data.excel_source import ExcelDataSource
from backtester.data.exchange_source import ExchangeDataSource
from backtester.data.synthetic import SyntheticDataSource

__all__ = [
    'DataSource',
    'CSVDataSource',
    'ExcelDataSource',
    'ExchangeDataSource',
    'SyntheticDataSource'
] 