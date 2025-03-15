#!/usr/bin/env python
"""
Unit tests for the core engine module.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtester.core.engine import Backtester
from backtester.core.results import BacktestResults
from backtester.data.base import DataSource
from backtester.strategy.base import Strategy
from backtester.portfolio.base import PortfolioManager
from backtester.events.event_bus import EventBus
from backtester.utils.constants import SignalType


class TestBacktester(unittest.TestCase):
    """Test cases for the Backtester class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects for dependencies
        self.data_source = MagicMock(spec=DataSource)
        self.strategy = MagicMock(spec=Strategy)
        self.portfolio_manager = MagicMock(spec=PortfolioManager)
        self.portfolio_manager.initial_capital = 10000.0
        self.event_bus = MagicMock(spec=EventBus)
        
        # Set up sample data
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 10))
        
        self.sample_data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(100, 1000, 10)
        }, index=dates)
        
        # Configure mocks
        self.data_source.get_data.return_value = self.sample_data
        
        # Create signals DataFrame
        dates = self.sample_data.index
        signals_df = pd.DataFrame({
            'signal': [SignalType.BUY, SignalType.HOLD, SignalType.HOLD, SignalType.SELL,
                      SignalType.HOLD, SignalType.BUY, SignalType.HOLD, SignalType.HOLD,
                      SignalType.SELL, SignalType.HOLD],
            'close': self.sample_data['close'].values
        }, index=dates)
        
        self.strategy.generate_signals.return_value = signals_df
        
        # Create a portfolio history DataFrame
        portfolio_history = pd.DataFrame({
            'portfolio_value': 10000 * np.cumprod(1 + np.random.normal(0.001, 0.005, 10)),
            'cash': 5000 * np.ones(10),
            'holdings': 5000 * np.cumprod(1 + np.random.normal(0.002, 0.01, 10))
        }, index=dates)
        
        self.portfolio_manager.get_portfolio_history.return_value = portfolio_history
    
    def test_initialization(self):
        """Test that the Backtester initializes correctly."""
        backtester = Backtester(
            data_source=self.data_source,
            strategy=self.strategy,
            portfolio_manager=self.portfolio_manager,
            event_bus=self.event_bus,
            transaction_costs=0.002,
            slippage=0.001
        )
        
        self.assertEqual(backtester.data_source, self.data_source)
        self.assertEqual(backtester.strategy, self.strategy)
        self.assertEqual(backtester.portfolio_manager, self.portfolio_manager)
        self.assertEqual(backtester.event_bus, self.event_bus)
        self.assertEqual(backtester.transaction_costs, 0.002)
        self.assertEqual(backtester.slippage, 0.001)
    
    @patch('backtester.core.engine.BacktestResults')
    def test_run(self, mock_results_class):
        """Test the run method of the Backtester."""
        # Configure mock results
        mock_results = MagicMock()
        mock_results_class.return_value = mock_results
        
        backtester = Backtester(
            data_source=self.data_source,
            strategy=self.strategy,
            portfolio_manager=self.portfolio_manager
        )
        
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 10)
        symbol = 'BTC-USD'
        
        results = backtester.run(start_date, end_date, symbol)
        
        # Verify that the data source was called with the correct parameters
        self.data_source.get_data.assert_called_once_with(start_date=start_date, end_date=end_date)
        
        # Verify that the strategy was called with the correct parameters
        self.strategy.generate_signals.assert_called_once()
        
        # Verify that the portfolio manager was called for each signal
        self.assertEqual(self.portfolio_manager.update_position.call_count, 10)
        
        # Verify that the results are of the correct type
        self.assertEqual(results, mock_results)
        
        # Verify that BacktestResults was called with the correct parameters
        mock_results_class.assert_called_once()
    
    @patch('backtester.core.engine.BacktestResults')
    def test_run_with_default_dates(self, mock_results_class):
        """Test the run method with default dates."""
        # Configure mock results
        mock_results = MagicMock()
        mock_results_class.return_value = mock_results
        
        backtester = Backtester(
            data_source=self.data_source,
            strategy=self.strategy,
            portfolio_manager=self.portfolio_manager
        )
        
        symbol = 'BTC-USD'
        
        results = backtester.run(None, None, symbol)
        
        # Verify that the data source was called with None for dates
        self.data_source.get_data.assert_called_once_with(start_date=None, end_date=None)
        
        # Verify that the strategy was called
        self.strategy.generate_signals.assert_called_once()
        
        # Verify that the portfolio manager was called for each signal
        self.assertEqual(self.portfolio_manager.update_position.call_count, 10)
        
        # Verify that the results are of the correct type
        self.assertEqual(results, mock_results)
        
        # Verify that BacktestResults was called with the correct parameters
        mock_results_class.assert_called_once()
    
    @patch('backtester.core.engine.BacktestResults')
    def test_run_with_event_bus(self, mock_results_class):
        """Test the run method with an event bus."""
        # Configure mock results
        mock_results = MagicMock()
        mock_results_class.return_value = mock_results
        
        backtester = Backtester(
            data_source=self.data_source,
            strategy=self.strategy,
            portfolio_manager=self.portfolio_manager,
            event_bus=self.event_bus
        )
        
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 10)
        symbol = 'BTC-USD'
        
        results = backtester.run(start_date, end_date, symbol)
        
        # Verify that events were emitted
        self.assertTrue(self.event_bus.emit.called)
        
        # Verify that the results are of the correct type
        self.assertEqual(results, mock_results)
        
        # Verify that BacktestResults was called with the correct parameters
        mock_results_class.assert_called_once()


if __name__ == '__main__':
    unittest.main() 