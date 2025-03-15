"""
Main backtester engine for the backtesting system.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

from backtester.data.base import DataSource
from backtester.strategy.base import Strategy
from backtester.portfolio.base import PortfolioManager
from backtester.utils.constants import SignalType
from backtester.events.event_bus import EventBus
from backtester.core.results import BacktestResults


class Backtester:
    """
    Main class for backtesting trading strategies.
    
    This class coordinates the backtesting process by connecting the data source,
    strategy, and portfolio manager modules.
    """
    
    def __init__(self, 
                data_source: DataSource, 
                strategy: Strategy, 
                portfolio_manager: PortfolioManager,
                event_bus: Optional[EventBus] = None,
                transaction_costs: float = 0.001,  # 0.1% by default
                slippage: float = 0.0005):  # 0.05% by default
        """
        Initialize the backtester.
        
        Args:
            data_source: Data source for market data
            strategy: Strategy for generating signals
            portfolio_manager: Portfolio manager for executing trades
            event_bus: Optional event bus for emitting events
            transaction_costs: Transaction costs as a fraction of trade value
            slippage: Slippage as a fraction of price
        """
        self.data_source = data_source
        self.strategy = strategy
        self.portfolio_manager = portfolio_manager
        self.event_bus = event_bus
        self.transaction_costs = transaction_costs
        self.slippage = slippage
        
        # Connect event bus to all components if provided
        if event_bus:
            self.data_source.event_bus = event_bus
            self.strategy.event_bus = event_bus
            self.portfolio_manager.event_bus = event_bus
    
    def run(self, 
           start_date: Optional[str] = None, 
           end_date: Optional[str] = None,
           symbol: str = "BTC-USD") -> BacktestResults:
        """
        Run the backtest.
        
        Args:
            start_date: Optional start date for the backtest
            end_date: Optional end date for the backtest
            symbol: Trading symbol
            
        Returns:
            BacktestResults object with the results of the backtest
        """
        # Emit backtest start event
        if self.event_bus:
            self.event_bus.emit("backtest_started", {
                "start_date": start_date,
                "end_date": end_date,
                "symbol": symbol,
                "strategy": self.strategy.__class__.__name__,
                "portfolio_manager": self.portfolio_manager.__class__.__name__
            })
        
        # Get data
        data = self.data_source.get_data(start_date=start_date, end_date=end_date)
        
        # Generate signals
        signals = self.strategy.generate_signals(data)
        
        # Execute trades
        portfolio_values = []
        prices = {}
        
        for idx, row in signals.iterrows():
            # Get current price
            price = row['close']
            prices[symbol] = price
            
            # Apply slippage to price based on signal direction
            if row['signal'] == SignalType.BUY:
                execution_price = price * (1 + self.slippage)
            elif row['signal'] == SignalType.SELL:
                execution_price = price * (1 - self.slippage)
            else:
                execution_price = price
            
            # Update portfolio
            trade_details = self.portfolio_manager.update_position(
                timestamp=idx,
                symbol=symbol,
                signal_type=row['signal'],
                price=execution_price,
                metadata={"transaction_costs": self.transaction_costs}
            )
            
            # Record portfolio value
            portfolio_value = self.portfolio_manager.get_portfolio_value(prices)
            portfolio_values.append({
                "timestamp": idx,
                "portfolio_value": portfolio_value
            })
        
        # Create results object
        results = BacktestResults(
            portfolio_values=pd.DataFrame(portfolio_values),
            trades=self.portfolio_manager.get_portfolio_history(),
            signals=signals,
            strategy_parameters=self.strategy.get_parameters(),
            initial_capital=self.portfolio_manager.initial_capital,
            symbol=symbol
        )
        
        # Emit backtest completed event
        if self.event_bus:
            self.event_bus.emit("backtest_completed", {
                "final_portfolio_value": results.portfolio_values["portfolio_value"].iloc[-1],
                "total_return_pct": results.total_return * 100,
                "num_trades": len(results.trades),
                "strategy": self.strategy.__class__.__name__
            })
        
        return results 