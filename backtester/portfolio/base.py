"""
Base class for portfolio managers in the backtesting system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

from backtester.utils.constants import SignalType
from backtester.events.event_bus import EventBus


class PortfolioManager(ABC):
    """
    Abstract base class for all portfolio managers.
    
    This class defines the interface that all portfolio managers must implement.
    Portfolio managers are responsible for managing positions and trades,
    decoupled from strategy and backtesting logic.
    """
    
    def __init__(self, initial_capital: float, event_bus: Optional[EventBus] = None):
        """
        Initialize the portfolio manager.
        
        Args:
            initial_capital: Initial capital for the portfolio
            event_bus: Optional event bus for emitting events
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.holdings = {}  # Symbol -> quantity
        self.trades = []
        self.event_bus = event_bus
    
    @abstractmethod
    def update_position(self, timestamp, symbol: str, signal_type: SignalType, 
                       price: float, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update portfolio positions based on a signal.
        
        Args:
            timestamp: Timestamp of the signal
            symbol: Trading symbol (e.g., 'BTC-USD')
            signal_type: Type of signal (BUY, SELL, HOLD)
            price: Current price of the asset
            metadata: Additional information about the signal
            
        Returns:
            Dictionary with details of the executed trade
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, price: float, 
                               signal_type: SignalType) -> float:
        """
        Calculate the position size for a trade.
        
        Args:
            symbol: Trading symbol
            price: Current price of the asset
            signal_type: Type of signal
            
        Returns:
            Quantity to trade
        """
        pass
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate the current portfolio value.
        
        Args:
            prices: Dictionary mapping symbols to current prices
            
        Returns:
            Total portfolio value (cash + holdings)
        """
        holdings_value = sum(self.holdings.get(symbol, 0) * prices.get(symbol, 0) 
                            for symbol in self.holdings)
        return self.current_capital + holdings_value
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get the portfolio history.
        
        Returns:
            DataFrame with portfolio history
        """
        return pd.DataFrame(self.trades)
    
    def _emit_trade_event(self, trade_details: Dict[str, Any]):
        """
        Emit a trade event if an event bus is available.
        
        Args:
            trade_details: Details of the executed trade
        """
        if self.event_bus:
            self.event_bus.emit("trade_executed", trade_details) 