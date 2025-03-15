"""
Base class for trading strategies in the backtesting system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

from backtester.utils.constants import SignalType
from backtester.events.event_bus import EventBus


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all strategies must implement.
    Strategies are responsible for generating buy/sell signals based on
    market data, fully decoupled from backtesting logic.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the strategy.
        
        Args:
            event_bus: Optional event bus for emitting events
        """
        self.event_bus = event_bus
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the input data.
        
        Args:
            data: Market data to analyze
            
        Returns:
            DataFrame with signals (BUY, SELL, HOLD) for each timestamp
        """
        pass
    
    def _emit_signal_event(self, timestamp, signal_type: SignalType, metadata: Dict[str, Any] = None):
        """
        Emit a signal event if an event bus is available.
        
        Args:
            timestamp: Timestamp of the signal
            signal_type: Type of signal (BUY, SELL, HOLD)
            metadata: Additional information about the signal
        """
        if self.event_bus:
            self.event_bus.emit("signal_generated", {
                "timestamp": timestamp,
                "signal_type": signal_type.name,
                "metadata": metadata or {}
            })
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the strategy parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        # Default implementation returns all public attributes
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and k != 'event_bus'} 