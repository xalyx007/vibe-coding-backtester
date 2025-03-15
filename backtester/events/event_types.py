"""
Event types for the backtesting system.
"""

from enum import Enum, auto
from typing import Dict, Any, Optional
from datetime import datetime


class EventType(Enum):
    """Types of events that can be emitted by the backtesting system."""
    DATA_LOADED = "data_loaded"
    SIGNAL_GENERATED = "signal_generated"
    TRADE_EXECUTED = "trade_executed"
    BACKTEST_STARTED = "backtest_started"
    BACKTEST_COMPLETED = "backtest_completed"
    PORTFOLIO_UPDATED = "portfolio_updated"
    ERROR = "error"
    CONFIG_UPDATED = "config_updated"


class Event:
    """
    Class representing an event in the backtesting system.
    
    This class provides a structured way to represent events
    that can be emitted by the backtesting system.
    """
    
    def __init__(self, 
                event_type: EventType, 
                data: Dict[str, Any],
                timestamp: Optional[datetime] = None):
        """
        Initialize the event.
        
        Args:
            event_type: Type of event
            data: Data associated with the event
            timestamp: Optional timestamp (defaults to current time)
        """
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a dictionary.
        
        Returns:
            Dictionary representation of the event
        """
        return {
            "type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, event_dict: Dict[str, Any]) -> 'Event':
        """
        Create an event from a dictionary.
        
        Args:
            event_dict: Dictionary representation of the event
            
        Returns:
            Event object
        """
        event_type = EventType(event_dict["type"])
        data = event_dict["data"]
        timestamp = datetime.fromisoformat(event_dict["timestamp"])
        
        return cls(event_type, data, timestamp) 