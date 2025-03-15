"""
Events Module for the Backtesting System.

This module facilitates event-based communication between the backend
and a separate frontend, as well as between internal modules.
"""

from backtester.events.event_bus import EventBus
from backtester.events.event_types import EventType, Event

__all__ = [
    'EventBus',
    'EventType',
    'Event',
] 