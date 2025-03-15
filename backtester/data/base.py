"""
Base class for data sources in the backtesting system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import pandas as pd

from backtester.utils.constants import TimeFrame
from backtester.events.event_bus import EventBus


class DataSource(ABC):
    """
    Abstract base class for all data sources.
    
    This class defines the interface that all data sources must implement.
    Data sources are responsible for loading and preprocessing market data
    from various sources (CSV, Excel, Exchange APIs, etc.).
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the data source.
        
        Args:
            event_bus: Optional event bus for emitting events
        """
        self.event_bus = event_bus
        self.data = None
    
    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load data from the source.
        
        Args:
            **kwargs: Source-specific parameters
            
        Returns:
            DataFrame containing the loaded data
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            data: Raw data to preprocess
            
        Returns:
            Preprocessed data
        """
        pass
    
    def get_data(self, 
                timeframe: Optional[TimeFrame] = None, 
                start_date: Optional[str] = None, 
                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get data for the specified timeframe and date range.
        
        Args:
            timeframe: Optional timeframe to resample data to
            start_date: Optional start date for filtering data
            end_date: Optional end date for filtering data
            
        Returns:
            DataFrame containing the requested data
        """
        if self.data is None:
            self.data = self.load_data()
            self.data = self.preprocess_data(self.data)
        
        # Filter by date if specified
        filtered_data = self.data
        if start_date:
            filtered_data = filtered_data[filtered_data.index >= start_date]
        if end_date:
            filtered_data = filtered_data[filtered_data.index <= end_date]
            
        # Resample to the requested timeframe if specified
        if timeframe:
            # Implementation depends on how the data is stored
            # This is a placeholder for actual resampling logic
            pass
            
        # Emit event if event bus is available
        if self.event_bus:
            self.event_bus.emit("data_loaded", {
                "data_shape": filtered_data.shape,
                "start_date": filtered_data.index.min(),
                "end_date": filtered_data.index.max()
            })
            
        return filtered_data 