"""
Synthetic data source for backtesting.

This module provides a data source that generates synthetic price data
for testing and validation purposes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from backtester.data.base import DataSource
from backtester.events.event_bus import EventBus


class SyntheticDataSource(DataSource):
    """
    A data source that generates synthetic price data for testing and validation.
    
    This data source generates synthetic price data with specified trend and volatility
    characteristics. It's useful for testing and validation purposes when real market
    data is not available or when specific price patterns are needed.
    
    Attributes:
        start_date (str): The start date for the synthetic data.
        end_date (str): The end date for the synthetic data.
        symbol (str): The symbol for the synthetic data.
        trend (float): The annualized trend (drift) of the price series.
        volatility (float): The annualized volatility of the price series.
        initial_price (float): The initial price of the series.
        trading_days (int): The number of trading days per year.
        seed (Optional[int]): Random seed for reproducibility.
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        symbol: str = "SYNTHETIC",
        trend: float = 0.05,
        volatility: float = 0.2,
        initial_price: float = 100.0,
        trading_days: int = 252,
        seed: Optional[int] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the synthetic data source.
        
        Args:
            start_date: The start date for the synthetic data (format: 'YYYY-MM-DD').
            end_date: The end date for the synthetic data (format: 'YYYY-MM-DD').
            symbol: The symbol for the synthetic data.
            trend: The annualized trend (drift) of the price series.
            volatility: The annualized volatility of the price series.
            initial_price: The initial price of the series.
            trading_days: The number of trading days per year.
            seed: Random seed for reproducibility.
            event_bus: Optional event bus for emitting events.
        """
        super().__init__(event_bus=event_bus)
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol
        self.trend = trend
        self.volatility = volatility
        self.initial_price = initial_price
        self.trading_days = trading_days
        self.seed = seed
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load synthetic data.
        
        This method generates synthetic price data based on the parameters
        specified during initialization.
        
        Args:
            **kwargs: Additional parameters (not used).
            
        Returns:
            DataFrame containing the synthetic data.
        """
        # Parse dates
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        
        # Calculate number of days
        days = (end_date - start_date).days + 1
        
        # Generate business days (Monday to Friday)
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday to Friday
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Calculate daily parameters
        daily_trend = self.trend / self.trading_days
        daily_volatility = self.volatility / np.sqrt(self.trading_days)
        
        # Generate random returns
        n = len(dates)
        returns = np.random.normal(daily_trend, daily_volatility, n)
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate prices
        prices = self.initial_price * cumulative_returns
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, daily_volatility / 2, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, daily_volatility / 2, n))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n).astype(int),
            'symbol': self.symbol
        })
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        return df
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the synthetic data.
        
        For synthetic data, no preprocessing is needed as the data is already
        in the correct format.
        
        Args:
            data: The synthetic data to preprocess.
            
        Returns:
            The preprocessed data (unchanged for synthetic data).
        """
        # No preprocessing needed for synthetic data
        return data
    
    def get_symbols(self) -> list:
        """
        Get the list of symbols available in this data source.
        
        Returns:
            A list containing the symbol for the synthetic data.
        """
        return [self.symbol]
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the synthetic data source.
        
        Returns:
            A dictionary containing metadata about the synthetic data source.
        """
        return {
            'type': 'synthetic',
            'start_date': self.start_date,
            'end_date': self.end_date,
            'symbol': self.symbol,
            'trend': self.trend,
            'volatility': self.volatility,
            'initial_price': self.initial_price,
            'trading_days': self.trading_days,
            'seed': self.seed
        } 