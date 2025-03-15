"""
Exchange Data Source for the Backtesting System.

This module provides functionality to load and process data from cryptocurrency exchanges.
"""

import pandas as pd
import ccxt
from backtester.data.base import DataSource

class ExchangeDataSource(DataSource):
    """
    Data source that loads market data from a cryptocurrency exchange.
    
    Attributes:
        exchange_id (str): ID of the exchange to use.
        symbol (str): Trading pair symbol.
        timeframe (str): Timeframe for the data.
        start_date (str): Start date for the data.
        end_date (str): End date for the data.
        exchange (ccxt.Exchange): The exchange instance.
        data (pd.DataFrame): The loaded data.
    """
    
    def __init__(self, exchange_id='binance', symbol='BTC/USDT', timeframe='1d', 
                 start_date=None, end_date=None, **kwargs):
        """
        Initialize the exchange data source.
        
        Args:
            exchange_id (str): ID of the exchange to use.
            symbol (str): Trading pair symbol.
            timeframe (str): Timeframe for the data.
            start_date (str): Start date for the data.
            end_date (str): End date for the data.
            **kwargs: Additional arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.exchange = getattr(ccxt, exchange_id)()
        
    def load_data(self, **kwargs):
        """
        Load data from the exchange.
        
        Args:
            **kwargs: Additional arguments to pass to the exchange API.
            
        Returns:
            pd.DataFrame: The loaded data.
        """
        # Convert dates to timestamps if provided
        since = None
        if self.start_date:
            since = pd.Timestamp(self.start_date).timestamp() * 1000
            
        # Fetch OHLCV data
        ohlcv = self.exchange.fetch_ohlcv(
            symbol=self.symbol,
            timeframe=self.timeframe,
            since=since,
            **kwargs
        )
        
        # Convert to DataFrame
        data = pd.DataFrame(
            ohlcv, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime and set as index
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the loaded data.
        
        Args:
            data (pd.DataFrame): Raw data to preprocess.
            
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        # Make a copy to avoid modifying the original
        processed_data = data.copy()
        
        # Filter by end date if provided
        if self.end_date:
            end_timestamp = pd.Timestamp(self.end_date)
            processed_data = processed_data[processed_data.index <= end_timestamp]
        
        # Handle missing values
        processed_data = processed_data.fillna(method='ffill')
        
        # Sort by date if not already sorted
        if not processed_data.index.is_monotonic_increasing:
            processed_data = processed_data.sort_index()
        
        return processed_data
    
    def load(self):
        """
        Load and preprocess data from the exchange.
        
        Returns:
            pd.DataFrame: The loaded and preprocessed data.
        """
        self.data = self.load_data()
        self.data = self.preprocess_data(self.data)
        return self.data
    
    def get_latest_price(self):
        """
        Get the latest price from the data.
        
        Returns:
            float: The latest price.
        """
        if self.data is None:
            self.load()
        return self.data['close'].iloc[-1]
    
    def get_latest_volume(self):
        """
        Get the latest volume from the data.
        
        Returns:
            float: The latest volume.
        """
        if self.data is None:
            self.load()
        return self.data['volume'].iloc[-1] 