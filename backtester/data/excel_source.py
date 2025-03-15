"""
Excel Data Source for the Backtesting System.

This module provides functionality to load and process data from Excel files.
"""

import pandas as pd
from backtester.data.base import DataSource

class ExcelDataSource(DataSource):
    """
    Data source that loads market data from an Excel file.
    
    Attributes:
        filepath (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to load.
        date_column (str): Name of the column containing dates.
        price_column (str): Name of the column containing price data.
        volume_column (str): Name of the column containing volume data.
        data (pd.DataFrame): The loaded data.
    """
    
    def __init__(self, filepath, sheet_name=0, date_column='date', price_column='close', volume_column='volume', **kwargs):
        """
        Initialize the Excel data source.
        
        Args:
            filepath (str): Path to the Excel file.
            sheet_name (str or int): Name or index of the sheet to load.
            date_column (str): Name of the column containing dates.
            price_column (str): Name of the column containing price data.
            volume_column (str): Name of the column containing volume data.
            **kwargs: Additional arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.date_column = date_column
        self.price_column = price_column
        self.volume_column = volume_column
        
    def load_data(self, **kwargs):
        """
        Load data from the Excel file.
        
        Args:
            **kwargs: Additional arguments to pass to pd.read_excel.
            
        Returns:
            pd.DataFrame: The loaded data.
        """
        # Merge kwargs with default parameters
        params = {'sheet_name': self.sheet_name}
        params.update(kwargs)
        
        # Load data from Excel
        data = pd.read_excel(self.filepath, **params)
        
        # Ensure date column is in datetime format
        if self.date_column in data.columns:
            data[self.date_column] = pd.to_datetime(data[self.date_column])
            data.set_index(self.date_column, inplace=True)
        
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
        
        # Ensure all required columns exist
        required_columns = [self.price_column, self.volume_column]
        for col in required_columns:
            if col not in processed_data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Convert column names to lowercase
        processed_data.columns = processed_data.columns.str.lower()
        
        # Handle missing values
        processed_data = processed_data.fillna(method='ffill')
        
        # Sort by date if not already sorted
        if not processed_data.index.is_monotonic_increasing:
            processed_data = processed_data.sort_index()
        
        return processed_data
    
    def load(self):
        """
        Load and preprocess data from the Excel file.
        
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
        return self.data[self.price_column].iloc[-1]
    
    def get_latest_volume(self):
        """
        Get the latest volume from the data.
        
        Returns:
            float: The latest volume.
        """
        if self.data is None:
            self.load()
        return self.data[self.volume_column].iloc[-1] 