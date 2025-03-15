#!/usr/bin/env python
"""
Generate sample data for the Modular Backtesting System examples.

This script generates synthetic OHLCV data for use with the examples.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Create the data directory if it doesn't exist
os.makedirs("data/sample", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def generate_daily_data(symbol, start_date, end_date, initial_price=100.0, volatility=0.01):
    """
    Generate daily OHLCV data for a single asset.
    
    Parameters
    ----------
    symbol : str
        The symbol of the asset.
    start_date : datetime
        The start date of the data.
    end_date : datetime
        The end date of the data.
    initial_price : float, optional
        The initial price of the asset. Default is 100.0.
    volatility : float, optional
        The volatility of the asset. Default is 0.01.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the generated data.
    """
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate returns
    returns = np.random.normal(0, volatility, len(dates))
    
    # Generate prices
    prices = initial_price * (1 + returns).cumprod()
    
    # Generate OHLCV data
    data = []
    for i, date in enumerate(dates):
        price = prices[i]
        open_price = price * (1 + np.random.normal(0, volatility / 2))
        high_price = max(price, open_price) * (1 + abs(np.random.normal(0, volatility)))
        low_price = min(price, open_price) * (1 - abs(np.random.normal(0, volatility)))
        close_price = price
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def generate_intraday_data(symbol, start_date, end_date, initial_price=100.0, volatility=0.005):
    """
    Generate intraday OHLCV data for a single asset.
    
    Parameters
    ----------
    symbol : str
        The symbol of the asset.
    start_date : datetime
        The start date of the data.
    end_date : datetime
        The end date of the data.
    initial_price : float, optional
        The initial price of the asset. Default is 100.0.
    volatility : float, optional
        The volatility of the asset. Default is 0.005.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the generated data.
    """
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Generate returns
    returns = np.random.normal(0, volatility, len(dates))
    
    # Generate prices
    prices = initial_price * (1 + returns).cumprod()
    
    # Generate OHLCV data
    data = []
    for i, date in enumerate(dates):
        # Skip non-trading hours (outside 9:00-17:00)
        if date.hour < 9 or date.hour >= 17:
            continue
            
        price = prices[i]
        open_price = price * (1 + np.random.normal(0, volatility / 2))
        high_price = max(price, open_price) * (1 + abs(np.random.normal(0, volatility)))
        low_price = min(price, open_price) * (1 - abs(np.random.normal(0, volatility)))
        close_price = price
        volume = np.random.randint(10000, 100000)
        
        data.append({
            'date': date.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def main():
    """Generate sample data files."""
    # Set date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    
    # Generate daily data for a single asset
    print("Generating sample_data.csv...")
    daily_data = generate_daily_data('SAMPLE', start_date, end_date)
    daily_data.to_csv('data/sample/sample_data.csv', index=False)
    
    # Generate daily data for multiple assets
    print("Generating multiple_assets.csv...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
    multiple_assets_data = pd.DataFrame()
    for symbol in symbols:
        symbol_data = generate_daily_data(
            symbol, 
            start_date, 
            end_date, 
            initial_price=np.random.randint(50, 200),
            volatility=np.random.uniform(0.005, 0.02)
        )
        multiple_assets_data = pd.concat([multiple_assets_data, symbol_data])
    
    multiple_assets_data.to_csv('data/sample/multiple_assets.csv', index=False)
    
    # Generate intraday data
    print("Generating intraday_data.csv...")
    intraday_start_date = datetime(2020, 1, 1)
    intraday_end_date = datetime(2020, 1, 10)
    intraday_data = generate_intraday_data('SAMPLE', intraday_start_date, intraday_end_date)
    intraday_data.to_csv('data/sample/intraday_data.csv', index=False)
    
    print("Sample data generation complete.")

if __name__ == '__main__':
    main() 