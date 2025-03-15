#!/usr/bin/env python
"""
Download Bitcoin Data from Yahoo Finance

This script downloads Bitcoin data from Yahoo Finance and saves it to CSV files.
It doesn't depend on the backtester package, so it can be run independently.
"""

import os
import sys
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_OUTPUT_DIR = "data/yahoo"
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_INTERVALS = ["1d", "1wk", "1mo"]


def download_btc_data(
    start_date=DEFAULT_START_DATE,
    end_date=DEFAULT_END_DATE,
    intervals=DEFAULT_INTERVALS,
    output_dir=DEFAULT_OUTPUT_DIR
):
    """
    Download Bitcoin data for multiple intervals.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    intervals : list
        List of intervals to download.
    output_dir : str
        Directory to save the downloaded data.
        
    Returns
    -------
    dict
        Dictionary of DataFrames containing the downloaded data.
    """
    btc_data = {}
    
    for interval in intervals:
        logger.info(f"Downloading BTC data with interval {interval}")
        
        # Create output directory
        interval_dir = os.path.join(output_dir, interval)
        os.makedirs(interval_dir, exist_ok=True)
        
        # Define output file path
        output_file = os.path.join(interval_dir, "BTC_USD.csv")
        
        try:
            # Download data from Yahoo Finance
            ticker = yf.Ticker("BTC-USD")
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            # Check if data is empty
            if data.empty:
                logger.warning(f"No data available for BTC-USD with interval {interval}")
                continue
                
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns to match our standard format
            data = data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add symbol column
            data['symbol'] = "BTC-USD"
            
            # Convert date to string format
            data['date'] = data['date'].dt.strftime('%Y-%m-%d')
            
            # Select and reorder columns
            data = data[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            # Save to CSV
            data.to_csv(output_file, index=False)
            logger.info(f"Saved data to {output_file}")
            
            # Store in dictionary
            btc_data[interval] = data
            
        except Exception as e:
            logger.error(f"Error downloading data for BTC-USD with interval {interval}: {e}")
    
    return btc_data


def main():
    """Main function to parse arguments and download data."""
    parser = argparse.ArgumentParser(description="Download Bitcoin data from Yahoo Finance")
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Start date in YYYY-MM-DD format (default: %(default)s)"
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="End date in YYYY-MM-DD format (default: %(default)s)"
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=DEFAULT_INTERVALS,
        help="Data intervals to download (default: %(default)s)"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the downloaded data (default: %(default)s)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Downloading Bitcoin data from {args.start_date} to {args.end_date}")
    download_btc_data(
        start_date=args.start_date,
        end_date=args.end_date,
        intervals=args.intervals,
        output_dir=args.output_dir
    )
    
    logger.info("Download completed successfully")


if __name__ == "__main__":
    main() 