#!/usr/bin/env python
"""
Download cryptocurrency data from Yahoo Finance.

This script downloads historical OHLCV data for cryptocurrencies from Yahoo Finance
and saves it to CSV files in the data directory.
"""

import os
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
DEFAULT_INTERVAL = "1d"  # 1d, 1wk, 1mo
DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD"]


def download_crypto_data(
    symbols,
    start_date,
    end_date,
    interval,
    output_dir
):
    """
    Download cryptocurrency data from Yahoo Finance.
    
    Parameters
    ----------
    symbols : list
        List of cryptocurrency symbols to download.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    interval : str
        Data interval (1d, 1wk, 1mo).
    output_dir : str
        Directory to save the downloaded data.
        
    Returns
    -------
    dict
        Dictionary of DataFrames containing the downloaded data.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    data_dict = {}
    
    for symbol in symbols:
        logger.info(f"Downloading data for {symbol} from {start_date} to {end_date} with interval {interval}")
        
        try:
            # Download data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            # Check if data is empty
            if data.empty:
                logger.warning(f"No data available for {symbol}")
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
            data['symbol'] = symbol
            
            # Convert date to string format
            data['date'] = data['date'].dt.strftime('%Y-%m-%d')
            
            # Select and reorder columns
            data = data[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            # Save to CSV
            output_file = os.path.join(output_dir, f"{symbol.replace('-', '_')}.csv")
            data.to_csv(output_file, index=False)
            logger.info(f"Saved data to {output_file}")
            
            # Store in dictionary
            data_dict[symbol] = data
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
    
    return data_dict


def download_btc_data(
    start_date=DEFAULT_START_DATE,
    end_date=DEFAULT_END_DATE,
    intervals=["1d", "1wk", "1mo"],
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
        interval_dir = os.path.join(output_dir, interval)
        data = download_crypto_data(
            symbols=["BTC-USD"],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            output_dir=interval_dir
        )
        btc_data[interval] = data.get("BTC-USD")
    
    return btc_data


def main():
    """Main function to parse arguments and download data."""
    parser = argparse.ArgumentParser(description="Download cryptocurrency data from Yahoo Finance")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="List of cryptocurrency symbols to download (default: %(default)s)"
    )
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
        "--interval",
        default=DEFAULT_INTERVAL,
        choices=["1d", "1wk", "1mo"],
        help="Data interval (default: %(default)s)"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the downloaded data (default: %(default)s)"
    )
    parser.add_argument(
        "--btc-only",
        action="store_true",
        help="Download only Bitcoin data with multiple intervals"
    )
    
    args = parser.parse_args()
    
    if args.btc_only:
        logger.info("Downloading Bitcoin data with multiple intervals")
        download_btc_data(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir
        )
    else:
        logger.info(f"Downloading data for {args.symbols}")
        download_crypto_data(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            output_dir=args.output_dir
        )
    
    logger.info("Download completed successfully")


if __name__ == "__main__":
    main() 