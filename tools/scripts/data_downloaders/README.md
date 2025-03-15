# Data Downloaders

This directory contains scripts for downloading market data from various sources for use with the Modular Backtesting System.

## Available Scripts

- `download_yahoo_crypto.py`: Downloads cryptocurrency data from Yahoo Finance

## Usage

### Download Cryptocurrency Data from Yahoo Finance

To download cryptocurrency data from Yahoo Finance, run:

```bash
# Install required dependencies
pip install yfinance pandas

# Download default cryptocurrencies (BTC, ETH, XRP, LTC, BCH) with daily interval
python scripts/data_downloaders/download_yahoo_crypto.py

# Download specific cryptocurrencies
python scripts/data_downloaders/download_yahoo_crypto.py --symbols BTC-USD ETH-USD

# Download data for a specific date range
python scripts/data_downloaders/download_yahoo_crypto.py --start-date 2020-01-01 --end-date 2021-12-31

# Download data with a specific interval (1d, 1wk, 1mo)
python scripts/data_downloaders/download_yahoo_crypto.py --interval 1wk

# Download only Bitcoin data with multiple intervals (daily, weekly, monthly)
python scripts/data_downloaders/download_yahoo_crypto.py --btc-only

# Specify output directory
python scripts/data_downloaders/download_yahoo_crypto.py --output-dir data/custom_dir
```

The downloaded data will be saved to CSV files in the specified output directory (default: `data/yahoo`).

## Data Format

The downloaded data follows the standard format used by the Modular Backtesting System:

- `date`: The date of the data point (format: YYYY-MM-DD)
- `symbol`: The symbol of the cryptocurrency
- `open`: The opening price
- `high`: The highest price
- `low`: The lowest price
- `close`: The closing price
- `volume`: The trading volume

## Adding New Data Sources

To add a new data source, create a new script in this directory following these guidelines:

1. Use a descriptive name for the script (e.g., `download_binance.py`)
2. Include a docstring at the top of the file explaining the purpose of the script
3. Use command-line arguments to make the script configurable
4. Include error handling to gracefully handle exceptions
5. Add logging to track the progress of the script
6. Save the downloaded data in the standard format used by the Modular Backtesting System
7. Update this README with usage instructions for the new script 