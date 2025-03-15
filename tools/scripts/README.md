# Scripts

This directory contains utility scripts for the Modular Backtesting System.

## Available Scripts

- `generate_sample_data.py`: Generates synthetic OHLCV data for use with the examples
- `download_and_test_btc.py`: Downloads Bitcoin data from Yahoo Finance and runs a simple backtest
- `data_downloaders/`: Directory containing scripts for downloading market data from various sources
  - `download_yahoo_crypto.py`: Downloads cryptocurrency data from Yahoo Finance

## Usage

### Generate Sample Data

To generate sample data for the examples, run:

```bash
python scripts/generate_sample_data.py
```

This will create the following files in the `data/sample` directory:

- `sample_data.csv`: Daily OHLCV data for a single asset
- `multiple_assets.csv`: Daily OHLCV data for multiple assets
- `intraday_data.csv`: Intraday OHLCV data for a single asset

### Download Cryptocurrency Data

To download cryptocurrency data from Yahoo Finance, run:

```bash
# Install required dependencies
pip install yfinance pandas

# Download default cryptocurrencies (BTC, ETH, XRP, LTC, BCH) with daily interval
python scripts/data_downloaders/download_yahoo_crypto.py

# Download only Bitcoin data with multiple intervals (daily, weekly, monthly)
python scripts/data_downloaders/download_yahoo_crypto.py --btc-only
```

See the [Data Downloaders README](data_downloaders/README.md) for more options.

### Download and Test Bitcoin Data

To download Bitcoin data and run a simple backtest to verify that everything is working correctly, run:

```bash
# Install required dependencies
pip install yfinance pandas matplotlib

# Download Bitcoin data and run a backtest
python scripts/download_and_test_btc.py
```

This script will:
1. Download Bitcoin data from Yahoo Finance
2. Verify that the data is valid
3. Run a simple backtest using the Moving Average Crossover strategy
4. Save the equity curve to `results/btc_test_equity_curve.png`

## Creating Your Own Scripts

You can create your own scripts to automate common tasks. Here are some ideas:

- Data preprocessing scripts
- Batch backtesting scripts
- Parameter optimization scripts
- Results analysis scripts
- Data downloaders for other sources

## Best Practices

When creating scripts, follow these best practices:

1. Include a docstring at the top of the file explaining the purpose of the script
2. Use command-line arguments to make the script configurable
3. Include error handling to gracefully handle exceptions
4. Add logging to track the progress of the script
5. Use a `main()` function to organize the script's logic
6. Make the script executable with `#!/usr/bin/env python` at the top of the file 