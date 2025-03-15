# Yahoo Finance Cryptocurrency Data

This directory contains cryptocurrency data downloaded from Yahoo Finance using the `download_yahoo_crypto.py` script.

## Data Structure

The data is organized as follows:

- `1d/`: Daily data
  - `BTC_USD.csv`: Bitcoin daily data
  - `ETH_USD.csv`: Ethereum daily data
  - ...
- `1wk/`: Weekly data
  - `BTC_USD.csv`: Bitcoin weekly data
  - ...
- `1mo/`: Monthly data
  - `BTC_USD.csv`: Bitcoin monthly data
  - ...

## Data Format

Each CSV file contains the following columns:

- `date`: The date of the data point (format: YYYY-MM-DD)
- `symbol`: The symbol of the cryptocurrency
- `open`: The opening price
- `high`: The highest price
- `low`: The lowest price
- `close`: The closing price
- `volume`: The trading volume

## Downloading Data

To download cryptocurrency data from Yahoo Finance, run:

```bash
# Install required dependencies
pip install yfinance pandas

# Download default cryptocurrencies (BTC, ETH, XRP, LTC, BCH) with daily interval
python scripts/data_downloaders/download_yahoo_crypto.py

# Download only Bitcoin data with multiple intervals (daily, weekly, monthly)
python scripts/data_downloaders/download_yahoo_crypto.py --btc-only
```

See the [Data Downloaders README](../../scripts/data_downloaders/README.md) for more options.

## Usage in Backtests

To use this data in your backtests, you can create a `CSVDataSource` object:

```python
from backtester.data import CSVDataSource

# Load daily Bitcoin data
data_source = CSVDataSource("data/yahoo/1d/BTC_USD.csv")

# Load weekly Bitcoin data
weekly_data_source = CSVDataSource("data/yahoo/1wk/BTC_USD.csv")
```

## Data Source

The data is sourced from [Yahoo Finance](https://finance.yahoo.com/), which provides free historical market data for various assets, including cryptocurrencies.

## Data Limitations

Please note the following limitations of Yahoo Finance data:

1. The data may not be as accurate as data from cryptocurrency exchanges
2. The data may have gaps or inconsistencies
3. The data is delayed and not suitable for real-time trading
4. The data may not include all cryptocurrencies

For more accurate backtesting, consider using data directly from cryptocurrency exchanges. 