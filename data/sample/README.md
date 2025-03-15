# Sample Data

This directory contains sample data files for use with the Modular Backtesting System examples.

## Data Files

- `sample_data.csv`: A sample CSV file containing daily OHLCV data for a fictional asset
- `multiple_assets.csv`: A sample CSV file containing daily OHLCV data for multiple assets
- `intraday_data.csv`: A sample CSV file containing intraday OHLCV data for a fictional asset

## Data Format

The sample data files follow a standard format with the following columns:

- `date`: The date of the data point (format: YYYY-MM-DD for daily data, YYYY-MM-DD HH:MM:SS for intraday data)
- `open`: The opening price
- `high`: The highest price
- `low`: The lowest price
- `close`: The closing price
- `volume`: The trading volume

## Usage

To use the sample data files in your backtests, you can create a `CSVDataSource` object:

```python
from backtester.data import CSVDataSource

# Load daily data
data_source = CSVDataSource("data/sample/sample_data.csv")

# Load intraday data
intraday_data_source = CSVDataSource(
    "data/sample/intraday_data.csv",
    date_format="%Y-%m-%d %H:%M:%S"
)

# Load multiple assets
multiple_assets_data_source = CSVDataSource(
    "data/sample/multiple_assets.csv",
    asset_column="symbol"
)
```

## Data Sources

The sample data files are synthetic and are provided for demonstration purposes only. In a real-world scenario, you would use data from sources such as:

- Financial data providers (e.g., Alpha Vantage, Yahoo Finance, IEX Cloud)
- Cryptocurrency exchanges (e.g., Binance, Coinbase, Kraken)
- Proprietary data sources

## Creating Your Own Data Files

You can create your own data files by following the same format as the sample data files. The files should be in CSV format with the columns described above.

If your data is in a different format, you can create a custom data source by inheriting from the `DataSource` class and implementing the required methods. See the [Data Module API Reference](../docs/api/data.md) for more information. 