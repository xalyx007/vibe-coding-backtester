# Sample Data

This directory contains sample data files for use with the Modular Backtesting System.

## Data Files

- `BTC-USD.csv`: Sample Bitcoin price data (not included in the repository)
- Add your own data files here

## Data Format

The backtesting system expects data files to be in CSV format with the following columns:

- `Date`: The timestamp (e.g., "2020-01-01")
- `Open`: The opening price
- `High`: The highest price during the period
- `Low`: The lowest price during the period
- `Close`: The closing price
- `Volume`: The trading volume

Example:

```
Date,Open,High,Low,Close,Volume
2020-01-01,7200.17,7254.33,7174.94,7200.85,500000000
2020-01-02,7200.85,7212.15,6850.36,6976.40,750000000
2020-01-03,6976.40,7413.72,6915.50,7344.88,900000000
```

## Obtaining Data

You can obtain historical price data from various sources:

1. **Yahoo Finance**: Download CSV files from [Yahoo Finance](https://finance.yahoo.com/)
2. **Cryptocurrency Exchanges**: Many exchanges provide historical data through their APIs
3. **Financial Data Providers**: Services like Alpha Vantage, Quandl, or IEX Cloud

## Using Your Own Data

To use your own data with the backtesting system:

1. Ensure your data is in the correct format (see above)
2. Place your CSV file in this directory
3. Update the file path in your backtesting script:

```python
data_source = CSVDataSource(
    file_path="data/YOUR_DATA_FILE.csv",
    date_format="%Y-%m-%d",
    timestamp_column="Date",
    open_column="Open",
    high_column="High",
    low_column="Low",
    close_column="Close",
    volume_column="Volume"
)
```

## Data Preprocessing

If your data requires preprocessing before use, you can:

1. Use pandas to clean and transform your data
2. Create a custom data source class that inherits from `DataSource`
3. Implement the preprocessing logic in the `load_data` method

Example:

```python
from backtester.data import DataSource
import pandas as pd

class CustomDataSource(DataSource):
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load_data(self):
        # Load the data
        data = pd.read_csv(self.file_path)
        
        # Preprocess the data
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Add custom indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        
        return data
```

## Note

Sample data files are not included in the repository to keep it lightweight. You will need to download or generate your own data files. 